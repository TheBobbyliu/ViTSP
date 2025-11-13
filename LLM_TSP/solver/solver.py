from __future__ import annotations
import sys
sys.path.append('../../')

import LLM_TSP.proc_logging
import multiprocessing as mp
import time
from queue import Empty, Full
from typing import Iterable, List, Tuple, Union, Any, Dict, Sequence
import argparse
import psutil
import os
from collections import Counter
from LLM_TSP.tsp import TravelingSalesmenProblem, SubTSP
from exact_concorde.exact_concorde import Concorde
# from exact_Gurobi.gurobi_model import GurobiTSPModel
from LLM_TSP.proc_logging import _configure_logging
import logging
from dataclasses import dataclass, asdict
import numpy as np                      
from LLM_TSP.llm import LLMRecorder


__all__ = [
    "evaluate_best_gain",
    "subproblem_scale_below_threshold",
    "non_overlapping",
    "subproblem_overlapping",
    "process_subTSP",
    # "solver_master",
    "reformulate_and_solve_subTSP",
]


Coordinate = Tuple[int, int, int, int]

@dataclass
class GlobalObjRecord:
    """
    One snapshot of the running objective trajectory produced
    by the solver / LLM pipeline.
    """
    # seconds since the true experiment start
    latency   : float
    # objective value *after* applying the sub‑solution
    new_obj : int
    # rectangular regions that defined the sub‑problem
    coords  : List[Coordinate]
    # |removed_nodes|
    num_nodes_removed : int
    # which LLM (or algo) proposed this sub‑problem
    llm_mode : Any
    # objective value *before* the sub‑problem was solved
    global_solution_version : int
    # which subprocess
    process_name: str = ""

class SubTSPTask:
    def __init__(self, id, current_route, removed_nodes, route_segments, coordinates, parent=False, child=False):
        self.id = id
        self.removed_nodes = removed_nodes  # Ensure uniqueness
        self.num_removed_nodes = len(removed_nodes)
        self.route_segments = route_segments
        self.coordinates = coordinates
        self.new_route = None
        self.new_obj = None
        self.solver_latency = None
        self.gain = None
        self.parent = parent
        self.child = child
        #TODO: the starting and ending index in the current route


logger = logging.getLogger(__name__)

def _strip_terminal_duplicate(route: Sequence[int]) -> List[int]:
    """
    Normalize a tour by removing the duplicated terminal node (e.g. [0,1,...,0]).
    Returns a list copy to avoid mutating the caller's route.
    """
    if not route:
        return []
    if len(route) > 1 and route[0] == route[-1]:
        return list(route[:-1])
    return list(route)


def _summarize(values: Sequence[int] | None, limit: int = 10) -> str:
    if not values:
        return "[]"
    seq = list(values)
    if len(seq) <= limit:
        return "[" + ", ".join(map(str, seq)) + "]"
    head = ", ".join(map(str, seq[:limit]))
    return f"[{head}, ... ({len(seq)} total)]"


def validate_route_nodes(reference_route: Sequence[int], candidate_route: Sequence[int]) -> tuple[bool, List[int], List[int]]:
    """
    Compare the multiset of vertices in two routes (ignoring the duplicated depot).
    Returns (is_valid, missing_nodes, extra_nodes).
    """
    ref_counter = Counter(_strip_terminal_duplicate(reference_route))
    cand_counter = Counter(_strip_terminal_duplicate(candidate_route))
    missing = sorted((ref_counter - cand_counter).elements())
    extra = sorted((cand_counter - ref_counter).elements())
    return len(missing) == 0 and len(extra) == 0, missing, extra

def evaluate_best_gain(results, current_obj):
    """
    Given a list of results where each result[1] is the obj value,
    return the best combination with the highest net gain,
    prioritizing lower index if multiple have same gain.
    """
    gain_tasks      = []
    no_impr_tasks   = []

    for task in results:
        gain = max(0, current_obj - task.new_obj)
        task.gain = gain
        print('current gains of %s is %f' % (task.coordinates, gain))

        if gain > 0:
            gain_tasks.append(task)
        else:
            no_impr_tasks.append(task)

    gain_tasks = sorted(gain_tasks, key=lambda task: (-task.gain, task.id)) # task with higher gain and lower id is ranked first

    return gain_tasks, no_impr_tasks

def splice_longest_subroute(par: "Child", sub: "Child") -> None:
    
    needle = set(sub.removed_nodes)
    if not needle:
        return                                   # nothing to do

    def first_last(route, items):
        """Return (first_idx, last_idx) where any item of *items* occurs."""
        first = last = None
        for i, v in enumerate(route):
            if v in items:
                if first is None:
                    first = i
                last = i
        if first is None:
            raise ValueError("Some removed‑nodes not found in a route.")
        return first, last

    # --- 1. donor segment ---------------------------------------------------
    d0, d1 = first_last(sub.new_route, needle)
    donor_segment = sub.new_route[d0:d1 + 1]

    # --- 2. parent segment to be replaced -----------------------------------
    p0, p1 = first_last(par.new_route, needle)

    # Optional sanity check: same multiset of vertices
    # if Counter(donor_segment) != Counter(par.new_route[p0:p1 + 1]):
    #     raise ValueError("Segment substitution would change vertex set")

    # --- 3. splice in one go -------------------------------------------------
    par.new_route[p0:p1 + 1] = donor_segment

def exclusive_include(childs_with_gain):
    # TODO: and consecutive sequence
    # all nodes are in the same segment, even though nodes are not consecutive
    for route_segment in childs_with_gain[0].route_segments:
        if set(route_segment) >= set(childs_with_gain[1].removed_nodes):
            pos = {value: idx for idx, value in enumerate(route_segment)}
            try:
                idxs = sorted(pos[x] for x in childs_with_gain[1].removed_nodes)  # KeyError if x not in full
            except KeyError:
                return False
            if len(idxs) != len(set(idxs)):
                return False
            return True
    else:
        return False


def evaluate_gain_contribution(tsp_instance, current_obj, current_route, gain_tasks, no_impr_tasks, parent_indicator, child_indicator):

    def has_child_gain_advantage_over_parent(child_tasks, parent_tasks):
        return (
                len(child_tasks) > 1 and
                sum(task.gain for task in child_tasks) > sum(task.gain for task in parent_tasks)
        )

    if len(gain_tasks) <= 1:
        return gain_tasks

    childs_with_gain = []
    parent_with_gain = []

    for task in gain_tasks:
        if task.child == True:
            childs_with_gain.append(task)
        elif task.parent == True:
            parent_with_gain.append(task)

    if has_child_gain_advantage_over_parent(childs_with_gain, parent_with_gain):
        if exclusive_include(childs_with_gain):

            splice_longest_subroute(childs_with_gain[0], childs_with_gain[1])
            childs_with_gain[0].new_obj = tsp_instance.calculate_total_distance(childs_with_gain[0].new_route)
            childs_with_gain[0].coordinates = childs_with_gain[0].coordinates + childs_with_gain[1].coordinates
            childs_with_gain[0].solver_latency = max(childs_with_gain[0].solver_latency, childs_with_gain[1].solver_latency)
            childs_with_gain[0].removed_nodes = childs_with_gain[0].removed_nodes.union(childs_with_gain[1].removed_nodes)
            childs_with_gain[0].num_removed_nodes = len(childs_with_gain[0].removed_nodes)
            childs_with_gain[0].gain =  max(0, current_obj - childs_with_gain[0].new_obj)
            gain_tasks = childs_with_gain

    else:
        gain_tasks = gain_tasks[:1]

    return gain_tasks

def subproblem_scale_below_threshold(args, unique_removed_node_count, removed_nodes_list):
    if unique_removed_node_count < args.max_node_for_solver:
        return True
    else:
        return False

def non_overlapping(args, unique_removed_node_count, removed_nodes_list):
    if unique_removed_node_count == len([node for sublist in removed_nodes_list for node in sublist]):
        return True
    else:
        return False
def subproblem_overlapping(args, unique_removed_node_count, removed_nodes_list):
    if args.select_sequence:
        return True
    if unique_removed_node_count < len(
        [node for sublist in removed_nodes_list for node in sublist]):
        return True
    else:
        return False

# edit
# def set_cpu_affinity(process, cpu_ids):
#     """
#     Set CPU affinity for a process.
#     """
#     p = psutil.Process(process.pid)
#     p.cpu_affinity(cpu_ids)  # Assign specific CPUs to the process

import platform
def set_cpu_affinity(process, cpu_ids):
    """
    Bind *proc* to given CPU cores if supported.
    On macOS this is a no-op, since cpu_affinity() isn't available.
    """
    system = platform.system()
    if system == "Linux":
        psutil.Process(process.pid).cpu_affinity(cpu_ids)
    elif system == "Windows":
        psutil.Process(process.pid).cpu_affinity(cpu_ids)
    elif system == "Darwin":  # macOS
        # macOS doesn't support cpu_affinity; optionally use 'taskpolicy' if you really need it
        # Example (non-binding): subprocess.run(["taskpolicy", "-c", str(cores[0]), str(proc.pid)])
        print(f"[pin] cpu_affinity not supported on macOS; skipping for PID {process.pid}")
        pass
    else:
        print(f"[pin] Unsupported OS: {system}")

def assign_proportional_cores_to_tasks(tasks: List[Any]) -> Dict[int, List[int]]:
    """
    Proportionally assign available CPU cores to the given list of tasks.
    """
    total_cores = os.cpu_count() or 1
    total_task_lengths = sum(task.num_removed_nodes for task in tasks)

    core_assignment = {}
    current_core = 4

    for i, task in enumerate(tasks):
        weight = task.num_removed_nodes / total_task_lengths if total_task_lengths > 0 else 1 / len(tasks)
        allocated_cores = max(1, int(round(weight * total_cores)))
        core_assignment[i] = list(range(current_core, min(current_core + allocated_cores, total_cores)))
        current_core += allocated_cores
        if current_core >= total_cores:
            break

    return core_assignment


def reformulate_and_solve_subTSP(args, tsp_instance, current_route, current_obj, route_segments, removed_nodes):

    sub_tsp = SubTSP(global_distance_mat=tsp_instance.distance_mat,
                     route_segments=route_segments,
                     free_nodes=removed_nodes,
                     starting_point=current_route[0])
    sub_tsp.reformulate_as_ATSP() # build an ATSP distance mat

    start_time = time.time()
    stsp_mat = sub_tsp.transform_partial_ATSP_into_STSP(ATSP_dist_mat=sub_tsp.distance_mat)
    print('Runtime for partial ATSP is: ', time.time() - start_time)

    start_time = time.time()
    # new_route, new_obj, solver_name = parallel_solvers(args, stsp_mat, sub_tsp, tsp_instance)
    # if args.solver_model == 'concorde' and len(removed_nodes) >= 5:
    # edited: use concorde all the time
    if args.solver_model == 'concorde':
        try:
            
            print("====== start concorde, time limit: ", args.SolverTimeLimit)
            sub_solver_model = Concorde(dist_matrix=stsp_mat)
            sub_solver_model.optimize(timelimit=args.SolverTimeLimit, verbose=False)
            optimized_sub_tsp = sub_tsp.filter_dummy_nodes(route=sub_solver_model.get_tsp_route(),
                                                           num_ori_nodes=len(sub_tsp.node_list))
            new_route = sub_tsp.resume_master_route(optimized_sub_tsp)
            new_obj = tsp_instance.calculate_total_distance(new_route)
            print("====== end concorde, time spent: ", time.time() - start_time)
        except Exception as e:
            # Log the error with additional context and fallback gracefully
            print(f"Error occurred during Concorde optimization: {e}")
            print("Falling back to current route and objective.")

            # Fallback to the current route and objective
            new_route = current_route
            new_obj = current_obj

    elif len(removed_nodes) < 5:
        sub_solver_model = GurobiTSPModel(nodes=sub_tsp.node_list,
                                          distance_mat=sub_tsp.distance_mat)
        # sub_solver_model.update_model_param(args)
        sub_solver_model.optimize()
        new_route = sub_tsp.resume_master_route(sub_solver_model.get_tsp_route()[:-1])
        new_obj = tsp_instance.calculate_total_distance(new_route)

    solver_latency = time.time() - start_time

    # Ensure the reconstructed tour is well-formed before handing it back.
    route_ok, missing, extra = validate_route_nodes(current_route, new_route)
    if not route_ok:
        logger.error(
            "Discarding sub-TSP solution due to node mismatch (missing=%s, extra=%s, removed=%s)",
            _summarize(missing),
            _summarize(extra),
            _summarize(removed_nodes),
        )
        new_route = current_route
        new_obj = current_obj

    return new_route, new_obj, solver_latency


def process_subTSP(args, task, tsp_instance, current_route, current_obj, result_queue):
   
    import queue  # Import the queue module

    removed_nodes, route_segments = task.removed_nodes, task.route_segments

    # Perform the sub-TSP computation
    results = reformulate_and_solve_subTSP(args, tsp_instance, current_route, current_obj, route_segments, removed_nodes)
    task.new_route      = results[0]
    task.new_obj        = results[1]
    task.solver_latency = results[2]
    # result_queue.put(results, block=False)  # Send the result back to the parent process
    try:
        result_queue.put(task, block=False)
    except queue.Full:
        print("Queue is full! Dropping result or retrying...")


def format_task_traj(task):
    coord_str = " ".join(
        f"<coordinates> x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max} </coordinates>"
        for x_min, x_max, y_min, y_max in task.coordinates
    )
    return (
        f"{coord_str}, number of nodes within the subrectangle={task.num_removed_nodes}, "
        f"travel distance reduction={round(task.gain, 2)} (the higher the better), "
        f"computation time for this subrectangle={round(task.solver_latency, 2)} sec \n"
    )

def sample_independent_subproblem(config, active_subproblems, gain_subproblem_queue, subproblem_re_queue, subproblem_ft_queue, traj_lock, current_route):

    queues_with_priority = [gain_subproblem_queue, subproblem_re_queue, subproblem_ft_queue]
    for queue in queues_with_priority:
        temp_buffer = []
        selected_subproblem = None

        while True:
            subproblem = queue.try_get()
            if subproblem is None:
                break
            # edited
            # try:
            #     subproblem = queue.get_nowait()
            # except Empty:
            #     break # does this put the subproblem back to the queue?
            
            # TODO: a subproblem can include multiple removed_nodes 
            if all(subproblems_are_independent(active_sp, subproblem, config, current_route) for active_sp in active_subproblems):
                selected_subproblem = subproblem
                print('Retrieved valid subproblems')
                break
            else:
                temp_buffer.append(subproblem)

        for sp in temp_buffer:
            queue.put(sp) # since we selectively choose a subproblem, the order in the queue is not important
        
        if selected_subproblem:
            return selected_subproblem
        
    return None

def sample_next_subproblem(config, active_subproblems, subproblem_re_queue, subproblem_ft_queue, traj_lock, current_route):

    queues_with_priority = [subproblem_re_queue, subproblem_ft_queue]
    for queue in queues_with_priority:
        temp_buffer = []
        selected_subproblem = None

        while True:
            try:
                subproblem = queue.get_nowait()
            except Empty:
                break # does this put the subproblem back to the queue?
            if subproblem:
                return subproblem

    return None


def _is_within_segment(removed_nodes, route_segment):
    # determine if a given (non-consecutive) node list is within a segment
    # i.e., the longest consecutive node sequence speficied by the starting and ending nodes should be within the segment   
    if set(route_segment) >= set(removed_nodes):
        pos = {value: idx for idx, value in enumerate(route_segment)}
        try:
            idxs = sorted(pos[x] for x in removed_nodes)  # KeyError if x not in full
        except KeyError:
            return False
        if len(idxs) != len(set(idxs)):
            return False
        return True

def subproblems_are_independent(running_subproblem, pending_subproblem, config, current_route):
   
    # merge all removed nodes of the running subproblem first 
    unique_removed_node = sorted(set(node for sublist in running_subproblem.removed_nodes_list for node in sublist))
    route_segments = config.tsp_instance.remove_edges_given_nodes(current_route, unique_removed_node)

    # then obtain the unified segments
    # if all removed_nodes is within any segment return true, else, return false
    for removed_nodes in pending_subproblem.removed_nodes_list:
        if not any(_is_within_segment(removed_nodes, route_segment) for route_segment in route_segments):
            return False
    return True
        

def subproblem_solver(subproblem, config):
    _configure_logging('solver')
    log = logging.getLogger()
    log.info("===== Subproblem solver initialized")

    removed_nodes_list = subproblem.removed_nodes_list
    route_segments_list = subproblem.route_segments_list
    coordinates_list = subproblem.coordinates_list
    record_id = getattr(subproblem, "record_id", None)
    recorder = None
    if record_id is not None and getattr(subproblem, "model_name", None):
        try:
            recorder = LLMRecorder(subproblem.model_name)
        except Exception as exc:
            log.warning("Failed to prepare recorder for %s: %s", subproblem.model_name, exc)
            recorder = None

    unique_removed_node = sorted(set(node for sublist in removed_nodes_list for node in sublist))

    with config.obj_lock:
        current_obj = config.global_obj.value
        log.info("start coords=%s obj_in=%s", subproblem.coordinates_list, current_obj)
    with config.sol_lock:
        current_route = list(config.global_sol)

    # ---- determine how to handle multiple selections    
    if subproblem_overlapping(config.args, len(unique_removed_node), removed_nodes_list):
        removed_nodes = unique_removed_node  # multiple subproblems get merged
        route_segments = config.tsp_instance.remove_edges_given_nodes(current_route, removed_nodes)
        task = SubTSPTask(id=0,
                            current_route=current_route,
                            removed_nodes=removed_nodes,
                            route_segments=route_segments,
                            coordinates=coordinates_list,
                            parent=True)
        tasks = [task]
        parent = True
        child = False
    else:
        if subproblem_scale_below_threshold(config.args, len(unique_removed_node), removed_nodes_list):
            tasks = []
            for idx, (removed_nodes, route_segments, coordinate) in enumerate(zip(removed_nodes_list, route_segments_list, coordinates_list), start=0):
                task = SubTSPTask(id=idx,
                                    current_route=current_route,
                                    removed_nodes=removed_nodes,
                                    route_segments=route_segments,
                                    coordinates=[coordinate],
                                    child=True)
                tasks.append(task)

            merged_removed_nodes = unique_removed_node
            merged_route_segments = config.tsp_instance.remove_edges_given_nodes(current_route, merged_removed_nodes)
            merged_id = tasks[-1].id + 1

            merged_task = SubTSPTask(id=merged_id,
                                        current_route=current_route,
                                        removed_nodes=merged_removed_nodes,
                                        route_segments=merged_route_segments,
                                        coordinates=coordinates_list,
                                        parent=True)

            tasks.append(merged_task)
            parent = True
            child = True
        else:
            # tasks = list(zip(removed_nodes_list, route_segments_list))
            tasks = []
            for idx, (removed_nodes, route_segments, coordinate) in enumerate(
                    zip(removed_nodes_list, route_segments_list, coordinates_list), start=0):
                task = SubTSPTask(id=idx,
                                    current_route=current_route,
                                    removed_nodes=removed_nodes,
                                    route_segments=route_segments,
                                    coordinates=[coordinate],
                                    child=True)
                tasks.append(task)

            parent = False
            child = True

    # core_map = assign_proportional_cores_to_tasks(tasks)
    processes = []
    result_queue = mp.Queue()

    start_time = time.time()

    # Spawn multiple processes
    log.info(f"Spawn {len(tasks)} process_subTSP tasks")
    for i, task in enumerate(tasks):
        # current_route, current_obj = list(global_sol), global_obj.value
        process = mp.Process(target=process_subTSP, args=(config.args, task, config.tsp_instance, current_route, current_obj, result_queue))
        process.start()
        processes.append(process)

        #TODO: Set CPU affinity based on the given CPU cores for concorde master
        # set_cpu_affinity(process, core_map[i])

    results = []
    completed_processes = 0
    num_processes = len(tasks)

    while completed_processes < num_processes:
        if not result_queue.empty():
            task = result_queue.get()
            results.append(task)
            completed_processes += 1

    log.info('Time spent in multiprocessing subTSPs: %f', time.time() - start_time)

    if results:
        gain_tasks, no_impr_tasks = evaluate_best_gain(results, current_obj)
        if gain_tasks:
            gain_tasks = evaluate_gain_contribution(config.tsp_instance, current_obj, current_route, gain_tasks, no_impr_tasks, parent, child)

        # always use the first task (either merged or as is)
        if gain_tasks:
            new_route, new_obj, solver_latency = gain_tasks[0].new_route, gain_tasks[0].new_obj, gain_tasks[0].solver_latency
            coordinates_list, removed_nodes, route_segments = gain_tasks[0].coordinates, gain_tasks[0].removed_nodes, gain_tasks[0].route_segments
        else:
            new_route, new_obj, solver_latency = current_route, current_obj, 0
            coordinates_list, removed_nodes, route_segments = subproblem.coordinates_list, [node for sublist in subproblem.removed_nodes_list for node in sublist], subproblem.route_segments_list
    else:
        new_route, new_obj, solver_latency = current_route, current_obj, 0
        coordinates_list, removed_nodes, route_segments = None, None, None

    route_valid, missing_nodes, extra_nodes = validate_route_nodes(current_route, new_route)
    if not route_valid:
        log.error(
            "Ignoring invalid sub-route (missing=%s, extra=%s, removed=%s)",
            _summarize(missing_nodes),
            _summarize(extra_nodes),
            _summarize(removed_nodes),
        )
        new_route, new_obj = current_route, current_obj

    # ---------------------------------
    # hill-climbing acceptance criteria
    # ---------------------------------
    with config.obj_lock, config.sol_lock:
        # if subproblem.solution_version != config.global_obj.value:
        #     log.info("Find different version of solution")
        # elif subproblem.solution_version == config.global_obj.value:
        old_obj = config.global_obj.value
        delta_obj = new_obj - old_obj

        if delta_obj < 0:
            if len(new_route) == len(config.global_sol):
                config.global_obj.value = new_obj
                config.global_sol[:] = new_route[:]
            else:
                if not removed_nodes:
                    removed_nodes = []
                log.error(
                    "Route update failed due to length mismatch: SKIP this update (got %s, expected %s), \nnew route:\n%s\nold route:\n%s\nremoved nodes:\n%s\n",
                    len(new_route),
                    len(config.global_sol),
                    ", ".join([str(x) for x in new_route]),
                    ", ".join([str(x) for x in config.global_sol]),
                    ", ".join([str(x) for x in removed_nodes]),
                )
                delta_obj = 0
                # assert False
        now = round(time.time() - config.t0 + config.warmstart_latency, 2)
        proc = mp.current_process()
        record = GlobalObjRecord(
            latency=now,
            new_obj=config.global_obj.value,
            coords=coordinates_list,
            num_nodes_removed=len(removed_nodes or []),
            llm_mode=subproblem.llm_source,
            global_solution_version=subproblem.current_obj,
            process_name=proc.name,
        )

        config.track_global_obj_queue.put(record)

        if config.args.keep_selection_trajectory:
            with config.traj_lock:
                if gain_tasks:
                    config.selection_traj.put(format_task_traj(gain_tasks[0]))
                for task in no_impr_tasks:
                    config.selection_traj.put(format_task_traj(task))

        objective_after = config.global_obj.value

    improved = delta_obj < 0

    log.info(f"task {record_id} finished, improved: {improved}")
    
    if recorder is not None and record_id is not None:
        try:
            recorder._update_record(
                record_id,
                {
                    "improved_objective": bool(improved),
                    "objective_delta": float(delta_obj),
                    "objective_before": float(old_obj),
                    "objective_after": float(objective_after),
                    "proposal_evaluated_at": time.time(),
                },
            )
            log.info("Updated record for %s", record_id)
        except Exception as exc:
            log.warning("Failed to log outcome for record %s: %s", record_id, exc)

    with config.obj_lock:
        log.info("updated obj %s→%s, using %s", old_obj, config.global_obj.value, subproblem.llm_source)

    log.info("Subproblem solver ended =====")


def subproblem_verifier(subproblem, config):
    _configure_logging('verifier')
    log = logging.getLogger()

    removed_nodes_list, route_segments_list, coordinates_list = subproblem.removed_nodes_list, subproblem.route_segments_list, subproblem.coordinates_list
    unique_removed_node = sorted(set(node for sublist in removed_nodes_list for node in sublist))

    with config.obj_lock:
        current_obj = config.global_obj.value
        log.info("start coords=%s obj_in=%s", subproblem.coordinates_list, current_obj)
    with config.sol_lock:
        current_route = list(config.global_sol)

    # ---- determine how to handle multiple selections    
    if subproblem_overlapping(config.args, len(unique_removed_node), removed_nodes_list):
            removed_nodes = unique_removed_node  # multiple subproblems get merged
            route_segments = config.tsp_instance.remove_edges_given_nodes(current_route, removed_nodes)
            task = SubTSPTask(id=0,
                              current_route=current_route,
                              removed_nodes=removed_nodes,
                              route_segments=route_segments,
                              coordinates=coordinates_list,
                              parent=True)
            tasks = [task]
            parent = True
            child = False
    else:
        if subproblem_scale_below_threshold(config.args, len(unique_removed_node), removed_nodes_list):
            tasks = []
            for idx, (removed_nodes, route_segments, coordinate) in enumerate(zip(removed_nodes_list, route_segments_list, coordinates_list), start=0):
                task = SubTSPTask(id=idx,
                                    current_route=current_route,
                                    removed_nodes=removed_nodes,
                                    route_segments=route_segments,
                                    coordinates=[coordinate],
                                    child=True)
                tasks.append(task)

            merged_removed_nodes = unique_removed_node
            merged_route_segments = config.tsp_instance.remove_edges_given_nodes(current_route, merged_removed_nodes)
            merged_id = tasks[-1].id + 1

            merged_task = SubTSPTask(id=merged_id,
                                        current_route=current_route,
                                        removed_nodes=merged_removed_nodes,
                                        route_segments=merged_route_segments,
                                        coordinates=coordinates_list,
                                        parent=True)

            tasks.append(merged_task)
            parent = True
            child = True
        else:
            # tasks = list(zip(removed_nodes_list, route_segments_list))
            tasks = []
            for idx, (removed_nodes, route_segments, coordinate) in enumerate(
                    zip(removed_nodes_list, route_segments_list, coordinates_list), start=0):
                task = SubTSPTask(id=idx,
                                    current_route=current_route,
                                    removed_nodes=removed_nodes,
                                    route_segments=route_segments,
                                    coordinates=[coordinate],
                                    child=True)
                tasks.append(task)

            parent = False
            child = True

    # core_map = assign_proportional_cores_to_tasks(tasks)
    processes = []
    result_queue = mp.Queue()

    start_time = time.time()

    # Spawn multiple processes
    for i, task in enumerate(tasks):
        # current_route, current_obj = list(global_sol), global_obj.value
        process = mp.Process(target=process_subTSP, args=(config.args, task, config.tsp_instance, current_route, current_obj, result_queue))
        process.start()
        processes.append(process)

        #TODO: Set CPU affinity based on the given CPU cores for concorde master
        # set_cpu_affinity(process, core_map[i])

    results = []
    completed_processes = 0
    num_processes = len(tasks)

    while completed_processes < num_processes:
        if not result_queue.empty():
            task = result_queue.get()
            results.append(task)
            completed_processes += 1

    log.info('Time spent in multiprocessing subTSPs: %f', time.time() - start_time)

    if results:
        gain_tasks, no_impr_tasks = evaluate_best_gain(results, current_obj)
        if gain_tasks:
            gain_tasks = evaluate_gain_contribution(config.tsp_instance, current_obj, current_route, gain_tasks, no_impr_tasks, parent, child)

        # always use the first task (either merged or as is)
        if gain_tasks:
            new_route, new_obj, solver_latency = gain_tasks[0].new_route, gain_tasks[0].new_obj, gain_tasks[0].solver_latency
            coordinates_list, removed_nodes, route_segments = gain_tasks[0].coordinates, gain_tasks[0].removed_nodes, gain_tasks[0].route_segments
        else:
            new_route, new_obj, solver_latency = current_route, current_obj, 0
            coordinates_list, removed_nodes, route_segments = subproblem.coordinates_list, [node for sublist in subproblem.removed_nodes_list for node in sublist], subproblem.route_segments_list
    else:
        new_route, new_obj, solver_latency = current_route, current_obj, 0
        coordinates_list, removed_nodes, route_segments = None, None, None

    # ---------------------------------
    # hill-climbing acceptance criteria
    # ---------------------------------
    with config.obj_lock, config.sol_lock:
        
        delta_obj = new_obj - config.global_obj.value

        if delta_obj < 0:
            config.gain_subproblem_queue.put(subproblem)
            print('Finding a gain subproblem!')
