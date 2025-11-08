from dataclasses import dataclass
import logging
import psutil
from typing import Iterable, List, Tuple, Union, Any, Dict, Optional
from multiprocessing import Queue
import queue
from helper.parse_llm_response import LLMTextParser
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import random

@dataclass
class Subproblem:
    removed_nodes_list: list
    route_segments_list: list
    coordinates_list: list
    current_obj: float
    solution_version: float
    llm_source: any
    record_id: Optional[int] = None
    model_name: Optional[str] = None


def get_all_from_queue(q: Queue):
    items = []
    while True:
        try:
            item = q.get_nowait()
            items.append(item)
        except queue.Empty:
            break
    return items

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
    total_task_lengths = sum(len(task[0]) for task in tasks)

    core_assignment = {}
    current_core = 0

    for i, task in enumerate(tasks):
        weight = len(task[0]) / total_task_lengths if total_task_lengths > 0 else 1 / len(tasks)
        allocated_cores = max(1, int(round(weight * total_cores)))
        core_assignment[i] = list(range(current_core, min(current_core + allocated_cores, total_cores)))
        current_core += allocated_cores
        if current_core >= total_cores:
            break

    return core_assignment

def parse_num_tag(text: str, as_float: bool = False) -> Union[int, float]:
    """
    Return the numeric value inside a <num> … </num> tag.

    Parameters
    ----------
    text : str
        The string that contains a single <num> … </num> element.
    as_float : bool, default False
        • False  → return an int if the content looks like an integer.
        • True   → always return a float.

    Raises
    ------
    ValueError
        If no <num> tag is present or the content is not a valid number.
    """
    m = _NUM_RE.search(text)
    if not m:
        raise ValueError("No <num> … </num> tag with a valid number found")

    number_str = m.group(1)
    return float(number_str) if as_float or "." in number_str else int(number_str)

def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(processName)s] %(message)s",
                        datefmt="%H:%M:%S",
                        force=True,
                        filename="logs/llm.log")

def llm_visual_selection(args, solution_plotter, current_route, pending_subproblem_queue, tsp_instance, prior_selection_traj,
                         llm_selector, backup_selector, llm_parser, num_subregion, task_label,
                         x_min, x_max, y_min, y_max, grid_res, traj_lock, num_nodes):

    #TODO: optimize the plot generation using GPU?
    tsp_plot = solution_plotter.plot_tsp_solution_plotly(args, current_route, tsp_instance.coords,
                                                         x_min=int(x_min), x_max=int(x_max),
                                                         y_min=int(y_min), y_max=int(y_max),
                                                         grid_resolution=int(grid_res), node_size=2, num_nodes=num_nodes)

    #TODO: input the buffer area to avoid selection
    current_selector = llm_selector.get_next_llm()

    pending_subproblems = peek_queue(pending_subproblem_queue, traj_lock)
    prior_selections = peek_queue(prior_selection_traj, traj_lock)

    pending_coords = [coord for s_p in pending_subproblems for coord in s_p.coordinates_list]
    print("Pending length is ", pending_coords)

    response_text = current_selector.vision_chat(fig=tsp_plot,
                                                 prior_selection=prior_selections,
                                                 num_region=num_subregion,
                                                 pending_coords=pending_coords,
                                                 x_min=int(x_min),
                                                 x_max=int(x_max),
                                                 y_min=int(y_min),
                                                 y_max=int(y_max)
                                                 )

    record_id = current_selector.last_record_id

    try:
        coordinates_list = llm_parser.parse_subrectangle_coordinates(response_text)
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to parse LLM response: %s", exc)
        coordinates_list = [(None, None, None, None)]

    if coordinates_list[0][0] is None:
        coordinates_list = backup_selector.generate_subrectangle(X_MIN=x_min,
                                                                 X_MAX=x_max,
                                                                 Y_MIN=y_min,
                                                                 Y_MAX=y_max,
                                                                 no_more_than=grid_res,
                                                                 num_region=num_subregion)

    if record_id is not None:
        current_selector.recorder.save_selection_plot(record_id, tsp_plot, coordinates_list)
        current_selector.recorder._update_record(record_id, {"task_name": str(task_label)})
    else:
        current_selector.recorder.save_selection_plot(record_id, tsp_plot, coordinates_list)

    return coordinates_list, record_id, current_selector.model_name

def handle_valid_subproblem_selection(args, removed_nodes, coordinates_list, solution_plotter, current_route,
                                            pending_subproblem_queue, tsp_instance, prior_selection, llm_selector, backup_selector,
                                            llm_parser, num_subregion,
                                            current_route_edges, X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RES, traj_lock,
                                            task_label):
    """
    - If the rectangle is too small (<5 nodes), it falls back to the RandomSelector to redraw a larger region over the entire instance.
    - If the rectangle is too large (>max), it “zooms in”: either asks the LLM again but confined to the oversized region (with higher resolution) or, if using the random selector, samples a
        smaller subrectangle inside the oversized one.
    - After each redraw, it recomputes removed_nodes/route_segments until the count is acceptable, then returns the sanitized route_segments, removed_nodes, and metadata (coordinates, record id,
        model name).
    """
    record_id = None
    model_name = None

    while len(removed_nodes) < 5 or len(removed_nodes) > args.max_node_for_solver:

        if len(removed_nodes) < 5:
            coordinates_list = backup_selector.generate_subrectangle(X_MIN=X_MIN,
                                                                     X_MAX=X_MAX,
                                                                     Y_MIN=Y_MIN,
                                                                     Y_MAX=Y_MAX,
                                                                     num_region=1
                                                                     )
            x_min, x_max, y_min, y_max = coordinates_list[0]
            record_id = None
            model_name = None

        elif len(removed_nodes) > args.max_node_for_solver:
            x_min, x_max, y_min, y_max = coordinates_list[0]
            print(f'coordinate list is {x_min, x_max, y_min, y_max}')

            grid_resolution = max((x_max - x_min), (y_max - y_min)) // args.gridding_resolution

            if args.fast_llm_model == 'random':
                coordinates_list = backup_selector.generate_subrectangle(X_MIN=x_min,
                                                                         X_MAX=x_max,
                                                                         Y_MIN=y_min,
                                                                         Y_MAX=y_max,
                                                                         num_region=1
                                                                         )
                time.sleep(5) # minimicing the arrival rate of OpenAI response
                record_id = None
                model_name = None
            else:
                # select a region from zoomed in region
                coordinates_list, new_record_id, new_model_name = llm_visual_selection(args, solution_plotter, current_route, pending_subproblem_queue,
                                                        tsp_instance,
                                                        prior_selection, llm_selector, backup_selector,
                                                        llm_parser, 1, task_label,
                                                        x_min, x_max, y_min, y_max, grid_resolution, traj_lock,
                                                        num_nodes = len(removed_nodes),)
                record_id = new_record_id
                model_name = new_model_name

            print(f'zoomed in coordinate list is {coordinates_list}')

        updated_route_edges, removed_nodes, route_segments = tsp_instance.remove_edges_in_subrectangle(current_route,
                                                                                                       current_route_edges,
                                                                                                       coordinates_list)

    return route_segments, removed_nodes, coordinates_list, record_id, model_name

def process_coordinate(coordinate, args, tsp_instance, current_route, current_route_edges,
                        pending_subproblem_queue, coordinates_list, solution_plotter, prior_selection,
                        llm_selector, backup_selector,
                        llm_parser, num_subregion, X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RES, traj_lock,
                        task_label, record_id=None, model_name=None, logger=None):

    coordinate = [coordinate]

    updated_route_edges, removed_nodes, route_segments = tsp_instance.remove_edges_in_subrectangle(
        current_route, current_route_edges, coordinate
    )
    origin_removed_nodes = removed_nodes
    if (len(removed_nodes) == 0) or (len(removed_nodes) > args.max_node_for_solver):
        route_segments, removed_nodes, coordinate, record_id, model_name = handle_valid_subproblem_selection(
            args, removed_nodes, coordinate, solution_plotter, current_route, pending_subproblem_queue,
            tsp_instance, prior_selection, llm_selector,
            backup_selector, llm_parser, args.llm_subproblem_selection, current_route_edges,
            X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RES, traj_lock, task_label
        )

    logger.info(f'original removed nodes is {len(origin_removed_nodes)}')
    logger.info(f'new removed nodes is {len(removed_nodes)}')
    logger.info(f'overlapping with the original one: {len(set(origin_removed_nodes).intersection(set(removed_nodes)))}')

    return removed_nodes, route_segments, coordinate, record_id, model_name

def llm_io(args, solution_plotter, current_route, current_route_edges, pending_subproblem_queue, tsp_instance, prior_selection, llm_selector,
                 backup_selector, llm_parser, num_subregion, task_label,
                 x_min, x_max, y_min, y_max, grid_res, num_nodes, traj_lock, logger):

    # await asyncio.sleep(random.uniform(10, 20))
    # await asyncio.sleep(latency)

    logger.info("===== LLM IO:")
    # --- Obtain subproblem selection
    # if True:
    if args.select_sequence:
        removed_nodes_list = []
        route_segments_list = []
        zoomed_in_coordinates_list = []
       
        for i in range(args.llm_subproblem_selection):
            removed_nodes = backup_selector.generate_sequence(current_route, randomize=args.random_selection, max_length=args.max_node_for_solver)
            removed_nodes_list.append(removed_nodes)
        time.sleep(5)
        record_id = None
        model_name = None
        
    else:

        if args.hard_coded_subrectangle:
            coordinates_list = [(8408, 8927, 5921, 9508), (7375, 8207, 7090, 9756)]
            time.sleep(5)
            record_id = None
            model_name = None
        elif args.fast_llm_model == 'random':
            coordinates_list = backup_selector.generate_subrectangle(X_MIN=x_min,
                                                                    X_MAX=x_max,
                                                                    Y_MIN=y_min,
                                                                    Y_MAX=y_max,
                                                                    no_more_than=grid_res,
                                                                    num_region=args.llm_subproblem_selection)
            time.sleep(5)
            record_id = None
            model_name = None

        else:
            coordinates_list, record_id, model_name = llm_visual_selection(args, solution_plotter, current_route, pending_subproblem_queue, tsp_instance, prior_selection,
                                llm_selector, backup_selector, llm_parser, args.llm_subproblem_selection, task_label,
                                x_min, x_max, y_min, y_max, grid_res, traj_lock, len(current_route), )


        # ---- Process the selection to ensure the selection is valid (does not cover too large area)
        removed_nodes_list = []
        route_segments_list = []
        zoomed_in_coordinates_list = []

        # Prepare a partial function to pass shared arguments
        process_func = partial(
            process_coordinate, args=args, tsp_instance=tsp_instance, current_route=current_route,
            current_route_edges=current_route_edges, pending_subproblem_queue=pending_subproblem_queue,
            coordinates_list=coordinates_list,
            solution_plotter=solution_plotter, prior_selection=prior_selection,
            llm_selector=llm_selector, backup_selector=backup_selector,
            llm_parser=llm_parser, num_subregion=args.llm_subproblem_selection, X_MIN=x_min, X_MAX=x_max,
            Y_MIN=y_min, Y_MAX=y_max, GRID_RES=grid_res, traj_lock=traj_lock,
            task_label=task_label, record_id=record_id, model_name=model_name, logger=logger
        )

        num_thread = len(coordinates_list)

        # Use ThreadPoolExecutor for parallel execution
        logger.info(f"Processing coordinates: {coordinates_list}")
        with ThreadPoolExecutor(max_workers=num_thread) as executor:
            results = list(executor.map(process_func,
                                        coordinates_list))  # elements within coordinates_list will be split dynamically to threads
        logger.info(f"Processing coordinates done")

        # Unpack results
        for removed_nodes, route_segments, coordinates_chunk, updated_record_id, updated_model_name in results:
            removed_nodes_list.append(removed_nodes)
            route_segments_list.append(route_segments)
            zoomed_in_coordinates_list.extend(coordinates_chunk)
            if updated_record_id is not None:
                record_id = updated_record_id
            if updated_model_name is not None:
                model_name = updated_model_name

    logger.info("LLM IO END =====")
    return removed_nodes_list, route_segments_list, zoomed_in_coordinates_list, record_id, model_name

    # return [random.randint(0, 100) for _ in range(5)]
    # nums = parse_num_tag(llm_model.chat())

    # return [nums]


def concorde_cpu(current_obj: int, sub_sol: List[int]) -> int:
    time.sleep(random.uniform(1, 2))
    return current_obj + sum(sub_sol)


def _chunk(lst: List[int], n_chunks: int) -> Iterable[List[int]]:
    k, m = divmod(len(lst), n_chunks)
    for i in range(n_chunks):
        start = i * k + min(i, m)
        end = (i + 1) * k + min(i + 1, m)
        yield lst[start:end]


def _safe_copy(proxy, default):
    """Return a plain object copy of a Manager proxy or *default* if broken."""
    try:
        return proxy() if callable(proxy) else proxy.value if hasattr(proxy, 'value') else list(proxy)
    except (BrokenPipeError, EOFError, ConnectionRefusedError, AttributeError):
        logging.getLogger().warning("manager proxy unavailable – returning default")
        return default

# ---------------------------------------------------------------------------
# LLM producer
# ---------------------------------------------------------------------------
def peek_queue(q, traj_lock):
    temp = []
    with traj_lock:
        while not q.empty():
            item = q.get()
            temp.append(item)
        # After peeking, put items back
        for item in temp:
            q.put(item)
    return temp

async def _llm_producer(llm_name, args, tsp_instance, llm_selector, pending_subproblem_queue, global_obj, global_sol, sol_lock, obj_lock, selection_traj, deadline, t0, traj_queue, traj_lock,
                        X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RES,
                        backup_selector, solution_plotter):

    _configure_logging()
    log = logging.getLogger()

    # ==== simulate the generation of LLM selection
    llm_parser = LLMTextParser()

    while time.time() < deadline:
        with sol_lock:
            current_route = list(global_sol)
        with obj_lock:
            current_obj = global_obj.value
        # with traj_lock:
        #     prior_selection = peek_queue(traj_queue)

        current_route_edges = tsp_instance.get_route_pair(current_route)
        removed_nodes_list, route_segments_list, coordinates_list, record_id, model_name = llm_io(args=args,
                                                                                 solution_plotter=solution_plotter,
                                                                                 current_route=current_route,
                                                                                 current_route_edges=current_route_edges,
                                                                                 pending_subproblem_queue=pending_subproblem_queue,
                                                                                 tsp_instance=tsp_instance,
                                                                                 prior_selection = selection_traj,
                                                                                 llm_selector=llm_selector,
                                                                                 backup_selector=backup_selector,
                                                                                 llm_parser=llm_parser,
                                                                                 num_subregion=args.llm_subproblem_selection,
                                                                                 task_label=str(args.instance_path),
                                                                                 x_min=X_MIN,
                                                                                 x_max=X_MAX,
                                                                                 y_min=Y_MIN,
                                                                                 y_max=Y_MAX,
                                                                                 grid_res=GRID_RES,
                                                                                 num_nodes=0,
                                                                                 traj_lock=traj_lock,
                                                                                 logger = log
                                                                                )

        try:
            subproblem = Subproblem(removed_nodes_list=removed_nodes_list,
                                    route_segments_list=route_segments_list,
                                    coordinates_list=coordinates_list,
                                    current_obj=current_obj,
                                    solution_version=None,
                                    llm_source=llm_name,
                                    record_id=record_id,
                                    model_name=model_name,
                                    )
            with traj_lock:
                pending_subproblem_queue.put(subproblem)
        except (BrokenPipeError, EOFError):
            log.warning("queue closed – producer exiting early")
            return
        log.info("produced %s from obj=%s sol_len=%s",
                 coordinates_list, current_obj, len(current_route))

    log.info("producer finished (deadline)")
