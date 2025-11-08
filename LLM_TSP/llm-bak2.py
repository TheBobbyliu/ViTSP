#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created on 10/30/24 8:25 PM
@File:llm.py
@Author:XXXX-6
@Contact: XXXX-1@XXXX-7.edu
'''
from openai import OpenAI
import requests
import base64
import io
import json
from pathlib import Path
import threading
from PIL import Image
import matplotlib.pyplot as plt
import cProfile
import pstats
# from pdf2image import convert_from_path
import time

MODEL_TYPES = {
    "gpt-4o": "gpt-4o",
    "o1": "o1",
    "qwen2.5-32b-reasoning": "Qwen/QwQ-32B",
    "qwen2.5-32b-v":"Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen2.5-7b-v":"Qwen/Qwen2.5-VL-7B-Instruct"
}


class LLMRecorder:
    """Handles persistent logging of LLM requests and responses per model."""

    def __init__(self, model_name: str, base_dir: str | Path = "llm_records"):
        model_key = (
            model_name.replace("/", "__")
            .replace(":", "_")
            .replace(" ", "_")
        )
        self.root = Path(base_dir) / model_key
        self.images_dir = self.root / "images"
        self.data_dir = self.root / "data"
        for path in (self.images_dir, self.data_dir):
            path.mkdir(parents=True, exist_ok=True)

        existing_ids = [
            int(p.stem) for p in self.data_dir.glob("*.json") if p.stem.isdigit()
        ]
        self._counter = max(existing_ids, default=0)
        self._lock = threading.Lock()

    def reserve_id(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    def image_path(self, record_id: int) -> Path:
        return self.images_dir / f"{record_id}.png"

    def data_path(self, record_id: int) -> Path:
        return self.data_dir / f"{record_id}.json"

    def save(self, record_id: int, image: Image.Image, payload: dict) -> None:
        image_path = self.image_path(record_id)
        data_path = self.data_path(record_id)

        warnings = []
        try:
            image.save(image_path, format="PNG")
        except Exception as exc:
            warnings.append(f"Failed to write image: {exc}")

        if warnings:
            payload["warnings"] = warnings

        payload["image_path"] = str(image_path)

        with data_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)


class RoundRobinLLMSelector:
    """
    In case there is a rate per minute
    Use round robin to avoid server rejection
    """
    def __init__(self, llm_instances: list):
        self.llms = llm_instances
        self.counter = 0

    def get_next_llm(self):
        llm = self.llms[self.counter]
        self.counter = (self.counter + 1) % len(self.llms)
        return llm


class toy_GPT:
    def __init__(self, api_key, model_name="gpt-4o"):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
    def chat(self):
        response = self.client.responses.create(
            model= self.model_name,
            input="Return a random number between 1 and 100. Return in the format of <num> [your number] </num>"
        )
        print(response.output_text)
        return response.output_text

class GPT:
    def __init__(self, api_key, model_name="gpt-4-vision-preview"):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.recorder = LLMRecorder(self.model_name)

    def _capture_response_snapshot(self, response):
        if response is None:
            return None
        for attr in ("model_dump", "to_dict", "dict"):
            getter = getattr(response, attr, None)
            if callable(getter):
                try:
                    return getter()
                except Exception:
                    continue
        return None

    def vision_chat(self, fig, prior_selection, num_region, pending_coords, x_min, x_max, y_min, y_max):

        region_mapping = {
            1: 'one',
            2: 'two',
            3: 'three',  # Add more mappings as needed
        }

        requested_region_count = num_region

        # Convert PIL Image to bytes
        buf = io.BytesIO()
        fig.write_image(buf, format="png", engine='kaleido')  # Requires kaleido
        buf.seek(0)  # Move to the start of the buffer
        image = Image.open(buf).convert("RGB")

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # Getting the base64 string for GPT-vision
        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')

        num_region_text = region_mapping.get(num_region, 'one')  # 'unknown' as default value

        pending_regions = ", ".join(
            f"<coordinates> x_min={coord[0]}, x_max={coord[1]}, y_min={coord[2]}, y_max={coord[3]} </coordinates>"
            for coord in pending_coords
        )
       
        extraction_prompts = f"""You are tasked with improving an existing solution to a Traveling Salesman Problem (TSP) by selecting a sub-region where the routes can be significantly optimized. 
        Carefully consider the locations of the nodes (in red) and connected routes (in black) in the initial solution on a map. The boundary of the map is x_min={x_min-10000}, x_max={x_max+10000}, y_min={y_min-10000}, y_max={y_max+10000}.
        Please return {num_region_text} non-overlapping sub-rectangle(s) that you believe would most reduce total travel distance from further optimization by a downstream TSP solver.
        Analyze the problem-specific distribution to do meaningful selection. Select areas as large as you could to cover more nodes, which can bring larger improvement. Remember, if you don't see significant improvement, try selecting larger areas that cover more nodes based on your analysis of the prior selection trajectory
        Keep your output very brief as the following template. Don't tell me you cannot view or analyze the map. I don't want an excuse:
        <coordinates> x_min= 1,000, x_max= 2,000, y_min= 1,000, y_max=2,000 </coordinates> 
        \n Avoid selecting the same regions as follows, which are pending optimization:
        {pending_regions}
        
        \n Below are some previous selection trajectory. Learn from the trajectory to improve your selection capability. Please avoid selecting the same subrectangle.
        {prior_selection}
        """
        record_id = self.recorder.reserve_id()
        disk_image_path = self.recorder.image_path(record_id)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": extraction_prompts
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        response = None
        response_text = None
        error_info = None
        response_snapshot = None

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            response_snapshot = self._capture_response_snapshot(response)
            response_text = response.choices[0].message.content
            print(response_text)
            return response_text

        except requests.exceptions.RequestException as e:
            error_info = f"{type(e).__name__}: {e}"
            print(f"Network error occurred: {e}")
        except ValueError as e:
            error_info = f"{type(e).__name__}: {e}"
            print(f"Failed to parse JSON: {e}")
        except (KeyError, IndexError) as e:
            error_info = f"{type(e).__name__}: {e}"
            print(f"Unexpected JSON structure: {e}")
        except Exception as e:
            error_info = f"{type(e).__name__}: {e}"
            print(f"An unexpected error occurred: {e}")
        finally:
            payload = {
                "id": record_id,
                "timestamp": time.time(),
                "model": self.model_name,
                "prompt": extraction_prompts,
                "requested_region_count": requested_region_count,
                "num_region_text": num_region_text,
                "pending_coords": pending_coords,
                "prior_selection": prior_selection,
                "bounds": {
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                },
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": extraction_prompts},
                            {"type": "image_path", "path": str(disk_image_path)},
                        ],
                    }
                ],
                "response_text": response_text,
                "error": error_info,
            }
            if response_snapshot is None:
                response_snapshot = self._capture_response_snapshot(response)
            if response_snapshot is not None:
                payload["raw_response"] = response_snapshot
            self.recorder.save(record_id, image, payload)


    def vision_chat_png(self, png, prior_selection, num_region, pending_coords, x_min, x_max, y_min, y_max):

        region_mapping = {
            1: 'one',
            2: 'two',
            3: 'three',  # Add more mappings as needed
        }

        requested_region_count = num_region

        # Convert PIL Image to bytes
        image = Image.open(png).convert("RGB")

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # Getting the base64 string for GPT-vision
        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')

        num_region_text = region_mapping.get(num_region, 'one')  # 'unknown' as default value

        pending_regions = ", ".join(
            f"<coordinates> x_min={coord[0]}, x_max={coord[1]}, y_min={coord[2]}, y_max={coord[3]} </coordinates>"
            for coord in pending_coords
        )
       
        extraction_prompts = f"""You are tasked with improving an existing solution to a Traveling Salesman Problem (TSP) by selecting a sub-region where the routes can be significantly optimized. 
        Carefully consider the locations of the nodes (in red) and connected routes (in black) in the initial solution on a map. The boundary of the map is x_min={x_min-10000}, x_max={x_max+10000}, y_min={y_min-10000}, y_max={y_max+10000}.
        Please return {num_region_text} non-overlapping sub-rectangle(s) that you believe would most reduce total travel distance from further optimization by a downstream TSP solver.
        Analyze the problem-specific distribution to do meaningful selection. Select areas as large as you could to cover more nodes, which can bring larger improvement. Remember, if you don't see significant improvement, try selecting larger areas that cover more nodes based on your analysis of the prior selection trajectory
        Keep your output very brief as the following template. Don't tell me you cannot view or analyze the map. I don't want an excuse:
        <coordinates> x_min= 1,000, x_max= 2,000, y_min= 1,000, y_max=2,000 </coordinates> 
        \n Avoid selecting the same regions as follows, which are pending optimization:
        {pending_regions}
        
        \n Below are some previous selection trajectory. Learn from the trajectory to improve your selection capability. Please avoid selecting the same subrectangle.
        {prior_selection}
        """
        record_id = self.recorder.reserve_id()
        disk_image_path = self.recorder.image_path(record_id)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": extraction_prompts
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        response = None
        response_text = None
        error_info = None
        response_snapshot = None

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            response_snapshot = self._capture_response_snapshot(response)
            response_text = response.choices[0].message.content
            print(response_text)
            return response_text

        except requests.exceptions.RequestException as e:
            error_info = f"{type(e).__name__}: {e}"
            print(f"Network error occurred: {e}")
        except ValueError as e:
            error_info = f"{type(e).__name__}: {e}"
            print(f"Failed to parse JSON: {e}")
        except (KeyError, IndexError) as e:
            error_info = f"{type(e).__name__}: {e}"
            print(f"Unexpected JSON structure: {e}")
        except Exception as e:
            error_info = f"{type(e).__name__}: {e}"
            print(f"An unexpected error occurred: {e}")
        finally:
            payload = {
                "id": record_id,
                "timestamp": time.time(),
                "model": self.model_name,
                "prompt": extraction_prompts,
                "requested_region_count": requested_region_count,
                "num_region_text": num_region_text,
                "pending_coords": pending_coords,
                "prior_selection": prior_selection,
                "bounds": {
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                },
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": extraction_prompts},
                            {"type": "image_path", "path": str(disk_image_path)},
                        ],
                    }
                ],
                "response_text": response_text,
                "error": error_info,
            }
            if response_snapshot is None:
                response_snapshot = self._capture_response_snapshot(response)
            if response_snapshot is not None:
                payload["raw_response"] = response_snapshot
            self.recorder.save(record_id, image, payload)
