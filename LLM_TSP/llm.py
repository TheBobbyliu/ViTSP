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
import logging
import random
import textwrap
from pathlib import Path
import threading
from typing import Iterable, List, Optional, Sequence
from PIL import Image
import plotly.graph_objects as go
from plotly.basedatatypes import BaseFigure
import cProfile
import pstats
# from pdf2image import convert_from_path
import time

try:
    from anthropic import Anthropic  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Anthropic = None

try:
    from kaleido.scopes.plotly import PlotlyScope
except ImportError:  # pragma: no cover - optional dependency
    PlotlyScope = None

try:
    from matplotlib.figure import Figure as MPLFigure
except ImportError:  # pragma: no cover - optional dependency
    MPLFigure = None

MODEL_TYPES = {
    "gpt-4o": "gpt-4o",
    "o1": "o1",
    "qwen2.5-32b-reasoning": "Qwen/QwQ-32B",
    "qwen2.5-32b-v":"Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen2.5-7b-v":"Qwen/Qwen2.5-VL-7B-Instruct"
}

_PLOTLY_SCOPE_LOCK = threading.Lock()
_PLOTLY_SCOPE: Optional["PlotlyScope"] = None


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
        self.figures_dir = self.root / "figures"
        for path in (self.images_dir, self.data_dir, self.figures_dir):
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

    def figure_path(self, record_id: int) -> Path:
        return self.figures_dir / f"{record_id}.png"

    def _update_record(self, record_id: int, updates: dict) -> None:
        data_path = self.data_path(record_id)
        try:
            with data_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        data.update(updates)
        with data_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

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

    def save_selection_plot(self, record_id: Optional[int], fig, coordinates: Iterable[Iterable[float]]) -> None:
        if record_id is None or fig is None:
            return

        try:
            annotated = go.Figure(fig)
        except Exception:
            try:
                annotated = go.Figure(fig.to_dict())
            except Exception as exc:
                logging.getLogger(__name__).warning("Failed to clone figure for record %s: %s", record_id, exc)
                return

        valid_rectangles: List[List[float]] = []
        for coord in coordinates or []:
            if coord is None or len(coord) != 4:
                continue
            x_min, x_max, y_min, y_max = coord
            try:
                x0, x1 = float(x_min), float(x_max)
                y0, y1 = float(y_min), float(y_max)
            except (TypeError, ValueError):
                continue
            valid_rectangles.append([x0, x1, y0, y1])
            annotated.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                line=dict(color="#FF5733", width=2),
                fillcolor="rgba(255, 87, 51, 0.18)",
            )

        if not valid_rectangles:
            return

        fig_path = self.figure_path(record_id)
        try:
            annotated.write_image(fig_path, engine="kaleido")
        except Exception as exc:
            logging.getLogger(__name__).warning("Failed to save figure %s: %s", fig_path, exc)
            return

        self._update_record(
            record_id,
            {
                "selection_figure": str(fig_path),
                "selected_coordinates": valid_rectangles,
            },
        )


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


class VisionLLMBase:
    """Shared utilities for vision-capable LLM interfaces."""

    REGION_WORDS = {1: "one", 2: "two", 3: "three"}

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.recorder = LLMRecorder(self.model_name)
        self._last_record_id: Optional[int] = None
        self._logger = logging.getLogger(__name__)

    @property
    def last_record_id(self) -> Optional[int]:
        return self._last_record_id

    def _reserve_record(self) -> int:
        record_id = self.recorder.reserve_id()
        self._last_record_id = record_id
        return record_id

    def _build_prompt(
        self,
        prior_selection: str,
        num_region: int,
        pending_coords: Sequence[Sequence[float]],
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> tuple[str, str, int]:
        num_region_text = self.REGION_WORDS.get(num_region, str(num_region))
        pending_regions = ", ".join(
            f"<coordinates> x_min={coord[0]}, x_max={coord[1]}, y_min={coord[2]}, y_max={coord[3]} </coordinates>"
            for coord in pending_coords
            if coord is not None and len(coord) == 4
        )

        # the boundary is used to be: x_min={x_min-10000}, x_max={x_max+10000}, y_min={y_min-10000}, y_max={y_max+10000}
        # according to the paper, we change it to 10% extra
        x_delta = int((x_max - x_min) / 10)
        y_delta = int((y_max - y_min) / 10)
        prompt = textwrap.dedent(
            f"""
            You are tasked with improving an existing solution to a Traveling Salesman Problem (TSP) by selecting a sub-region where the routes can be significantly optimized.
            Carefully consider the locations of the nodes (in red) and connected routes (in black) in the initial solution on a map. The boundary of the map is x_min={x_min-x_delta}, x_max={x_max+x_delta}, y_min={y_min-y_delta}, y_max={y_max+y_delta}.
            Please return {num_region_text} non-overlapping sub-rectangle(s) that you believe would most reduce total travel distance from further optimization by a downstream TSP solver.
            Analyze the problem-specific distribution to do meaningful selection. Select areas as large as you could to cover more nodes, which can bring larger improvement. Remember, if you don't see significant improvement, try selecting larger areas that cover more nodes based on your analysis of the prior selection trajectory
            Keep your output very brief as the following template. Don't tell me you cannot view or analyze the map. I don't want an excuse:
            <coordinates> x_min= 1,000, x_max= 2,000, y_min= 1,000, y_max=2,000 </coordinates>

            Avoid selecting the same regions as follows, which are pending optimization:
            {pending_regions}

            Below are some previous selection trajectory. Learn from the trajectory to improve your selection capability. Please avoid selecting the same subrectangle.
            {prior_selection}
            """
        ).strip()
        return prompt, num_region_text, num_region

    @staticmethod
    def _image_from_figure(fig) -> Image.Image:
        buf = io.BytesIO()
        if isinstance(fig, BaseFigure):
            image_bytes = VisionLLMBase._plotly_figure_to_png(fig)
            buf.write(image_bytes)
        elif MPLFigure is not None and isinstance(fig, MPLFigure):
            fig.savefig(buf, format="png", bbox_inches="tight")
        elif hasattr(fig, "savefig"):
            fig.savefig(buf, format="png")
        else:
            fig.write_image(buf, format="png", engine="kaleido")
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
        return image

    @staticmethod
    def _plotly_figure_to_png(fig: BaseFigure) -> bytes:
        if PlotlyScope is None:
            return fig.to_image(format="png", engine="kaleido")

        global _PLOTLY_SCOPE
        try:
            with _PLOTLY_SCOPE_LOCK:
                if _PLOTLY_SCOPE is None:
                    _PLOTLY_SCOPE = PlotlyScope()
                scope = _PLOTLY_SCOPE
                return scope.transform(fig.to_dict(), format="png")
        except Exception as exc:  # pragma: no cover - defensive fallback
            logging.getLogger(__name__).warning(
                "PlotlyScope transform failed (%s); falling back to fig.to_image.",
                exc,
            )
            return fig.to_image(format="png", engine="kaleido")

    @staticmethod
    def _image_from_png(png: Path | str) -> Image.Image:
        image = Image.open(png).convert("RGB")
        return image

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

    @staticmethod
    def _capture_response_snapshot(response):
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

    def _log_interaction(
        self,
        *,
        record_id: int,
        image: Image.Image,
        prompt: str,
        requested_region_count: int,
        num_region_text: str,
        pending_coords: Sequence[Sequence[float]],
        prior_selection: str,
        bounds: dict,
        response_text: Optional[str],
        error_info: Optional[str],
        raw_response,
        assistant_messages: Optional[List[dict]] = None,
        request_latency: Optional[float] = None,
    ) -> None:
        disk_image_path = self.recorder.image_path(record_id)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_path", "path": str(disk_image_path)},
                ],
            }
        ]
        if assistant_messages:
            messages.extend(assistant_messages)

        normalized_pending = [
            list(coord) for coord in pending_coords if coord is not None and len(coord) == 4
        ]

        payload = {
            "id": record_id,
            "timestamp": time.time(),
            "model": self.model_name,
            "prompt": prompt,
            "requested_region_count": requested_region_count,
            "num_region_text": num_region_text,
            "pending_coords": normalized_pending,
            "prior_selection": prior_selection,
            "bounds": bounds,
            "messages": messages,
            "response_text": response_text,
            "error": error_info,
        }
        if raw_response is not None:
            payload["raw_response"] = raw_response
        if request_latency is not None:
            payload["request_latency"] = request_latency

        self.recorder.save(record_id, image, payload)

    def _run_with_image_common(
        self,
        image: Image.Image,
        prior_selection: str,
        num_region: int,
        pending_coords: Sequence[Sequence[float]] | None,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        invoke_fn,
    ) -> Optional[str]:
        pending_coords = list(pending_coords or [])
        prompt, num_region_text, requested_region_count = self._build_prompt(
            prior_selection, num_region, pending_coords, x_min, x_max, y_min, y_max
        )
        record_id = self._reserve_record()
        base64_image = self._image_to_base64(image)

        start_time = time.time()
        invoke_result = invoke_fn(prompt, base64_image)
        request_latency = time.time() - start_time
        assistant_messages: Optional[List[dict]] = None
        if isinstance(invoke_result, tuple) and len(invoke_result) == 4:
            response_text, error_info, raw_snapshot, assistant_messages = invoke_result
        else:
            response_text, error_info, raw_snapshot = invoke_result

        bounds = {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        }

        self._log_interaction(
            record_id=record_id,
            image=image,
            prompt=prompt,
            requested_region_count=requested_region_count,
            num_region_text=num_region_text,
            pending_coords=pending_coords,
            prior_selection=prior_selection,
            bounds=bounds,
            response_text=response_text,
            error_info=error_info,
            raw_response=raw_snapshot,
            assistant_messages=assistant_messages,
            request_latency=request_latency,
        )

        return response_text


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

class GPT(VisionLLMBase):
    def __init__(self, api_key, model_name="gpt-4-vision-preview", base_url = ""):
        super().__init__(model_name)
        self.api_key = api_key
        if base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)

    def _invoke(self, prompt: str, base64_image: str) -> tuple[Optional[str], Optional[str], Optional[dict]]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]

        response = None
        response_text = None
        error_info = None
        raw_snapshot = None

        try:
            if 'gpt-5' in self.model_name:
                print("reasoning effort: minimal")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    reasoning_effort='minimal'
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
            raw_snapshot = self._capture_response_snapshot(response)
            response_text = response.choices[0].message.content
        except requests.exceptions.RequestException as exc:
            error_info = f"{type(exc).__name__}: {exc}"
            self._logger.error("Network error when querying %s: %s", self.model_name, exc)
        except ValueError as exc:
            error_info = f"{type(exc).__name__}: {exc}"
            self._logger.error("Failed to parse response from %s: %s", self.model_name, exc)
        except (KeyError, IndexError) as exc:
            error_info = f"{type(exc).__name__}: {exc}"
            self._logger.error("Unexpected response structure from %s: %s", self.model_name, exc)
        except Exception as exc:  # pragma: no cover - defensive
            error_info = f"{type(exc).__name__}: {exc}"
            self._logger.error("Unhandled error during %s call: %s", self.model_name, exc)

        return response_text, error_info, raw_snapshot

    def vision_chat(self, fig, prior_selection, num_region, pending_coords, x_min, x_max, y_min, y_max):
        image = self._image_from_figure(fig)
        return self._run_with_image_common(
            image,
            prior_selection,
            num_region,
            pending_coords,
            x_min,
            x_max,
            y_min,
            y_max,
            self._invoke,
        )

    def vision_chat_png(self, png, prior_selection, num_region, pending_coords, x_min, x_max, y_min, y_max):
        image = self._image_from_png(png)
        return self._run_with_image_common(
            image,
            prior_selection,
            num_region,
            pending_coords,
            x_min,
            x_max,
            y_min,
            y_max,
            self._invoke,
        )


class RandomRegionLLM(VisionLLMBase):
    """Deterministic substitute that proposes random regions without calling an external model."""

    def __init__(
        self,
        model_name: str = "random-region-generator",
        seed: Optional[int] = None,
        min_fraction: float = 0.15,
        max_fraction: float = 0.45,
    ):
        super().__init__(model_name)
        self._rng = random.Random(seed)
        self._min_fraction = min_fraction
        self._max_fraction = max_fraction

    def _generate_rectangles(
        self,
        num_region: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        pending_coords: Sequence[Sequence[float]],
    ) -> List[tuple[int, int, int, int]]:
        if num_region <= 0:
            return []

        width_range = max(x_max - x_min, 1.0)
        height_range = max(y_max - y_min, 1.0)

        min_width = max(width_range * self._min_fraction, 1.0)
        max_width = max(width_range * self._max_fraction, min_width)
        min_height = max(height_range * self._min_fraction, 1.0)
        max_height = max(height_range * self._max_fraction, min_height)

        blocked = {
            tuple(int(round(v)) for v in coord)
            for coord in pending_coords
            if coord is not None and len(coord) == 4
        }
        generated: List[tuple[int, int, int, int]] = []
        generated_set = set()

        attempts = 0
        max_attempts = max(num_region, 1) * 25
        while len(generated) < num_region and attempts < max_attempts:
            attempts += 1

            width = min(self._rng.uniform(min_width, max_width), width_range)
            height = min(self._rng.uniform(min_height, max_height), height_range)

            if width_range <= 1.0:
                x0 = x_min
            else:
                x0 = self._rng.uniform(x_min, x_max - width)
            if height_range <= 1.0:
                y0 = y_min
            else:
                y0 = self._rng.uniform(y_min, y_max - height)

            x1 = x0 + width
            y1 = y0 + height

            rect = (
                int(round(min(x0, x1))),
                int(round(max(x0, x1))),
                int(round(min(y0, y1))),
                int(round(max(y0, y1))),
            )

            x0i, x1i, y0i, y1i = rect
            x0i = max(x0i, int(round(x_min)))
            x1i = min(x1i, int(round(x_max)))
            y0i = max(y0i, int(round(y_min)))
            y1i = min(y1i, int(round(y_max)))

            if x1i <= x0i or y1i <= y0i:
                continue

            normalized = (x0i, x1i, y0i, y1i)
            if normalized in blocked or normalized in generated_set:
                continue

            generated.append(normalized)
            generated_set.add(normalized)

        if not generated:
            fallback_x0 = int(round(x_min))
            fallback_x1 = int(round(x_max))
            fallback_y0 = int(round(y_min))
            fallback_y1 = int(round(y_max))
            if fallback_x1 <= fallback_x0:
                fallback_x1 = fallback_x0 + 1
            if fallback_y1 <= fallback_y0:
                fallback_y1 = fallback_y0 + 1
            generated.append((fallback_x0, fallback_x1, fallback_y0, fallback_y1))

        return generated

    def _invoke_random(
        self,
        prompt: str,
        _base64_image: str,
        *,
        num_region: int,
        pending_coords: Sequence[Sequence[float]],
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ):
        rectangles = self._generate_rectangles(num_region, x_min, x_max, y_min, y_max, pending_coords)
        response_text = "\n".join(
            f"<coordinates> x_min={rect[0]}, x_max={rect[1]}, y_min={rect[2]}, y_max={rect[3]} </coordinates>"
            for rect in rectangles
        )
        assistant_messages = [
            {"role": "assistant", "content": [{"type": "text", "text": response_text or ""}]}
        ]
        raw_snapshot = {
            "generated_rectangles": [list(rect) for rect in rectangles],
            "prompt": prompt,
        }
        return response_text, None, raw_snapshot, assistant_messages

    def vision_chat(self, fig, prior_selection, num_region, pending_coords, x_min, x_max, y_min, y_max):
        image = self._image_from_figure(fig)
        pending_list = list(pending_coords or [])
        return self._run_with_image_common(
            image,
            prior_selection,
            num_region,
            pending_list,
            x_min,
            x_max,
            y_min,
            y_max,
            lambda prompt, base64_image: self._invoke_random(
                prompt,
                base64_image,
                num_region=num_region,
                pending_coords=pending_list,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            ),
        )

    def vision_chat_png(self, png, prior_selection, num_region, pending_coords, x_min, x_max, y_min, y_max):
        image = self._image_from_png(png)
        pending_list = list(pending_coords or [])
        return self._run_with_image_common(
            image,
            prior_selection,
            num_region,
            pending_list,
            x_min,
            x_max,
            y_min,
            y_max,
            lambda prompt, base64_image: self._invoke_random(
                prompt,
                base64_image,
                num_region=num_region,
                pending_coords=pending_list,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            ),
        )


class ClaudeAI(VisionLLMBase):
    def __init__(self, api_key, model_name: str = "claude-3-opus-20240229", max_tokens: int = 1024):
        if Anthropic is None:
            raise ImportError("anthropic package is required to use ClaudeAI.")
        super().__init__(model_name)
        self.client = Anthropic(api_key=api_key)
        self.max_tokens = max_tokens

    def _invoke(self, prompt: str, base64_image: str) -> tuple[Optional[str], Optional[str], Optional[dict]]:
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image},
            },
        ]
        response = None
        response_text = None
        error_info = None
        raw_snapshot = None

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": content}],
            )
            raw_snapshot = self._capture_response_snapshot(response)
            blocks = getattr(response, "content", None)
            if blocks:
                parts = []
                for block in blocks:
                    block_type = getattr(block, "type", None)
                    text_value = getattr(block, "text", "")
                    if block_type == "text" and text_value:
                        parts.append(text_value)
                joined = "\n".join(parts).strip()
                response_text = joined if joined else None
        except Exception as exc:  # pragma: no cover - defensive
            error_info = f"{type(exc).__name__}: {exc}"
            self._logger.error("Claude vision request failed: %s", exc)

        return response_text, error_info, raw_snapshot

    def vision_chat(self, fig, prior_selection, num_region, pending_coords, x_min, x_max, y_min, y_max):
        image = self._image_from_figure(fig)
        return self._run_with_image_common(
            image,
            prior_selection,
            num_region,
            pending_coords,
            x_min,
            x_max,
            y_min,
            y_max,
            self._invoke,
        )

    def vision_chat_png(self, png, prior_selection, num_region, pending_coords, x_min, x_max, y_min, y_max):
        image = self._image_from_png(png)
        return self._run_with_image_common(
            image,
            prior_selection,
            num_region,
            pending_coords,
            x_min,
            x_max,
            y_min,
            y_max,
            self._invoke,
        )
