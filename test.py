import argparse
import ast
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

DEFAULT_SYSTEM_PROMPT = "You are a careful and consistent visual comparison expert."
DEFAULT_REASONING_PROMPT = "You are a careful visual reasoning assistant."
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "384"))
DEFAULT_FOLLOWUP_MAX_NEW_TOKENS = int(os.environ.get("FOLLOWUP_MAX_NEW_TOKENS", "768"))
DEFAULT_MAX_LONG_EDGE = int(os.environ.get("MAX_LONG_EDGE", "1344"))
DEFAULT_GPU_ID = int(os.environ.get("GPU_ID", "0"))
DEFAULT_SEED = int(os.environ.get("SEED", "42"))
DEFAULT_TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.01"))
DEFAULT_TOP_P = float(os.environ.get("TOP_P", "0.1"))
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "5"))
ALLOWED_DIMENSIONS = ("semantic", "structure", "lowlevel")
ALLOWED_MODES = ("simple", "cot", "score")


@dataclass
class PromptLibrary:
    semantic: str
    structure: str
    lowlevel: str
    fidelity: str
    semantic_followup: str
    lowlevel_followup: str

    @classmethod
    def from_dir(cls, prompt_dir: Path) -> "PromptLibrary":
        return cls(
            semantic=(prompt_dir / "semantic.txt").read_text(encoding="utf-8"),
            structure=(prompt_dir / "structure.txt").read_text(encoding="utf-8"),
            lowlevel=(prompt_dir / "lowlevel.txt").read_text(encoding="utf-8"),
            fidelity=(prompt_dir / "fidelity.txt").read_text(encoding="utf-8"),
            semantic_followup=(prompt_dir / "semantic_followup.txt").read_text(encoding="utf-8"),
            lowlevel_followup=(prompt_dir / "lowlevel_followup.txt").read_text(encoding="utf-8"),
        )


@dataclass
class InferenceConfig:
    ckpt: str
    gpu_id: int = DEFAULT_GPU_ID
    seed: int = DEFAULT_SEED
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    followup_max_new_tokens: int = DEFAULT_FOLLOWUP_MAX_NEW_TOKENS
    max_long_edge: int = DEFAULT_MAX_LONG_EDGE
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    reasoning_prompt: str = DEFAULT_REASONING_PROMPT


def set_deterministic(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


class LocalJudge:
    def __init__(self, config: InferenceConfig, prompts: PromptLibrary) -> None:
        self.config = config
        self.prompts = prompts
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
        set_deterministic(config.seed)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda"):
            torch.cuda.set_device(0)
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_num_threads(1)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.ckpt,
            torch_dtype="auto",
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(config.ckpt)
        self.model.eval()
        self.model.requires_grad_(False)

    def generate_one(self, messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
        with torch.inference_mode():
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs = []
            for msg in messages:
                for item in msg.get("content", []):
                    if isinstance(item, dict) and item.get("type") == "image":
                        image_inputs.append(item["image"])

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                num_beams=1,
                use_cache=True,
            )
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            return self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

    def run_case(
        self,
        input_image: str,
        output_image: str,
        task_prompt: str,
        modes: Sequence[str],
        dimensions: Sequence[str],
    ) -> Dict[str, Any]:
        before = safe_open_image(input_image)
        after = safe_open_image(output_image)
        before, after = preprocess_pair_resize(before, after, self.config.max_long_edge)

        mode_results: Dict[str, Dict[str, Any]] = {}
        display_texts: Dict[str, str] = {}
        for mode in modes:
            mode_result, display_text = self._run_one_mode(
                before=before,
                after=after,
                task_prompt=task_prompt,
                mode=mode,
                dimensions=dimensions,
                input_image=input_image,
                output_image=output_image,
            )
            mode_results[mode] = mode_result
            display_texts[mode] = display_text

        if len(modes) == 1:
            compact_result: Dict[str, Any] = mode_results[modes[0]]
        else:
            compact_result = mode_results

        return {
            "result": compact_result,
            "display_texts": display_texts,
        }

    def _run_one_mode(
        self,
        before: Image.Image,
        after: Image.Image,
        task_prompt: str,
        mode: str,
        dimensions: Sequence[str],
        input_image: str,
        output_image: str,
    ) -> Tuple[Dict[str, Any], str]:
        parsed: Dict[str, Any] = {}
        skip_lowlevel = is_restoration_prompt(task_prompt)

        if mode in {"simple", "cot"}:
            if "semantic" in dimensions:
                semantic_raw = self.generate_one(
                    build_messages(self.config.system_prompt, before, after, task_prompt, self.prompts.semantic),
                    self.config.max_new_tokens,
                )
                parsed["Semantic"] = extract_first_json_obj(semantic_raw)

            if "structure" in dimensions:
                structure_raw = self.generate_one(
                    build_messages(self.config.system_prompt, before, after, task_prompt, self.prompts.structure),
                    self.config.max_new_tokens,
                )
                parsed["Structure"] = extract_first_json_obj(structure_raw)

            if "lowlevel" in dimensions:
                if skip_lowlevel:
                    parsed["Low-Level"] = {
                        "answer": "NULL",
                        "problem": "NULL",
                        "skipped": True,
                        "reason": "Prompt indicates restoration task; low-level branch skipped.",
                    }
                else:
                    lowlevel_raw = self.generate_one(
                        build_messages(self.config.system_prompt, before, after, task_prompt, self.prompts.lowlevel),
                        self.config.max_new_tokens,
                    )
                    parsed["Low-Level"] = extract_first_json_obj(lowlevel_raw)

            if mode == "cot":
                parsed["Semantic_Followup"], _ = self._semantic_followup(
                    parsed, before, after, task_prompt, dimensions
                )
                parsed["Low-Level_Followup"], _ = self._lowlevel_followup(
                    parsed, before, after, task_prompt, dimensions, skip_lowlevel
                )

        if mode == "score":
            fidelity_raw = self.generate_one(
                build_messages(self.config.system_prompt, before, after, task_prompt, self.prompts.fidelity),
                self.config.max_new_tokens,
            )
            parsed["Fidelity"] = extract_first_json_obj(fidelity_raw)

        display_rows = build_display_rows(parsed, mode, dimensions)
        if mode == "score":
            compact_result = {
                "Score": parsed.get("Fidelity"),
            }
        elif mode == "simple":
            compact_result = {}
            if "semantic" in dimensions:
                compact_result["Semantic"] = parsed.get("Semantic")
            if "structure" in dimensions:
                compact_result["Structure"] = parsed.get("Structure")
            if "lowlevel" in dimensions:
                compact_result["Low-Level"] = parsed.get("Low-Level")
        else:
            compact_result = {}
            if "semantic" in dimensions:
                compact_result["Semantic"] = parsed.get("Semantic")
            if "structure" in dimensions:
                compact_result["Structure"] = parsed.get("Structure")
            if "lowlevel" in dimensions:
                compact_result["Low-Level"] = parsed.get("Low-Level")
            if "semantic" in dimensions:
                compact_result["Semantic_think"] = parsed.get("Semantic_Followup")
            if "lowlevel" in dimensions:
                compact_result["Low-Level_think"] = parsed.get("Low-Level_Followup")
        return compact_result, "\n".join(display_rows)

    def _semantic_followup(
        self,
        parsed: Dict[str, Any],
        before: Image.Image,
        after: Image.Image,
        task_prompt: str,
        dimensions: Sequence[str],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if "semantic" not in dimensions:
            return None, None
        semantic = parsed.get("Semantic")
        if not semantic or semantic.get("answer") != "No":
            return None, None

        template = self.prompts.semantic_followup.format(
            task_prompt=task_prompt,
            problem_json=json.dumps(normalize_problem_list(semantic.get("problem")), ensure_ascii=False),
        )
        raw = self.generate_one(
            build_messages(self.config.reasoning_prompt, before, after, task_prompt, template),
            self.config.followup_max_new_tokens,
        )
        return _canonicalize_followup_obj(extract_first_json_obj(raw)), raw

    def _lowlevel_followup(
        self,
        parsed: Dict[str, Any],
        before: Image.Image,
        after: Image.Image,
        task_prompt: str,
        dimensions: Sequence[str],
        skip_lowlevel: bool,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if "lowlevel" not in dimensions:
            return None, None
        if skip_lowlevel:
            return {"skipped": True, "reason": "Prompt indicates restoration task; low-level followup skipped."}, None

        lowlevel = parsed.get("Low-Level")
        if not lowlevel or lowlevel.get("answer") != "No":
            return None, None

        template = self.prompts.lowlevel_followup.format(
            task_prompt=task_prompt,
            problem_json=json.dumps(normalize_problem_list(lowlevel.get("problem")), ensure_ascii=False),
        )
        raw = self.generate_one(
            build_messages(self.config.reasoning_prompt, before, after, task_prompt, template),
            self.config.followup_max_new_tokens,
        )
        return _canonicalize_followup_obj(extract_first_json_obj(raw)), raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local image-pair tests from a JSONL file.")
    parser.add_argument("--jsonl", required=True, help="Input JSONL file.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint directory.")
    parser.add_argument("--mode", default="simple")
    parser.add_argument("--dimensions", default="semantic,structure,lowlevel")
    parser.add_argument("--output-jsonl", default="results.jsonl", help="Output JSONL path.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gpu-id", type=int, default=DEFAULT_GPU_ID)
    parser.add_argument("--prompt-dir", default="prompts")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--followup-max-new-tokens", type=int, default=DEFAULT_FOLLOWUP_MAX_NEW_TOKENS)
    parser.add_argument("--max-long-edge", type=int, default=DEFAULT_MAX_LONG_EDGE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser.parse_args()


def parse_dimensions(value: str) -> List[str]:
    dims = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in dims if item not in ALLOWED_DIMENSIONS]
    if invalid:
        raise ValueError(f"Invalid dimensions: {invalid}")
    if not dims:
        raise ValueError("At least one dimension is required.")
    return dims


def parse_modes(value: str) -> List[str]:
    modes = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in modes if item not in ALLOWED_MODES]
    if invalid:
        raise ValueError(f"Invalid modes: {invalid}")
    if not modes:
        raise ValueError("At least one mode is required.")
    return modes


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_number} is not a JSON object.")
            yield item


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_case(item: Dict[str, Any], line_index: int) -> Dict[str, Any]:
    case_id = item.get("id")
    input_image = item.get("input_image") or item.get("before_image")
    output_image = item.get("output_image") or item.get("after_image")
    if case_id is None or str(case_id) == "":
        raise ValueError(f"Missing id in JSONL line {line_index}.")
    if not input_image or not output_image:
        raise ValueError(f"Missing input_image/output_image in JSONL line {line_index}.")
    return {
        "id": str(case_id),
        "input_image": str(input_image),
        "output_image": str(output_image),
        "prompt": str(item.get("prompt", "")),
    }


def safe_open_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def preprocess_pair_resize(img1: Image.Image, img2: Image.Image, max_long_edge: int) -> Tuple[Image.Image, Image.Image]:
    if img1.mode != "RGB":
        img1 = img1.convert("RGB")
    if img2.mode != "RGB":
        img2 = img2.convert("RGB")

    max_long = max(max(img1.size), max(img2.size))
    if max_long > max_long_edge:
        scale = float(max_long_edge) / float(max_long)

        def resize_keep_ar(image: Image.Image) -> Image.Image:
            width, height = image.size
            return image.resize(
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                resample=Image.BICUBIC,
            )

        img1 = resize_keep_ar(img1)
        img2 = resize_keep_ar(img2)

    if max(img1.size) >= max(img2.size):
        target = img1
        other = img2
        target_is_img1 = True
    else:
        target = img2
        other = img1
        target_is_img1 = False
    other = other.resize(target.size, resample=Image.BICUBIC)
    return (target, other) if target_is_img1 else (other, target)


def build_messages(
    system_prompt: str,
    ref_img: Image.Image,
    cand_img: Image.Image,
    task_prompt: str,
    template: str,
) -> List[Dict[str, Any]]:
    task_text = f"The task prompt is: {task_prompt}\n{template}"
    messages: List[Dict[str, Any]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "The first image "},
                {"type": "image", "image": ref_img},
                {"type": "text", "text": ": Before processing.\nThe second image "},
                {"type": "image", "image": cand_img},
                {"type": "text", "text": ": After processing.\n"},
                {"type": "text", "text": task_text},
            ],
        }
    )
    return messages


def _extract_balanced_brace_blocks(text: str) -> List[str]:
    blocks: List[str] = []
    if not text:
        return blocks
    stripped = text.strip()
    stripped = re.sub(r"^\s*```json\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"^\s*```\s*", "", stripped)
    stripped = re.sub(r"\s*```\s*$", "", stripped)

    start = stripped.find("{")
    while start != -1:
        depth = 0
        in_string = False
        string_char = ""
        escaped = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == string_char:
                    in_string = False
                continue
            if char in ["'", '"']:
                in_string = True
                string_char = char
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(stripped[start:index + 1])
                    start = stripped.find("{", index + 1)
                    break
        else:
            break
    return blocks


def extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    blocks = _extract_balanced_brace_blocks(text) or [text.strip()]
    for block in blocks:
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def normalize_problem_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in [None, "NULL"]:
        return []
    return [str(value)]


def is_restoration_prompt(prompt: str) -> bool:
    prompt_lower = prompt.strip().lower()
    return any(token in prompt_lower for token in ("restore", "restoration", "recover", "recovery"))


def _canonicalize_followup_obj(obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return obj
    out = dict(obj)
    if "answer" not in out and "problem" in out:
        out["answer"] = out.pop("problem")
    return out


def _stringify_problem(value: Any) -> str:
    if value in [None, "NULL"]:
        return "NULL"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "NULL"
    if isinstance(value, dict):
        parts = [f"{key}: {item}" for key, item in value.items() if item not in [None, "", "NULL", {}, []]]
        return "; ".join(parts) if parts else "NULL"
    return str(value)


def _format_main_branch(name: str, obj: Optional[Dict[str, Any]]) -> str:
    if obj is None:
        return f"{name}: Not run"
    answer = obj.get("answer")
    if answer == "Yes":
        return f"{name}: No issue"
    if answer == "No":
        return f"{name}: Issue detected, {_stringify_problem(obj.get('problem'))}"
    if answer == "NULL":
        return f"{name}: Skipped"
    return f"{name}: Parse failed"


def _format_followup_branch(name: str, obj: Optional[Dict[str, Any]]) -> str:
    if obj is None:
        return f"{name}: Not triggered"
    if isinstance(obj, dict) and obj.get("skipped"):
        return f"{name}: Skipped"
    normalized = _canonicalize_followup_obj(obj)
    think = normalized.get("think") if isinstance(normalized, dict) else None
    answer = normalized.get("answer") if isinstance(normalized, dict) else None
    if think in [None, "", "NULL"] and answer in [None, "", "NULL", {}, []]:
        return f"{name}: No issue"
    return f"{name}: {{think: {json.dumps(think, ensure_ascii=False) if isinstance(think, (dict, list)) else think}, answer: {json.dumps(answer, ensure_ascii=False) if isinstance(answer, (dict, list)) else answer}}}"


def build_display_rows(parsed: Dict[str, Any], mode: str, dimensions: Sequence[str]) -> List[str]:
    rows = [f"Mode: {mode}", f"Dimensions: {', '.join(dimensions)}"]
    if mode in {"simple", "cot"}:
        if "semantic" in dimensions:
            rows.append(_format_main_branch("Semantic", parsed.get("Semantic")))
        if "structure" in dimensions:
            rows.append(_format_main_branch("Structure", parsed.get("Structure")))
        if "lowlevel" in dimensions:
            rows.append(_format_main_branch("Low-Level", parsed.get("Low-Level")))
        if mode == "cot":
            if "semantic" in dimensions:
                rows.append(_format_followup_branch("Semantic Followup", parsed.get("Semantic_Followup")))
            if "lowlevel" in dimensions:
                rows.append(_format_followup_branch("Low-Level Followup", parsed.get("Low-Level_Followup")))
    if mode == "score":
        fidelity = parsed.get("Fidelity")
        score = fidelity.get("score") if isinstance(fidelity, dict) else None
        rows.append(f"Fidelity Score: {score if score is not None else 'NULL'}")
    return rows


def main() -> None:
    args = parse_args()
    prompt_dir = Path(args.prompt_dir)
    prompts = PromptLibrary.from_dir(prompt_dir)
    config = InferenceConfig(
        ckpt=args.ckpt,
        gpu_id=args.gpu_id,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        followup_max_new_tokens=args.followup_max_new_tokens,
        max_long_edge=args.max_long_edge,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    judge = LocalJudge(config, prompts)
    modes = parse_modes(args.mode)
    dimensions = parse_dimensions(args.dimensions)

    input_rows = list(iter_jsonl(Path(args.jsonl)))
    output_rows: List[Dict[str, Any]] = []
    for line_index, raw_case in enumerate(
        tqdm(input_rows, desc="Running cases", unit="case"),
        start=1,
    ):
        case = normalize_case(raw_case, line_index)
        result = judge.run_case(
            input_image=case["input_image"],
            output_image=case["output_image"],
            task_prompt=case["prompt"],
            modes=modes,
            dimensions=dimensions,
        )
        output_rows.append(
            {
                "id": case["id"],
                "input_image": case["input_image"],
                "output_image": case["output_image"],
                "prompt": case["prompt"],
                "result": result["result"],
            }
        )

    write_jsonl(Path(args.output_jsonl), output_rows)
    print(f"Saved {len(output_rows)} results to {args.output_jsonl}")


if __name__ == "__main__":
    main()
