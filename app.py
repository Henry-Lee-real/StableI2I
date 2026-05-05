import os
import json
import re
import ast
import tempfile
from typing import List, Dict, Any, Optional

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/petrelfs/lijiayang1/stableI2I/ckpt")
GPU_ID = int(os.environ.get("GPU_ID", "0"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "10004"))
SYSTEM_PROMPT = "You are a careful and consistent visual comparison expert."
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
FOLLOWUP_MAX_NEW_TOKENS = int(os.environ.get("FOLLOWUP_MAX_NEW_TOKENS", "1024"))
MAX_LONG_EDGE = int(os.environ.get("MAX_LONG_EDGE", "1344"))

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

# =========================
# Fill these examples
# =========================
EXAMPLE_CASES = [
    {
        "name": "Example 1",
        "input_image": "example/000155856.jpg",
        "output_image": "example/000155856_dup2.png",
        "prompt": "Add a wooden bench along the path.",
    },
    {
        "name": "Example 2",
        "input_image": "example/000369014.jpg",
        "output_image": "example/000369014_out.png",
        "prompt": "Change the snowy environment surrounding the house to a beach setting with sand and palm trees.",
    },
    {
        "name": "Example 3",
        "input_image": "example/0164.png",
        "output_image": "example/0164_out.png",
        "prompt": "Restore the image.",
    },
    {
        "name": "Example 4",
        "input_image": "example/0341.png",
        "output_image": "example/0341_out.png",
        "prompt": "Restore the image.",
    },
]


PROMPT_SEMANTIC = """Please determine whether the regions that should remain unchanged between the pre- and post-processed images exhibit any semantic errors, that is, whether there are additions, deletions, or modifications of semantic content relative to the pre-processed image.
If no specific task type is provided, simply determine whether the two images are completely identical.
Return your decision in a single line of valid JSON with the format:
{"answer": "Yes", "problem": "NULL"} if the images are consistent,
otherwise {"answer": "No", "problem": ["add", "replace", "remove"]}.
"No" is used whenever any potential inconsistency is detected."""


PROMPT_STRUCTURE = """Please determine whether the texture-consistent regions that should remain unchanged between the pre- and post-process images are indeed consistent (e.g., unchanged areas remain identical, or for restoration tasks, whether the overall texture and color are consistent).
If no specific task type is given, simply judge whether the two images are identical.
Major inconsistencies generally fall into two categories: structural misalignment, texture repainting.
Return your decision in a single line of valid JSON with the format: {"answer": "Yes", "problem": "NULL"} if the images are consistent, otherwise {"answer": "No", "problem": ["misalignment", "repainting"]}, where the "problem" field should reflect the most dominant issue observed between the two images."""


PROMPT_LOWLEVEL = """Please determine whether the pre-processed image has undergone any low-level degradation or shift after processing. Specifically, check whether there is any degradation (e.g., blur, noise), color cast, or newly introduced artifacts. If the processing described in the task type is explicitly related to any of the above (e.g., denoising, deblurring, color correction, artifact removal), then this case should be ignored.
If no specific task type is given, simply judge whether the two images are identical.
Return your decision in a single line of valid JSON with the format:
In the ignored case, output:{"answer": "NULL", "problem": "NULL"},
{"answer": "Yes", "problem": "NULL"} if the images are consistent,
otherwise {"answer": "No", "problem": ["noise", "blur", "color cast", "exposure degradation", "artifact"]}."""


FIDELITY_PROMPT = """Based on the task prompt, evaluate the fidelity of the after-processed image relative to the before image (i.e., whether non-target content that should remain unchanged is preserved, and whether the requested edit/restoration is performed without introducing unintended semantic/structural/visual drift). Give a score from 0 to 10 (higher is better).
Return only one line of valid JSON:
{"score": XX}
"""


SEMANTIC_FOLLOWUP_TEMPLATE = """The task prompt is: XXX

Assume this example DOES contain issues: semantic drift has occurred within regions that should be preserved.
Initial Drift type(s): YYY

Your task is to analyze the problem strictly in two stages. The second stage MUST be derived from the first stage.

1) Preservation analysis (think):
   - Identify the intended edit target region(s) according to the task prompt.
   - Explicitly state which changes are allowed or can be ignored because they fall inside the intended edit scope.
   - Identify the regions/elements that must be preserved (non-edit regions), and list them as a concrete checklist with brief justification.
   - Carefully verify the Initial Drift type(s): YYY against the preserved regions.
     * By default, assume they are correct.
     * Only if you are confident a drift type does NOT actually apply to violations on preserved regions, you may discard it.

2) Final answer reporting (answer):
   - Report ONLY issues that violate the preservation checklist defined in the think stage.
   - Every reported issue must be grounded in the preserved regions identified above.
   - Anything stated as allowed or ignorable in the think stage MUST NOT appear here.
   - Use only drift type keys that you have confirmed as applicable after the think stage.
   - Do NOT include drift types that do not have real violations.
   - If no valid remaining issue exists, output an empty object for answer: {}.

Output MUST be a single valid JSON object and nothing else:
{
  "think": "...",
  "answer": {
    "add": "...",
    "replace": "...",
    "remove": "..."
  }
}
"""


LOWLEVEL_FOLLOWUP_TEMPLATE = """The first image <image>: Before processing.
The second image <image>: After processing.
The task prompt is: XXX

You are evaluating whether the AFTER image introduces unintended LOW-LEVEL degradation/shift in regions that should be preserved.

Candidate degradation type(s) (use ONLY these keys if applicable): YYY

Your task is to analyze strictly in two stages:

1) Preservation & scope analysis (think):
   - Identify the intended target region(s) implied by the task prompt.
   - State which changes are allowed ONLY if they occur strictly inside the intended target region(s).
   - Clarify that low-level degradations (noise/blur/color cast/exposure issues/artifacts) are NOT intended unless the task prompt explicitly requests low-level enhancement/removal.
   - Provide a checklist of what must be preserved (non-target regions/elements) with brief justification.
   - If the task prompt explicitly requests low-level processing (e.g., denoise/deblur/color correction/exposure enhancement/artifact removal), then low-level changes consistent with that request and confined to the intended scope may be treated as allowed; state this in "think".

2) Final answer reporting (answer):
   - Report ONLY low-level degradations that violate the scope above (i.e., occur in preserved regions or exceed intended scope).
   - If no violation is found, output an empty object: answer = {}.
   - Use ONLY the keys provided in YYY. Do not invent new keys.
   - For each key you include, describe: where it appears, how it differs from BEFORE, and the visual symptom.

Output MUST be a single valid JSON object and nothing else:
{
  "think": "...",
  "answer": {
    "noise": "...",
    "blur": "...",
    "color cast": "...",
    "exposure degradation": "...",
    "artifact": "..."
  }
}
"""


def _extract_balanced_brace_blocks(s: str) -> List[str]:
    blocks: List[str] = []
    if not s:
        return blocks
    t = s.strip()
    t = re.sub(r"^\s*```json\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*```\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    start = t.find("{")
    while start != -1:
        depth = 0
        in_str = False
        str_ch = ""
        escape = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == str_ch:
                    in_str = False
                continue
            else:
                if ch in ["'", '"']:
                    in_str = True
                    str_ch = ch
                    continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(t[start:i + 1])
                    start = t.find("{", i + 1)
                    break
        else:
            break
    return blocks


def extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    blocks = _extract_balanced_brace_blocks(text)
    if not blocks:
        blocks = [text.strip()]
    for b in blocks:
        try:
            obj = json.loads(b)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(b)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def normalize_problem_list(x):
    if isinstance(x, list):
        return x
    if x is None or x == "NULL":
        return []
    return [str(x)]


def obj_to_score(obj: Optional[Dict[str, Any]]) -> Optional[int]:
    if not obj or "score" not in obj:
        return None
    try:
        return max(0, min(10, int(round(float(obj.get("score"))))))
    except Exception:
        return None


def answer_to_vote(obj: Optional[Dict[str, Any]]) -> Optional[int]:
    if not obj or "answer" not in obj:
        return None
    ans = str(obj.get("answer")).strip()
    if ans == "Yes":
        return 1
    if ans == "No":
        return 0
    if ans == "NULL":
        return None
    return None


def votes_to_score(votes: List[Optional[int]], default_if_all_null: float = 0.0) -> float:
    valid = [v for v in votes if v is not None]
    if not valid:
        return float(default_if_all_null)
    return float(sum(valid)) / float(len(valid))


def _is_null_like(x) -> bool:
    return x is None or x == "NULL"


def _stringify_problem(x) -> str:
    if _is_null_like(x):
        return "NULL"
    if isinstance(x, list):
        if len(x) == 0:
            return "NULL"
        return ", ".join(str(v) for v in x)
    if isinstance(x, dict):
        parts = []
        for k, v in x.items():
            if v in [None, "", "NULL", {}, []]:
                continue
            parts.append(f"{k}: {v}")
        return "; ".join(parts) if parts else "NULL"
    return str(x)


def _canonicalize_followup_obj(obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return obj
    out = dict(obj)
    if "answer" not in out and "problem" in out:
        out["answer"] = out.pop("problem")
    return out


def _format_main_branch(name: str, obj: Optional[Dict[str, Any]], null_text: str = "Skipped") -> str:
    if obj is None:
        return f"{name}: No issue"

    ans = obj.get("answer")

    if ans == "Yes":
        return f"{name}: No issue"
    if ans == "No":
        return f"{name}: Issue detected, {_stringify_problem(obj.get('problem'))}"
    if ans == "NULL":
        return f"{name}: {null_text}"

    return f"{name}: No issue"


def _inline_value_for_cot(x: Any) -> str:
    if x in [None, "", "NULL"]:
        return "NULL"
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def _format_followup_branch(
    name: str,
    obj: Optional[Dict[str, Any]],
    null_text: str = "No issue",
    skipped_text: str = "Skipped"
) -> str:
    if obj is None:
        return f"{name}: {null_text}"

    if isinstance(obj, dict) and obj.get("skipped"):
        return f"{name}: {skipped_text}"

    obj = _canonicalize_followup_obj(obj)

    think = None
    answer = None
    if isinstance(obj, dict):
        think = obj.get("think")
        answer = obj.get("answer")

    if think in [None, "", "NULL"] and answer in [None, "", "NULL", {}, []]:
        return f"{name}: {null_text}"

    return f"{name}: {{think: {_inline_value_for_cot(think)}, answer: {_inline_value_for_cot(answer)}}}"


def build_display_rows(parsed: Dict[str, Any], fidelity_score: Optional[int]) -> List[str]:
    rows = []
    rows.append(_format_main_branch("1. Semantic Level", parsed.get("Semantic"), null_text="Skipped"))
    rows.append(_format_main_branch("2. Structure Level", parsed.get("Structure"), null_text="Skipped"))
    rows.append(_format_main_branch("3. Low-Level Perception", parsed.get("Low-Level"), null_text="Skipped"))
    rows.append(_format_followup_branch("4. Semantic Level COT", parsed.get("Semantic_Followup"), null_text="No issue", skipped_text="No issue"))
    rows.append(_format_followup_branch("5. Low-Level Perception COT", parsed.get("Low-Level_Followup"), null_text="No issue", skipped_text="Skipped"))

    if fidelity_score is None:
        rows.append("6. Score: NULL (max 10)")
    else:
        rows.append(f"6. Score: {fidelity_score} (max 10)")

    return rows

def is_restoration_prompt(prompt: str) -> bool:
    if not prompt:
        return False
    p = str(prompt).strip().lower()
    restoration_keywords = [
        "restore",
        "restoration",
        "recover",
        "recovery",
    ]
    return any(k in p for k in restoration_keywords)


def build_messages(system_prompt: str, ref_img, cand_img, prompt_text: str, template: str):
    task_text = f"The task prompt is: {prompt_text}\n{template}"
    messages = []
    if system_prompt.strip():
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        })
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "The first image "},
            {"type": "image", "image": ref_img},
            {"type": "text", "text": ": Before processing.\nThe second image "},
            {"type": "image", "image": cand_img},
            {"type": "text", "text": ": After processing.\n"},
            {"type": "text", "text": task_text},
        ],
    })
    return messages


def preprocess_pair_resize(img1, img2, max_long_edge: int = 1344):
    if img1.mode != "RGB":
        img1 = img1.convert("RGB")
    if img2.mode != "RGB":
        img2 = img2.convert("RGB")

    def long_edge(w, h):
        return max(w, h)

    w1, h1 = img1.size
    w2, h2 = img2.size
    max_long = max(long_edge(w1, h1), long_edge(w2, h2))
    if max_long > max_long_edge:
        scale = float(max_long_edge) / float(max_long)

        def resize_keep_ar(im, s):
            w, h = im.size
            nw = max(1, int(round(w * s)))
            nh = max(1, int(round(h * s)))
            return im.resize((nw, nh), resample=Image.BICUBIC)

        img1 = resize_keep_ar(img1, scale)
        img2 = resize_keep_ar(img2, scale)

    w1, h1 = img1.size
    w2, h2 = img2.size
    if max(w1, h1) >= max(w2, h2):
        target = img1
        other = img2
        target_is_img1 = True
    else:
        target = img2
        other = img1
        target_is_img1 = False

    other = other.resize(target.size, resample=Image.BICUBIC)
    return (target, other) if target_is_img1 else (other, target)


torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_num_threads(1)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto"
).to("cuda:0")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model.eval()
model.requires_grad_(False)


def generate_one_local(messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
    with torch.inference_mode():
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = []
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image" and "image" in item:
                    image_inputs.append(item["image"])
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=1.0,
        )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        return processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]


def safe_open_image(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def run_inference(input_image: str, output_image: str, prompt: str) -> Dict[str, Any]:
    ref_img = safe_open_image(input_image)
    cand_img = safe_open_image(output_image)
    ref_img, cand_img = preprocess_pair_resize(ref_img, cand_img, MAX_LONG_EDGE)

    raw_outputs: Dict[str, Optional[str]] = {}
    parsed: Dict[str, Any] = {}

    msg = build_messages(SYSTEM_PROMPT, ref_img, cand_img, prompt, PROMPT_SEMANTIC)
    raw_outputs["Semantic"] = generate_one_local(msg, MAX_NEW_TOKENS)
    parsed["Semantic"] = extract_first_json_obj(raw_outputs["Semantic"])

    msg = build_messages(SYSTEM_PROMPT, ref_img, cand_img, prompt, PROMPT_STRUCTURE)
    raw_outputs["Structure"] = generate_one_local(msg, MAX_NEW_TOKENS)
    parsed["Structure"] = extract_first_json_obj(raw_outputs["Structure"])

    skip_lowlevel = is_restoration_prompt(prompt)

    if skip_lowlevel:
        raw_outputs["Low-Level"] = None
        parsed["Low-Level"] = {
            "answer": "NULL",
            "problem": "NULL",
            "skipped": True,
            "reason": "Prompt indicates restoration task; low-level branch skipped."
        }
    else:
        msg = build_messages(SYSTEM_PROMPT, ref_img, cand_img, prompt, PROMPT_LOWLEVEL)
        raw_outputs["Low-Level"] = generate_one_local(msg, MAX_NEW_TOKENS)
        parsed["Low-Level"] = extract_first_json_obj(raw_outputs["Low-Level"])

    semantic_obj = parsed["Semantic"]
    if semantic_obj and semantic_obj.get("answer") == "No":
        problems = normalize_problem_list(semantic_obj.get("problem"))
        injected = SEMANTIC_FOLLOWUP_TEMPLATE.replace("XXX", prompt).replace("YYY", json.dumps(problems, ensure_ascii=False))
        msg = build_messages("You are a careful visual reasoning assistant.", ref_img, cand_img, prompt, injected)
        raw_outputs["Semantic_Followup"] = generate_one_local(msg, FOLLOWUP_MAX_NEW_TOKENS)
        parsed["Semantic_Followup"] = _canonicalize_followup_obj(
            extract_first_json_obj(raw_outputs["Semantic_Followup"])
        )
    else:
        raw_outputs["Semantic_Followup"] = None
        parsed["Semantic_Followup"] = None

    if skip_lowlevel:
        raw_outputs["Low-Level_Followup"] = None
        parsed["Low-Level_Followup"] = {
            "skipped": True,
            "reason": "Prompt indicates restoration task; low-level followup skipped."
        }
    else:
        low_obj = parsed["Low-Level"]
        if low_obj and low_obj.get("answer") == "No":
            problems = normalize_problem_list(low_obj.get("problem"))
            injected = LOWLEVEL_FOLLOWUP_TEMPLATE.replace("XXX", prompt).replace("YYY", json.dumps(problems, ensure_ascii=False))
            msg = build_messages("You are a careful visual reasoning assistant.", ref_img, cand_img, prompt, injected)
            raw_outputs["Low-Level_Followup"] = generate_one_local(msg, FOLLOWUP_MAX_NEW_TOKENS)
            parsed["Low-Level_Followup"] = _canonicalize_followup_obj(
                extract_first_json_obj(raw_outputs["Low-Level_Followup"])
            )
        else:
            raw_outputs["Low-Level_Followup"] = None
            parsed["Low-Level_Followup"] = None

    msg = build_messages(SYSTEM_PROMPT, ref_img, cand_img, prompt, FIDELITY_PROMPT)
    raw_outputs["Fidelity"] = generate_one_local(msg, MAX_NEW_TOKENS)
    parsed["Fidelity"] = extract_first_json_obj(raw_outputs["Fidelity"])
    fidelity_score = obj_to_score(parsed["Fidelity"])

    votes = [
        answer_to_vote(parsed["Semantic"]),
        answer_to_vote(parsed["Structure"]),
        answer_to_vote(parsed["Low-Level"]),
    ]

    display_rows = build_display_rows(parsed, fidelity_score)

    return {
        "prompt": prompt,
        "images": [input_image, output_image],
        "assessment": {
            "Semantic": parsed["Semantic"],
            "Structure": parsed["Structure"],
            "Low-Level": parsed["Low-Level"],
        },
        "semantic_followup": {
            "triggered": raw_outputs["Semantic_Followup"] is not None,
            "analysis": parsed["Semantic_Followup"],
        },
        "low_level_followup": {
            "triggered": raw_outputs["Low-Level_Followup"] is not None and not skip_lowlevel,
            "analysis": parsed["Low-Level_Followup"],
        },
        "fidelity": {
            "score": fidelity_score,
            "judgement": parsed["Fidelity"],
        },
        "display_rows": display_rows,
        "display_text": "\n".join(display_rows),
        "score_12": votes_to_score(votes[:2]),
        "score_123": votes_to_score(votes[:3]),
        "raw_model_outputs": raw_outputs,
        "skip_lowlevel": skip_lowlevel,
    }


class InferRequest(BaseModel):
    input_image: str
    output_image: str
    prompt: str = ""


app = FastAPI(title="StableI2I_PLUS Web UI")


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>StableI2I_PLUS Demo</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: #121933;
      --panel-2: #192246;
      --text: #eef2ff;
      --muted: #aab4d6;
      --line: rgba(255,255,255,0.08);
      --accent: #6d8cff;
      --accent-2: #8b5cf6;
      --ok: #22c55e;
      --err: #ef4444;
      --shadow: 0 18px 50px rgba(0,0,0,0.28);
      --radius: 18px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Inter, Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(109,140,255,0.20), transparent 30%),
        radial-gradient(circle at top right, rgba(139,92,246,0.18), transparent 28%),
        linear-gradient(180deg, #0a0f1f 0%, #0e1327 100%);
      color: var(--text);
    }

    .wrap {
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px 18px 40px;
    }

    .hero {
      margin-bottom: 22px;
    }

    .hero h1 {
      margin: 0 0 8px;
      font-size: 32px;
      line-height: 1.15;
      letter-spacing: -0.02em;
    }

    .hero p {
      margin: 0;
      color: var(--muted);
      font-size: 15px;
    }

    .panel {
      background: rgba(18,25,51,0.88);
      backdrop-filter: blur(10px);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 20px;
    }

    .grid {
      display: grid;
      grid-template-columns: 1.05fr 1.35fr;
      gap: 20px;
      align-items: start;
    }

    .section-title {
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #cdd7ff;
      margin-bottom: 12px;
    }

    .row {
      margin-bottom: 14px;
    }

    label {
      display: block;
      font-weight: 600;
      margin-bottom: 7px;
      color: #e6ebff;
      font-size: 14px;
    }

    input[type=file], textarea, button, select {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid rgba(255,255,255,0.09);
      border-radius: 14px;
      outline: none;
    }

    textarea, input[type=file] {
      background: rgba(255,255,255,0.04);
      color: var(--text);
      padding: 12px 13px;
      font-size: 14px;
    }

    select {
      background: #ffffff;
      color: #111111;
      padding: 12px 13px;
      font-size: 14px;
      font-weight: 700;
      border: 1px solid rgba(255,255,255,0.09);
    }

    select:focus, textarea:focus, input[type=file]:focus {
      border-color: #6d8cff;
      box-shadow: 0 0 0 3px rgba(109,140,255,0.18);
    }

    select option {
      background: #ffffff;
      color: #111111;
    }

    textarea {
      min-height: 92px;
      resize: vertical;
    }

    input[type=file] {
      padding: 10px 12px;
    }

    .subgrid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .preview-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }

    .preview-card {
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      overflow: hidden;
    }

    .preview-head {
      padding: 12px 14px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      font-weight: 700;
      color: #dfe6ff;
      background: rgba(255,255,255,0.02);
    }

    .preview-box {
      width: 100%;
      height: 360px;
      display: flex;
      align-items: center;
      justify-content: center;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0.03));
      color: #95a2cc;
      position: relative;
      overflow: hidden;
    }

    .preview-box img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      display: none;
    }

    .preview-placeholder {
      font-size: 14px;
      color: #97a3c9;
      padding: 0 12px;
      text-align: center;
    }

    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 4px;
    }

    button {
      border: 0;
      padding: 13px 16px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      transition: 0.18s ease;
      color: white;
    }

    .btn-primary {
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      box-shadow: 0 10px 26px rgba(109,140,255,0.28);
    }

    .btn-secondary {
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.10);
      color: #edf2ff;
    }

    button:hover {
      transform: translateY(-1px);
      filter: brightness(1.04);
    }

    .server-note {
      margin-top: 12px;
      font-size: 13px;
      color: var(--muted);
    }

    .status {
      margin-top: 16px;
      min-height: 22px;
      color: #dce4ff;
      font-size: 14px;
    }

    .status.ok { color: #86efac; }
    .status.err { color: #fca5a5; }

    .result-panel {
      margin-top: 22px;
    }

    .result-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 10px;
      gap: 12px;
    }

    .result-title {
      font-size: 18px;
      font-weight: 800;
      margin: 0;
    }

    .badge {
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      color: #dce4ff;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.08);
    }

    pre {
      margin: 0;
      background: #090d19;
      color: #f6f7fb;
      padding: 18px;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.06);
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.75;
      font-size: 14px;
      min-height: 280px;
    }

    @media (max-width: 980px) {
      .grid {
        grid-template-columns: 1fr;
      }
      .preview-grid,
      .subgrid,
      .button-row {
        grid-template-columns: 1fr;
      }
      .preview-box {
        height: 280px;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>StableI2I Demo</h1>
      <p>Choose an example or upload your own before/after pair. Selecting an example will automatically fill paths, prompt, and image previews.</p>
    </div>

    <div class="panel">
      <div class="grid">
        <div>
          <div class="section-title">Inputs</div>

          <div class="row">
            <label for="example_select">Default Examples</label>
            <select id="example_select">
              <option value="">Select an example</option>
            </select>
          </div>

          <div class="subgrid">
            <div class="row">
              <label for="path_input_image">Input Image Path</label>
              <textarea id="path_input_image" placeholder="/absolute/path/to/input.png"></textarea>
            </div>
            <div class="row">
              <label for="path_output_image">Output Image Path</label>
              <textarea id="path_output_image" placeholder="/absolute/path/to/output.png"></textarea>
            </div>
          </div>

          <div class="row">
            <label for="input_image">Input / Before Image Upload</label>
            <input id="input_image" name="input_image" type="file" accept="image/*">
          </div>

          <div class="row">
            <label for="output_image">Output / After Image Upload</label>
            <input id="output_image" name="output_image" type="file" accept="image/*">
          </div>

          <div class="row">
            <label for="prompt">Prompt</label>
            <textarea id="prompt" name="prompt" placeholder="Restore the image."></textarea>
          </div>

          <div class="button-row">
            <button class="btn-primary" type="submit" form="infer-form">Run by Upload</button>
            <button class="btn-secondary" type="button" id="run_by_path">Run by Path / Example</button>
          </div>

          <div class="server-note">Server: __HOST__:__PORT__</div>
          <div id="status" class="status"></div>
        </div>

        <div>
          <div class="section-title">Preview</div>
          <div class="preview-grid">
            <div class="preview-card">
              <div class="preview-head">Before Preview</div>
              <div class="preview-box">
                <span id="before_placeholder" class="preview-placeholder">No image selected</span>
                <img id="before_preview" alt="Before preview">
              </div>
            </div>
            <div class="preview-card">
              <div class="preview-head">After Preview</div>
              <div class="preview-box">
                <span id="after_placeholder" class="preview-placeholder">No image selected</span>
                <img id="after_preview" alt="After preview">
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <form id="infer-form" style="display:none;"></form>

    <div class="panel result-panel">
      <div class="result-head">
        <h2 class="result-title">Result Summary</h2>
        <div class="badge">6 Rows</div>
      </div>
      <pre id="result">Waiting to run...</pre>
    </div>
  </div>

  <script>
    const form = document.getElementById('infer-form');
    const statusDiv = document.getElementById('status');
    const resultPre = document.getElementById('result');

    const inputFile = document.getElementById('input_image');
    const outputFile = document.getElementById('output_image');

    const beforePreview = document.getElementById('before_preview');
    const afterPreview = document.getElementById('after_preview');
    const beforePlaceholder = document.getElementById('before_placeholder');
    const afterPlaceholder = document.getElementById('after_placeholder');

    const exampleSelect = document.getElementById('example_select');
    const pathInputImage = document.getElementById('path_input_image');
    const pathOutputImage = document.getElementById('path_output_image');
    const promptInput = document.getElementById('prompt');

    function setStatus(msg, type = '') {
      statusDiv.textContent = msg;
      statusDiv.className = 'status ' + type;
    }

    function showPreview(imgEl, placeholderEl, src) {
      if (!src) {
        imgEl.src = '';
        imgEl.style.display = 'none';
        placeholderEl.style.display = 'block';
        placeholderEl.textContent = 'No image selected';
        return;
      }
      imgEl.src = src;
      imgEl.style.display = 'block';
      placeholderEl.style.display = 'none';
    }

    function bindPreview(fileInput, imgEl, placeholderEl, pathField = null) {
      fileInput.addEventListener('change', () => {
        const file = fileInput.files && fileInput.files[0];
        if (!file) {
          showPreview(imgEl, placeholderEl, '');
          return;
        }

        if (pathField) {
          pathField.value = '';
        }

        const reader = new FileReader();
        reader.onload = (e) => {
          showPreview(imgEl, placeholderEl, e.target.result);
        };
        reader.readAsDataURL(file);
      });
    }

    function setPreviewFromPath(path, imgEl, placeholderEl) {
      if (!path) {
        showPreview(imgEl, placeholderEl, '');
        return;
      }
      showPreview(imgEl, placeholderEl, '/preview_image?path=' + encodeURIComponent(path));
    }

    function renderSummary(data) {
      if (data.display_text) {
        resultPre.textContent = data.display_text;
        return;
      }
      if (data.display_rows && Array.isArray(data.display_rows)) {
        resultPre.textContent = data.display_rows.join('\\n');
        return;
      }
      resultPre.textContent = JSON.stringify(data, null, 2);
    }

    bindPreview(inputFile, beforePreview, beforePlaceholder, pathInputImage);
    bindPreview(outputFile, afterPreview, afterPlaceholder, pathOutputImage);

    async function loadExamples() {
      try {
        const resp = await fetch('/examples');
        const data = await resp.json();
        const examples = data.examples || [];

        examples.forEach((ex, idx) => {
          const opt = document.createElement('option');
          opt.value = idx;
          opt.textContent = ex.name || `Example ${idx + 1}`;
          exampleSelect.appendChild(opt);
        });

        exampleSelect.addEventListener('change', () => {
          const idx = exampleSelect.value;
          if (idx === '') return;

          const ex = examples[Number(idx)];
          pathInputImage.value = ex.input_image || '';
          pathOutputImage.value = ex.output_image || '';
          promptInput.value = ex.prompt || '';

          inputFile.value = '';
          outputFile.value = '';

          setPreviewFromPath(ex.input_image || '', beforePreview, beforePlaceholder);
          setPreviewFromPath(ex.output_image || '', afterPreview, afterPlaceholder);

          setStatus(`Loaded ${ex.name || 'example'}.`, 'ok');
        });
      } catch (e) {
        console.error(e);
        setStatus('Failed to load examples.', 'err');
      }
    }

    loadExamples();

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      setStatus('Running by upload...', '');
      resultPre.textContent = 'Running...';

      const formData = new FormData();
      if (inputFile.files[0]) formData.append('input_image', inputFile.files[0]);
      if (outputFile.files[0]) formData.append('output_image', outputFile.files[0]);
      formData.append('prompt', promptInput.value || '');

      if (!inputFile.files[0] || !outputFile.files[0]) {
        setStatus('Please upload both images, or use Run by Path / Example.', 'err');
        return;
      }

      try {
        const resp = await fetch('/infer_upload', {
          method: 'POST',
          body: formData
        });
        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data.detail || 'Request failed');
        }
        renderSummary(data);
        setStatus('Done.', 'ok');
      } catch (err) {
        setStatus('Error: ' + err.message, 'err');
        resultPre.textContent = 'Request failed.';
      }
    });

    document.getElementById('run_by_path').addEventListener('click', async () => {
      const inputImage = pathInputImage.value.trim();
      const outputImage = pathOutputImage.value.trim();
      const prompt = promptInput.value;

      if (!inputImage || !outputImage) {
        setStatus('Please select an example or fill both image paths.', 'err');
        return;
      }

      setPreviewFromPath(inputImage, beforePreview, beforePlaceholder);
      setPreviewFromPath(outputImage, afterPreview, afterPlaceholder);

      setStatus('Running by path/example...', '');
      resultPre.textContent = 'Running...';

      try {
        const resp = await fetch('/infer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            input_image: inputImage,
            output_image: outputImage,
            prompt: prompt
          })
        });
        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data.detail || 'Request failed');
        }
        renderSummary(data);
        setStatus('Done.', 'ok');
      } catch (err) {
        setStatus('Error: ' + err.message, 'err');
        resultPre.textContent = 'Request failed.';
      }
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE.replace("__HOST__", HOST).replace("__PORT__", str(PORT))


@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH, "gpu": GPU_ID, "host": HOST, "port": PORT}


@app.get("/examples")
def get_examples():
    return {"examples": EXAMPLE_CASES}


@app.get("/preview_image")
def preview_image(path: str):
    if not path:
        raise HTTPException(status_code=400, detail="Empty image path")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Image not found: {path}")
    return FileResponse(path)


@app.post("/infer")
def infer(req: InferRequest):
    try:
        return run_inference(req.input_image, req.output_image, req.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer_upload")
async def infer_upload(
    input_image: UploadFile = File(...),
    output_image: UploadFile = File(...),
    prompt: str = Form(""),
):
    suffix1 = os.path.splitext(input_image.filename or "input.png")[-1] or ".png"
    suffix2 = os.path.splitext(output_image.filename or "output.png")[-1] or ".png"
    temp1 = tempfile.NamedTemporaryFile(delete=False, suffix=suffix1)
    temp2 = tempfile.NamedTemporaryFile(delete=False, suffix=suffix2)
    temp1_path = temp1.name
    temp2_path = temp2.name
    try:
        temp1.write(await input_image.read())
        temp2.write(await output_image.read())
        temp1.close()
        temp2.close()
        result = run_inference(temp1_path, temp2_path, prompt)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [temp1_path, temp2_path]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        workers=1,
    )

"""
srun --jobid=6061225 -n 1 bash -lc '
source /mnt/petrelfs/lijiayang1/anaconda3/etc/profile.d/conda.sh
conda activate artimuse
cd /mnt/hwfile/lijiayang1/stableI2I
export MODEL_PATH=/mnt/petrelfs/lijiayang1/stableI2I/ckpt
export GPU_ID=0
export HOST=10.140.60.137
export PORT=10004
python api.py
'
"""
