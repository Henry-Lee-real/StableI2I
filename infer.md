# Inference Guide

This document describes the input JSONL format and the command-line arguments for `test.py`.

## Input JSONL

The input file must be a `.jsonl` file.

Rules:
- One case per line
- Each line must be a valid JSON object
- `id` is required and will be copied to the output exactly
- `input_image` and `output_image` are required
- `prompt` is optional, default is `""`

Minimal format:

```json
{"id":"case-1","input_image":"path/to/before.png","output_image":"path/to/after.png","prompt":"Add a wooden bench along the path."}
```

Compatible aliases:

- `before_image` can be used instead of `input_image`
- `after_image` can be used instead of `output_image`

Example:

```json
{"id":"case-1","input_image":"example/000155856.jpg","output_image":"example/000155856_dup2.png","prompt":"Add a wooden bench along the path."}
{"id":"case-2","input_image":"example/0164.png","output_image":"example/0164_out.png","prompt":"Restore the image."}
```

## Main Command

```bash
python test.py --jsonl examples/sample.jsonl --ckpt path/to/ckpt --mode cot --dimensions semantic,structure,lowlevel --output-jsonl outputs/results.jsonl
```

## Arguments

### Required

- `--jsonl`
  Path to the input JSONL file.

- `--ckpt`
  Path to the local model checkpoint.

### Common

- `--mode`
  Comma-separated mode list.
  Supported values:
  - `simple`
  - `cot`
  - `score`

  Examples:

  ```bash
  --mode simple
  --mode cot
  --mode score
  --mode simple,cot,score
  ```

- `--dimensions`
  Comma-separated dimension list.
  Supported values:
  - `semantic`
  - `structure`
  - `lowlevel`

  Examples:

  ```bash
  --dimensions semantic,structure,lowlevel
  --dimensions semantic,structure
  --dimensions semantic
  ```

- `--output-jsonl`
  Output result file path.

  Example:

  ```bash
  --output-jsonl outputs/results.jsonl
  ```

### Optional Inference Parameters

- `--gpu-id`
  Visible CUDA device id.

- `--seed`
  Random seed used for stable sampling.

- `--prompt-dir`
  Prompt template directory. Default is `prompts`.

- `--max-new-tokens`
  Max tokens for main branches.

- `--followup-max-new-tokens`
  Max tokens for follow-up reasoning branches.

- `--max-long-edge`
  Resize long edge upper bound before inference.

- `--temperature`
  Sampling temperature. Default is `0.01`.

- `--top-p`
  Top-p sampling threshold. Default is `0.1`.

- `--top-k`
  Top-k sampling threshold. Default is `5`.

## Mode Behavior

### `simple`

Runs only the selected main dimensions.

Output:

```json
"result": {
  "Semantic": {...},
  "Structure": {...},
  "Low-Level": {...}
}
```

### `cot`

Runs the selected main dimensions and, when needed, follow-up reasoning.

Output:

```json
"result": {
  "Semantic": {...},
  "Structure": {...},
  "Low-Level": {...},
  "Semantic_think": {...},
  "Low-Level_think": {...}
}
```

### `score`

Runs only the fidelity scoring branch.

Output:

```json
"result": {
  "Score": {
    "score": 8
  }
}
```

### Multiple modes

If multiple modes are passed, the output is grouped by mode name:

```json
"result": {
  "simple": {...},
  "cot": {...},
  "score": {...}
}
```

## Example Commands

Run `cot` with all three dimensions:

```bash
python test.py --jsonl examples/sample.jsonl --ckpt path/to/ckpt --mode cot --dimensions semantic,structure,lowlevel --output-jsonl outputs/results.jsonl
```

Run `simple` with only semantic and structure:

```bash
python test.py --jsonl examples/sample.jsonl --ckpt path/to/ckpt --mode simple --dimensions semantic,structure --output-jsonl outputs/simple_results.jsonl
```

Run score only:

```bash
python test.py --jsonl examples/sample.jsonl --ckpt path/to/ckpt --mode score --dimensions semantic --output-jsonl outputs/score_results.jsonl
```

Run all modes once:

```bash
python test.py --jsonl examples/sample.jsonl --ckpt path/to/ckpt --mode simple,cot,score --dimensions semantic,structure,lowlevel --output-jsonl outputs/all_results.jsonl
```

## Output Notes

- The terminal shows a single global `tqdm` progress bar.
- The output directory is created automatically if it does not exist.
- Output `id` is copied directly from the input JSONL.
