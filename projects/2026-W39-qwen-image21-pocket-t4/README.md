# Qwen-Image-2.1 Pocket 0.8B prompt rewriter — Colab T4 probe

The linked Pocket model is a **text-only prompt rewriter** for Qwen-Image-2.1, not a smaller image-generation model. This probe evaluates the combined pipeline: direct prompt → Qwen-Image-2.1 versus Pocket rewrite → the same Qwen-Image-2.1.

- Pocket model: [ML-Intern-lab/Qwen-Image-2.1-PE-T2I-Pocket-0.8B](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-PE-T2I-Pocket-0.8B)
- Base image model: [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)
- T4 image runtime: [Qwen-Image-2.1 Uncensored GGUF Q4_K_M](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF) with the leejet ComfyUI-GGUF node

## Method

Three fixed prompts were each rendered twice with the same seed and settings: 768-ish output size following Pocket's aspect-ratio choice, 12 steps, cfg 2.5, `res_multistep/simple`, seed 42, Q4_K_M image weights with fp16 enabled and the ComfyUI compiler disabled. Each pair uses the same aspect ratio to isolate the text rewrite. The Pocket model used its card's sampling settings (temperature 1, top-p 0.95, top-k 20, seed 0).

The T4 was Tesla T4, torch 2.11.0+cu130, Python 3.13.15. All 6 renders and 3 JSON rewrites succeeded. Full machine-readable timing, prompts, and outputs are in `results/timings.json`; images are in `results/`.

## Results

| Request | Direct render | Pocket rewrite render | Visual note |
|---|---:|---:|---|
| Cat in a tiny yellow raincoat | 126.1 s (cold) | 78.1 s | Both are strong images. Direct preserves the requested yellow coat; Pocket changes it to a black leather coat and invents yellow forehead marks. |
| “FREE GPU LAB” neon sign | 87.1 s | 87.1 s | Both spell the sign correctly. Pocket produces a cleaner, more legible sign; direct has a more atmospheric street scene. |
| Red bicycle at a bakery door | 75.1 s | 78.1 s | Both follow the request. Pocket gives a more polished product-photo composition, with some invented bicycle details. |

Pocket rewrites took **16.15–21.32 s** each on this T4 (19.33 s mean), after a **126.1 s model load**. It produced 323–410 tokens per prompt. Warm image renders averaged **81.1 s direct** and **81.1 s after rewrite**; the rewriting stage adds its own latency rather than making image generation faster.

## Takeaway

The image quality is promising: the sign and bicycle examples benefit from the richer descriptions, and text rendering remains correct. But this is not a consistent quality win. The cat example shows a material loss of prompt fidelity (yellow raincoat becomes black), and the rewriter adds unsupported details. For a workflow where fidelity to user-specified colors, clothing, and objects matters, review or constrain the rewritten prompt before rendering. Three prompts are a small subjective sample, not a broad quality evaluation.

The Pocket card specifies non-commercial research use under the Qwen Research License; check its `LICENSE` and `NOTICE` before use beyond this evaluation.
