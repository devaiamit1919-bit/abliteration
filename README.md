# Abliteration: Removing Refusal from LLMs

A hands-on learning project implementing the **abliteration** technique — removing the refusal behavior from safety-tuned LLMs without retraining, based on [mlabonne's blog post](https://huggingface.co/blog/mlabonne/abliteration).

## How It Works

Safety-tuned LLMs encode "refusal" as a specific **direction** in their internal representations (the residual stream). Abliteration:

1. Collects model activations on harmful vs harmless prompts
2. Computes the mean difference to find the "refusal direction"
3. Removes that direction from the model's weights permanently

## Notebooks

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 1 | `01_setup_and_data.ipynb` | Install deps, load datasets, confirm model refuses harmful prompts |
| 2 | `02_collect_activations.ipynb` | Capture internal activations using TransformerLens hooks |
| 3 | `03_find_refusal_direction.ipynb` | Compute and visualize the refusal direction per layer |
| 4 | `04_inference_intervention.ipynb` | Temporarily remove refusal via hooks during generation |
| 5 | `05_permanent_abliteration.ipynb` | Permanently modify weights and save the abliterated model |

Each notebook includes inline concept explanations and visualizations — no prior knowledge of transformer internals or linear algebra required.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then open `01_setup_and_data.ipynb` and run through the notebooks in order.

## Model

Uses `Qwen/Qwen2-0.5B-Instruct` (~1GB) — small enough to run on a Mac Mini M2 with 16GB RAM.

## References

- [Uncensor any LLM with abliteration](https://huggingface.co/blog/mlabonne/abliteration) — mlabonne's blog post
- [Refusal in LLMs is mediated by a single direction](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) — Arditi et al.
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — mechanistic interpretability library
- [FailSpy's abliterator](https://github.com/FailSpy/abliterator) — production-ready implementation
