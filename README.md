This repository is part of the thesis project **Training Neural Causal Models With Adversarial Training** (2022).

The code contains an implementation of the causal discovery technique proposed as part of the project. Concretely, it allows you to train a generative model known as a **neural causal model (NCM)**. The NCM is backed by a **soft adjacency matrix** of edge beliefs, which (ideally) converge to the edges of the true causal graph underlying the dataset on which the model is trained.

The model can be run from the command line like so:

```
python3 run.py --structure CHAIN --type LINEAR --num_nodes 4
```

This runs the model on a dataset constructed by a randomly generated **structural causal model (SCM)** with a chain graph structure, four nodes, and linear functional relationships. These parameters can be adjusted as follows:

* `structure`: `CHAIN`, `COLLIDER`, `BIDIAG`, `TREE`, `JUNGLE`, `FULL`, `ER-1`, `ER-2`
* `type`: `LINEAR`, `NONLINEAR`
* `num_nodes`: Any integer greater than 1
