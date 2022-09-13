## GPED

This repository is a re-implementation of GPED. The correction is not guaranteed, and a careful check is needed.

### How to use this code?

Steps:

1. Changing the selection operator to GPED. GPED needs to pass an argument representing the real target value.
    ```python
   toolbox.register("select", selGPED, real_target=real)
    ```
2. In the evaluation function, we need to assign prediction values to each individual for calculating the semantic
   matrix.
    ```python
       individual.prediction = prediction
    ```

### Reference

```bibtex
@article{chen2020preserving,
    title = {Preserving population diversity based on transformed semantics in genetic programming for symbolic regression},
    author = {Chen, Qi and Xue, Bing and Zhang, Mengjie},
    journal = {IEEE Transactions on Evolutionary Computation},
    volume = {25},
    number = {3},
    pages = {433--447},
    year = {2020},
    publisher = {IEEE}
}
```