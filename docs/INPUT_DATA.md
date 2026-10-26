# Input data

[Documentation index](README.md) · [Main README](../README.md)

EFSC accepts a pandas DataFrame through `EFSC.fit`, or a CSV file through
`EFSC.fit_csv`. Data must be in raw long format: each row is one observation
inside a unit-time cell, so repeated unit-time pairs represent the dataset in
that cell.

```text
outcome,unit,time
3.2,A,2010
4.1,A,2010
1.5,A,2011
2.8,B,2010
...
```

A unit-time pair omitted from the file is a missing panel cell. Every observed
cell may contain a different number of observations. Every included unit and
period must have at least one observed cell because a coordinate that never
appears in the input cannot be learned.

Numeric time labels must be increasing and equally spaced; EFSC infers their
step automatically. Parseable dates are placed in chronological order.
Arbitrary labels require an explicit `time_order`.

A compact 24-observation toy CSV file and a complete file-based Gaussian
example are available in
[`examples/gaussian_imputation_quickstart.py`](../examples/gaussian_imputation_quickstart.py). Run it
from the repository root with:

```bash
python examples/gaussian_imputation_quickstart.py
```

The larger examples in the [imputation gallery](../examples/IMPUTATION.md)
simulate 16-by-48 panels in memory; the linked toy CSV files are intended only
to make the raw input format easy to inspect.

The analogous causal-mode example is
[`examples/gaussian_causal_quickstart.py`](../examples/gaussian_causal_quickstart.py):

```bash
python examples/gaussian_causal_quickstart.py
```
