cgnet (pytorch 1.1 compatible)
===

The changes are:

In `feature/geometry.py`:

```
# return torch.BoolTensor(np.eye(n, dtype=np.bool)) # pytorch >=1.2
return torch.ByteTensor(np.eye(n, dtype=np.bool)) # pytorch 1.1
```

In `network/nnet.py`:

```
# energy = torch.sum(energy, axis=1) # pytorch >=1.2
energy = energy.sum(dim=1) # pytorch 1.1
```

```
# energy = torch.sum(energy, axis=-2) # pytorch >=1.2
energy = energy.sum(dim=-2) # pytorch 1.1
```

in `tests/test_geometry_core.py`:

```
# masked_neighbors_torch[~g_torch.to_type(
#     new_neighbors_mask_torch, g_torch.bool)] = -1 # pytorch >=1.2
masked_neighbors_torch[~g_torch.to_type(
    new_neighbors_mask_torch, torch.ByteTensor)] = -1 # pytorch 1.1
```

Dependencies
---
Required:
+ ``numpy``
+ ``pytorch`` (1.2 or higher)
+ ``scipy``

Optional:
+ ``mdtraj`` (for ``cgnet.molecule`` only)
+ ``pandas`` (for ``cgnet.molecule`` only)
+ ``sklearn`` (for testing)
+ ``Jupyter`` (for ``examples``)

Usage
---
Clone the repository:
```
git clone git@github.com:coarse-graining/cgnet.git
```

Install any missing dependencies, and then run:
```
cd cgnet
python setup.py install
```

Cite
---
Based off the CGnet paper,

```bibtex
@article{wang2019machine,
  title={Machine learning of coarse-grained molecular dynamics force fields},
  author={Wang, Jiang and Olsson, Simon and Wehmeyer, Christoph and Pérez, Adrià and Charron, Nicholas E and de Fabritiis, Gianni and Noé, Frank and Clementi, Cecilia},
  journal={ACS Central Science},
  year={2019},
  publisher={ACS Publications},
  doi={10.1021/acscentsci.8b00913}
}
```
