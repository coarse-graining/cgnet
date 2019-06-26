cgnet
===

In development!

Requirements
---
+ ``numpy``
+ ``pytorch`` (1.0 or higher)
+ ``scipy``
+ ``mdtraj`` (for ``cgnet.molecule`` only)
+ ``pandas`` (for ``cgnet.molecule`` only)
+ ``sklearn`` (for testing)

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
