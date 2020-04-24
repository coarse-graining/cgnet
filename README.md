cgnet
=====

In development!

Dependencies
------------
Required:
+ `numpy`
+ `pytorch` (1.2 or higher)
+ `scipy`

Optional:
+ `mdtraj` (for `cgnet.molecule` only)
+ `pandas` (for `cgnet.molecule` only)
+ `sklearn` (for testing)
+ `Jupyter` (for `examples`)

Usage
-----
Clone the repository:
```
git clone git@github.com:coarse-graining/cgnet.git
```

Install any missing dependencies, and then run:
```
cd cgnet
python setup.py install
```

Notes
-----
For backwards compatibility with software before the incorporations of variable size utilities and Langevin dynamics, please use the `invariable` branch.

For compatibility with `pytorch==1.1`, please use the `pytorch-1.1` branch. This branch currently does not include the updates for variable size and Langevin dynamics, nor some normalization options.

Cite
----
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
