cgnet
=====

Coarse graining for molecular dymamics ([preprint](https://arxiv.org/abs/2007.11412))

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
+ `matplotlib` (for `examples`)

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
For compatibility with `pytorch==1.1`, please use the `pytorch-1.1` branch. This branch currently does not include the updates for variable size and Langevin dynamics, nor some normalization options.

Cite
----
Please cite our [paper](https://doi.org/10.1063/5.0026133) in J Chem Phys:

```bibtex
@article{husic2020coarse,
  title={Coarse graining molecular dynamics with graph neural networks},
  author={Husic, Brooke E and Charron, Nicholas E and Lemm, Dominik and Wang, Jiang and P{\'e}rez, Adri{\`a} and Majewski, Maciej and Kr{\"a}mer, Andreas and Chen, Yaoyi and Olsson, Simon and de Fabritiis, Gianni and Noe{\'e}, Frank and Clementi, Cecilia},
  journal={The Journal of Chemical Physics},
  volume={153},
  number={19},
  pages={194101},
  year={2020},
  publisher={AIP Publishing LLC}
}
```

Various methods are based off the following papers. CGnet:

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

SchNet:

```bibtex
@article{schutt2018schnetpack,
  title={SchNetPack: A deep learning toolbox for atomistic systems},
  author={Schutt, KT and Kessel, Pan and Gastegger, Michael and Nicoli, KA and Tkatchenko, Alexandre and Müller, K-R},
  journal={Journal of Chemical Theory and Computation},
  volume={15},
  number={1},
  pages={448--455},
  year={2018},
  publisher={ACS Publications}
}
```
