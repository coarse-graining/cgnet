The alanine dipeptide coordinates and forces are a subset of the ones used in the CGnet paper [1]. The simulation was performed with the AMBER ff99SB-ILDN force field [2] at 300 K.

The subset included in this repository contain 10,000 data points at 10 ps intervals, whereas the dataset in the CGnet paper has 1,000,000 data points at 1 ps intervals.

The file `ala2_coordinates.npy` is a np.ndarray of shape `(10000, 5, 3)`, for 10,000 frames, five coarse-grained beads corresponding to the five backbone atoms C (ACE-1), N (ALA-2), CA (ALA-2), C (ALA-2), N (NME-3), and three dimensions. The file `ala2_forces.npy` is a np.ndarray of the same shape containing the corresponding forces for each frame, bead, and dimension.

[1] Wang, J., Olsson, S., Wehmeyer, C., Pérez, A., Charron, N. E., de Fabritiis, G., Noé, F., Clementi, C. (2019). Machine Learning of Coarse-Grained Molecular Dynamics Force Fields. _ACS Central Science._ https://doi.org/10.1021/acscentsci.8b00913

[2] K. Lindorff-Larsen, S. Piana, K. Palmo, P. Maragakis, J. L. Klepeis, R. O. Dror, and D. E. Shaw. (2010). Improved side-chain torsion potentials for the Amber ff99SB protein force field. _Proteins: Struct., Funct., Bioinf._ 78, 1950 (2010). http://dx.doi.org/10.1002/prot.22711
