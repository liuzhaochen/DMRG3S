# DMRG3S

DMRG3S is a DMRG solver implementing a variant of strictly single-site DMRG with subspace expansion [1], combined with a randomized SVD strategy [3], 
and fully based on [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl).

based on [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl).

Related approaches also includes CBE algorithm [2,4,5].

The code has only been tested for limited cases. Please use it with caution and validate results carefully.
## Example
Spinfull free Fermion on $(N_x,N_y)=(10,4)$ clylinder at filling $N=0.9N_xN_y$ with $U(1)_s\times U(1)_s$ symmetries. 
See the directory [example/square](https://github.com/liuzhaochen/DMRG3S/tree/main/example/square) for detailed scripts and parameters.

Energy vs. discarded weight:
<div align="center">
	<img src="https://github.com/liuzhaochen/DMRG3S/blob/main/example/square/weight_fitting.jpg" width="500">
</div>

Energy vs. square of inverse bond dimension:
<div align="center">
	<img src="https://github.com/liuzhaochen/DMRG3S/blob/main/example/square/D%5E2Fitting.jpg" width="500">
</div>

## Reference
[1][Strictly single-site DMRG algorithm with subspace expansion](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.91.155115)

[2][Controlled Bond Expansion for Density Matrix Renormalization Group Ground State Search at Single-Site Costs](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.130.246402)

[3][Comment on "Controlled Bond Expansion for Density Matrix Renormalization Group Ground State Search at Single-Site Costs"](https://arxiv.org/abs/2403.00562)

[4][Reply to comment on "Controlled bond expansion for Density Matrix Renormalization Group ground state search at single-site costs"](https://arxiv.org/abs/2501.12291)

[5][CBEAlgorithms](https://github.com/ShimpeiGoto/CBEAlgorithms)
