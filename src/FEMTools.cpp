#include <dolfin.h>

using namespace dolfin;

//DOI:10.1007/978-3-642-00605-0
//Flux-corrected-transport finite element method
int FEMFCT_Lk_D_Compute(Mat A, Mat &Dk, Mat &Lk) {
	//Input: A
	//Output: D, Lk

	MatZeroEntries(Dk);
	MatZeroEntries(Lk);

	Mat ATr;
	MatTranspose(A, MAT_INITIAL_MATRIX, &ATr);

	PetscInt A_FromRow;
	PetscInt A_ToRow;

	MatGetOwnershipRange(A, &A_FromRow, &A_ToRow);

	for (PetscInt i = A_FromRow; i < A_ToRow; i = i + 1) {
		const PetscScalar* a_i;
		const PetscInt* colsA;
		PetscInt ncolsA;

		const PetscScalar* a_j;
		const PetscInt* colsATr;
		PetscInt ncolsATr;

		std::vector<PetscScalar> d_i;
		std::vector<PetscInt> d_ind;

		MatGetRow(A, i, &ncolsA, &colsA, &a_i);
		MatGetRow(ATr, i, &ncolsATr, &colsATr, &a_j);

		//Nonzero entries of A for computations
		//Only nonzero and upper triangle entries of matrix D is needed to be set
		PetscInt j = 0;
		PetscInt k = 0;
		PetscScalar tmpval = 0;
		while (j < ncolsA && k < ncolsATr) { 
			if (colsA[j] < colsATr[k]) {
				if (colsA[j] > i) {
					tmpval = -1*std::max(double(0), a_i[j]);
					d_ind.push_back(colsA[j]);
					d_i.push_back(tmpval);
				}
				j = j + 1;
			}
			else if (colsATr[k] < colsA[j]) {
				if (colsATr[k] > i) {
					tmpval = -1*std::max(double(0), a_j[k]);
					d_ind.push_back(colsATr[k]);
					d_i.push_back(tmpval);
				}
				k = k + 1;
			}
			else {
				if (colsATr[k] > i) {
					tmpval = -1*std::max(double(0), std::max(a_i[j], a_j[k]));
					d_ind.push_back(colsATr[k]);
					d_i.push_back(tmpval);
				}
				k = k + 1;
				j = j + 1;
			}
		}
		while(j < ncolsA) {
			if (colsA[j] > i) {
				tmpval = -1*std::max(double(0), a_i[j]);
				d_ind.push_back(colsA[j]);
				d_i.push_back(tmpval);
			}
			j = j + 1;
		}
		while(k < ncolsATr) {
			if (colsATr[k] > i) {
				tmpval = -1*std::max(double(0), a_j[k]);
				d_ind.push_back(colsATr[k]);
				d_i.push_back(tmpval);
			}
			k = k + 1;
		}
		MatRestoreRow(A, i, &ncolsA, &colsA, &a_i);
		MatRestoreRow(ATr, i, &ncolsATr, &colsATr, &a_j);

		MatSetValues(Dk, 1, &i, d_ind.size(), d_ind.data(), d_i.data(), INSERT_VALUES);
		MatSetValues(Dk, d_ind.size(), d_ind.data(), 1, &i, d_i.data(), INSERT_VALUES);

		d_i.clear();
		d_i.shrink_to_fit();
		d_ind.clear();
		d_ind.shrink_to_fit();
	}

	MatAssemblyBegin(Dk, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(Dk, MAT_FINAL_ASSEMBLY);

	MatDestroy(&ATr);

	Vec vtmp;//setting the diagonal of D
	MatCreateVecs(A, NULL, &vtmp);
	VecSet(vtmp, 0);

	PetscBarrier(NULL);

	MatDiagonalSet(Dk, vtmp, INSERT_VALUES);

	PetscBarrier(NULL);

	MatGetRowSum(Dk, vtmp);
	VecScale(vtmp, -1);

	PetscBarrier(NULL);

	MatDiagonalSet(Dk, vtmp, INSERT_VALUES);

	PetscBarrier(NULL);

	VecDestroy(&vtmp);

	MatCopy(A, Lk, SAME_NONZERO_PATTERN);
	MatAXPY(Lk, 1, Dk, SAME_NONZERO_PATTERN);

	PetscBarrier(NULL);

	return 0;
}

//DOI:10.1016/j.cma.2008.08.016
//Flux-corrected-transport finite element method
int FEMFCT_dt_Compute(Mat ML, Mat Lk, PetscReal &dt) {
	//Input: ML, Lk
	//Output: dt

	Vec vtmp1, vtmp2;

	MatCreateVecs(ML, NULL, &vtmp1);

	PetscBarrier(NULL);

	VecDuplicate(vtmp1, &vtmp2);

	MatGetDiagonal(ML, vtmp1);
	MatGetDiagonal(Lk, vtmp2);

	PetscBarrier(NULL);

	VecPointwiseDivide(vtmp1, vtmp1, vtmp2);

	PetscBarrier(NULL);

	VecMin(vtmp1, NULL, &dt);
	dt = 2*dt;
	if(dt>1e-3)
		dt = std::floor(1e3*dt)/1e3;

	return 0;
}

//DOI:10.1016/j.cma.2008.08.016
//Flux-corrected-transport finite element method
int FEMFCT_fStar_Compute(Mat ML, Mat MC, Mat D1, Mat D0, Mat L0, Vec c0, Vec b0, PetscReal dt, Mat &r, Vec &fStar) {
	//Input: ML, MC, D1, D0, L0, c0, b0, dt
	//Output: r, f*

	MatZeroEntries(r);
	VecSet(fStar, 0);

	Vec v1Over2, vtmp;

	MatCreateVecs(L0, NULL, &v1Over2);

	PetscBarrier(NULL);

	VecDuplicate(v1Over2, &vtmp);
	VecSet(v1Over2, 0);
	VecSet(vtmp, 0);

	PetscBarrier(NULL);

	//v_half
	VecCopy(b0, v1Over2);
	MatMult(L0, c0, vtmp);

	PetscBarrier(NULL);

	VecAXPY(v1Over2, -1, vtmp);
	MatGetDiagonal(ML, vtmp);
	VecPointwiseDivide(v1Over2, v1Over2, vtmp);

	PetscBarrier(NULL);

	//each processor needs other processors indices
	VecScatter par2seq;
	Vec c0_SEQ, v1Over2_SEQ;

	VecScatterCreateToAll(c0, &par2seq, &c0_SEQ);
	VecScatterBegin(par2seq, c0, c0_SEQ, INSERT_VALUES, SCATTER_FORWARD);
	VecScatterEnd(par2seq, c0, c0_SEQ, INSERT_VALUES, SCATTER_FORWARD);

	PetscBarrier(NULL);

	VecScatterDestroy(&par2seq);

	VecScatterCreateToAll(v1Over2, &par2seq, &v1Over2_SEQ);
	VecScatterBegin(par2seq, v1Over2, v1Over2_SEQ, INSERT_VALUES, SCATTER_FORWARD);
	VecScatterEnd(par2seq, v1Over2, v1Over2_SEQ, INSERT_VALUES, SCATTER_FORWARD);

	PetscBarrier(NULL);

	VecScatterDestroy(&par2seq);
	VecDestroy(&vtmp);
	VecDestroy(&v1Over2);

	//r
	PetscInt L0_FromRow;
	PetscInt L0_ToRow;

	MatGetOwnershipRange(L0, &L0_FromRow, &L0_ToRow);

	Vec Rp, Rn;
	VecDuplicate(c0_SEQ, &Rp);
	VecSet(Rp, 1);
	VecDuplicate(c0_SEQ, &Rn);
	VecSet(Rn, 1);

	const PetscScalar* c0_i;
	const PetscScalar* v_i;

	VecGetArrayRead(c0_SEQ, &c0_i);
	VecGetArrayRead(v1Over2_SEQ, &v_i);

	for (PetscInt i = L0_FromRow; i < L0_ToRow; i = i + 1) {
		const PetscScalar* mc_i;
		const PetscInt* colsMC;
		PetscInt ncolsMC;

		const PetscScalar* d1_i;
		const PetscScalar* d0_i;
		const PetscInt* colsD0;
		PetscInt ncolsD0;

		MatGetRow(MC, i, &ncolsMC, &colsMC, &mc_i);
		MatGetRow(D0, i, &ncolsD0, &colsD0, &d0_i);
		MatGetRow(D1, i, NULL, NULL, &d1_i);

		std::vector<PetscScalar> ri;
		std::vector<PetscInt> ri_ind;
		PetscInt j = 0;
		PetscInt k = 0;

		PetscScalar Ppi = 0;
		PetscScalar Pni = 0;
		PetscScalar Qpi = c0_i[colsD0[k]] + 0.5*dt*v_i[colsD0[k]] - c0_i[i] - 0.5*dt*v_i[i];
		PetscScalar Qni = c0_i[colsD0[k]] + 0.5*dt*v_i[colsD0[k]] - c0_i[i] - 0.5*dt*v_i[i];
		PetscScalar mi = 0;

		while (j < ncolsMC && k < ncolsD0) { 
			if (colsMC[j] < colsD0[k]) {
				ri_ind.push_back(colsMC[j]);
				ri.push_back(dt*mc_i[j]*(v_i[i]-v_i[colsMC[j]]));
				if (colsMC[j] != i) {
					Ppi = Ppi + std::max(double(0), dt*mc_i[j]*(v_i[i]-v_i[colsMC[j]]));
					Pni = Pni + std::min(double(0), dt*mc_i[j]*(v_i[i]-v_i[colsMC[j]]));
					Qpi = std::max(Qpi, c0_i[colsMC[j]] + 0.5*dt*v_i[colsMC[j]] - c0_i[i] - 0.5*dt*v_i[i]);
					Qni = std::min(Qni, c0_i[colsMC[j]] + 0.5*dt*v_i[colsMC[j]] - c0_i[i] - 0.5*dt*v_i[i]);
				}
				mi = mi + mc_i[j];
				j = j + 1;
			}
			else if (colsD0[k] < colsMC[j]) {
				ri_ind.push_back(colsD0[k]);
				ri.push_back(-0.5*dt*(dt*d1_i[k]*(v_i[i]-v_i[colsD0[k]]) + d1_i[k]*(c0_i[i]-c0_i[colsD0[k]]) + d0_i[k]*(c0_i[i]-c0_i[colsD0[k]])));
				if (colsD0[k] != i) {
					Ppi = Ppi + std::max(double(0), -0.5*dt*(dt*d1_i[k]*(v_i[i]-v_i[colsD0[k]]) + d1_i[k]*(c0_i[i]-c0_i[colsD0[k]]) + d0_i[k]*(c0_i[i]-c0_i[colsD0[k]])));
					Pni = Pni + std::min(double(0), -0.5*dt*(dt*d1_i[k]*(v_i[i]-v_i[colsD0[k]]) + d1_i[k]*(c0_i[i]-c0_i[colsD0[k]]) + d0_i[k]*(c0_i[i]-c0_i[colsD0[k]])));
					Qpi = std::max(Qpi, c0_i[colsD0[k]] + 0.5*dt*v_i[colsD0[k]] - c0_i[i] - 0.5*dt*v_i[i]);
					Qni = std::min(Qni, c0_i[colsD0[k]] + 0.5*dt*v_i[colsD0[k]] - c0_i[i] - 0.5*dt*v_i[i]);
				}
				k = k + 1;
			}
			else {
				ri_ind.push_back(colsD0[k]);
				ri.push_back(dt*mc_i[j]*(v_i[i]-v_i[colsMC[j]])-0.5*dt*(dt*d1_i[k]*(v_i[i]-v_i[colsD0[k]]) + d1_i[k]*(c0_i[i]-c0_i[colsD0[k]]) + d0_i[k]*(c0_i[i]-c0_i[colsD0[k]])));
				if (colsD0[k] != i) {
					Ppi = Ppi + std::max(double(0), dt*mc_i[j]*(v_i[i]-v_i[colsMC[j]])-0.5*dt*(dt*d1_i[k]*(v_i[i]-v_i[colsD0[k]]) + d1_i[k]*(c0_i[i]-c0_i[colsD0[k]]) + d0_i[k]*(c0_i[i]-c0_i[colsD0[k]])));
					Pni = Pni + std::min(double(0), dt*mc_i[j]*(v_i[i]-v_i[colsMC[j]])-0.5*dt*(dt*d1_i[k]*(v_i[i]-v_i[colsD0[k]]) + d1_i[k]*(c0_i[i]-c0_i[colsD0[k]]) + d0_i[k]*(c0_i[i]-c0_i[colsD0[k]])));
					Qpi = std::max(Qpi, c0_i[colsD0[k]] + 0.5*dt*v_i[colsD0[k]] - c0_i[i] - 0.5*dt*v_i[i]);
					Qni = std::min(Qni, c0_i[colsD0[k]] + 0.5*dt*v_i[colsD0[k]] - c0_i[i] - 0.5*dt*v_i[i]);
				}
				mi = mi + mc_i[j];
				k = k + 1;
				j = j + 1;
			}
		}
		while(j < ncolsMC) {
			ri_ind.push_back(colsMC[j]);
			ri.push_back(dt*mc_i[j]*(v_i[i]-v_i[colsMC[j]]));
			if (colsMC[j] != i) {
				Ppi = Ppi + std::max(double(0), dt*mc_i[j]*(v_i[i]-v_i[colsMC[j]]));
				Pni = Pni + std::min(double(0), dt*mc_i[j]*(v_i[i]-v_i[colsMC[j]]));
				Qpi = std::max(Qpi, c0_i[colsMC[j]] + 0.5*dt*v_i[colsMC[j]] - c0_i[i] - 0.5*dt*v_i[i]);
				Qni = std::min(Qni, c0_i[colsMC[j]] + 0.5*dt*v_i[colsMC[j]] - c0_i[i] - 0.5*dt*v_i[i]);
			}
			mi = mi + mc_i[j];
			j = j + 1;
		}
		while(k < ncolsD0) {
			ri_ind.push_back(colsD0[k]);
			ri.push_back(-0.5*dt*(dt*d1_i[k]*(v_i[i]-v_i[colsD0[k]]) + d1_i[k]*(c0_i[i]-c0_i[colsD0[k]]) + d0_i[k]*(c0_i[i]-c0_i[colsD0[k]])));
			if (colsD0[k] != i) {
				Ppi = Ppi + std::max(double(0), -0.5*dt*(dt*d1_i[k]*(v_i[i]-v_i[colsD0[k]]) + d1_i[k]*(c0_i[i]-c0_i[colsD0[k]]) + d0_i[k]*(c0_i[i]-c0_i[colsD0[k]])));
				Pni = Pni + std::min(double(0), -0.5*dt*(dt*d1_i[k]*(v_i[i]-v_i[colsD0[k]]) + d1_i[k]*(c0_i[i]-c0_i[colsD0[k]]) + d0_i[k]*(c0_i[i]-c0_i[colsD0[k]])));
				Qpi = std::max(Qpi, c0_i[colsD0[k]] + 0.5*dt*v_i[colsD0[k]] - c0_i[i] - 0.5*dt*v_i[i]);
				Qni = std::min(Qni, c0_i[colsD0[k]] + 0.5*dt*v_i[colsD0[k]] - c0_i[i] - 0.5*dt*v_i[i]);
			}
			k = k + 1;
		}
		Qpi = std::max(double(0), Qpi);
		Qni = std::min(double(0), Qni);
		MatSetValues(r, 1, &i, ri_ind.size(), ri_ind.data(), ri.data(), INSERT_VALUES);

		if (std::abs(Ppi) > 1e-23) {
			VecSetValue(Rp, i, std::min(1.0, (mi*Qpi)/Ppi), INSERT_VALUES);
		}
		if (std::abs(Pni) > 1e-23) {
			VecSetValue(Rn, i, std::min(1.0, (mi*Qni)/Pni), INSERT_VALUES);
		}

		MatRestoreRow(MC, i, &ncolsMC, &colsMC, &mc_i);
		MatRestoreRow(D0, i, &ncolsD0, &colsD0, &d0_i);
		MatRestoreRow(D1, i, NULL, NULL, &d1_i);

		ri_ind.clear();
		ri.shrink_to_fit();
	}

	VecAssemblyBegin(Rp);
	VecAssemblyEnd(Rp);
	VecAssemblyBegin(Rn);
	VecAssemblyEnd(Rn);

	MatAssemblyBegin(r, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(r, MAT_FINAL_ASSEMBLY);

	VecRestoreArrayRead(c0_SEQ, &c0_i);
	VecRestoreArrayRead(v1Over2_SEQ, &v_i);

	VecDestroy(&c0_SEQ);
	VecDestroy(&v1Over2_SEQ);

	// Zalesak's algorithm
	const PetscScalar* Rp_i;
	const PetscScalar* Rn_i;

	VecGetArrayRead(Rp, &Rp_i);
	VecGetArrayRead(Rn, &Rn_i);

	for (PetscInt i = L0_FromRow; i < L0_ToRow; i = i + 1) {
		const PetscScalar* r_i;
		const PetscInt* colsr;
		PetscInt ncolsr;

		MatGetRow(r, i, &ncolsr, &colsr, &r_i);

		PetscScalar fi = 0;

		for (PetscInt j = 0; j < ncolsr; j = j + 1) {
			if (r_i[j]>0)
				fi = fi + std::min(Rp_i[i], Rn_i[colsr[j]])*r_i[j];
			else
				fi = fi + std::min(Rn_i[i], Rp_i[colsr[j]])*r_i[j];
		}
		if (std::abs(fi)<1e23 && std::abs(fi)>1e-23)
			VecSetValue(fStar, i, fi, INSERT_VALUES);
		MatRestoreRow(r, i, &ncolsr, &colsr, &r_i);
	}

	VecAssemblyBegin(fStar);
	VecAssemblyEnd(fStar);

	VecRestoreArrayRead(Rp, &Rp_i);
	VecRestoreArrayRead(Rn, &Rn_i);

	VecDestroy(&Rp);
	VecDestroy(&Rn);

	return 0;
}

//DOI:10.1016/j.cma.2008.08.016
//Flux-corrected-transport finite element method
int FEMFCT_ML_Compute(Mat MC, Mat &ML) {
	//Input: MC
	//Output: ML

	MatZeroEntries(ML);

	Vec vtmp;
	MatCreateVecs(MC, NULL, &vtmp);

	PetscBarrier(NULL);

	MatGetRowSum(MC, vtmp);

	//ML
	MatDiagonalSet(ML, vtmp, INSERT_VALUES);

	PetscBarrier(NULL);

	VecDestroy(&vtmp);

	return 0;
}
