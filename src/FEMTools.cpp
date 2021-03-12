#include <dolfin.h>

using namespace dolfin;

//DOI:10.1016/j.cma.2008.08.016
//Computation of symmetric matrix D from stiffness matrix A in Algebraic Flux-corrected finite element method
//In our paper notation of matrix was changed to Gamma
int AFC_D_Compute(Mat A, Mat D) {
	//Input: A
	//Output: D

	MatZeroEntries(D);

	Mat ATr;
	MatTranspose(A, MAT_INITIAL_MATRIX, &ATr);

	PetscInt A_FromRow;
	PetscInt A_ToRow;

	MatGetOwnershipRange(A, &A_FromRow, &A_ToRow);

	const PetscScalar* a_i;
	const PetscInt* colsA;
	PetscInt ncolsA;
	const PetscScalar* a_j;
	const PetscInt* colsATr;
	PetscInt ncolsATr;
	PetscInt j;
	PetscInt k;
	PetscScalar tmpval;
	std::vector<PetscScalar> d_i;
	std::vector<PetscInt> d_ind;

	PetscBarrier(NULL);

	for (PetscInt i = A_FromRow; i < A_ToRow; i = i + 1) {
		MatGetRow(A, i, &ncolsA, &colsA, &a_i);
		MatGetRow(ATr, i, &ncolsATr, &colsATr, &a_j);

		//Nonzero entries of A for computations
		//Only nonzero and upper triangle entries of matrix D is needed to be set
		j = 0;
		k = 0;
		tmpval = 0;

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
		MatSetValues(D, 1, &i, d_ind.size(), d_ind.data(), d_i.data(), INSERT_VALUES);
		MatSetValues(D, d_ind.size(), d_ind.data(), 1, &i, d_i.data(), INSERT_VALUES);
		MatRestoreRow(A, i, &ncolsA, &colsA, &a_i);
		MatRestoreRow(ATr, i, &ncolsATr, &colsATr, &a_j);

		d_i.clear();
		d_i.shrink_to_fit();
		d_ind.clear();
		d_ind.shrink_to_fit();
	}

	MatDestroy(&ATr);

	Vec vtmp;//setting the diagonal of D
	MatCreateVecs(A, NULL, &vtmp);
	VecSet(vtmp, 0);

	PetscBarrier(NULL);

	MatDiagonalSet(D, vtmp, INSERT_VALUES);

	PetscBarrier(NULL);

	MatGetRowSum(D, vtmp);
	VecScale(vtmp, -1);

	PetscBarrier(NULL);

	MatDiagonalSet(D, vtmp, INSERT_VALUES);

	PetscBarrier(NULL);

	VecDestroy(&vtmp);

	return 0;
}

//DOI:10.1016/j.cma.2008.08.016
//L = A + D or L= A + Gamma in our paper
int AFC_L_Compute(Mat A, Mat D, Mat L) {
	//Input: A, D
	//Output: L

	PetscBarrier(NULL);

	MatCopy(A, L, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(L, 1, D, DIFFERENT_NONZERO_PATTERN);

	return 0;
}

//DOI:10.1016/j.cma.2008.08.016
//Computation of time step in Algebraic Flux-corrected finite element method for evolutionary convection-diffusion-reaction equation
double AFC_dt_Compute(Mat ML, Mat Lk) {
	//Input: ML, Lk
	//Output: dt

	PetscScalar dt=0;
	Vec vtmp1, vtmp2;

	MatCreateVecs(ML, NULL, &vtmp1);

	PetscBarrier(NULL);

	VecDuplicate(vtmp1, &vtmp2);

	//to make non-positive elements of Lk ineffective
	VecSet(vtmp1, 1e-13);//a truncated value of almost zero
	MatGetDiagonal(Lk, vtmp2);
	VecPointwiseMax(vtmp2, vtmp2, vtmp1);

	MatGetDiagonal(ML, vtmp1);
	
	PetscBarrier(NULL);
	
	VecPointwiseDivide(vtmp1, vtmp1, vtmp2);

	PetscBarrier(NULL);

	VecMin(vtmp1, NULL, &dt);
	dt = 2*dt;
	
	//round off time step
	if(dt>1e-3)
		dt = std::floor(1e3*dt)/1e3;
		
	PetscBarrier(NULL);

	VecDestroy(&vtmp1);
	VecDestroy(&vtmp2);

	return dt;
}

//DOI:10.1016/j.cma.2008.08.016
//Computation of the corrected residual (anti-diffusive alpha*residual low-order approximation) term in Algebraic Flux-corrected finite element method for evolutionary convection-diffusion-reaction equation
//Note that in linear AFC the low-order r*=alpha*r=alpha*beta_tilde can be explicitly computed
int LinAFC_fStar_Compute(Mat ML, Mat MC, Mat D1, Mat D0, Mat A0, Vec c0, Vec b0, PetscReal dt, Mat r, Vec fStar, bool isresidual=false) {
	//Input: ML, MC, D1, D0, A0, c0, b0, dt
	//Output: r, f*
	//flag: isresidual=true means the aim is just to calculate the normal residual without applying AFC

	MatZeroEntries(r);//the matrix r will be low-order estimate beta_tilde in paper
	VecSet(fStar, 0);

	Vec c_tilde, vtmp;

	MatCreateVecs(A0, NULL, &c_tilde);

	PetscBarrier(NULL);

	VecDuplicate(c_tilde, &vtmp);
	VecSet(c_tilde, 0);
	VecSet(vtmp, 0);

	PetscBarrier(NULL);

	//c_tilde
	VecCopy(b0, c_tilde);//in our model b0 represents I_i^{k-1}
	VecScale(c_tilde, -1);
	MatMult(A0, c0, vtmp);
	MatMultAdd(D0, c0, vtmp, vtmp);//vtmp=(A0+D0)c0=L0c0

	PetscBarrier(NULL);

	VecAXPY(c_tilde, 1, vtmp);
	MatGetDiagonal(ML, vtmp);
	VecPointwiseDivide(c_tilde, c_tilde, vtmp);//Inverse of M_L
	//c_tilde = =inv(ML)[(A0+D0)c0+I0]=inv(ML)(L0c0-I0)
	VecScale(c_tilde, -0.5*dt);//c_tilde = =-0.5*dt*inv(ML)[(A0+D0)c0+I0]=-0.5*dt*inv(ML)(L0c0-I0)
	VecAXPY(c_tilde, 1, c0);//c_tilde = =c0-0.5*dt*inv(ML)[(A0+D0)c0+I0]=c0-0.5*dt*inv(ML)(L0c0-I0)

	VecGhostUpdateBegin(c_tilde, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(c_tilde, INSERT_VALUES, SCATTER_FORWARD);

	PetscBarrier(NULL);

	//each processor needs other processors indices
	VecScatter par2seq;
	Vec c0_SEQ, c_tilde_SEQ;

	VecScatterCreateToAll(c0, &par2seq, &c0_SEQ);
	VecScatterBegin(par2seq, c0, c0_SEQ, INSERT_VALUES, SCATTER_FORWARD);
	VecScatterEnd(par2seq, c0, c0_SEQ, INSERT_VALUES, SCATTER_FORWARD);

	PetscBarrier(NULL);

	VecScatterDestroy(&par2seq);

	VecScatterCreateToAll(c_tilde, &par2seq, &c_tilde_SEQ);
	VecScatterBegin(par2seq, c_tilde, c_tilde_SEQ, INSERT_VALUES, SCATTER_FORWARD);
	VecScatterEnd(par2seq, c_tilde, c_tilde_SEQ, INSERT_VALUES, SCATTER_FORWARD);

	PetscBarrier(NULL);

	VecScatterDestroy(&par2seq);
	VecDestroy(&vtmp);
	VecDestroy(&c_tilde);

	//start of computing requirements for Zalesak's algorithm

	//r (low-order approximation beta_tilde)
	PetscInt A0_FromRow;
	PetscInt A0_ToRow;

	MatGetOwnershipRange(A0, &A0_FromRow, &A0_ToRow);

	Vec Rp, Rn;
	VecDuplicate(c0_SEQ, &Rp);
	VecSet(Rp, 1);
	VecDuplicate(c0_SEQ, &Rn);
	VecSet(Rn, 1);

	const PetscScalar* c0_i;
	const PetscScalar* ct_i;
	const PetscScalar* mc_i;
	const PetscInt* colsMC;
	PetscInt ncolsMC;
	const PetscScalar* d1_i;
	const PetscScalar* d0_i;
	const PetscInt* colsD0;
	PetscInt ncolsD0;
	PetscInt j;
	PetscInt k;
	PetscScalar Ppi;
	PetscScalar Pni;
	PetscScalar Qpi;
	PetscScalar Qni;
	PetscScalar mi;
	std::vector<PetscScalar> ri;
	std::vector<PetscInt> ri_ind;

	VecGetArrayRead(c0_SEQ, &c0_i);
	VecGetArrayRead(c_tilde_SEQ, &ct_i);

	for (PetscInt i = A0_FromRow; i < A0_ToRow; i = i + 1) {
		MatGetRow(MC, i, &ncolsMC, &colsMC, &mc_i);
		MatGetRow(D0, i, &ncolsD0, &colsD0, &d0_i);
		MatGetRow(D1, i, NULL, NULL, &d1_i);//has the same structure as D0

		j = 0;
		k = 0;
		Ppi = 0;
		Pni = 0;
		if (colsMC[j] < colsD0[k]) {
			Qpi = ct_i[colsMC[j]] - ct_i[i];//u_tilde or low-order approximation c_tilde in our paper
			Qni = ct_i[colsMC[j]] - ct_i[i];//u_tilde or low-order approximation c_tilde in our paper
		}
		else {
			Qpi = ct_i[colsD0[k]] - ct_i[i];//u_tilde or low-order approximation c_tilde in our paper
			Qni = ct_i[colsD0[k]] - ct_i[i];//u_tilde or low-order approximation c_tilde in our paper
		}
		mi = 0;

		while (j < ncolsMC && k < ncolsD0) { 
			if (colsMC[j] < colsD0[k]) {
				ri_ind.push_back(colsMC[j]);
				ri.push_back(mc_i[j]*(2*(ct_i[i]-ct_i[colsMC[j]])-(c0_i[i]-c0_i[colsMC[j]]))-mc_i[j]*(c0_i[i]-c0_i[colsMC[j]]));
				if (colsMC[j] != i) {
					Ppi = Ppi + std::max(double(0), mc_i[j]*(2*(ct_i[i]-ct_i[colsMC[j]])-(c0_i[i]-c0_i[colsMC[j]]))-mc_i[j]*(c0_i[i]-c0_i[colsMC[j]]));//r or low-order approximation beta_tilde in our paper depends on nonzero structure
					Pni = Pni + std::min(double(0), mc_i[j]*(2*(ct_i[i]-ct_i[colsMC[j]])-(c0_i[i]-c0_i[colsMC[j]]))-mc_i[j]*(c0_i[i]-c0_i[colsMC[j]]));//r or low-order approximation beta_tilde in our paper depends on nonzero structure
					Qpi = std::max(Qpi, ct_i[colsMC[j]] - ct_i[i]);
					Qni = std::min(Qni, ct_i[colsMC[j]] - ct_i[i]);
				}
				mi = mi + mc_i[j];// M_L matrix diagonal which is required in computing R+ and R-; preferred to computed again rather than read from ML diretly
				j = j + 1;
			}
			else if (colsD0[k] < colsMC[j]) {
				ri_ind.push_back(colsD0[k]);
				ri.push_back(-0.5*dt*d1_i[k]*(2*(ct_i[i]-ct_i[colsD0[k]])-(c0_i[i]-c0_i[colsD0[k]])) -0.5*dt*d0_i[k]*(c0_i[i]-c0_i[colsD0[k]]));
				if (colsD0[k] != i) {
					Ppi = Ppi + std::max(double(0), -0.5*dt*d1_i[k]*(2*(ct_i[i]-ct_i[colsD0[k]])-(c0_i[i]-c0_i[colsD0[k]])) -0.5*dt*d0_i[k]*(c0_i[i]-c0_i[colsD0[k]]));
					Pni = Pni + std::min(double(0), -0.5*dt*d1_i[k]*(2*(ct_i[i]-ct_i[colsD0[k]])-(c0_i[i]-c0_i[colsD0[k]])) -0.5*dt*d0_i[k]*(c0_i[i]-c0_i[colsD0[k]]));
					Qpi = std::max(Qpi, ct_i[colsD0[k]] - ct_i[i]);
					Qni = std::min(Qni, ct_i[colsD0[k]] - ct_i[i]);
				}
				k = k + 1;
			}
			else {
				ri_ind.push_back(colsD0[k]);
				ri.push_back(mc_i[j]*(2*(ct_i[i]-ct_i[colsMC[j]])-(c0_i[i]-c0_i[colsMC[j]]))-mc_i[j]*(c0_i[i]-c0_i[colsMC[j]])-0.5*dt*d1_i[k]*(2*(ct_i[i]-ct_i[colsD0[k]])-(c0_i[i]-c0_i[colsD0[k]])) -0.5*dt*d0_i[k]*(c0_i[i]-c0_i[colsD0[k]]));
				if (colsD0[k] != i) {
					Ppi = Ppi + std::max(double(0), mc_i[j]*(2*(ct_i[i]-ct_i[colsMC[j]])-(c0_i[i]-c0_i[colsMC[j]]))-mc_i[j]*(c0_i[i]-c0_i[colsMC[j]])-0.5*dt*d1_i[k]*(2*(ct_i[i]-ct_i[colsD0[k]])-(c0_i[i]-c0_i[colsD0[k]])) -0.5*dt*d0_i[k]*(c0_i[i]-c0_i[colsD0[k]]));
					Pni = Pni + std::min(double(0), mc_i[j]*(2*(ct_i[i]-ct_i[colsMC[j]])-(c0_i[i]-c0_i[colsMC[j]]))-mc_i[j]*(c0_i[i]-c0_i[colsMC[j]])-0.5*dt*d1_i[k]*(2*(ct_i[i]-ct_i[colsD0[k]])-(c0_i[i]-c0_i[colsD0[k]])) -0.5*dt*d0_i[k]*(c0_i[i]-c0_i[colsD0[k]]));
					Qpi = std::max(Qpi, ct_i[colsD0[k]] - ct_i[i]);
					Qni = std::min(Qni, ct_i[colsD0[k]] - ct_i[i]);
				}
				mi = mi + mc_i[j];
				k = k + 1;
				j = j + 1;
			}
		}
		while(j < ncolsMC) {
			ri_ind.push_back(colsMC[j]);
			ri.push_back(mc_i[j]*(2*(ct_i[i]-ct_i[colsMC[j]])-(c0_i[i]-c0_i[colsMC[j]]))-mc_i[j]*(c0_i[i]-c0_i[colsMC[j]]));
			if (colsMC[j] != i) {
				Ppi = Ppi + std::max(double(0), mc_i[j]*(2*(ct_i[i]-ct_i[colsMC[j]])-(c0_i[i]-c0_i[colsMC[j]]))-mc_i[j]*(c0_i[i]-c0_i[colsMC[j]]));
				Pni = Pni + std::min(double(0), mc_i[j]*(2*(ct_i[i]-ct_i[colsMC[j]])-(c0_i[i]-c0_i[colsMC[j]]))-mc_i[j]*(c0_i[i]-c0_i[colsMC[j]]));
				Qpi = std::max(Qpi, ct_i[colsMC[j]] - ct_i[i]);
				Qni = std::min(Qni, ct_i[colsMC[j]] - ct_i[i]);
			}
			mi = mi + mc_i[j];
			j = j + 1;
		}
		while(k < ncolsD0) {
			ri_ind.push_back(colsD0[k]);
			ri.push_back(-0.5*dt*d1_i[k]*(2*(ct_i[i]-ct_i[colsD0[k]])-(c0_i[i]-c0_i[colsD0[k]])) -0.5*dt*d0_i[k]*(c0_i[i]-c0_i[colsD0[k]]));
			if (colsD0[k] != i) {
				Ppi = Ppi + std::max(double(0), -0.5*dt*d1_i[k]*(2*(ct_i[i]-ct_i[colsD0[k]])-(c0_i[i]-c0_i[colsD0[k]])) -0.5*dt*d0_i[k]*(c0_i[i]-c0_i[colsD0[k]]));
				Pni = Pni + std::min(double(0), -0.5*dt*d1_i[k]*(2*(ct_i[i]-ct_i[colsD0[k]])-(c0_i[i]-c0_i[colsD0[k]])) -0.5*dt*d0_i[k]*(c0_i[i]-c0_i[colsD0[k]]));
				Qpi = std::max(Qpi, ct_i[colsD0[k]] - ct_i[i]);
				Qni = std::min(Qni, ct_i[colsD0[k]] - ct_i[i]);
			}
			k = k + 1;
		}
		Qpi = std::max(double(0), Qpi);
		Qni = std::min(double(0), Qni);
		MatSetValues(r, 1, &i, ri_ind.size(), ri_ind.data(), ri.data(), INSERT_VALUES);

		//At the end of each row we are ready to calculate R+,R-
		VecSetValue(Rp, i, std::min(1.0, (mi*Qpi)/Ppi), INSERT_VALUES);
		VecSetValue(Rn, i, std::min(1.0, (mi*Qni)/Pni), INSERT_VALUES);

		MatRestoreRow(MC, i, &ncolsMC, &colsMC, &mc_i);
		MatRestoreRow(D0, i, &ncolsD0, &colsD0, &d0_i);
		MatRestoreRow(D1, i, NULL, NULL, &d1_i);

		ri.clear();
		ri.shrink_to_fit();
		ri_ind.clear();
		ri_ind.shrink_to_fit();
	}

	PetscBarrier(NULL);

	VecAssemblyBegin(Rp);
	VecAssemblyEnd(Rp);
	VecGhostUpdateBegin(Rp, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Rp, INSERT_VALUES, SCATTER_FORWARD);
	VecAssemblyBegin(Rn);
	VecAssemblyEnd(Rn);
	VecGhostUpdateBegin(Rn, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Rn, INSERT_VALUES, SCATTER_FORWARD);

	MatAssemblyBegin(r, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(r, MAT_FINAL_ASSEMBLY);
	
	//end of computing requirements for Zalesak's algorithm

	PetscBarrier(NULL);

	VecRestoreArrayRead(c0_SEQ, &c0_i);
	VecRestoreArrayRead(c_tilde_SEQ, &ct_i);

	VecDestroy(&c0_SEQ);
	VecDestroy(&c_tilde_SEQ);

	// Zalesak's algorithm
	const PetscScalar* Rp_i;
	const PetscScalar* Rn_i;
	const PetscScalar* r_i;
	const PetscInt* colsr;
	PetscInt ncolsr;

	VecGetArrayRead(Rp, &Rp_i);
	VecGetArrayRead(Rn, &Rn_i);

	PetscScalar fi;
	for (PetscInt i = A0_FromRow; i < A0_ToRow; i = i + 1) {
		MatGetRow(r, i, &ncolsr, &colsr, &r_i);

		fi = double(0);

		for (PetscInt j = 0; j < ncolsr; j = j + 1) {
			if (isresidual)
				fi = fi + r_i[j];//all alpha_ij=1, usual FEM
			else {
				if (r_i[j]>double(0)) {
					fi = fi + std::min(Rp_i[i], Rn_i[colsr[j]])*r_i[j];
				}
				else {
					fi = fi + std::min(Rn_i[i], Rp_i[colsr[j]])*r_i[j];
				}
			}
		}
		VecSetValue(fStar, i, fi, INSERT_VALUES);
		MatRestoreRow(r, i, &ncolsr, &colsr, &r_i);
	}

	PetscBarrier(NULL);

	VecAssemblyBegin(fStar);
	VecAssemblyEnd(fStar);
	VecGhostUpdateBegin(fStar, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(fStar, INSERT_VALUES, SCATTER_FORWARD);

	VecRestoreArrayRead(Rp, &Rp_i);
	VecRestoreArrayRead(Rn, &Rn_i);

	PetscBarrier(NULL);

	VecDestroy(&Rp);
	VecDestroy(&Rn);

	return 0;
}

//DOI:10.1016/j.cma.2008.08.016
//Computation of the anti-diffusive term alpha with help of residual low-order approximation in Algebraic Flux-corrected finite element method for evolutionary convection-diffusion-reaction equation
//Note that in nonlinear AFC the low-order r*=alpha*r=alpha*beta_tilde can not be explicitly computed so we just have an alpha matrix from here and then multiply it by r
int nonLinAFC_alpha_Compute(Mat ML, Mat MC, Mat D1, Mat D0, Mat A0, Vec c0, Vec b0, PetscReal dt, Mat r, Mat alpha) {
	//Input: ML, MC, D1, D0, A0, c0, b0, dt
	//Output: r and alpha

	MatZeroEntries(r);//the matrix r will be populated by low-order estimate beta_tilde in paper
	MatZeroEntries(alpha);

	Vec c_tilde, vtmp;

	MatCreateVecs(A0, NULL, &c_tilde);

	PetscBarrier(NULL);

	VecDuplicate(c_tilde, &vtmp);
	VecSet(c_tilde, 0);
	VecSet(vtmp, 0);

	PetscBarrier(NULL);

	//c_tilde
	VecCopy(b0, c_tilde);//in our model b0 represents I_i^{k-1}
	VecScale(c_tilde, -1);
	MatMult(A0, c0, vtmp);
	MatMultAdd(D0, c0, vtmp, vtmp);//vtmp=(A0+D0)c0=L0c0

	PetscBarrier(NULL);

	VecAXPY(c_tilde, 1, vtmp);
	MatGetDiagonal(ML, vtmp);
	VecPointwiseDivide(c_tilde, c_tilde, vtmp);//Inverse of M_L
	//c_tilde = =inv(ML)[(A0+D0)c0+I0]=inv(ML)(L0c0-I0)
	VecScale(c_tilde, -0.5*dt);//c_tilde = =-0.5*dt*inv(ML)[(A0+D0)c0+I0]=-0.5*dt*inv(ML)(L0c0-I0)
	VecAXPY(c_tilde, 1, c0);//c_tilde = =c0-0.5*dt*inv(ML)[(A0+D0)c0+I0]=c0-0.5*dt*inv(ML)(L0c0-I0)

	PetscBarrier(NULL);

	//synchronizing ghost values
	VecGhostUpdateBegin(c0, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(c0, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateBegin(c_tilde, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(c_tilde, INSERT_VALUES, SCATTER_FORWARD);

	//each processor needs other processors indices
	VecScatter par2seq;
	Vec c0_SEQ, c_tilde_SEQ;

	VecScatterCreateToAll(c0, &par2seq, &c0_SEQ);
	VecScatterBegin(par2seq, c0, c0_SEQ, INSERT_VALUES, SCATTER_FORWARD);
	VecScatterEnd(par2seq, c0, c0_SEQ, INSERT_VALUES, SCATTER_FORWARD);

	PetscBarrier(NULL);

	VecScatterDestroy(&par2seq);

	VecScatterCreateToAll(c_tilde, &par2seq, &c_tilde_SEQ);
	VecScatterBegin(par2seq, c_tilde, c_tilde_SEQ, INSERT_VALUES, SCATTER_FORWARD);
	VecScatterEnd(par2seq, c_tilde, c_tilde_SEQ, INSERT_VALUES, SCATTER_FORWARD);

	PetscBarrier(NULL);

	VecScatterDestroy(&par2seq);
	VecDestroy(&vtmp);
	VecDestroy(&c_tilde);

	//start of computing requirements for Zalesak's algorithm

	//r (low-order approximation beta_tilde)
	PetscInt A0_FromRow;
	PetscInt A0_ToRow;

	MatGetOwnershipRange(A0, &A0_FromRow, &A0_ToRow);

	Vec Rp, Rn, vtmp_SEQ;
	VecDuplicate(c0_SEQ, &Rp);
	VecSet(Rp, 1);
	VecDuplicate(c0_SEQ, &Rn);
	VecSet(Rn, 1);

	const PetscScalar* c0_i;
	const PetscScalar* ct_i;
	const PetscScalar* mc_i;
	const PetscInt* colsMC;
	PetscInt ncolsMC;
	const PetscScalar* d1_i;
	const PetscScalar* d0_i;
	const PetscInt* colsD0;
	PetscInt ncolsD0;
	PetscInt j;
	PetscInt k;
	PetscScalar Ppi;
	PetscScalar Pni;
	PetscScalar Qpi;
	PetscScalar Qni;
	PetscScalar mi;
	std::vector<PetscScalar> ri;
	std::vector<PetscInt> ri_ind;

	VecGetArrayRead(c0_SEQ, &c0_i);
	VecGetArrayRead(c_tilde_SEQ, &ct_i);

	PetscBarrier(NULL);

	for (PetscInt i = A0_FromRow; i < A0_ToRow; i = i + 1) {
		MatGetRow(MC, i, &ncolsMC, &colsMC, &mc_i);
		MatGetRow(D0, i, &ncolsD0, &colsD0, &d0_i);
		MatGetRow(D1, i, NULL, NULL, &d1_i);//has the same structure as D0

		j = 0;
		k = 0;
		Ppi = 0;
		Pni = 0;
		if (colsMC[j] < colsD0[k]) {
			Qpi = ct_i[colsMC[j]] - ct_i[i];//u_tilde or low-order approximation c_tilde in our paper
			Qni = ct_i[colsMC[j]] - ct_i[i];//u_tilde or low-order approximation c_tilde in our paper
		}
		else {
			Qpi = ct_i[colsD0[k]] - ct_i[i];//u_tilde or low-order approximation c_tilde in our paper
			Qni = ct_i[colsD0[k]] - ct_i[i];//u_tilde or low-order approximation c_tilde in our paper
		}
		mi = 0;

		while (j < ncolsMC && k < ncolsD0) { 
			if (colsMC[j] < colsD0[k]) {
				ri_ind.push_back(colsMC[j]);
				ri.push_back(mc_i[j]*(2*(ct_i[i]-ct_i[colsMC[j]])-(c0_i[i]-c0_i[colsMC[j]]))-mc_i[j]*(c0_i[i]-c0_i[colsMC[j]]));
				if (colsMC[j] != i) {
					Ppi = Ppi + std::max(double(0), mc_i[j]*(2*(ct_i[i]-ct_i[colsMC[j]])-(c0_i[i]-c0_i[colsMC[j]]))-mc_i[j]*(c0_i[i]-c0_i[colsMC[j]]));//r or low-order approximation beta_tilde in our paper depends on nonzero structure
					Pni = Pni + std::min(double(0), mc_i[j]*(2*(ct_i[i]-ct_i[colsMC[j]])-(c0_i[i]-c0_i[colsMC[j]]))-mc_i[j]*(c0_i[i]-c0_i[colsMC[j]]));//r or low-order approximation beta_tilde in our paper depends on nonzero structure
					Qpi = std::max(Qpi, ct_i[colsMC[j]] - ct_i[i]);
					Qni = std::min(Qni, ct_i[colsMC[j]] - ct_i[i]);
				}
				mi = mi + mc_i[j];// M_L matrix diagonal which is required in computing R+ and R-; preferred to computed again rather than read from ML diretly
				j = j + 1;
			}
			else if (colsD0[k] < colsMC[j]) {
				ri_ind.push_back(colsD0[k]);
				ri.push_back(-0.5*dt*d1_i[k]*(2*(ct_i[i]-ct_i[colsD0[k]])-(c0_i[i]-c0_i[colsD0[k]])) -0.5*dt*d0_i[k]*(c0_i[i]-c0_i[colsD0[k]]));
				if (colsD0[k] != i) {
					Ppi = Ppi + std::max(double(0), -0.5*dt*d1_i[k]*(2*(ct_i[i]-ct_i[colsD0[k]])-(c0_i[i]-c0_i[colsD0[k]])) -0.5*dt*d0_i[k]*(c0_i[i]-c0_i[colsD0[k]]));
					Pni = Pni + std::min(double(0), -0.5*dt*d1_i[k]*(2*(ct_i[i]-ct_i[colsD0[k]])-(c0_i[i]-c0_i[colsD0[k]])) -0.5*dt*d0_i[k]*(c0_i[i]-c0_i[colsD0[k]]));
					Qpi = std::max(Qpi, ct_i[colsD0[k]] - ct_i[i]);
					Qni = std::min(Qni, ct_i[colsD0[k]] - ct_i[i]);
				}
				k = k + 1;
			}
			else {
				ri_ind.push_back(colsD0[k]);
				ri.push_back(mc_i[j]*(2*(ct_i[i]-ct_i[colsMC[j]])-(c0_i[i]-c0_i[colsMC[j]]))-mc_i[j]*(c0_i[i]-c0_i[colsMC[j]])-0.5*dt*d1_i[k]*(2*(ct_i[i]-ct_i[colsD0[k]])-(c0_i[i]-c0_i[colsD0[k]])) -0.5*dt*d0_i[k]*(c0_i[i]-c0_i[colsD0[k]]));
				if (colsD0[k] != i) {
					Ppi = Ppi + std::max(double(0), mc_i[j]*(2*(ct_i[i]-ct_i[colsMC[j]])-(c0_i[i]-c0_i[colsMC[j]]))-mc_i[j]*(c0_i[i]-c0_i[colsMC[j]])-0.5*dt*d1_i[k]*(2*(ct_i[i]-ct_i[colsD0[k]])-(c0_i[i]-c0_i[colsD0[k]])) -0.5*dt*d0_i[k]*(c0_i[i]-c0_i[colsD0[k]]));
					Pni = Pni + std::min(double(0), mc_i[j]*(2*(ct_i[i]-ct_i[colsMC[j]])-(c0_i[i]-c0_i[colsMC[j]]))-mc_i[j]*(c0_i[i]-c0_i[colsMC[j]])-0.5*dt*d1_i[k]*(2*(ct_i[i]-ct_i[colsD0[k]])-(c0_i[i]-c0_i[colsD0[k]])) -0.5*dt*d0_i[k]*(c0_i[i]-c0_i[colsD0[k]]));
					Qpi = std::max(Qpi, ct_i[colsD0[k]] - ct_i[i]);
					Qni = std::min(Qni, ct_i[colsD0[k]] - ct_i[i]);
				}
				mi = mi + mc_i[j];
				k = k + 1;
				j = j + 1;
			}
		}
		while(j < ncolsMC) {
			ri_ind.push_back(colsMC[j]);
			ri.push_back(mc_i[j]*(2*(ct_i[i]-ct_i[colsMC[j]])-(c0_i[i]-c0_i[colsMC[j]]))-mc_i[j]*(c0_i[i]-c0_i[colsMC[j]]));
			if (colsMC[j] != i) {
				Ppi = Ppi + std::max(double(0), mc_i[j]*(2*(ct_i[i]-ct_i[colsMC[j]])-(c0_i[i]-c0_i[colsMC[j]]))-mc_i[j]*(c0_i[i]-c0_i[colsMC[j]]));
				Pni = Pni + std::min(double(0), mc_i[j]*(2*(ct_i[i]-ct_i[colsMC[j]])-(c0_i[i]-c0_i[colsMC[j]]))-mc_i[j]*(c0_i[i]-c0_i[colsMC[j]]));
				Qpi = std::max(Qpi, ct_i[colsMC[j]] - ct_i[i]);
				Qni = std::min(Qni, ct_i[colsMC[j]] - ct_i[i]);
			}
			mi = mi + mc_i[j];
			j = j + 1;
		}
		while(k < ncolsD0) {
			ri_ind.push_back(colsD0[k]);
			ri.push_back(-0.5*dt*d1_i[k]*(2*(ct_i[i]-ct_i[colsD0[k]])-(c0_i[i]-c0_i[colsD0[k]])) -0.5*dt*d0_i[k]*(c0_i[i]-c0_i[colsD0[k]]));
			if (colsD0[k] != i) {
				Ppi = Ppi + std::max(double(0), -0.5*dt*d1_i[k]*(2*(ct_i[i]-ct_i[colsD0[k]])-(c0_i[i]-c0_i[colsD0[k]])) -0.5*dt*d0_i[k]*(c0_i[i]-c0_i[colsD0[k]]));
				Pni = Pni + std::min(double(0), -0.5*dt*d1_i[k]*(2*(ct_i[i]-ct_i[colsD0[k]])-(c0_i[i]-c0_i[colsD0[k]])) -0.5*dt*d0_i[k]*(c0_i[i]-c0_i[colsD0[k]]));
				Qpi = std::max(Qpi, ct_i[colsD0[k]] - ct_i[i]);
				Qni = std::min(Qni, ct_i[colsD0[k]] - ct_i[i]);
			}
			k = k + 1;
		}
		Qpi = std::max(double(0), Qpi);
		Qni = std::min(double(0), Qni);
		MatSetValues(r, 1, &i, ri_ind.size(), ri_ind.data(), ri.data(), INSERT_VALUES);

		//At the end of each row we are ready to calculate R+,R-
		VecSetValue(Rp, i, std::min(1.0, (mi*Qpi)/Ppi), INSERT_VALUES);
		VecSetValue(Rn, i, std::min(1.0, (mi*Qni)/Pni), INSERT_VALUES);

		MatRestoreRow(MC, i, &ncolsMC, &colsMC, &mc_i);
		MatRestoreRow(D0, i, &ncolsD0, &colsD0, &d0_i);
		MatRestoreRow(D1, i, NULL, NULL, &d1_i);

		ri.clear();
		ri.shrink_to_fit();
		ri_ind.clear();
		ri_ind.shrink_to_fit();
	}

	PetscBarrier(NULL);

	VecAssemblyBegin(Rp);
	VecAssemblyEnd(Rp);
	VecGhostUpdateBegin(Rp, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Rp, INSERT_VALUES, SCATTER_FORWARD);
	VecAssemblyBegin(Rn);
	VecAssemblyEnd(Rn);
	VecGhostUpdateBegin(Rn, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Rn, INSERT_VALUES, SCATTER_FORWARD);

	MatAssemblyBegin(r, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(r, MAT_FINAL_ASSEMBLY);

	// P+,P-,Q+ and Q- are computed row by row to obtain the only decisives R+,R-
	// r now is populated by low-order approximation (beta_tilde in our paper)
	//end of computing requirements for Zalesak's algorithm

	PetscBarrier(NULL);

	VecRestoreArrayRead(c0_SEQ, &c0_i);
	VecRestoreArrayRead(c_tilde_SEQ, &ct_i);

	VecDestroy(&c0_SEQ);
	VecDestroy(&c_tilde_SEQ);

	//insurance that R+ and R- are in [0,1]
	VecDuplicate(Rp, &vtmp_SEQ);
	VecSet(vtmp_SEQ, 0);
	VecPointwiseMax(Rp, Rp, vtmp_SEQ);
	VecPointwiseMax(Rn, Rn, vtmp_SEQ);
	VecSet(vtmp_SEQ, 1);
	VecPointwiseMin(Rp, Rp, vtmp_SEQ);
	VecPointwiseMin(Rn, Rn, vtmp_SEQ);
	VecGhostUpdateBegin(Rp, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Rp, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateBegin(Rn, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Rn, INSERT_VALUES, SCATTER_FORWARD);

	PetscBarrier(NULL);

	VecDestroy(&vtmp_SEQ);

	// Zalesak's algorithm
	// Based on the last line of (8) alpha is computed from R+,R- and r(in our paper beta_tilde)
	const PetscScalar* Rp_i;
	const PetscScalar* Rn_i;
	const PetscScalar* r_i;
	const PetscInt* colsr;
	PetscInt ncolsr;
	std::vector<PetscScalar> alphai;
	std::vector<PetscInt> alphai_ind;

	VecGetArrayRead(Rp, &Rp_i);
	VecGetArrayRead(Rn, &Rn_i);

	for (PetscInt i = A0_FromRow; i < A0_ToRow; i = i + 1) {
		MatGetRow(r, i, &ncolsr, &colsr, &r_i);//a row of matrix beta_tilde (r)
		//we will read the beta_tilde row by row and then transfer it to alpha after we finished by a row
		for (PetscInt j = 0; j < ncolsr; j = j + 1) {
			if (r_i[j]>double(0)) {
				alphai_ind.push_back(colsr[j]);//in paper for alpha m=i and n=colsr[j]
				alphai.push_back(std::min(Rp_i[i], Rn_i[colsr[j]]));
			}
			else {
				alphai_ind.push_back(colsr[j]);
				alphai.push_back(std::min(Rn_i[i], Rp_i[colsr[j]]));
			}
		}
		MatSetValues(alpha, 1, &i, alphai_ind.size(), alphai_ind.data(), alphai.data(), INSERT_VALUES);
		MatRestoreRow(r, i, &ncolsr, &colsr, &r_i);

		alphai.clear();
		alphai.shrink_to_fit();
		alphai_ind.clear();
		alphai_ind.shrink_to_fit();
	}

	PetscBarrier(NULL);

	VecRestoreArrayRead(Rp, &Rp_i);
	VecRestoreArrayRead(Rn, &Rn_i);

	MatAssemblyBegin(alpha, MAT_FINAL_ASSEMBLY);//it is now the matrix of all alpha_mn
	MatAssemblyEnd(alpha, MAT_FINAL_ASSEMBLY);

	PetscBarrier(NULL);

	VecDestroy(&Rp);
	VecDestroy(&Rn);

	return 0;
}

//DOI:10.1016/j.cma.2008.08.016
//Computation of the lumped matrix M_L in Algebraic Flux-corrected finite element method for evolutionary convection-diffusion-reaction equation
int AFC_ML_Compute(Mat MC, Mat ML) {
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

int nonLinAFC_LinSys_Construct(Mat ML, Mat MC, Mat G1, Mat G0, Mat A1, Mat A0, Mat alpha, Mat tmp, Vec c0, Vec I1, Vec I0, double dt, Mat LinSysLhs, Vec LinSysRhs) {
	//Input: ML, MC, G1, G0, A1, A0, alpha, tmp, c0, I1, I0, dt
	//Output: LinSysLhs, LinSysRhs for a system Ax=b

	//b
	MatCopy(MC, tmp, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(tmp, -1, ML, DIFFERENT_NONZERO_PATTERN);
	MatPointwiseMult(alpha, tmp, LinSysLhs);//-alpha(ML-MC)
	MatZeroEntries(tmp);
	MatPointwiseMult(alpha, G0, tmp);
	MatAXPY(LinSysLhs, 0.5*dt, tmp, DIFFERENT_NONZERO_PATTERN);//-alpha(ML-MC)+0.5dt*alpha*Gamma0
	MatZeroEntries(tmp);
	MatCopy(A0, tmp, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(tmp, 1, G0, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(LinSysLhs, -0.5*dt, tmp, DIFFERENT_NONZERO_PATTERN);//-0.5dt*L0-alpha(ML-MC)+0.5dt*alpha*Gamma0
	MatAXPY(LinSysLhs, 1, ML, DIFFERENT_NONZERO_PATTERN);//ML-0.5dt*L0-alpha(ML-MC)+0.5dt*alpha*Gamma0
	MatMult(LinSysLhs, c0, LinSysRhs);//(ML-0.5dt*L0-alpha(ML-MC)+0.5dt*alpha*Gamma0)*c0

	PetscBarrier((PetscObject)LinSysRhs);

	VecAXPY(LinSysRhs, 0.5*dt, I0);
	VecAXPY(LinSysRhs, 0.5*dt, I1);//(ML-0.5dt*L0-alpha(ML-MC)+0.5dt*alpha*Gamma0)*c0+0.5dt*I1+0.5dt*I0

	PetscBarrier(NULL);

	VecGhostUpdateBegin(LinSysRhs, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(LinSysRhs, INSERT_VALUES, SCATTER_FORWARD);

	//A
	MatZeroEntries(tmp);
	MatZeroEntries(LinSysLhs);
	MatCopy(MC, tmp, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(tmp, -1, ML, DIFFERENT_NONZERO_PATTERN);
	MatPointwiseMult(alpha, tmp, LinSysLhs);//-alpha(ML-MC)
	MatZeroEntries(tmp);
	MatPointwiseMult(alpha, G1, tmp);
	MatAXPY(LinSysLhs, -0.5*dt, tmp, DIFFERENT_NONZERO_PATTERN);//-alpha(ML-MC)-0.5dt*alpha*Gamma1
	MatZeroEntries(tmp);
	MatCopy(A1, tmp, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(tmp, 1, G1, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(LinSysLhs, 0.5*dt, tmp, DIFFERENT_NONZERO_PATTERN);//0.5dt*L1-alpha(ML-MC)-0.5dt*alpha*Gamma1
	MatAXPY(LinSysLhs, 1, ML, DIFFERENT_NONZERO_PATTERN);//ML+0.5dt*L1-alpha(ML-MC)-0.5dt*alpha*Gamma1

	PetscBarrier((PetscObject)LinSysLhs);

	return 0;
}
