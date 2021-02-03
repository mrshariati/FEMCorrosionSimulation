#include <dolfin.h>

using namespace dolfin;

int funcRemoveNeg(dolfin::Function &func) {
	Vec vtmp;
	VecDuplicate(as_type<const dolfin::PETScVector>(func.vector())->vec(), &vtmp);
	VecSet(vtmp, 0);
	VecPointwiseMax(as_type<const dolfin::PETScVector>(func.vector())->vec(), as_type<const dolfin::PETScVector>(func.vector())->vec(), vtmp);// negative concentrations set to zero

	PetscBarrier(NULL);

	VecDestroy(&vtmp);
	return 0;
}

int funcsLinSum(std::vector<int> zi, std::vector<dolfin::Function> ci, dolfin::Function &lsum) {
	*(lsum.vector())= 0;
	for (size_t i=0; i<ci.size(); i = i + 1)
		*(lsum.vector())+=*(*(ci[i].vector())*zi[i]);
	return 0;
}

int VecSetOnLocalDOFs(std::vector<size_t> DOFsSet, Vec v, double val) {
	std::vector<double> valvec(DOFsSet.size(), val);
	for (size_t i=0; i<DOFsSet.size(); i = i + 1)
		VecSetValueLocal(v, DOFsSet[i], val, INSERT_VALUES);
	VecAssemblyBegin(v);
	VecAssemblyEnd(v);
	VecGhostUpdateBegin(v, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(v, INSERT_VALUES, SCATTER_FORWARD);
	valvec.clear();
	valvec.shrink_to_fit();
	return 0;
}

int VecSetOnDOFs(std::vector<size_t> DOFsSet, Vec v, double val) {
	std::vector<double> valvec(DOFsSet.size(), val);
	for (size_t i=0; i<DOFsSet.size(); i = i + 1)
		VecSetValue(v, DOFsSet[i], val, INSERT_VALUES);
	VecAssemblyBegin(v);
	VecAssemblyEnd(v);
	VecGhostUpdateBegin(v, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(v, INSERT_VALUES, SCATTER_FORWARD);
	valvec.clear();
	valvec.shrink_to_fit();
	return 0;
}

template <class myType>
int SharedTypeVectorDestructor(std::vector<std::shared_ptr<myType>> &vshrtype) {
	for (int i = 0; i<vshrtype.size(); i = i + 1)
		vshrtype[i].reset();
	vshrtype.clear();
	vshrtype.shrink_to_fit();
	return 0;
}

int FunctionFilterAvg(Vec func, dolfin::FunctionSpace Vh, dolfin::Mesh mesh) {
	const PetscScalar* f_i;
	const unsigned int *nb;
	std::vector<std::size_t> NeighbourhoodVertices;
	double avg;
	PetscInt VecLocalsize, ConnectedEdgesize, n;

	VecGetLocalSize(func, &VecLocalsize);
	VecGetArrayRead(func, &f_i);
	std::vector<std::size_t> dof2v = dolfin::dof_to_vertex_map(Vh);
	std::vector<int> v2dof = dolfin::vertex_to_dof_map(Vh);
	mesh.init(0,1);

	//filter out negative values by average of their positive neighbours
	for (std::size_t i=0; i<VecLocalsize; i=i+1) {
		if (f_i[i]<0) {
			avg=0;
			n=0;
			nb = Vertex(mesh, dof2v[i]).entities(1);
			ConnectedEdgesize = Vertex(mesh, dof2v[i]).num_entities(1);
			for (std::size_t j=0; j<ConnectedEdgesize; j=j+1) {
				if ((Edge(mesh, dof2v[j]).entities(0))[0]!=dof2v[i]) {
					NeighbourhoodVertices.push_back(v2dof[(Edge(mesh, dof2v[j]).entities(0))[0]]);
				}
				else {
					NeighbourhoodVertices.push_back(v2dof[(Edge(mesh, dof2v[j]).entities(0))[1]]);
				}
			}
			for (std::size_t k=0; k<NeighbourhoodVertices.size(); k=k+1) {
				if (f_i[NeighbourhoodVertices[k]]>=0) {
					avg = avg + f_i[NeighbourhoodVertices[k]];
					n = n+1;
				}
			}
			if (n>0) {
				VecSetValueLocal(func, i, avg/n, INSERT_VALUES);
			}
		}
	}
	VecAssemblyBegin(func);
	VecAssemblyEnd(func);
	VecGhostUpdateBegin(func, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(func, INSERT_VALUES, SCATTER_FORWARD);

	PetscBarrier(NULL);

	VecRestoreArrayRead(func, &f_i);

	NeighbourhoodVertices.clear();
	NeighbourhoodVertices.shrink_to_fit();
	dof2v.clear();
	dof2v.shrink_to_fit();
	v2dof.clear();
	v2dof.shrink_to_fit();

	return 0;
}

int NodesIndices2LocalDOFs(dolfin::FunctionSpace Vh, dolfin::Mesh mesh, std::vector<std::size_t> &IndexSet, std::vector<std::size_t> &DOFsSet) {
	auto tmpDOFsSet = Vh.dofmap()->entity_dofs(mesh, 0, IndexSet);
	for(std::size_t i=0; i<tmpDOFsSet.size(); i=i+1) {
		DOFsSet.push_back(tmpDOFsSet[i]);
	}
	tmpDOFsSet.clear();
	tmpDOFsSet.shrink_to_fit();
	return 0;
}

int NodesIndices2DOFs(dolfin::FunctionSpace Vh, dolfin::Mesh mesh, std::vector<std::size_t> &IndexSet, std::vector<std::size_t> &DOFsSet) {
	auto tmpDOFsSet = Vh.dofmap()->entity_dofs(mesh, 0, IndexSet);
	std::vector<std::size_t> l2g_dofmap;
	Vh.dofmap()->tabulate_local_to_global_dofs(l2g_dofmap);
	for(std::size_t i=0; i<tmpDOFsSet.size(); i=i+1) {
		DOFsSet.push_back(l2g_dofmap[tmpDOFsSet[i]]);
	}
	tmpDOFsSet.clear();
	tmpDOFsSet.shrink_to_fit();
	l2g_dofmap.clear();
	l2g_dofmap.shrink_to_fit();
	return 0;
}

int NodesIndex_on_Subdomain(std::shared_ptr<dolfin::SubDomain> myDomain, std::shared_ptr<dolfin::Mesh> mesh, std::vector<std::size_t> &IndexSet) {
	dolfin::MeshFunction<std::size_t> SubdomainMesh(mesh, 0);
	SubdomainMesh.set_all(0);
	myDomain->mark(SubdomainMesh, 1);
	dolfin::SubsetIterator itsub(SubdomainMesh, 1);
	for(itsub; !itsub.end(); ++itsub) {
		IndexSet.push_back(itsub->index());
	}
	return 0;
}

int NodesIndex_on_Subdomain(std::vector<std::shared_ptr<dolfin::SubDomain>> myDomains, std::shared_ptr<dolfin::Mesh> mesh, std::vector<std::size_t> &IndexSet) {
	dolfin::MeshFunction<std::size_t> SubdomainMesh(mesh, 0);
	SubdomainMesh.set_all(0);
	for(size_t i=0; i<myDomains.size(); i++) {
		myDomains[i]->mark(SubdomainMesh, 1);
	}
	dolfin::SubsetIterator itsub(SubdomainMesh, 1);
	for(itsub; !itsub.end(); ++itsub) {
		IndexSet.push_back(itsub->index());
	}
	return 0;
}

int Vector_of_NonConstFunctionGenerator(std::shared_ptr<dolfin::FunctionSpace> Vh, std::vector<std::shared_ptr<dolfin::Function>> &myfuncs, std::vector<bool> isconst, std::vector<double> constvalue) {
	std::size_t j = 0;
	for (std::size_t i = 0; i < isconst.size(); i = i + 1) {
		if (isconst[i]) {
			myfuncs.push_back(std::make_shared<dolfin::Function>(Vh));
			myfuncs[i]->interpolate(dolfin::Constant(constvalue[j]));
			j = j + 1;
		}
		else
			myfuncs.push_back(std::make_shared<dolfin::Function>(Vh));
	}
	return 0;
}

int Vector_of_ConstFunctionGenerator(std::shared_ptr<dolfin::FunctionSpace> Vh, std::vector<std::shared_ptr<dolfin::Constant>> &myfuncs, std::vector<double> constvalue) {
	for (std::size_t i = 0; i < constvalue.size(); i = i + 1) {
		myfuncs.push_back(std::make_shared<dolfin::Constant>(constvalue[i]));
	}
	return 0;
}

int Vector_of_ConstFunctionGenerator(std::shared_ptr<dolfin::FunctionSpace> Vh, std::vector<std::shared_ptr<dolfin::GenericFunction>> &myfuncs, std::vector<double> constvalue) {
	for (std::size_t i = 0; i < constvalue.size(); i = i + 1) {
		myfuncs.push_back(std::make_shared<dolfin::Constant>(constvalue[i]));
	}
	return 0;
}

int FixNaNValues(Mat A) {
	PetscInt A_FromRow;
	PetscInt A_ToRow;

	MatGetOwnershipRange(A, &A_FromRow, &A_ToRow);

	const PetscScalar* a_i;
	const PetscInt* colsA;
	PetscInt ncolsA;
	std::vector<PetscScalar> z_i;
	std::vector<PetscInt> z_ind;

	for (PetscInt i = A_FromRow; i < A_ToRow; i = i + 1) {
		MatGetRow(A, i, &ncolsA, &colsA, &a_i);
		for (PetscInt j=0; j < ncolsA; j = j + 1) {
			if (!PetscIsNormalReal(a_i[j])) {
				z_ind.push_back(colsA[j]);
				z_i.push_back(0);
			}
		}
		MatSetValues(A, 1, &i, z_i.size(), z_ind.data(), z_i.data(), INSERT_VALUES);
		MatRestoreRow(A, i, &ncolsA, &colsA, &a_i);
		z_i.clear();
		z_i.shrink_to_fit();
		z_ind.clear();
		z_ind.shrink_to_fit();
	}
	MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

	return 0;
}

int FixNaNValues(Vec b) {
	PetscInt lsize;
	VecGetLocalSize(b, &lsize);

	std::vector<PetscScalar> z_i;
	std::vector<PetscInt> z_ind;
	const PetscScalar* b_i;
	VecGetArrayRead(b, &b_i);

	for (PetscInt j = 0; j < lsize; j = j + 1) {
		if (!PetscIsNormalReal(b_i[j])) {
			z_ind.push_back(j);
			z_i.push_back(0);
		}
	}

	VecSetValuesLocal(b, z_i.size(), z_ind.data(), z_i.data(), INSERT_VALUES);
	VecRestoreArrayRead(b, &b_i);
	VecAssemblyBegin(b);
	VecAssemblyEnd(b);
	VecGhostUpdateBegin(b, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(b, INSERT_VALUES, SCATTER_FORWARD);

	return 0;
}

int Trunc2Precision(dolfin::PETScMatrix &A, PetscScalar trc) {
	trc = std::abs(trc);

	PetscInt A_FromRow;
	PetscInt A_ToRow;

	MatGetOwnershipRange(A.mat(), &A_FromRow, &A_ToRow);

	const PetscScalar* a_i;
	const PetscInt* colsA;
	PetscInt ncolsA;
	std::vector<PetscScalar> z_i;
	std::vector<PetscInt> z_ind;

	for (PetscInt i = A_FromRow; i < A_ToRow; i = i + 1) {
		MatGetRow(A.mat(), i, &ncolsA, &colsA, &a_i);
		for (PetscInt j=0; j < ncolsA; j = j + 1) {
			if ((std::abs(a_i[j])<trc && a_i[j]!=0) || (!PetscIsNormalReal(a_i[j]) && a_i[j]!=0)) {
				z_ind.push_back(colsA[j]);
				z_i.push_back(0);
			}
		}
		MatSetValues(A.mat(), 1, &i, z_i.size(), z_ind.data(), z_i.data(), INSERT_VALUES);
		MatRestoreRow(A.mat(), i, &ncolsA, &colsA, &a_i);
		z_i.clear();
		z_i.shrink_to_fit();
		z_ind.clear();
		z_ind.shrink_to_fit();
	}
	MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

	return 0;
}

int Trunc2Precision(dolfin::PETScVector &b, PetscScalar trc) {
	trc = std::abs(trc);

	PetscInt lsize;
	VecGetLocalSize(b.vec(), &lsize);

	std::vector<PetscScalar> z_i;
	std::vector<PetscInt> z_ind;
	const PetscScalar* b_i;
	VecGetArrayRead(b.vec(), &b_i);

	for (PetscInt j = 0; j < lsize; j = j + 1) {
		if ((std::abs(b_i[j])<trc && b_i[j]!=0) || (!PetscIsNormalReal(b_i[j]) && b_i[j]!=0)) {
			z_ind.push_back(j);
			z_i.push_back(0);
		}
	}

	VecSetValuesLocal(b.vec(), z_i.size(), z_ind.data(), z_i.data(), INSERT_VALUES);
	VecRestoreArrayRead(b.vec(), &b_i);
	VecAssemblyBegin(b.vec());
	VecAssemblyEnd(b.vec());
	VecGhostUpdateBegin(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(b.vec(), INSERT_VALUES, SCATTER_FORWARD);

	z_i.clear();
	z_i.shrink_to_fit();
	z_ind.clear();
	z_ind.shrink_to_fit();

	return 0;
}


int Weak2Matrix(dolfin::Form a, dolfin::Form L, std::vector<dolfin::DirichletBC> DBCs, dolfin::PETScMatrix &A, dolfin::PETScVector &b) {
	if (!(A.empty()))
		A.zero();
	if (!(b.empty()))
		b.zero();
	dolfin::assemble(A, a);
	dolfin::assemble(b, L);
	for (std::size_t i = 0; i < DBCs.size(); i = i + 1) {
		DBCs[i].apply(A, b);
	}
	return 0;
}

int Weak2Matrix(dolfin::Form a, std::vector<dolfin::DirichletBC> DBCs, dolfin::PETScMatrix &A, dolfin::PETScVector &b) {
	if (!(A.empty()))
		A.zero();
	dolfin::assemble(A, a);
	for (std::size_t i = 0; i < DBCs.size(); i = i + 1) {
		DBCs[i].apply(A, b);
	}
	return 0;
}

int Weak2Matrix(dolfin::Form a, std::vector<dolfin::DirichletBC> DBCs, dolfin::PETScMatrix &A) {
	if (!(A.empty()))
		A.zero();
	dolfin::assemble(A, a);
	for (std::size_t i = 0; i < DBCs.size(); i = i + 1) {
		DBCs[i].apply(A);
	}
	return 0;
}

int Weak2Matrix(dolfin::Form L, std::vector<dolfin::DirichletBC> DBCs, dolfin::PETScVector &b) {
	if (!(b.empty()))
		b.zero();
	dolfin::assemble(b, L);
	for (std::size_t i = 0; i < DBCs.size(); i = i + 1) {
		DBCs[i].apply(b);
	}
	return 0;
}

int myDirichletBCGenerator(std::shared_ptr<dolfin::FunctionSpace> Vh, std::vector<std::shared_ptr<dolfin::GenericFunction>> BCfs, std::vector<std::shared_ptr<dolfin::SubDomain>> SDs, std::vector<dolfin::DirichletBC> &DBCs) {
	for (std::size_t i = 0; i < SDs.size(); i = i + 1) {
		DBCs.push_back(dolfin::DirichletBC(Vh, BCfs[i], SDs[i], "geometric"));
	}
	return 0;
}

int WeakAssign(dolfin::Form &a, std::vector<std::string> names, std::vector<std::shared_ptr<dolfin::GenericFunction>> fs) {
	for (std::size_t i = 0; i < names.size(); i = i + 1) {
		a.set_coefficient(names[i], fs[i]);
	}
	return 0;
}

int MatPointwiseMult(Mat alpha, Mat beta, Mat r) {//alpha and r or beta and r can be similar. This useful function was not implemented in Petsc
	//Input: alpha, beta
	//Output: r

	MatZeroEntries(r);

	PetscInt alpha_FromRow;
	PetscInt alpha_ToRow;

	MatGetOwnershipRange(alpha, &alpha_FromRow, &alpha_ToRow);

	const PetscScalar* alpha_i;
	const PetscInt* colsalpha;
	PetscInt ncolsalpha;
	const PetscScalar* beta_i;
	const PetscInt* colsbeta;
	PetscInt ncolsbeta;
	PetscInt j;
	PetscInt k;
	std::vector<PetscScalar> ri;
	std::vector<PetscInt> ri_ind;

	PetscBarrier(NULL);

	for (PetscInt i = alpha_FromRow; i < alpha_ToRow; i = i + 1) {
		MatGetRow(alpha, i, &ncolsalpha, &colsalpha, &alpha_i);
		MatGetRow(beta, i, &ncolsbeta, &colsbeta, &beta_i);

		j = 0;
		k = 0;

		while (j < ncolsalpha && k < ncolsbeta) { 
			if (colsalpha[j] < colsbeta[k]) {
				j = j + 1;
			}
			else if (colsbeta[k] < colsalpha[j]) {
				k = k + 1;
			}
			else {
				ri_ind.push_back(colsalpha[j]);
				ri.push_back(alpha_i[j]*beta_i[k]);
				k = k + 1;
				j = j + 1;
			}
		}
		MatSetValues(r, 1, &i, ri_ind.size(), ri_ind.data(), ri.data(), INSERT_VALUES);
		MatRestoreRow(alpha, i, &ncolsalpha, &colsalpha, &alpha_i);
		MatRestoreRow(beta, i, &ncolsbeta, &colsbeta, &beta_i);

		ri.clear();
		ri.shrink_to_fit();
		ri_ind.clear();
		ri_ind.shrink_to_fit();
	}

	PetscBarrier(NULL);

	MatAssemblyBegin(r, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(r, MAT_FINAL_ASSEMBLY);

	return 0;
}
