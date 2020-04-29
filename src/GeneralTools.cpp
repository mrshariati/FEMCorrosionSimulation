#include <dolfin.h>
#include <boost/mpi.hpp>

using namespace dolfin;

int funcRemoveNeg(dolfin::Function &func) {
	Vec vtmp;
	VecDuplicate(as_type<const dolfin::PETScVector>(func.vector())->vec(), &vtmp);
	VecSet(vtmp, 0);
	VecPointwiseMax(as_type<const dolfin::PETScVector>(func.vector())->vec(), as_type<const dolfin::PETScVector>(func.vector())->vec(), vtmp);// negative concentrations set to zero
	VecDestroy(&vtmp);
	return 0;
}

int funcsLinSum(std::vector<int> zi, std::vector<dolfin::Function> ci, dolfin::Function &lsum) {
	*(lsum.vector())= 0;
	for (size_t i=0; i<ci.size(); i = i + 1)
		*(lsum.vector())+=*(*(ci[i].vector())*zi[i]);
	return 0;
}

int VecSetOnDOFs(std::vector<size_t> DOFsSet, Vec &v, double val) {
	std::vector<double> valvec(DOFsSet.size(), val);
	for (size_t i=0; i<DOFsSet.size(); i = i + 1)
		VecSetValue(v, DOFsSet[i], val, INSERT_VALUES);
	VecAssemblyBegin(v);
	VecAssemblyEnd(v);
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

int FixNaNValues(Mat &A) {
	Mat D;
	MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &D);

	PetscInt r_startA;
	PetscInt r_endA;

	MatGetOwnershipRange(A, &r_startA, &r_endA);

	for (PetscInt i = r_startA; i < r_endA; i = i + 1) {

		const PetscScalar* a_i;
		const PetscInt* colsA;
		PetscInt ncolsA;

		std::vector<PetscScalar> z_i;
		std::vector<PetscInt> z_ind;

		MatGetRow(A, i, &ncolsA, &colsA, &a_i);

		for (PetscInt j=0; j < ncolsA; j = j + 1) {
			if (PetscIsNanReal(a_i[j])) {
				z_ind.push_back(colsA[j]);
				z_i.push_back(0);
			}
		}
		MatSetValues(D, 1, &i, z_i.size(), z_ind.data(), z_i.data(), INSERT_VALUES);
		MatRestoreRow(A, i, &ncolsA, &colsA, &a_i);
	}
	MatAssemblyBegin(D,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(D,MAT_FINAL_ASSEMBLY);

	MatCopy(D, A, SAME_NONZERO_PATTERN);

	MatDestroy(&D);

	return 0;
}

int FixNaNValues(Vec &b) {
	const PetscScalar* b_i;
	VecGetArrayRead(b, &b_i);

	PetscInt lsize;

	VecGetLocalSize(b, &lsize);

	std::vector<PetscScalar> z_i;
	std::vector<PetscInt> z_ind;
	for (PetscInt i = 0; i < lsize; i = i + 1) {
		if (PetscIsNanReal(b_i[i])) {
			z_ind.push_back(i);
			z_i.push_back(0);
		}
	}

	VecSetValues(b, z_i.size(), z_ind.data(), z_i.data(), INSERT_VALUES);

	VecAssemblyBegin(b);
	VecAssemblyEnd(b);

	VecRestoreArrayRead(b, &b_i);

	return 0;
}

int myLinearSystemAssembler(dolfin::Form a, dolfin::Form L, std::vector<dolfin::DirichletBC> DBCs, dolfin::PETScMatrix &A, dolfin::PETScVector &b) {
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

int myLinearSystemAssembler(dolfin::Form a, std::vector<dolfin::DirichletBC> DBCs, dolfin::PETScMatrix &A) {
	if (!(A.empty()))
		A.zero();
	dolfin::assemble(A, a);
	for (std::size_t i = 0; i < DBCs.size(); i = i + 1) {
		DBCs[i].apply(A);
	}
	return 0;
}

int myLinearSystemAssembler(dolfin::Form L, std::vector<dolfin::DirichletBC> DBCs, dolfin::PETScVector &b) {
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

int myFormAssigner(dolfin::Form &a, std::vector<std::string> names, std::vector<std::shared_ptr<dolfin::GenericFunction>> fs) {
	for (std::size_t i = 0; i < names.size(); i = i + 1) {
		a.set_coefficient(names[i], fs[i]);
	}
	return 0;
}