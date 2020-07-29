#include <dolfin.h>
#include <fstream>
#include "src/FEMTools.cpp"
#include "src/CorrosionTools.cpp"
#include "src/MeshTools.cpp"

using namespace dolfin;

int main(int argc,char ** args) {

	PetscInitialize(&argc, &args, NULL, NULL);

	PetscMPIInt prcID;
	MPI_Comm_rank(PETSC_COMM_WORLD, &prcID);

	Mat A, L, D, MC, ML;
	PetscScalar v=1;
	int j=1;
	MatCreateSeqAIJ(PETSC_COMM_WORLD,4,4,4,0,&A);
	for (int i = 0; i<4; i++) {
		MatSetValues(A,1,&i,1,&i,&v,INSERT_VALUES);
		MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);
	}
	MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

	//prellocation (union of nonzero patterns)
	Mat ATr;
	MatTranspose(A, MAT_INITIAL_MATRIX, &ATr);
	MatDuplicate(A, MAT_COPY_VALUES, &L);
	MatAXPY(L, 1, ATr, DIFFERENT_NONZERO_PATTERN);
	MatSetOption(L, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);

	PetscBarrier(NULL);

	//prellocation
	MatDuplicate(L, MAT_DO_NOT_COPY_VALUES, &D);

	FEMFCT_Lk_D_Compute(A, D, L);

	//MatView(A, PETSC_VIEWER_STDOUT_SELF);
	//MatView(D, PETSC_VIEWER_STDOUT_SELF);
	//MatView(L, PETSC_VIEWER_STDOUT_SELF);

	//FEMFCT_ML_Compute(A, D);

	MatView(D, PETSC_VIEWER_STDOUT_SELF);

	if(prcID==0) {
	std::vector<double> Phi,iMg,iAl;
	PolarizationDataAssign(Phi, iMg, iAl, "PolarizationData/UData.txt", "PolarizationData/MgPolData.txt", "PolarizationData/AlPolData.txt");
	std::cout<<Phi[0]<<", "<<iMg[150]<<", "<<iAl[iAl.size()-1]<<std::endl;

	std::cout<<iMg[iAl.size()-1]-0.1<<", est:"<<Current2ElectricField(Phi, iMg, iMg[iAl.size()-1]-0.1)<<", exact: "<<Phi[iAl.size()-1]<<std::endl;
	std::cout<<iAl[100]<<", est:"<<Current2ElectricField(Phi, iAl, iAl[100])<<", exact: "<<Phi[100]<<std::endl;
	std::cout<<iAl[100]+0.01<<", est:"<<Current2ElectricField(Phi, iAl, iAl[100]+0.01)<<", exact: "<<Phi[100]<<std::endl;
	}

	//Mesh_XDMF2XML("Gapmesh.xdmf", "Gapmesh.xml");

	PetscFinalize();
//std::vector<std::size_t> l2g_dofmap;Vh->dofmap()->tabulate_local_to_global_dofs(l2g_dofmap);
//auto shdof = Vh->dofmap()->shared_nodes();for (auto& i: shdof) for (std::size_t j;j<(i.second).size();j++) std::cout<<"dof: "<<l2g_dofmap[i.first]<<" with prc: "<<(i.second)[j]<<std::endl;

//dolfin::File ff_bh(PETSC_COMM_WORLD, "Results/Ii.pvd");
//ff_bh<<(*std::make_shared<dolfin::Function>(Vh, (std::make_shared<dolfin::PETScVector>(Ii))->operator*(1)));
	return 0;
}
