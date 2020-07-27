#include <dolfin.h>
#include <fstream>
#include "src/FEMTools.cpp"
#include "src/GeneralTools.cpp"
#include "src/CorrosionTools.cpp"
#include "src/MeshTools.cpp"
#include "src/Poisson.h"
#include "src/MassMatrix.h"
#include "src/StiffnessMatrix.h"

using namespace dolfin;

int main(int argc,char ** args) {

	PetscInitialize(&argc, &args, NULL, NULL);

	PetscMPIInt prcID;
	MPI_Comm_rank(PETSC_COMM_WORLD, &prcID);

	time_t start, end;
	if(prcID==0) {
		time(&start);
	}

	//----The points that is needed in a corrosion rectangle model----
	std::vector<dolfin::Point> ps;

	RectPointsGenerator(0.04, 0.5, ps);//RectPointsGenerator(1, 2, ps);
	ps.push_back(ps[0] + (ps[3]-ps[0])*0.3 + (ps[3]-ps[0])*0.01);
	ps.push_back(ps[0] + (ps[3]-ps[0])*0.3);

	//----Creating the rectangle from points and specifying the boundaries----
	//parameters["ghost_mode"] = "shared_facet";
	auto mesh = std::make_shared<dolfin::Mesh>(PETSC_COMM_WORLD, "Gapmesh.xml");

	//RectMeshGenerator(PETSC_COMM_WORLD, *mesh, ps[0], ps[2], 0.01, "crossed");
	//myMeshRefiner(mesh, std::make_shared<CircularDomain>(ps[5], 0.01));

	std::vector<std::shared_ptr<dolfin::SubDomain>> bcs;
	bcs.push_back(std::make_shared<RectBorderLine>(ps[0], ps[1]));
	bcs.push_back(std::make_shared<RectBorderLine>(ps[1], ps[2]));
	bcs.push_back(std::make_shared<RectBorderLine>(ps[2], ps[3]));
	bcs.push_back(std::make_shared<RectBorderLine>(ps[3], ps[4]));
	bcs.push_back(std::make_shared<RectBorderLine>(ps[5], ps[0]));
	bcs.push_back(std::make_shared<RectBorderLine>(ps[4], ps[5]));//only for Gapmesh

	PetscBarrier(NULL);

	//--Points information are no longer needed--
	ps.clear(); ps.shrink_to_fit();

	//----Creating the variational formulations in the corrosion model----
	auto Vh = std::make_shared<StiffnessMatrix::FunctionSpace>(mesh);

	auto a = std::make_shared<Poisson::BilinearForm>(Vh, Vh);
	auto L = std::make_shared<Poisson::LinearForm>(Vh);

	auto MC = std::make_shared<MassMatrix::BilinearForm>(Vh, Vh);
	auto A = std::make_shared<StiffnessMatrix::BilinearForm>(Vh, Vh);
	auto f = std::make_shared<StiffnessMatrix::LinearForm>(Vh);

	PetscBarrier(NULL);

	//----Creating the functions that we apply to variational formulations in the corrosion model----
	auto zerofunc = std::make_shared<dolfin::Function>(Vh);
	zerofunc->interpolate(dolfin::Constant(0));

	//Electrical field
	std::vector<std::shared_ptr<dolfin::Function>> EFfuncs;
	std::vector<bool> isconst = {0};
	Vector_of_NonConstFunctionGenerator(Vh, EFfuncs, isconst, {});

	//Mg
	std::vector<std::shared_ptr<dolfin::Function>> Mgfuncs;
	isconst.clear(); isconst.shrink_to_fit();
	isconst = {1, 0};
	std::vector<double> constvalue = {0};
	Vector_of_NonConstFunctionGenerator(Vh, Mgfuncs, isconst, constvalue);

	std::vector<std::shared_ptr<dolfin::GenericFunction>> Mgconsts;
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {0.71e-9, 2};
	Vector_of_ConstFunctionGenerator(Vh, Mgconsts, constvalue);

	//OH
	std::vector<std::shared_ptr<dolfin::Function>> OHfuncs;
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {1e-4};
	Vector_of_NonConstFunctionGenerator(Vh, OHfuncs, isconst, constvalue);

	std::vector<std::shared_ptr<dolfin::GenericFunction>> OHconsts;
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {5.27e-9, -1};
	Vector_of_ConstFunctionGenerator(Vh, OHconsts, constvalue);


	//H
	std::vector<std::shared_ptr<dolfin::Function>> Hfuncs;
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {1e-4};
	Vector_of_NonConstFunctionGenerator(Vh, Hfuncs, isconst, constvalue);

	std::vector<std::shared_ptr<dolfin::GenericFunction>> Hconsts;
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {9.31e-9, 1};
	Vector_of_ConstFunctionGenerator(Vh, Hconsts, constvalue);

	//Na
	std::vector<std::shared_ptr<dolfin::Function>> Nafuncs;
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {17};
	Vector_of_NonConstFunctionGenerator(Vh, Nafuncs, isconst, constvalue);

	std::vector<std::shared_ptr<dolfin::GenericFunction>> Naconsts;
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {1.33e-9, 1};
	Vector_of_ConstFunctionGenerator(Vh, Naconsts, constvalue);

	//Cl
	std::vector<std::shared_ptr<dolfin::Function>> Clfuncs;
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {17};
	Vector_of_NonConstFunctionGenerator(Vh, Clfuncs, isconst, constvalue);

	std::vector<std::shared_ptr<dolfin::GenericFunction>> Clconsts;
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {2.03e-9, -1};
	Vector_of_ConstFunctionGenerator(Vh, Clconsts, constvalue);

	//--Function generation information is no longer needed--
	isconst.clear(); isconst.shrink_to_fit();
	constvalue.clear(); constvalue.shrink_to_fit();

	//----Deciesion on type and assignment of the boudaries (Neumann, Dirichlet, ...)----
	std::vector<dolfin::DirichletBC> DBCs;
	std::vector<double> PhiPolData, MgPolData, AlPolData;
	myDirichletBCGenerator(Vh, {std::make_shared<dolfin::Constant>(-1.517), std::make_shared<dolfin::Constant>(-0.595)}, {bcs[3], bcs[4]}, DBCs);
	PolarizationDataAssign(PhiPolData, MgPolData, AlPolData, "PolarizationData/UData.txt", "PolarizationData/MgPolData.txt", "PolarizationData/AlPolData.txt");

	std::vector<std::size_t> NodesOnAlElectrode;
	std::vector<std::size_t> NodesOnMgElectrode;
	std::vector<std::size_t> DOFsSetOnAlElectrode;
	std::vector<std::size_t> DOFsSetOnMgElectrode;
	NodesIndex_on_Subdomain(bcs[4], mesh, NodesOnAlElectrode);
	NodesIndex_on_Subdomain(bcs[3], mesh, NodesOnMgElectrode);
	NodesIndices2LocalDOFs(*Vh, *mesh, NodesOnAlElectrode, DOFsSetOnAlElectrode);
	NodesIndices2LocalDOFs(*Vh, *mesh, NodesOnMgElectrode, DOFsSetOnMgElectrode);

	PetscBarrier(NULL);
//std::vector<std::size_t> l2g_dofmap;Vh->dofmap()->tabulate_local_to_global_dofs(l2g_dofmap);
//auto shdof = Vh->dofmap()->shared_nodes();for (auto& i: shdof) for (std::size_t j;j<(i.second).size();j++) std::cout<<"dof: "<<l2g_dofmap[i.first]<<" with prc: "<<(i.second)[j]<<std::endl;
	//--Nodes indices and domain information is no longer needed--
	NodesOnAlElectrode.clear(); NodesOnAlElectrode.shrink_to_fit();
	NodesOnMgElectrode.clear(); NodesOnMgElectrode.shrink_to_fit();
	SharedTypeVectorDestructor(bcs);

	//----Assembling the final linear systems and solve them and storing the solution----
	//Poisson
	myFormAssigner(*L, {"f", "g"}, {zerofunc, zerofunc});
	auto A_P = std::make_shared<dolfin::PETScMatrix>(PETSC_COMM_WORLD);
	auto b_P = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	myLinearSystemAssembler(*a, *L, DBCs, *A_P, *b_P);
	Vec BoundaryPhi;

	if(prcID==0) {
		list_krylov_solver_methods();
		list_krylov_solver_preconditioners();
		list_linear_solver_methods();
		list_lu_solver_methods();
	}

	std::string PSolverMethod = "mumps";

	PetscBarrier(NULL);

	auto PSolver = std::make_shared<dolfin::PETScLUSolver>(PETSC_COMM_WORLD, PSolverMethod);
	PSolver->set_operator(*A_P);
	PSolver->solve(*EFfuncs[0]->vector(), *b_P);

	PetscScalar minval = as_type<const dolfin::PETScVector>(EFfuncs[0]->vector())->min();
	PetscScalar maxval = as_type<const dolfin::PETScVector>(EFfuncs[0]->vector())->max();

	if(prcID==0) {
		std::cout<<"Solver for poisson is set to: "<<PSolverMethod<<std::endl;
		std::cout<<"Min EF:"<<minval<<std::endl;
		std::cout<<"Max EF:"<<maxval<<std::endl<<std::endl;
	}

	auto ff_P = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/Electric Field.pvd");
	ff_P->operator<<(*EFfuncs[0]);

	//Nernst-Planck
	auto A_MC = std::make_shared<dolfin::PETScMatrix>(PETSC_COMM_WORLD);
	auto A1_NP = std::make_shared<dolfin::PETScMatrix>(PETSC_COMM_WORLD);
	Mat A_ML, rij;
	Vec Ii, change;

	double dt = 1;
	double t = 0;
	double alfa = 1e-7/dt;//change of variable
	double change_norm;
	std::size_t s = 5;
	std::size_t totalsteps = 12*3600*1;

	myLinearSystemAssembler(*MC, {}, *A_MC);

	PetscBarrier(NULL);

	//prellocation
	MatDuplicate(A_MC->mat(), MAT_DO_NOT_COPY_VALUES, &A_ML);
	MatCreateVecs(A_MC->mat(), NULL, &Ii);
	VecDuplicate(Ii, &BoundaryPhi);
	VecDuplicate(Ii, &change);
	VecSet(Ii, 0);
	VecSet(BoundaryPhi, 0);
	VecSet(change, 0);

	FEMFCT_ML_Compute(A_MC->mat(), A_ML);

	//Mg
	Mat L0_Mg, L1_Mg, D0_Mg, D1_Mg;
	Vec fStar;
	auto b0_Mg = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	auto b1_Mg = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);

	iMg(DOFsSetOnMgElectrode, t, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), 0.55, 0.4, 1e-7, Ii, BoundaryPhi, PhiPolData, MgPolData);//A/m^2

//dolfin::File ff_bh(PETSC_COMM_WORLD, "Results/Ii.pvd");
//ff_bh<<(*std::make_shared<dolfin::Function>(Vh, (std::make_shared<dolfin::PETScVector>(Ii))->operator*(1)));

	myFormAssigner(*A, {"Di", "zi"}, Mgconsts);
	myFormAssigner(*A, {"phi", "alpha"}, {EFfuncs[0], std::make_shared<dolfin::Constant>(alfa)});
	myFormAssigner(*f, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, (std::make_shared<dolfin::PETScVector>(Ii))->operator*(-1*5.182045e-6))});//stochimetry coefficient divided by z in equation (3)

	PetscBarrier(NULL);

	myLinearSystemAssembler(*A, {}, *A1_NP);
	myLinearSystemAssembler(*f, {}, *b0_Mg);

	//prellocation (union of nonzero patterns)
	Mat ATr;
	MatTranspose(A1_NP->mat(), MAT_INITIAL_MATRIX, &ATr);
	MatDuplicate(A1_NP->mat(), MAT_COPY_VALUES, &L0_Mg);
	MatAXPY(L0_Mg, 1, ATr, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(L0_Mg, 1, A_MC->mat(), DIFFERENT_NONZERO_PATTERN);
	MatSetOption(L0_Mg, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);

	PetscBarrier(NULL);

	//prellocation
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &rij);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &L1_Mg);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &D0_Mg);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &D1_Mg);
	VecDuplicate(Ii, &fStar);
	MatDestroy(&ATr);

	FEMFCT_Lk_D_Compute(A1_NP->mat(), D0_Mg, L0_Mg);
	MatCopy(D0_Mg, D1_Mg, SAME_NONZERO_PATTERN);//D1_Mg=D0_Mg
	FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_Mg, D0_Mg, L0_Mg, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), b0_Mg->vec(), dt, rij, fStar);

	Mat A_Mg;
	Mat b_Mg;
	Vec b_NP;
	//prellocation
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &A_Mg);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &b_Mg);
	VecDuplicate(Ii, &b_NP);
	MatZeroEntries(A_Mg);
	MatZeroEntries(b_Mg);
	VecSet(b_NP, 0);

	PetscBarrier(NULL);

	MatCopy(A_ML, A_Mg, SAME_NONZERO_PATTERN);
	MatAXPY(A_Mg, 0.5*dt, L0_Mg, SAME_NONZERO_PATTERN);//L0=L1

	MatCopy(A_ML, b_Mg, SAME_NONZERO_PATTERN);
	MatAXPY(b_Mg, -0.5*dt, L0_Mg, SAME_NONZERO_PATTERN);
	MatMult(b_Mg, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), b_NP);
	VecScale(b_NP, std::exp(alfa*dt));
	VecAXPY(b_NP, 0.5*dt, b0_Mg->vec());//b0=0
	VecAXPY(b_NP, 1, fStar);

	PetscBarrier(NULL);

	KSP myNPSolver;
	PC myNPConditioner;
	KSPCreate(PETSC_COMM_WORLD, &myNPSolver);
	KSPSetType(myNPSolver, KSPGMRES);
	KSPSetInitialGuessNonzero(myNPSolver, PETSC_TRUE);
	KSPGetPC(myNPSolver, &myNPConditioner);
	PCSetType(myNPConditioner, PCSOR);
	KSPSetUp(myNPSolver);

	auto NPSolver = std::make_shared<dolfin::PETScKrylovSolver>(myNPSolver);
	NPSolver->set_operator(dolfin::PETScMatrix(A_Mg));
	NPSolver->solve(*Mgfuncs[1]->vector(), dolfin::PETScVector(b_NP));

	std::string NPSolverMethod = "gmres";

	auto ff_Mg = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/Mg Concentration.pvd");
	ff_Mg->operator<<(*Mgfuncs[0]);

	//OH
	Mat L0_OH, L1_OH, D0_OH, D1_OH;
	Vec RHOH;
	auto b0_OH = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	auto b1_OH = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);	

	//prellocation
	VecDuplicate(Ii, &RHOH);

	iOH(DOFsSetOnAlElectrode, t, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), 0.55, 0.4, 1e-7, Ii, BoundaryPhi, PhiPolData, AlPolData);//A/m^2

	myFormAssigner(*A, {"Di", "zi"}, OHconsts);
	myFormAssigner(*f, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, (std::make_shared<dolfin::PETScVector>(Ii))->operator*(-1*1.0364e-5))});

	PetscBarrier(NULL);

	myLinearSystemAssembler(*A, {}, *A1_NP);
	myLinearSystemAssembler(*f, {}, *b0_OH);

	//prellocation (union of nonzero patterns)
	MatTranspose(A1_NP->mat(), MAT_INITIAL_MATRIX, &ATr);
	MatDuplicate(A1_NP->mat(), MAT_COPY_VALUES, &L0_OH);
	MatAXPY(L0_OH, 1, ATr, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(L0_OH, 1, A_ML, DIFFERENT_NONZERO_PATTERN);
	MatSetOption(L0_OH, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);

	PetscBarrier(NULL);

	//prellocation
	MatDuplicate(L0_OH, MAT_DO_NOT_COPY_VALUES, &L1_OH);
	MatDuplicate(L0_OH, MAT_DO_NOT_COPY_VALUES, &D0_OH);
	MatDuplicate(L0_OH, MAT_DO_NOT_COPY_VALUES, &D1_OH);
	MatDestroy(&ATr);

	FEMFCT_Lk_D_Compute(A1_NP->mat(), D0_OH, L0_OH);
	MatCopy(D0_OH, D1_OH, SAME_NONZERO_PATTERN);//D1_OH=D0_OH
	FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_OH, D0_OH, L0_OH, as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), b0_OH->vec(), dt, rij, fStar);

	Mat A_OH;
	Mat b_OH;
	//prellocations
	MatDuplicate(L0_OH, MAT_DO_NOT_COPY_VALUES, &A_OH);
	MatDuplicate(L0_OH, MAT_DO_NOT_COPY_VALUES, &b_OH);
	MatZeroEntries(A_OH);
	MatZeroEntries(b_OH);
	VecSet(b_NP, 0);

	PetscBarrier(NULL);

	MatCopy(A_ML, A_OH, SAME_NONZERO_PATTERN);
	MatAXPY(A_OH, 0.5*dt, L0_OH, SAME_NONZERO_PATTERN);//L0=L1

	MatCopy(A_ML, b_OH, SAME_NONZERO_PATTERN);
	MatAXPY(b_OH, -0.5*dt, L0_OH, SAME_NONZERO_PATTERN);
	MatMult(b_OH, as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), b_NP);
	VecScale(b_NP, std::exp(alfa*dt));
	VecAXPY(b_NP, 0.5*dt, b0_OH->vec());//b0=0
	VecAXPY(b_NP, 1, fStar);

	PetscBarrier(NULL);

	NPSolver->set_operator(dolfin::PETScMatrix(A_OH));
	NPSolver->solve(*OHfuncs[1]->vector(), dolfin::PETScVector(b_NP));

	auto ff_OH = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/OH Concentration.pvd");
	ff_OH->operator<<(*OHfuncs[0]);

	//H
	Mat L0_H, L1_H, D0_H, D1_H;
	auto b0_H = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	auto b1_H = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);

	myFormAssigner(*A, {"Di", "zi"}, Hconsts);

	PetscBarrier(NULL);

	myLinearSystemAssembler(*A, {}, *A1_NP);
	myLinearSystemAssembler(*f, {}, *b0_H);

	//prellocation (union of nonzero patterns)
	MatTranspose(A1_NP->mat(), MAT_INITIAL_MATRIX, &ATr);
	MatDuplicate(A1_NP->mat(), MAT_COPY_VALUES, &L0_H);
	MatAXPY(L0_H, 1, ATr, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(L0_H, 1, A_ML, DIFFERENT_NONZERO_PATTERN);
	MatSetOption(L0_H, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);

	PetscBarrier(NULL);

	//prellocation
	MatDuplicate(L0_H, MAT_DO_NOT_COPY_VALUES, &L1_H);
	MatDuplicate(L0_H, MAT_DO_NOT_COPY_VALUES, &D0_H);
	MatDuplicate(L0_H, MAT_DO_NOT_COPY_VALUES, &D1_H);
	MatDestroy(&ATr);

	FEMFCT_Lk_D_Compute(A1_NP->mat(), D0_H, L0_H);
	MatCopy(D0_H, D1_H, SAME_NONZERO_PATTERN);//D1_H=D0_H
	FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_H, D0_H, L0_H, as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), b0_H->vec(), dt, rij, fStar);

	Mat A_H;
	Mat b_H;
	//prellocation
	MatDuplicate(L0_H, MAT_DO_NOT_COPY_VALUES, &A_H);
	MatDuplicate(L0_H, MAT_DO_NOT_COPY_VALUES, &b_H);
	MatZeroEntries(A_H);
	MatZeroEntries(b_H);
	VecSet(b_NP, 0);

	PetscBarrier(NULL);

	MatCopy(A_ML, A_H, SAME_NONZERO_PATTERN);
	MatAXPY(A_H, 0.5*dt, L0_H, SAME_NONZERO_PATTERN);//L0=L1

	MatCopy(A_ML, b_H, SAME_NONZERO_PATTERN);
	MatAXPY(b_H, -0.5*dt, L0_H, SAME_NONZERO_PATTERN);
	MatMult(b_H, as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), b_NP);
	VecScale(b_NP, std::exp(alfa*dt));
	VecAXPY(b_NP, 0.5*dt, b0_H->vec());//b0=0
	VecAXPY(b_NP, 1, fStar);

	PetscBarrier(NULL);

	NPSolver->set_operator(dolfin::PETScMatrix(A_H));
	NPSolver->solve(*Hfuncs[1]->vector(), dolfin::PETScVector(b_NP));

	auto ff_H = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/H Concentration.pvd");
	ff_H->operator<<(*Hfuncs[0]);

	WaterDissociation(as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), RHOH);

	VecAXPY(as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->vec(), dt, RHOH);
	VecAXPY(as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->vec(), dt, RHOH);

	//Na
	Mat L0_Na, L1_Na, D0_Na, D1_Na;
	auto b0_Na = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	auto b1_Na = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);

	myFormAssigner(*A, {"Di", "zi"}, Naconsts);
	myFormAssigner(*A, {"alpha"}, {zerofunc});
	myFormAssigner(*f, {"Ii"}, {zerofunc});

	PetscBarrier(NULL);

	myLinearSystemAssembler(*A, {}, *A1_NP);
	myLinearSystemAssembler(*f, {}, *b0_Na);

	//prellocation (union of nonzero patterns)
	MatTranspose(A1_NP->mat(), MAT_INITIAL_MATRIX, &ATr);
	MatDuplicate(A1_NP->mat(), MAT_COPY_VALUES, &L0_Na);
	MatAXPY(L0_Na, 1, ATr, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(L0_Na, 1, A_ML, DIFFERENT_NONZERO_PATTERN);
	MatSetOption(L0_Na, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);

	PetscBarrier(NULL);

	//prellocation
	MatDuplicate(L0_Na, MAT_DO_NOT_COPY_VALUES, &L1_Na);
	MatDuplicate(L0_Na, MAT_DO_NOT_COPY_VALUES, &D0_Na);
	MatDuplicate(L0_Na, MAT_DO_NOT_COPY_VALUES, &D1_Na);
	MatDestroy(&ATr);

	FEMFCT_Lk_D_Compute(A1_NP->mat(), D0_Na, L0_Na);
	MatCopy(D0_Na, D1_Na, SAME_NONZERO_PATTERN);//D1_Na=D0_Na
	FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_Na, D0_Na, L0_Na, as_type<const dolfin::PETScVector>(Nafuncs[0]->vector())->vec(), b0_Na->vec(), dt, rij, fStar);

	Mat A_Na;
	Mat b_Na;
	//prellocations
	MatDuplicate(L0_Na, MAT_DO_NOT_COPY_VALUES, &A_Na);
	MatDuplicate(L0_Na, MAT_DO_NOT_COPY_VALUES, &b_Na);
	MatZeroEntries(A_Na);
	MatZeroEntries(b_Na);
	VecSet(b_NP, 0);

	PetscBarrier(NULL);

	MatCopy(A_ML, A_Na, SAME_NONZERO_PATTERN);
	MatAXPY(A_Na, 0.5*dt, L0_Na, SAME_NONZERO_PATTERN);//L0=L1

	MatCopy(A_ML, b_Na, SAME_NONZERO_PATTERN);
	MatAXPY(b_Na, -0.5*dt, L0_Na, SAME_NONZERO_PATTERN);
	MatMult(b_Na, as_type<const dolfin::PETScVector>(Nafuncs[0]->vector())->vec(), b_NP);
	VecAXPY(b_NP, 0.5*dt, b0_Na->vec());//b0=0
	VecAXPY(b_NP, 1, fStar);

	PetscBarrier(NULL);

	NPSolver->set_operator(dolfin::PETScMatrix(A_Na));
	NPSolver->solve(*Nafuncs[1]->vector(), dolfin::PETScVector(b_NP));

	auto ff_Na = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/Na Concentration.pvd");
	ff_Na->operator<<(*Nafuncs[0]);

	//Cl
	Mat L0_Cl, L1_Cl, D0_Cl, D1_Cl;
	auto b0_Cl = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	auto b1_Cl = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);

	myFormAssigner(*A, {"Di", "zi"}, Clconsts);

	PetscBarrier(NULL);

	myLinearSystemAssembler(*A, {}, *A1_NP);
	myLinearSystemAssembler(*f, {}, *b0_Cl);

	//prellocation (union of nonzero patterns)
	MatTranspose(A1_NP->mat(), MAT_INITIAL_MATRIX, &ATr);
	MatDuplicate(A1_NP->mat(), MAT_COPY_VALUES, &L0_Cl);
	MatAXPY(L0_Cl, 1, ATr, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(L0_Cl, 1, A_ML, DIFFERENT_NONZERO_PATTERN);
	MatSetOption(L0_Cl, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);

	PetscBarrier(NULL);

	//prellocation
	MatDuplicate(L0_Cl, MAT_DO_NOT_COPY_VALUES, &L1_Cl);
	MatDuplicate(L0_Cl, MAT_DO_NOT_COPY_VALUES, &D0_Cl);
	MatDuplicate(L0_Cl, MAT_DO_NOT_COPY_VALUES, &D1_Cl);
	MatDestroy(&ATr);

	FEMFCT_Lk_D_Compute(A1_NP->mat(), D0_Cl, L0_Cl);
	MatCopy(D0_Cl, D1_Cl, SAME_NONZERO_PATTERN);//D1_Cl=D0_Cl
	FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_Cl, D0_Cl, L0_Cl, as_type<const dolfin::PETScVector>(Clfuncs[0]->vector())->vec(), b0_Cl->vec(), dt, rij, fStar);

	Mat A_Cl;
	Mat b_Cl;
	//prellocations
	MatDuplicate(L0_Cl, MAT_DO_NOT_COPY_VALUES, &A_Cl);
	MatDuplicate(L0_Cl, MAT_DO_NOT_COPY_VALUES, &b_Cl);
	MatZeroEntries(A_Cl);
	MatZeroEntries(b_Cl);
	VecSet(b_NP, 0);

	PetscBarrier(NULL);

	MatCopy(A_ML, A_Cl, SAME_NONZERO_PATTERN);
	MatAXPY(A_Cl, 0.5*dt, L0_Cl, SAME_NONZERO_PATTERN);//L0=L1

	MatCopy(A_ML, b_Cl, SAME_NONZERO_PATTERN);
	MatAXPY(b_Cl, -0.5*dt, L0_Cl, SAME_NONZERO_PATTERN);
	MatMult(b_Cl, as_type<const dolfin::PETScVector>(Clfuncs[0]->vector())->vec(), b_NP);
	VecAXPY(b_NP, 0.5*dt, b0_Cl->vec());//b0=0
	VecAXPY(b_NP, 1, fStar);

	PetscBarrier(NULL);

	NPSolver->set_operator(dolfin::PETScMatrix(A_Cl));
	NPSolver->solve(*Clfuncs[1]->vector(), dolfin::PETScVector(b_NP));

	auto ff_Cl = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/Cl Concentration.pvd");
	ff_Cl->operator<<(*Clfuncs[0]);

	PetscBarrier(NULL);

	//pH
	auto pH = std::make_shared<dolfin::Function>(Vh);
	pHCompute(*OHfuncs[0], *pH, false);
	auto ff_pH = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/pH.pvd");
	ff_pH->operator<<(*pH);

	//update step

	//Gummel iteration
	VecCopy(as_type<const dolfin::PETScVector>(Mgfuncs[1]->vector())->vec(), change);
	VecAXPY(change, -1, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec());
	VecNorm(change, NORM_INFINITY, &change_norm);

	//poisson
	auto sumfunc = std::make_shared<dolfin::Function>(Vh);
	sumfunc->interpolate(dolfin::Constant(0));
	funcsLinSum({2, 1, -1, 1, -1}, {*Mgfuncs[1], *Nafuncs[1], *Clfuncs[1], *Hfuncs[1], *OHfuncs[1]}, *sumfunc);
	DBCs[0].set_value(std::make_shared<dolfin::Function>(Vh, std::make_shared<dolfin::PETScVector>(BoundaryPhi)));
	DBCs[1].set_value(std::make_shared<dolfin::Function>(Vh, std::make_shared<dolfin::PETScVector>(BoundaryPhi)));

	//Nernst-Planck
	*(Mgfuncs[0]->vector()) = *(Mgfuncs[1]->vector());
	*(Hfuncs[0]->vector()) = *(Hfuncs[1]->vector());
	*(OHfuncs[0]->vector()) = *(OHfuncs[1]->vector());
	*(Nafuncs[0]->vector()) = *(Nafuncs[1]->vector());
	*(Clfuncs[0]->vector()) = *(Clfuncs[1]->vector());
std::cout<<"norm MgVec: "<<change_norm<<std::endl;
	while (change_norm>1e+4) {
		//Poisson
		myFormAssigner(*L, {"f"}, {sumfunc});
		myLinearSystemAssembler(*L, DBCs, *b_P);

		PetscBarrier(NULL);

		PSolver->solve(*EFfuncs[0]->vector(), *b_P);

		//Mg
		myFormAssigner(*A, {"Di", "zi"}, Mgconsts);
		myFormAssigner(*A, {"phi", "alpha"}, {EFfuncs[0], std::make_shared<dolfin::Constant>(alfa)});
		myFormAssigner(*f, {"Ii"}, {zerofunc});

		PetscBarrier(NULL);

		myLinearSystemAssembler(*A, {}, *A1_NP);
		myLinearSystemAssembler(*f, {}, *b1_Mg);

		FEMFCT_Lk_D_Compute(A1_NP->mat(), D1_Mg, L1_Mg);
		FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_Mg, D0_Mg, L0_Mg, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), b0_Mg->vec(), dt, rij, fStar);

		MatZeroEntries(A_Mg);
		MatZeroEntries(b_Mg);
		VecSet(b_NP, 0);

		PetscBarrier(NULL);

		MatCopy(A_ML, A_Mg, SAME_NONZERO_PATTERN);
		MatAXPY(A_Mg, 0.5*dt, L1_Mg, SAME_NONZERO_PATTERN);

		MatCopy(A_ML, b_Mg, SAME_NONZERO_PATTERN);
		MatAXPY(b_Mg, -0.5*dt, L0_Mg, SAME_NONZERO_PATTERN);
		MatMult(b_Mg, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), b_NP);
		VecScale(b_NP, std::exp(alfa*dt));
		VecAXPY(b_NP, std::exp(alfa*dt)*0.5*dt, b0_Mg->vec());
		VecAXPY(b_NP, 0.5*dt, b1_Mg->vec());
		VecAXPY(b_NP, 1, fStar);

		PetscBarrier(NULL);

		NPSolver->set_operator(dolfin::PETScMatrix(A_Mg));
		NPSolver->solve(*Mgfuncs[1]->vector(), dolfin::PETScVector(b_NP));

		//OH
		myFormAssigner(*A, {"Di", "zi"}, OHconsts);

		PetscBarrier(NULL);

		myLinearSystemAssembler(*A, {}, *A1_NP);
		myLinearSystemAssembler(*f, {}, *b1_OH);

		FEMFCT_Lk_D_Compute(A1_NP->mat(), D1_OH, L1_OH);
		FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_OH, D0_OH, L0_OH, as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), b0_OH->vec(), dt, rij, fStar);

		MatZeroEntries(A_OH);
		MatZeroEntries(b_OH);
		VecSet(b_NP, 0);

		PetscBarrier(NULL);

		MatCopy(A_ML, A_OH, SAME_NONZERO_PATTERN);
		MatAXPY(A_OH, 0.5*dt, L1_OH, SAME_NONZERO_PATTERN);

		MatCopy(A_ML, b_OH, SAME_NONZERO_PATTERN);
		MatAXPY(b_OH, -0.5*dt, L0_OH, SAME_NONZERO_PATTERN);
		MatMult(b_OH, as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), b_NP);
		VecScale(b_NP, std::exp(alfa*dt));
		VecAXPY(b_NP, std::exp(alfa*dt)*0.5*dt, b0_OH->vec());
		VecAXPY(b_NP, 0.5*dt, b1_OH->vec());
		VecAXPY(b_NP, 1, fStar);

		PetscBarrier(NULL);

		NPSolver->set_operator(dolfin::PETScMatrix(A_OH));
		NPSolver->solve(*OHfuncs[1]->vector(), dolfin::PETScVector(b_NP));

		//H
		myFormAssigner(*A, {"Di", "zi"}, Hconsts);

		PetscBarrier(NULL);

		myLinearSystemAssembler(*A, {}, *A1_NP);
		myLinearSystemAssembler(*f, {}, *b1_H);

		FEMFCT_Lk_D_Compute(A1_NP->mat(), D1_H, L1_H);
		FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_H, D0_H, L0_H, as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), b0_H->vec(), dt, rij, fStar);

		MatZeroEntries(A_H);
		MatZeroEntries(b_H);
		VecSet(b_NP, 0);

		PetscBarrier(NULL);

		MatCopy(A_ML, A_H, SAME_NONZERO_PATTERN);
		MatAXPY(A_H, 0.5*dt, L1_H, SAME_NONZERO_PATTERN);

		MatCopy(A_ML, b_H, SAME_NONZERO_PATTERN);
		MatAXPY(b_H, -0.5*dt, L0_H, SAME_NONZERO_PATTERN);
		MatMult(b_H, as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), b_NP);
		VecScale(b_NP, std::exp(alfa*dt));
		VecAXPY(b_NP, std::exp(alfa*dt)*0.5*dt, b0_H->vec());
		VecAXPY(b_NP, 0.5*dt, b1_H->vec());
		VecAXPY(b_NP, 1, fStar);

		PetscBarrier(NULL);

		NPSolver->set_operator(dolfin::PETScMatrix(A_H));
		NPSolver->solve(*Hfuncs[1]->vector(), dolfin::PETScVector(b_NP));

		//Na
		myFormAssigner(*A, {"Di", "zi"}, Naconsts);
		myFormAssigner(*A, {"alpha"}, {zerofunc});

		PetscBarrier(NULL);

		myLinearSystemAssembler(*A, {}, *A1_NP);
		myLinearSystemAssembler(*f, {}, *b1_Na);

		FEMFCT_Lk_D_Compute(A1_NP->mat(), D1_Na, L1_Na);
		FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_Na, D0_Na, L0_Na, as_type<const dolfin::PETScVector>(Nafuncs[0]->vector())->vec(), b0_Na->vec(), dt, rij, fStar);

		MatZeroEntries(A_Na);
		MatZeroEntries(b_Na);
		VecSet(b_NP, 0);

		PetscBarrier(NULL);

		MatCopy(A_ML, A_Na, SAME_NONZERO_PATTERN);
		MatAXPY(A_Na, 0.5*5*dt, L1_Na, SAME_NONZERO_PATTERN);

		MatCopy(A_ML, b_Na, SAME_NONZERO_PATTERN);
		MatAXPY(b_Na, -0.5*5*dt, L0_Na, SAME_NONZERO_PATTERN);
		MatMult(b_Na, as_type<const dolfin::PETScVector>(Nafuncs[0]->vector())->vec(), b_NP);
		VecAXPY(b_NP, 0.5*5*dt, b0_Na->vec());
		VecAXPY(b_NP, 0.5*5*dt, b1_Na->vec());
		VecAXPY(b_NP, 1, fStar);

		PetscBarrier(NULL);

		NPSolver->set_operator(dolfin::PETScMatrix(A_Na));
		NPSolver->solve(*Nafuncs[1]->vector(), dolfin::PETScVector(b_NP));

		//Cl
		myFormAssigner(*A, {"Di", "zi"}, Clconsts);

		PetscBarrier(NULL);

		myLinearSystemAssembler(*A, {}, *A1_NP);
		myLinearSystemAssembler(*f, {}, *b1_Cl);

		FEMFCT_Lk_D_Compute(A1_NP->mat(), D1_Cl, L1_Cl);
		FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_Cl, D0_Cl, L0_Cl, as_type<const dolfin::PETScVector>(Clfuncs[0]->vector())->vec(), b0_Cl->vec(), dt, rij, fStar);

		MatZeroEntries(A_Cl);
		MatZeroEntries(b_Cl);
		VecSet(b_NP, 0);

		PetscBarrier(NULL);

		MatCopy(A_ML, A_Cl, SAME_NONZERO_PATTERN);
		MatAXPY(A_Cl, 0.5*5*dt, L1_Cl, SAME_NONZERO_PATTERN);

		MatCopy(A_ML, b_Cl, SAME_NONZERO_PATTERN);
		MatAXPY(b_Cl, -0.5*5*dt, L0_Cl, SAME_NONZERO_PATTERN);
		MatMult(b_Cl, as_type<const dolfin::PETScVector>(Clfuncs[0]->vector())->vec(), b_NP);
		VecAXPY(b_NP, 0.5*5*dt, b0_Cl->vec());
		VecAXPY(b_NP, 0.5*5*dt, b1_Cl->vec());
		VecAXPY(b_NP, 1, fStar);

		PetscBarrier(NULL);

		NPSolver->set_operator(dolfin::PETScMatrix(A_Cl));
		NPSolver->solve(*Clfuncs[1]->vector(), dolfin::PETScVector(b_NP));

		PetscBarrier(NULL);

		//update step

		//Gummel iteration
		VecCopy(as_type<const dolfin::PETScVector>(Mgfuncs[1]->vector())->vec(), change);
		VecAXPY(change, -1, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec());
		VecNorm(change, NORM_INFINITY, &change_norm);
std::cout<<"breakpoint "<<change_norm<<std::endl;
		//poisson
		funcsLinSum({2, 1, -1, 1, -1}, {*Mgfuncs[1], *Nafuncs[1], *Clfuncs[1], *Hfuncs[1], *OHfuncs[1]}, *sumfunc);
		DBCs[0].set_value(std::make_shared<dolfin::Function>(Vh, std::make_shared<dolfin::PETScVector>(BoundaryPhi)));
		DBCs[1].set_value(std::make_shared<dolfin::Function>(Vh, std::make_shared<dolfin::PETScVector>(BoundaryPhi)));

		//Nernst-Planck
		*(Mgfuncs[0]->vector()) = *(Mgfuncs[1]->vector());
		*(Hfuncs[0]->vector()) = *(Hfuncs[1]->vector());
		*(OHfuncs[0]->vector()) = *(OHfuncs[1]->vector());
		*(Nafuncs[0]->vector()) = *(Nafuncs[1]->vector());
		*(Clfuncs[0]->vector()) = *(Clfuncs[1]->vector());

		VecCopy(b1_Mg->vec(), b0_Mg->vec());
		VecCopy(b1_H->vec(), b0_H->vec());
		VecCopy(b1_OH->vec(), b0_OH->vec());
		VecCopy(b1_Na->vec(), b0_Na->vec());
		VecCopy(b1_Cl->vec(), b0_Cl->vec());

		MatCopy(L1_Mg, L0_Mg, SAME_NONZERO_PATTERN);
		MatCopy(L1_H, L0_H, SAME_NONZERO_PATTERN);
		MatCopy(L1_OH, L0_OH, SAME_NONZERO_PATTERN);
		MatCopy(L1_Na, L0_Na, SAME_NONZERO_PATTERN);
		MatCopy(L1_Cl, L0_Cl, SAME_NONZERO_PATTERN);

		MatCopy(D1_Mg, D0_Mg, SAME_NONZERO_PATTERN);
		MatCopy(D1_H, D0_H, SAME_NONZERO_PATTERN);
		MatCopy(D1_OH, D0_OH, SAME_NONZERO_PATTERN);
		MatCopy(D1_Na, D0_Na, SAME_NONZERO_PATTERN);
		MatCopy(D1_Cl, D0_Cl, SAME_NONZERO_PATTERN);

		PetscBarrier(NULL);
	}

	//Gummel results
	//Poisson
	minval = as_type<const dolfin::PETScVector>(EFfuncs[0]->vector())->min();
	maxval = as_type<const dolfin::PETScVector>(EFfuncs[0]->vector())->max();

	if(prcID==0) {
		std::cout<<"Solver for poisson is set to: "<<PSolverMethod<<std::endl;
		std::cout<<"Min EF:"<<minval<<std::endl;
		std::cout<<"Max EF:"<<maxval<<std::endl<<std::endl;
	}

	ff_P->operator<<(*EFfuncs[0]);

	//Mg
	minval = as_type<const dolfin::PETScVector>(Mgfuncs[1]->vector())->min();
	maxval = as_type<const dolfin::PETScVector>(Mgfuncs[1]->vector())->max();

	if(prcID==0) {
		std::cout<<"Solver for Nernst-Planck (Mg) is set to: "<<NPSolverMethod<<std::endl;
		std::cout<<"Min Mg:"<<minval<<std::endl;
		std::cout<<"Max Mg:"<<maxval<<std::endl<<std::endl;
	}

	ff_Mg->operator<<(*Mgfuncs[1]);

	//OH
	minval = as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->min();
	maxval = as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->max();

	if(prcID==0) {
		std::cout<<"Solver for Nernst-Planck (OH) is set to: "<<NPSolverMethod<<std::endl;
		std::cout<<"Min OH:"<<minval<<std::endl;
		std::cout<<"Max OH:"<<maxval<<std::endl<<std::endl;
	}

	ff_OH->operator<<(*OHfuncs[1]);

	//H
	minval = as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->min();
	maxval = as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->max();

	if(prcID==0) {
		std::cout<<"Solver for Nernst-Planck (H) is set to: "<<NPSolverMethod<<std::endl;
		std::cout<<"Min H:"<<minval<<std::endl;
		std::cout<<"Max H:"<<maxval<<std::endl<<std::endl;
	}

	ff_H->operator<<(*Hfuncs[1]);

	//Na
	minval = as_type<const dolfin::PETScVector>(Nafuncs[1]->vector())->min();
	maxval = as_type<const dolfin::PETScVector>(Nafuncs[1]->vector())->max();

	if(prcID==0) {
		std::cout<<"Solver for Nernst-Planck (Na) is set to: "<<NPSolverMethod<<std::endl;
		std::cout<<"Min Na:"<<minval<<std::endl;
		std::cout<<"Max Na:"<<maxval<<std::endl<<std::endl;
	}

	ff_Na->operator<<(*Nafuncs[1]);

	//Cl
	minval = as_type<const dolfin::PETScVector>(Clfuncs[1]->vector())->min();
	maxval = as_type<const dolfin::PETScVector>(Clfuncs[1]->vector())->max();

	if(prcID==0) {
		std::cout<<"Solver for Nernst-Planck (Cl) is set to: "<<NPSolverMethod<<std::endl;
		std::cout<<"Min Cl:"<<minval<<std::endl;
		std::cout<<"Max Cl:"<<maxval<<std::endl<<std::endl;
	}

	ff_Cl->operator<<(*Clfuncs[1]);

	//pH
	pHCompute(*OHfuncs[1], *pH, false);
	ff_pH->operator<<(*pH);

	//time
	t = dt + t;
std::cin>>change_norm;
	//24 hour simulation
	for (std::size_t i=1; i<=totalsteps; i = i + 1) {//totalsteps

		//Poisson
		myFormAssigner(*L, {"f"}, {sumfunc});
		myLinearSystemAssembler(*L, DBCs, *b_P);

		PetscBarrier(NULL);

		PSolver->solve(*EFfuncs[0]->vector(), *b_P);

		//Mg
		iMg(DOFsSetOnMgElectrode, t, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), 0.55, 0.4, 1e-7, Ii, BoundaryPhi, PhiPolData, MgPolData);//A/m^2

		myFormAssigner(*A, {"Di", "zi"}, Mgconsts);
		myFormAssigner(*A, {"phi", "alpha"}, {EFfuncs[0], std::make_shared<dolfin::Constant>(alfa)});
		myFormAssigner(*f, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, (std::make_shared<dolfin::PETScVector>(Ii))->operator*(-1*5.182045e-6))});

		PetscBarrier(NULL);

		myLinearSystemAssembler(*A, {}, *A1_NP);
		myLinearSystemAssembler(*f, {}, *b1_Mg);

		FEMFCT_Lk_D_Compute(A1_NP->mat(), D1_Mg, L1_Mg);
		FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_Mg, D0_Mg, L0_Mg, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), b0_Mg->vec(), dt, rij, fStar);

		MatZeroEntries(A_Mg);
		MatZeroEntries(b_Mg);
		VecSet(b_NP, 0);

		PetscBarrier(NULL);

		MatCopy(A_ML, A_Mg, SAME_NONZERO_PATTERN);
		MatAXPY(A_Mg, 0.5*dt, L1_Mg, SAME_NONZERO_PATTERN);

		MatCopy(A_ML, b_Mg, SAME_NONZERO_PATTERN);
		MatAXPY(b_Mg, -0.5*dt, L0_Mg, SAME_NONZERO_PATTERN);
		MatMult(b_Mg, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), b_NP);
		VecScale(b_NP, std::exp(alfa*dt));
		VecAXPY(b_NP, std::exp(alfa*dt)*0.5*dt, b0_Mg->vec());
		VecAXPY(b_NP, 0.5*dt, b1_Mg->vec());
		VecAXPY(b_NP, 1, fStar);

		PetscBarrier(NULL);

		NPSolver->set_operator(dolfin::PETScMatrix(A_Mg));
		NPSolver->solve(*Mgfuncs[1]->vector(), dolfin::PETScVector(b_NP));

		//OH
		iOH(DOFsSetOnAlElectrode, t, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), 0.55, 0.5, 1e-7, Ii, BoundaryPhi, PhiPolData, AlPolData);//A/m^2

		myFormAssigner(*A, {"Di", "zi"}, OHconsts);
		myFormAssigner(*f, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, (std::make_shared<dolfin::PETScVector>(Ii))->operator*(-1*1.0364e-5))});

		PetscBarrier(NULL);

		myLinearSystemAssembler(*A, {}, *A1_NP);
		myLinearSystemAssembler(*f, {}, *b1_OH);

		FEMFCT_Lk_D_Compute(A1_NP->mat(), D1_OH, L1_OH);
		FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_OH, D0_OH, L0_OH, as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), b0_OH->vec(), dt, rij, fStar);

		MatZeroEntries(A_OH);
		MatZeroEntries(b_OH);
		VecSet(b_NP, 0);

		PetscBarrier(NULL);

		MatCopy(A_ML, A_OH, SAME_NONZERO_PATTERN);
		MatAXPY(A_OH, 0.5*dt, L1_OH, SAME_NONZERO_PATTERN);

		MatCopy(A_ML, b_OH, SAME_NONZERO_PATTERN);
		MatAXPY(b_OH, -0.5*dt, L0_OH, SAME_NONZERO_PATTERN);
		MatMult(b_OH, as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), b_NP);
		VecScale(b_NP, std::exp(alfa*dt));
		VecAXPY(b_NP, std::exp(alfa*dt)*0.5*dt, b0_OH->vec());
		VecAXPY(b_NP, 0.5*dt, b1_OH->vec());
		VecAXPY(b_NP, 1, fStar);

		PetscBarrier(NULL);

		NPSolver->set_operator(dolfin::PETScMatrix(A_OH));
		NPSolver->solve(*OHfuncs[1]->vector(), dolfin::PETScVector(b_NP));

		//H
		myFormAssigner(*A, {"Di", "zi"}, Hconsts);

		PetscBarrier(NULL);

		myLinearSystemAssembler(*A, {}, *A1_NP);
		myLinearSystemAssembler(*f, {}, *b1_H);

		FEMFCT_Lk_D_Compute(A1_NP->mat(), D1_H, L1_H);
		FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_H, D0_H, L0_H, as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), b0_H->vec(), dt, rij, fStar);

		MatZeroEntries(A_H);
		MatZeroEntries(b_H);
		VecSet(b_NP, 0);

		PetscBarrier(NULL);

		MatCopy(A_ML, A_H, SAME_NONZERO_PATTERN);
		MatAXPY(A_H, 0.5*dt, L1_H, SAME_NONZERO_PATTERN);

		MatCopy(A_ML, b_H, SAME_NONZERO_PATTERN);
		MatAXPY(b_H, -0.5*dt, L0_H, SAME_NONZERO_PATTERN);
		MatMult(b_H, as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), b_NP);
		VecScale(b_NP, std::exp(alfa*dt));
		VecAXPY(b_NP, std::exp(alfa*dt)*0.5*dt, b0_H->vec());
		VecAXPY(b_NP, 0.5*dt, b1_H->vec());
		VecAXPY(b_NP, 1, fStar);

		PetscBarrier(NULL);

		NPSolver->set_operator(dolfin::PETScMatrix(A_H));
		NPSolver->solve(*Hfuncs[1]->vector(), dolfin::PETScVector(b_NP));

		WaterDissociation(as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), RHOH);

		VecAXPY(as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->vec(), dt, RHOH);
		VecAXPY(as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->vec(), dt, RHOH);

		if (i%5 == 0) {
			//Na
			myFormAssigner(*A, {"Di", "zi"}, Naconsts);
			myFormAssigner(*A, {"alpha"}, {zerofunc});
			myFormAssigner(*f, {"Ii"}, {zerofunc});

			PetscBarrier(NULL);

			myLinearSystemAssembler(*A, {}, *A1_NP);
			myLinearSystemAssembler(*f, {}, *b1_Na);

			FEMFCT_Lk_D_Compute(A1_NP->mat(), D1_Na, L1_Na);
			FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_Na, D0_Na, L0_Na, as_type<const dolfin::PETScVector>(Nafuncs[0]->vector())->vec(), b0_Na->vec(), dt, rij, fStar);

			MatZeroEntries(A_Na);
			MatZeroEntries(b_Na);
			VecSet(b_NP, 0);

			PetscBarrier(NULL);

			MatCopy(A_ML, A_Na, SAME_NONZERO_PATTERN);
			MatAXPY(A_Na, 0.5*5*dt, L1_Na, SAME_NONZERO_PATTERN);

			MatCopy(A_ML, b_Na, SAME_NONZERO_PATTERN);
			MatAXPY(b_Na, -0.5*5*dt, L0_Na, SAME_NONZERO_PATTERN);
			MatMult(b_Na, as_type<const dolfin::PETScVector>(Nafuncs[0]->vector())->vec(), b_NP);
			VecAXPY(b_NP, 0.5*5*dt, b0_Na->vec());
			VecAXPY(b_NP, 0.5*5*dt, b1_Na->vec());
			VecAXPY(b_NP, 1, fStar);

			PetscBarrier(NULL);

			NPSolver->set_operator(dolfin::PETScMatrix(A_Na));
			NPSolver->solve(*Nafuncs[1]->vector(), dolfin::PETScVector(b_NP));

			//Cl
			myFormAssigner(*A, {"Di", "zi"}, Clconsts);

			PetscBarrier(NULL);

			myLinearSystemAssembler(*A, {}, *A1_NP);
			myLinearSystemAssembler(*f, {}, *b1_Cl);

			FEMFCT_Lk_D_Compute(A1_NP->mat(), D1_Cl, L1_Cl);
			FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_Cl, D0_Cl, L0_Cl, as_type<const dolfin::PETScVector>(Clfuncs[0]->vector())->vec(), b0_Cl->vec(), dt, rij, fStar);

			MatZeroEntries(A_Cl);
			MatZeroEntries(b_Cl);
			VecSet(b_NP, 0);

			PetscBarrier(NULL);

			MatCopy(A_ML, A_Cl, SAME_NONZERO_PATTERN);
			MatAXPY(A_Cl, 0.5*5*dt, L1_Cl, SAME_NONZERO_PATTERN);

			MatCopy(A_ML, b_Cl, SAME_NONZERO_PATTERN);
			MatAXPY(b_Cl, -0.5*5*dt, L0_Cl, SAME_NONZERO_PATTERN);
			MatMult(b_Cl, as_type<const dolfin::PETScVector>(Clfuncs[0]->vector())->vec(), b_NP);
			VecAXPY(b_NP, 0.5*5*dt, b0_Cl->vec());
			VecAXPY(b_NP, 0.5*5*dt, b1_Cl->vec());
			VecAXPY(b_NP, 1, fStar);

			PetscBarrier(NULL);

			NPSolver->set_operator(dolfin::PETScMatrix(A_Cl));
			NPSolver->solve(*Clfuncs[1]->vector(), dolfin::PETScVector(b_NP));
		}

		PetscBarrier(NULL);

		//update step

		//Gummel iteration
		VecCopy(as_type<const dolfin::PETScVector>(Mgfuncs[1]->vector())->vec(), change);
		VecAXPY(change, -1, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec());
		VecNorm(change, NORM_INFINITY, &change_norm);

		//poisson
		funcsLinSum({2, 1, -1, 1, -1}, {*Mgfuncs[1], *Nafuncs[1], *Clfuncs[1], *Hfuncs[1], *OHfuncs[1]}, *sumfunc);
		DBCs[0].set_value(std::make_shared<dolfin::Function>(Vh, std::make_shared<dolfin::PETScVector>(BoundaryPhi)));
		DBCs[1].set_value(std::make_shared<dolfin::Function>(Vh, std::make_shared<dolfin::PETScVector>(BoundaryPhi)));

		//Nernst-Planck
		*(Mgfuncs[0]->vector()) = *(Mgfuncs[1]->vector());
		*(Hfuncs[0]->vector()) = *(Hfuncs[1]->vector());
		*(OHfuncs[0]->vector()) = *(OHfuncs[1]->vector());
		*(Nafuncs[0]->vector()) = *(Nafuncs[1]->vector());
		*(Clfuncs[0]->vector()) = *(Clfuncs[1]->vector());

		VecCopy(b1_Mg->vec(), b0_Mg->vec());
		VecCopy(b1_H->vec(), b0_H->vec());
		VecCopy(b1_OH->vec(), b0_OH->vec());
		VecCopy(b1_Na->vec(), b0_Na->vec());
		VecCopy(b1_Cl->vec(), b0_Cl->vec());

		MatCopy(L1_Mg, L0_Mg, SAME_NONZERO_PATTERN);
		MatCopy(L1_H, L0_H, SAME_NONZERO_PATTERN);
		MatCopy(L1_OH, L0_OH, SAME_NONZERO_PATTERN);
		MatCopy(L1_Na, L0_Na, SAME_NONZERO_PATTERN);
		MatCopy(L1_Cl, L0_Cl, SAME_NONZERO_PATTERN);

		MatCopy(D1_Mg, D0_Mg, SAME_NONZERO_PATTERN);
		MatCopy(D1_H, D0_H, SAME_NONZERO_PATTERN);
		MatCopy(D1_OH, D0_OH, SAME_NONZERO_PATTERN);
		MatCopy(D1_Na, D0_Na, SAME_NONZERO_PATTERN);
		MatCopy(D1_Cl, D0_Cl, SAME_NONZERO_PATTERN);

		while (change_norm>1e-6) {
			//Poisson
			myFormAssigner(*L, {"f"}, {sumfunc});
			myLinearSystemAssembler(*L, DBCs, *b_P);

			PetscBarrier(NULL);

			PSolver->solve(*EFfuncs[0]->vector(), *b_P);

			//Mg
			myFormAssigner(*A, {"Di", "zi"}, Mgconsts);
			myFormAssigner(*A, {"phi", "alpha"}, {EFfuncs[0], std::make_shared<dolfin::Constant>(alfa)});
			myFormAssigner(*f, {"Ii"}, {zerofunc});

			PetscBarrier(NULL);

			myLinearSystemAssembler(*A, {}, *A1_NP);
			myLinearSystemAssembler(*f, {}, *b1_Mg);

			FEMFCT_Lk_D_Compute(A1_NP->mat(), D1_Mg, L1_Mg);
			FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_Mg, D0_Mg, L0_Mg, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), b0_Mg->vec(), dt, rij, fStar);

			MatZeroEntries(A_Mg);
			MatZeroEntries(b_Mg);
			VecSet(b_NP, 0);

			PetscBarrier(NULL);

			MatCopy(A_ML, A_Mg, SAME_NONZERO_PATTERN);
			MatAXPY(A_Mg, 0.5*dt, L1_Mg, SAME_NONZERO_PATTERN);

			MatCopy(A_ML, b_Mg, SAME_NONZERO_PATTERN);
			MatAXPY(b_Mg, -0.5*dt, L0_Mg, SAME_NONZERO_PATTERN);
			MatMult(b_Mg, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), b_NP);
			VecScale(b_NP, std::exp(alfa*dt));
			VecAXPY(b_NP, std::exp(alfa*dt)*0.5*dt, b0_Mg->vec());
			VecAXPY(b_NP, 0.5*dt, b1_Mg->vec());
			VecAXPY(b_NP, 1, fStar);

			PetscBarrier(NULL);

			KSPSetOperators(myNPSolver, A_Mg, A_Mg);
			KSPSolve(myNPSolver, b_NP, as_type<const dolfin::PETScVector>(Mgfuncs[1]->vector())->vec());

			//OH
			myFormAssigner(*A, {"Di", "zi"}, OHconsts);

			PetscBarrier(NULL);

			myLinearSystemAssembler(*A, {}, *A1_NP);
			myLinearSystemAssembler(*f, {}, *b1_OH);

			FEMFCT_Lk_D_Compute(A1_NP->mat(), D1_OH, L1_OH);
			FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_OH, D0_OH, L0_OH, as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), b0_OH->vec(), dt, rij, fStar);

			MatZeroEntries(A_OH);
			MatZeroEntries(b_OH);
			VecSet(b_NP, 0);

			PetscBarrier(NULL);

			MatCopy(A_ML, A_OH, SAME_NONZERO_PATTERN);
			MatAXPY(A_OH, 0.5*dt, L1_OH, SAME_NONZERO_PATTERN);

			MatCopy(A_ML, b_OH, SAME_NONZERO_PATTERN);
			MatAXPY(b_OH, -0.5*dt, L0_OH, SAME_NONZERO_PATTERN);
			MatMult(b_OH, as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), b_NP);
			VecScale(b_NP, std::exp(alfa*dt));
			VecAXPY(b_NP, std::exp(alfa*dt)*0.5*dt, b0_OH->vec());
			VecAXPY(b_NP, 0.5*dt, b1_OH->vec());
			VecAXPY(b_NP, 1, fStar);

			PetscBarrier(NULL);

			KSPSetOperators(myNPSolver, A_OH, A_OH);
			KSPSolve(myNPSolver, b_NP, as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->vec());

			//H
			myFormAssigner(*A, {"Di", "zi"}, Hconsts);

			PetscBarrier(NULL);

			myLinearSystemAssembler(*A, {}, *A1_NP);
			myLinearSystemAssembler(*f, {}, *b1_H);

			FEMFCT_Lk_D_Compute(A1_NP->mat(), D1_H, L1_H);
			FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_H, D0_H, L0_H, as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), b0_H->vec(), dt, rij, fStar);

			MatZeroEntries(A_H);
			MatZeroEntries(b_H);
			VecSet(b_NP, 0);

			PetscBarrier(NULL);

			MatCopy(A_ML, A_H, SAME_NONZERO_PATTERN);
			MatAXPY(A_H, 0.5*dt, L1_H, SAME_NONZERO_PATTERN);

			MatCopy(A_ML, b_H, SAME_NONZERO_PATTERN);
			MatAXPY(b_H, -0.5*dt, L0_H, SAME_NONZERO_PATTERN);
			MatMult(b_H, as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), b_NP);
			VecScale(b_NP, std::exp(alfa*dt));
			VecAXPY(b_NP, std::exp(alfa*dt)*0.5*dt, b0_H->vec());
			VecAXPY(b_NP, 0.5*dt, b1_H->vec());
			VecAXPY(b_NP, 1, fStar);

			PetscBarrier(NULL);

			KSPSetOperators(myNPSolver, A_H, A_H);
			KSPSolve(myNPSolver, b_NP, as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->vec());

			if (i%5 == 0) {
				//Na
				myFormAssigner(*A, {"Di", "zi"}, Naconsts);
				myFormAssigner(*A, {"alpha"}, {zerofunc});
				myFormAssigner(*f, {"Ii"}, {zerofunc});

				PetscBarrier(NULL);

				myLinearSystemAssembler(*A, {}, *A1_NP);
				myLinearSystemAssembler(*f, {}, *b1_Na);

				FEMFCT_Lk_D_Compute(A1_NP->mat(), D1_Na, L1_Na);
				FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_Na, D0_Na, L0_Na, as_type<const dolfin::PETScVector>(Nafuncs[0]->vector())->vec(), b0_Na->vec(), dt, rij, fStar);

				MatZeroEntries(A_Na);
				MatZeroEntries(b_Na);
				VecSet(b_NP, 0);

				PetscBarrier(NULL);
	
				MatCopy(A_ML, A_Na, SAME_NONZERO_PATTERN);
				MatAXPY(A_Na, 0.5*5*dt, L1_Na, SAME_NONZERO_PATTERN);

				MatCopy(A_ML, b_Na, SAME_NONZERO_PATTERN);
				MatAXPY(b_Na, -0.5*5*dt, L0_Na, SAME_NONZERO_PATTERN);
				MatMult(b_Na, as_type<const dolfin::PETScVector>(Nafuncs[0]->vector())->vec(), b_NP);
				VecAXPY(b_NP, 0.5*5*dt, b0_Na->vec());
				VecAXPY(b_NP, 0.5*5*dt, b1_Na->vec());
				VecAXPY(b_NP, 1, fStar);

				PetscBarrier(NULL);

				KSPSetOperators(myNPSolver, A_Na, A_Na);
				KSPSolve(myNPSolver, b_NP, as_type<const dolfin::PETScVector>(Nafuncs[1]->vector())->vec());

				//Cl
				myFormAssigner(*A, {"Di", "zi"}, Clconsts);

				PetscBarrier(NULL);

				myLinearSystemAssembler(*A, {}, *A1_NP);
				myLinearSystemAssembler(*f, {}, *b1_Cl);

				FEMFCT_Lk_D_Compute(A1_NP->mat(), D1_Cl, L1_Cl);
				FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D1_Cl, D0_Cl, L0_Cl, as_type<const dolfin::PETScVector>(Clfuncs[0]->vector())->vec(), b0_Cl->vec(), dt, rij, fStar);

				MatZeroEntries(A_Cl);
				MatZeroEntries(b_Cl);
				VecSet(b_NP, 0);

				PetscBarrier(NULL);

				MatCopy(A_ML, A_Cl, SAME_NONZERO_PATTERN);
				MatAXPY(A_Cl, 0.5*5*dt, L1_Cl, SAME_NONZERO_PATTERN);

				MatCopy(A_ML, b_Cl, SAME_NONZERO_PATTERN);
				MatAXPY(b_Cl, -0.5*5*dt, L0_Cl, SAME_NONZERO_PATTERN);
				MatMult(b_Cl, as_type<const dolfin::PETScVector>(Clfuncs[0]->vector())->vec(), b_NP);
				VecAXPY(b_NP, 0.5*5*dt, b0_Cl->vec());
				VecAXPY(b_NP, 0.5*5*dt, b1_Cl->vec());
				VecAXPY(b_NP, 1, fStar);

				PetscBarrier(NULL);

				KSPSetOperators(myNPSolver, A_Cl, A_Cl);
				KSPSolve(myNPSolver, b_NP, as_type<const dolfin::PETScVector>(Clfuncs[1]->vector())->vec());
			}

			PetscBarrier(NULL);

			//update step

			//Gummel iteration
			VecCopy(as_type<const dolfin::PETScVector>(Mgfuncs[1]->vector())->vec(), change);
			VecAXPY(change, -1, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec());
			VecNorm(change, NORM_INFINITY, &change_norm);

			//poisson
			funcsLinSum({2, 1, -1, 1, -1}, {*Mgfuncs[1], *Nafuncs[1], *Clfuncs[1], *Hfuncs[1], *OHfuncs[1]}, *sumfunc);
			DBCs[0].set_value(std::make_shared<dolfin::Function>(Vh, std::make_shared<dolfin::PETScVector>(BoundaryPhi)));
			DBCs[1].set_value(std::make_shared<dolfin::Function>(Vh, std::make_shared<dolfin::PETScVector>(BoundaryPhi)));

			//Nernst-Planck
			*(Mgfuncs[0]->vector()) = *(Mgfuncs[1]->vector());
			*(Hfuncs[0]->vector()) = *(Hfuncs[1]->vector());
			*(OHfuncs[0]->vector()) = *(OHfuncs[1]->vector());
			*(Nafuncs[0]->vector()) = *(Nafuncs[1]->vector());
			*(Clfuncs[0]->vector()) = *(Clfuncs[1]->vector());

			VecCopy(b1_Mg->vec(), b0_Mg->vec());
			VecCopy(b1_H->vec(), b0_H->vec());
			VecCopy(b1_OH->vec(), b0_OH->vec());
			VecCopy(b1_Na->vec(), b0_Na->vec());
			VecCopy(b1_Cl->vec(), b0_Cl->vec());

			MatCopy(L1_Mg, L0_Mg, SAME_NONZERO_PATTERN);
			MatCopy(L1_H, L0_H, SAME_NONZERO_PATTERN);
			MatCopy(L1_OH, L0_OH, SAME_NONZERO_PATTERN);
			MatCopy(L1_Na, L0_Na, SAME_NONZERO_PATTERN);
			MatCopy(L1_Cl, L0_Cl, SAME_NONZERO_PATTERN);

			MatCopy(D1_Mg, D0_Mg, SAME_NONZERO_PATTERN);
			MatCopy(D1_H, D0_H, SAME_NONZERO_PATTERN);
			MatCopy(D1_OH, D0_OH, SAME_NONZERO_PATTERN);
			MatCopy(D1_Na, D0_Na, SAME_NONZERO_PATTERN);
			MatCopy(D1_Cl, D0_Cl, SAME_NONZERO_PATTERN);

		}

		//Gummel results
		//Poisson
		minval = as_type<const dolfin::PETScVector>(EFfuncs[0]->vector())->min();
		maxval = as_type<const dolfin::PETScVector>(EFfuncs[0]->vector())->max();

		if(prcID==0) {
			std::cout<<"Solver for poisson is set to: "<<PSolverMethod<<std::endl;
			std::cout<<"Min EF:"<<minval<<std::endl;
			std::cout<<"Max EF:"<<maxval<<std::endl<<std::endl;
		}

		if (i%s == 0)
			ff_P->operator<<(*EFfuncs[0]);

		//Mg
		minval = as_type<const dolfin::PETScVector>(Mgfuncs[1]->vector())->min();
		maxval = as_type<const dolfin::PETScVector>(Mgfuncs[1]->vector())->max();

		if(prcID==0) {
			std::cout<<"Solver for Nernst-Planck (Mg) is set to: "<<NPSolverMethod<<std::endl;
			std::cout<<"Min Mg:"<<minval<<std::endl;
			std::cout<<"Max Mg:"<<maxval<<std::endl<<std::endl;
		}

		if (i%s == 0)
			ff_Mg->operator<<(*Mgfuncs[1]);

		//OH
		minval = as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->min();
		maxval = as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->max();

		if(prcID==0) {
			std::cout<<"Solver for Nernst-Planck (OH) is set to: "<<NPSolverMethod<<std::endl;
			std::cout<<"Min OH:"<<minval<<std::endl;
			std::cout<<"Max OH:"<<maxval<<std::endl<<std::endl;
		}

		if (i%s == 0) {
			ff_OH->operator<<(*OHfuncs[1]);
		}


		//H
		minval = as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->min();
		maxval = as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->max();

		if(prcID==0) {
			std::cout<<"Solver for Nernst-Planck (H) is set to: "<<NPSolverMethod<<std::endl;
			std::cout<<"Min H:"<<minval<<std::endl;
			std::cout<<"Max H:"<<maxval<<std::endl<<std::endl;
		}

		if (i%s == 0)
			ff_H->operator<<(*Hfuncs[1]);

		if (i%s == 0) {
			//Na
			minval = as_type<const dolfin::PETScVector>(Nafuncs[1]->vector())->min();
			maxval = as_type<const dolfin::PETScVector>(Nafuncs[1]->vector())->max();

			if(prcID==0) {
				std::cout<<"Solver for Nernst-Planck (Na) is set to: "<<NPSolverMethod<<std::endl;
				std::cout<<"Min Na:"<<minval<<std::endl;
				std::cout<<"Max Na:"<<maxval<<std::endl<<std::endl;
			}

			ff_Na->operator<<(*Nafuncs[1]);
			//Cl
			minval = as_type<const dolfin::PETScVector>(Clfuncs[1]->vector())->min();
			maxval = as_type<const dolfin::PETScVector>(Clfuncs[1]->vector())->max();

			if(prcID==0) {
				std::cout<<"Solver for Nernst-Planck (Cl) is set to: "<<NPSolverMethod<<std::endl;
				std::cout<<"Min Cl:"<<minval<<std::endl;
				std::cout<<"Max Cl:"<<maxval<<std::endl<<std::endl;
			}

			ff_Cl->operator<<(*Clfuncs[1]);
		}

		if (i%s == 0) {
			//pH
			pHCompute(*OHfuncs[1], *pH, false);
			ff_pH->operator<<(*pH);
		}

		t = dt + t;

		//storage step
		if (i == (1*3600*1)) {
			s = 20;
		}
		if (i == (3*3600*1)) {
			s = 100;
		}
		if (i == (10*3600*1)) {
			s = 300;
		}

		PetscBarrier(NULL);
	}

	if(prcID==0) {
		time(&end);
		std::cout<<"total exc time: "<<double(end-start)<<std::endl;
	}

	PetscBarrier(NULL);

	//--Electric field information is no longer needed--
	DBCs.clear(); DBCs.shrink_to_fit();
	A_P.reset();
	b_P.reset();
	PSolver.reset();

	//clean the memory of vectors
	DOFsSetOnAlElectrode.clear();
	DOFsSetOnAlElectrode.shrink_to_fit();
	DOFsSetOnMgElectrode.clear();
	DOFsSetOnMgElectrode.shrink_to_fit();
	PhiPolData.clear();
	PhiPolData.shrink_to_fit();
	MgPolData.clear();
	MgPolData.shrink_to_fit();
	AlPolData.clear();
	AlPolData.shrink_to_fit();

	//clean the shared pointers
	ff_P.reset();
	A_MC.reset();
	A1_NP.reset();
	NPSolver.reset();
	b0_Mg.reset();
	b1_Mg.reset();
	b0_H.reset();
	b1_H.reset();
	b0_OH.reset();
	b1_OH.reset();
	b0_Na.reset();
	b1_Na.reset();
	b0_Cl.reset();
	b1_Cl.reset();
	ff_Mg.reset();
	ff_H.reset();
	ff_OH.reset();
	ff_Na.reset();
	ff_Cl.reset();
	ff_pH.reset();

	SharedTypeVectorDestructor(EFfuncs);
	SharedTypeVectorDestructor(Mgfuncs);
	SharedTypeVectorDestructor(Mgconsts);
	SharedTypeVectorDestructor(OHfuncs);
	SharedTypeVectorDestructor(OHconsts);
	SharedTypeVectorDestructor(Hfuncs);
	SharedTypeVectorDestructor(Hconsts);
	SharedTypeVectorDestructor(Nafuncs);
	SharedTypeVectorDestructor(Naconsts);
	SharedTypeVectorDestructor(Clfuncs);
	SharedTypeVectorDestructor(Clconsts);

	a.reset();
	L.reset();
	MC.reset();
	A.reset();
	f.reset();
	zerofunc.reset();
	pH.reset();
	sumfunc.reset();
	Vh.reset();
	mesh.reset();

	//clean the memory of Petsc objects
	KSPDestroy(&myNPSolver);

	MatDestroy(&rij);
	VecDestroy(&BoundaryPhi);
	VecDestroy(&Ii);
	VecDestroy(&change);
	VecDestroy(&fStar);
	VecDestroy(&b_NP);
	VecDestroy(&RHOH);

	MatDestroy(&A_ML);
	MatDestroy(&L0_Mg);
	MatDestroy(&L1_Mg);
	MatDestroy(&D0_Mg);
	MatDestroy(&D1_Mg);
	MatDestroy(&A_Mg);
	MatDestroy(&b_Mg);

	MatDestroy(&L0_H);
	MatDestroy(&L1_H);
	MatDestroy(&D0_H);
	MatDestroy(&D1_H);
	MatDestroy(&A_H);
	MatDestroy(&b_H);

	MatDestroy(&L0_OH);
	MatDestroy(&L1_OH);
	MatDestroy(&D0_OH);
	MatDestroy(&D1_OH);
	MatDestroy(&A_OH);
	MatDestroy(&b_OH);

	MatDestroy(&L0_Na);
	MatDestroy(&L1_Na);
	MatDestroy(&D0_Na);
	MatDestroy(&D1_Na);
	MatDestroy(&A_Na);
	MatDestroy(&b_Na);

	MatDestroy(&L0_Cl);
	MatDestroy(&L1_Cl);
	MatDestroy(&D0_Cl);
	MatDestroy(&D1_Cl);
	MatDestroy(&A_Cl);
	MatDestroy(&b_Cl);

	PetscFinalize();

	return 0;
}
