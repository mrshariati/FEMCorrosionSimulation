#include <dolfin.h>
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

	RectPointsGenerator(0.04, 2, ps);//RectPointsGenerator(1, 2, ps);
	ps.push_back((ps[0]+ps[3])/2 + dolfin::Point(DOLFIN_EPS, 0));
	ps.push_back((ps[0]+ps[3])/2);

	//----Creating the rectangle from points and specifying the boundaries----
	parameters["ghost_mode"] = "shared_vetex";
	auto mesh = std::make_shared<dolfin::Mesh>(PETSC_COMM_WORLD);
	RectMeshGenerator(PETSC_COMM_WORLD, *mesh, ps[0], ps[2], 0.05, "crossed");

	std::vector<std::shared_ptr<dolfin::SubDomain>> bcs;
	bcs.push_back(std::make_shared<RectBorderLine>(ps[0], ps[1]));
	bcs.push_back(std::make_shared<RectBorderLine>(ps[1], ps[2]));
	bcs.push_back(std::make_shared<RectBorderLine>(ps[2], ps[3]));
	bcs.push_back(std::make_shared<RectBorderLine>(ps[3], ps[4]));
	bcs.push_back(std::make_shared<RectBorderLine>(ps[5], ps[0]));

	//Mesh refinement (does not support parallelism)
	/*myMeshRefiner(mesh, std::make_shared<CircularDomain>(ps[0], 0.5));
	myMeshRefiner(mesh, std::make_shared<CircularDomain>(ps[3], 0.5));
	myMeshRefiner(mesh, std::make_shared<CircularDomain>(ps[5], 0.5));
	myMeshRefiner(mesh, std::make_shared<CircularDomain>(ps[5], 0.4));
	myMeshRefiner(mesh, std::make_shared<CircularDomain>(ps[0], 0.2));
	myMeshRefiner(mesh, std::make_shared<CircularDomain>(ps[3], 0.2));
	myMeshRefiner(PETSC_COMM_WORLD, mesh, std::make_shared<CircularDomain>(ps[5], 0.04));
	myMeshRefiner(PETSC_COMM_WORLD, mesh, std::make_shared<TwoPointsSpace2D>(ps[0], ps[3]+Point(0, 0.04)));
	myMeshRefiner(PETSC_COMM_WORLD, mesh, std::make_shared<TwoPointsSpace2D>(ps[0], ps[3]+Point(0, 0.03)));
	myMeshRefiner(PETSC_COMM_WORLD, mesh, std::make_shared<TwoPointsSpace2D>(ps[0], ps[3]+Point(0, 0.02)));
	myMeshRefiner(PETSC_COMM_WORLD, mesh, std::make_shared<TwoPointsSpace2D>(ps[0], ps[3]+Point(0, 0.01)));*/

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

	//H
	std::vector<std::shared_ptr<dolfin::Function>> Hfuncs;
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {1e-4};
	Vector_of_NonConstFunctionGenerator(Vh, Hfuncs, isconst, constvalue);

	std::vector<std::shared_ptr<dolfin::GenericFunction>> Hconsts;
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {9.31e-9, 1};
	Vector_of_ConstFunctionGenerator(Vh, Hconsts, constvalue);

	//OH
	std::vector<std::shared_ptr<dolfin::Function>> OHfuncs;
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {1e-4};
	Vector_of_NonConstFunctionGenerator(Vh, OHfuncs, isconst, constvalue);

	std::vector<std::shared_ptr<dolfin::GenericFunction>> OHconsts;
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {5.27e-9, -1};
	Vector_of_ConstFunctionGenerator(Vh, OHconsts, constvalue);

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

	//O2
	std::vector<std::shared_ptr<dolfin::Function>> O2funcs;
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {0.233};
	Vector_of_NonConstFunctionGenerator(Vh, O2funcs, isconst, constvalue);

	std::vector<std::shared_ptr<dolfin::GenericFunction>> O2consts;
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {2.4e-9, 0};
	Vector_of_ConstFunctionGenerator(Vh, O2consts, constvalue);

	//--Function generation information is no longer needed--
	isconst.clear(); isconst.shrink_to_fit();
	constvalue.clear(); constvalue.shrink_to_fit();

	//----Deciesion on type and assignment of the boudaries (Neumann, Dirichlet, ...)----
	std::vector<dolfin::DirichletBC> DBCs;
	myDirichletBCGenerator(Vh, {std::make_shared<dolfin::Constant>(-1.517), std::make_shared<dolfin::Constant>(-0.595)}, {bcs[3], bcs[4]}, DBCs);

	std::vector<std::size_t> NodesOnAlElectrode;
	std::vector<std::size_t> NodesOnMgElectrode;
	std::vector<std::size_t> NodesOnBoundary;//for RHOH
	std::vector<std::size_t> DOFsSetOnAlElectrode;
	std::vector<std::size_t> DOFsSetOnMgElectrode;
	std::vector<std::size_t> DOFsSetOnBoundary;//for RHOH
	NodesIndex_on_Subdomain(bcs[4], mesh, NodesOnAlElectrode);
	NodesIndex_on_Subdomain(bcs[3], mesh, NodesOnMgElectrode);
	NodesIndex_on_Subdomain({bcs[0], bcs[1], bcs[2], bcs[3], bcs[4]}, mesh, NodesOnBoundary);
	NodesIndices2DOFs(*Vh, *mesh, NodesOnAlElectrode, DOFsSetOnAlElectrode);
	NodesIndices2DOFs(*Vh, *mesh, NodesOnMgElectrode, DOFsSetOnMgElectrode);
	NodesIndices2DOFs(*Vh, *mesh, NodesOnBoundary, DOFsSetOnBoundary);

	PetscBarrier(NULL);

	//--Nodes indices and domain information is no longer needed--
	NodesOnAlElectrode.clear(); NodesOnAlElectrode.shrink_to_fit();
	NodesOnMgElectrode.clear(); NodesOnMgElectrode.shrink_to_fit();
	NodesOnBoundary.clear(); NodesOnBoundary.shrink_to_fit();
	SharedTypeVectorDestructor(bcs);

	//----Assembling the final linear systems and solve them and storing the solution----
	//Poisson
	myFormAssigner(*L, {"f"}, {zerofunc});
	auto A_P = std::make_shared<dolfin::PETScMatrix>(PETSC_COMM_WORLD);
	auto b_P = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	myLinearSystemAssembler(*a, *L, DBCs, *A_P, *b_P);

	if(prcID==0) {
		list_krylov_solver_methods();
		list_krylov_solver_preconditioners();
		list_linear_solver_methods();
		list_lu_solver_methods();
	}

	std::string PSolverMethod = "mumps";
	std::string NPSolverMethod = "minres";

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

	PetscBarrier(NULL);

	//Nernst-Planck
	Vec Ii;
	double t = 1e-2;
	double dt = 1e-2;
	std::size_t s = 100;
	std::size_t totalsteps = 10*60*100 + 50*60*10;// + 30*60*1; //10 minutes dt=1e-2 + 50 minutes dt=1e-1

	auto cMg = std::make_shared<dolfin::PETScVector>(as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec());
	auto cH = std::make_shared<dolfin::PETScVector>(as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec());
	auto cOH = std::make_shared<dolfin::PETScVector>(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec());

	auto A_MC = std::make_shared<dolfin::PETScMatrix>(PETSC_COMM_WORLD);
	Mat A_ML;
	myLinearSystemAssembler(*MC, {}, *A_MC);

	MatDuplicate(A_MC->mat(), MAT_DO_NOT_COPY_VALUES, &A_ML);

	FEMFCT_ML_Compute(A_MC->mat(), A_ML);

	auto A_FCT_FEM = std::make_shared<dolfin::PETScMatrix>(PETSC_COMM_WORLD);
	Mat rij;

	PetscBarrier(NULL);

	//Mg
	MatCreateVecs(A_MC->mat(), NULL, &Ii);
	BoundaryCurrent(DOFsSetOnMgElectrode, t, cMg->vec(), cOH->vec(), cH->vec(), 0.55, 0.1, 0.01, Ii);//A/m^2
	VecScale(Ii, 0.02);//0.0004 We have an active length of 20mm

	myFormAssigner(*A, {"Di", "zi"}, Mgconsts);
	myFormAssigner(*A, {"phi"}, {EFfuncs[0]});
	myFormAssigner(*f, {"Ii", "Ri"}, {std::make_shared<dolfin::Function>(Vh, (std::make_shared<dolfin::PETScVector>(Ii))->operator*(-0.5)), zerofunc});

	Mat L0_Mg, L1_Mg;
	Mat D_Mg;
	Vec fStar;
	auto b0_Mg = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	auto b1_Mg = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	//prellocations
	VecDuplicate(Ii, &fStar);

	myLinearSystemAssembler(*A, {}, *A_FCT_FEM);
	myLinearSystemAssembler(*f, {}, *b0_Mg);

	//prellocations
	Mat ATr;
	MatCreateTranspose(A_FCT_FEM->mat(), &ATr);
	MatDuplicate(A_FCT_FEM->mat(), MAT_DO_NOT_COPY_VALUES, &rij);
	MatZeroEntries(rij);
	MatSetOption(rij, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE);
	MatAXPY(rij, 1, ATr, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(rij, 1, A_MC->mat(), DIFFERENT_NONZERO_PATTERN);

	PetscBarrier(NULL);

	MatZeroEntries(rij);
	MatSetOption(rij, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE);
	MatDuplicate(rij, MAT_DO_NOT_COPY_VALUES, &L0_Mg);
	MatDuplicate(rij, MAT_DO_NOT_COPY_VALUES, &L1_Mg);
	MatDuplicate(rij, MAT_DO_NOT_COPY_VALUES, &D_Mg);
	MatDestroy(&ATr);

	FEMFCT_Lk_D_Compute(A_FCT_FEM->mat(), D_Mg, L0_Mg);
	FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D_Mg, L0_Mg, cMg->vec(), b0_Mg->vec(), dt, rij, fStar);

	Mat A_Mg;
	Mat b_Mg;
	Vec b_NP;
	//prellocations
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &A_Mg);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &b_Mg);
	VecDuplicate(Ii, &b_NP);
	VecSet(b_NP, 0);

	MatCopy(A_ML, A_Mg, SAME_NONZERO_PATTERN);
	MatAXPY(A_Mg, 0.5*dt, L0_Mg, SAME_NONZERO_PATTERN);//L0=L1

	MatCopy(A_ML, b_Mg, SAME_NONZERO_PATTERN);
	MatAXPY(b_Mg, -0.5*dt, L0_Mg, SAME_NONZERO_PATTERN);
	MatMult(b_Mg, cMg->vec(), b_NP);
	VecAXPY(b_NP, 0.5*dt, b0_Mg->vec());//b0=0
	VecAXPY(b_NP, 1, fStar);

	PetscBarrier(NULL);

	auto A_NPSolver = std::make_shared<dolfin::PETScMatrix>(A_Mg);
	auto b_NPSolver = std::make_shared<dolfin::PETScVector>(b_NP);

	auto ff_Mg = std::make_shared<dolfin::File>("Results/Mg Concentration.pvd");
	ff_Mg->operator<<(*Mgfuncs[0]);

	auto NPSolver = std::make_shared<dolfin::PETScKrylovSolver>(PETSC_COMM_WORLD, NPSolverMethod, "petsc_amg");
	NPSolver->set_operator(A_NPSolver);
	NPSolver->solve(*Mgfuncs[1]->vector(), *b_NPSolver);

	minval = as_type<const dolfin::PETScVector>(Mgfuncs[1]->vector())->min();
	maxval = as_type<const dolfin::PETScVector>(Mgfuncs[1]->vector())->max();

	if(prcID==0) {
		std::cout<<"Solver for Nernst-Planck (Mg) is set to: "<<NPSolverMethod<<std::endl;
		std::cout<<"Min Mg:"<<minval<<std::endl;
		std::cout<<"Max Mg:"<<maxval<<std::endl<<std::endl;
	}

	ff_Mg->operator<<(*Mgfuncs[1]);

	//H
	Vec RHOH;
	VecDuplicate(Ii, &RHOH);
	WaterDissociation(cH->vec(), cOH->vec(), RHOH);
	VecSetOnDOFs(DOFsSetOnBoundary, RHOH, 0);//Boundaries should be removed from RHOH

	myFormAssigner(*A, {"Di", "zi"}, Hconsts);
	myFormAssigner(*f, {"Ii", "Ri"}, {std::make_shared<dolfin::Function>(Vh, (std::make_shared<dolfin::PETScVector>(Ii))->operator*(1)), std::make_shared<dolfin::Function>(Vh, (std::make_shared<dolfin::PETScVector>(RHOH))->operator*(1))});

	Mat L0_H, L1_H;
	Mat D_H;
	auto b0_H = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	auto b1_H = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	//prellocations
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &L0_H);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &L1_H);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &D_H);

	myLinearSystemAssembler(*A, {}, *A_FCT_FEM);
	myLinearSystemAssembler(*f, {}, *b0_H);

	FEMFCT_Lk_D_Compute(A_FCT_FEM->mat(), D_H, L0_H);
	FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D_H, L0_H, cH->vec(), b0_H->vec(), dt, rij, fStar);

	Mat A_H;
	Mat b_H;
	//prellocations
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &A_H);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &b_H);
	VecSet(b_NP, 0);

	MatCopy(A_ML, A_H, SAME_NONZERO_PATTERN);
	MatAXPY(A_H, 0.5*dt, L0_H, SAME_NONZERO_PATTERN);//L0=L1

	MatCopy(A_ML, b_H, SAME_NONZERO_PATTERN);
	MatAXPY(b_H, -0.5*dt, L0_H, SAME_NONZERO_PATTERN);
	MatMult(b_H, cH->vec(), b_NP);
	VecAXPY(b_NP, 0.5*dt, b0_H->vec());//b0=0

	VecAXPY(b_NP, 1, fStar);

	PetscBarrier(NULL);

	MatCopy(A_H, A_NPSolver->mat(), SAME_NONZERO_PATTERN);
	VecCopy(b_NP, b_NPSolver->vec());

	auto ff_H = std::make_shared<dolfin::File>("Results/H Concentration.pvd");
	ff_H->operator<<(*Hfuncs[0]);

	NPSolver->set_operator(A_NPSolver);
	NPSolver->solve(*Hfuncs[1]->vector(), *b_NPSolver);

	minval = as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->min();
	maxval = as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->max();

	if(prcID==0) {
		std::cout<<"Solver for Nernst-Planck (H) is set to: "<<NPSolverMethod<<std::endl;
		std::cout<<"Min H:"<<minval<<std::endl;
		std::cout<<"Max H:"<<maxval<<std::endl<<std::endl;
	}

	ff_H->operator<<(*Hfuncs[1]);

	//OH
	//reaction limiter on Al
	/*Vec Water;
	VecDuplicate(Ii, &Water);
	VecSet(Water, 55.555);*/
	BoundaryCurrent(DOFsSetOnAlElectrode, t, cMg->vec(), cOH->vec(), as_type<const dolfin::PETScVector>(O2funcs[0]->vector())->vec(), 0.55, 0.1, 0.01, Ii);//A/m^2
	VecScale(Ii, 0.02);//0.0004 We have an active length of 20mm

	myFormAssigner(*A, {"Di", "zi"}, OHconsts);
	myFormAssigner(*f, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, (std::make_shared<dolfin::PETScVector>(Ii))->operator*(-1))});

	Mat L0_OH, L1_OH;
	Mat D_OH;
	auto b0_OH = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	auto b1_OH = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	//prellocations
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &L0_OH);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &L1_OH);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &D_OH);

	myLinearSystemAssembler(*A, {}, *A_FCT_FEM);
	myLinearSystemAssembler(*f, {}, *b0_OH);

	FEMFCT_Lk_D_Compute(A_FCT_FEM->mat(), D_OH, L0_OH);
	FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D_OH, L0_OH, cOH->vec(), b0_OH->vec(), dt, rij, fStar);

	Mat A_OH;
	Mat b_OH;
	//prellocations
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &A_OH);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &b_OH);
	VecSet(b_NP, 0);

	MatCopy(A_ML, A_OH, SAME_NONZERO_PATTERN);
	MatAXPY(A_OH, 0.5*dt, L0_OH, SAME_NONZERO_PATTERN);//L0=L1

	MatCopy(A_ML, b_OH, SAME_NONZERO_PATTERN);
	MatAXPY(b_OH, -0.5*dt, L0_OH, SAME_NONZERO_PATTERN);
	MatMult(b_OH, cOH->vec(), b_NP);
	VecAXPY(b_NP, 0.5*dt, b0_OH->vec());//b0=0
	VecAXPY(b_NP, 1, fStar);

	PetscBarrier(NULL);

	MatCopy(A_OH, A_NPSolver->mat(), SAME_NONZERO_PATTERN);
	VecCopy(b_NP, b_NPSolver->vec());

	auto ff_OH = std::make_shared<dolfin::File>("Results/OH Concentration.pvd");
	ff_OH->operator<<(*OHfuncs[0]);

	NPSolver->set_operator(A_NPSolver);
	NPSolver->solve(*OHfuncs[1]->vector(), *b_NPSolver);

	minval = as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->min();
	maxval = as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->max();

	if(prcID==0) {
		std::cout<<"Solver for Nernst-Planck (OH) is set to: "<<NPSolverMethod<<std::endl;
		std::cout<<"Min OH:"<<minval<<std::endl;
		std::cout<<"Max OH:"<<maxval<<std::endl<<std::endl;
	}

	ff_OH->operator<<(*OHfuncs[1]);

	//Na
	myFormAssigner(*A, {"Di", "zi"}, Naconsts);
	myFormAssigner(*f, {"Ii", "Ri"}, {zerofunc, zerofunc});

	Mat L0_Na, L1_Na;
	Mat D_Na;
	auto b0_Na = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	auto b1_Na = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	//prellocations
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &L0_Na);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &L1_Na);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &D_Na);

	myLinearSystemAssembler(*A, {}, *A_FCT_FEM);
	myLinearSystemAssembler(*f, {}, *b0_Na);

	FEMFCT_Lk_D_Compute(A_FCT_FEM->mat(), D_Na, L0_Na);
	FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D_Na, L0_Na, as_type<const dolfin::PETScVector>(Nafuncs[0]->vector())->vec(), b0_Na->vec(), dt, rij, fStar);

	Mat A_Na;
	Mat b_Na;
	//prellocations
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &A_Na);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &b_Na);
	VecSet(b_NP, 0);

	MatCopy(A_ML, A_Na, SAME_NONZERO_PATTERN);
	MatAXPY(A_Na, 0.5*dt, L0_Na, SAME_NONZERO_PATTERN);//L0=L1

	MatCopy(A_ML, b_Na, SAME_NONZERO_PATTERN);
	MatAXPY(b_Na, -0.5*dt, L0_Na, SAME_NONZERO_PATTERN);
	MatMult(b_Na, as_type<const dolfin::PETScVector>(Nafuncs[0]->vector())->vec(), b_NP);
	VecAXPY(b_NP, 0.5*dt, b0_Na->vec());//b0=0
	VecAXPY(b_NP, 1, fStar);

	PetscBarrier(NULL);

	MatCopy(A_Na, A_NPSolver->mat(), SAME_NONZERO_PATTERN);
	VecCopy(b_NP, b_NPSolver->vec());

	auto ff_Na = std::make_shared<dolfin::File>("Results/Na Concentration.pvd");
	ff_Na->operator<<(*Nafuncs[0]);

	NPSolver->set_operator(A_NPSolver);
	NPSolver->solve(*Nafuncs[1]->vector(), *b_NPSolver);

	minval = as_type<const dolfin::PETScVector>(Nafuncs[1]->vector())->min();
	maxval = as_type<const dolfin::PETScVector>(Nafuncs[1]->vector())->max();

	if(prcID==0) {
		std::cout<<"Solver for Nernst-Planck (Na) is set to: "<<NPSolverMethod<<std::endl;
		std::cout<<"Min Na:"<<minval<<std::endl;
		std::cout<<"Max Na:"<<maxval<<std::endl<<std::endl;
	}

	ff_Na->operator<<(*Nafuncs[1]);

	//Cl
	myFormAssigner(*A, {"Di", "zi"}, Clconsts);

	Mat L0_Cl, L1_Cl;
	Mat D_Cl;
	auto b0_Cl = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	auto b1_Cl = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	//prellocations
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &L0_Cl);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &L1_Cl);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &D_Cl);

	myLinearSystemAssembler(*A, {}, *A_FCT_FEM);
	myLinearSystemAssembler(*f, {}, *b0_Cl);

	FEMFCT_Lk_D_Compute(A_FCT_FEM->mat(), D_Cl, L0_Cl);
	FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D_Cl, L0_Cl, as_type<const dolfin::PETScVector>(Clfuncs[0]->vector())->vec(), b0_Cl->vec(), dt, rij, fStar);

	Mat A_Cl;
	Mat b_Cl;
	//prellocations
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &A_Cl);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &b_Cl);
	VecSet(b_NP, 0);

	MatCopy(A_ML, A_Cl, SAME_NONZERO_PATTERN);
	MatAXPY(A_Cl, 0.5*dt, L0_Cl, SAME_NONZERO_PATTERN);//L0=L1

	MatCopy(A_ML, b_Cl, SAME_NONZERO_PATTERN);
	MatAXPY(b_Cl, -0.5*dt, L0_Cl, SAME_NONZERO_PATTERN);
	MatMult(b_Cl, as_type<const dolfin::PETScVector>(Clfuncs[0]->vector())->vec(), b_NP);
	VecAXPY(b_NP, 0.5*dt, b0_Cl->vec());//b0=0
	VecAXPY(b_NP, 1, fStar);

	PetscBarrier(NULL);

	MatCopy(A_Cl, A_NPSolver->mat(), SAME_NONZERO_PATTERN);
	VecCopy(b_NP, b_NPSolver->vec());

	auto ff_Cl = std::make_shared<dolfin::File>("Results/Cl Concentration.pvd");
	ff_Cl->operator<<(*Clfuncs[0]);

	NPSolver->set_operator(A_NPSolver);
	NPSolver->solve(*Clfuncs[1]->vector(), *b_NPSolver);

	minval = as_type<const dolfin::PETScVector>(Clfuncs[1]->vector())->min();
	maxval = as_type<const dolfin::PETScVector>(Clfuncs[1]->vector())->max();

	if(prcID==0) {
		std::cout<<"Solver for Nernst-Planck (Cl) is set to: "<<NPSolverMethod<<std::endl;
		std::cout<<"Min Cl:"<<minval<<std::endl;
		std::cout<<"Max Cl:"<<maxval<<std::endl<<std::endl;
	}

	ff_Cl->operator<<(*Clfuncs[1]);

	//O2
	myFormAssigner(*A, {"Di", "zi"}, O2consts);
	myFormAssigner(*f, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, (std::make_shared<dolfin::PETScVector>(Ii))->operator*(0.25))});

	Mat L0_O2, L1_O2;
	Mat D_O2;
	auto b0_O2 = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	auto b1_O2 = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	//prellocations
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &L0_O2);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &L1_O2);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &D_O2);

	myLinearSystemAssembler(*A, {}, *A_FCT_FEM);
	myLinearSystemAssembler(*f, {}, *b0_O2);

	FEMFCT_Lk_D_Compute(A_FCT_FEM->mat(), D_O2, L0_O2);

	Mat A_O2;
	Mat b_O2;
	//prellocations
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &A_O2);
	MatDuplicate(L0_Mg, MAT_DO_NOT_COPY_VALUES, &b_O2);
	VecSet(b_NP, 0);

	MatCopy(A_ML, A_O2, SAME_NONZERO_PATTERN);
	MatAXPY(A_O2, 0.5*dt, L0_O2, SAME_NONZERO_PATTERN);//L0=L1

	MatCopy(A_ML, b_O2, SAME_NONZERO_PATTERN);
	MatAXPY(b_O2, -0.5*dt, L0_O2, SAME_NONZERO_PATTERN);
	MatMult(b_O2, as_type<const dolfin::PETScVector>(O2funcs[0]->vector())->vec(), b_NP);
	VecAXPY(b_NP, 0.5*dt, b0_O2->vec());//b0=0

	PetscBarrier(NULL);

	MatCopy(A_O2, A_NPSolver->mat(), SAME_NONZERO_PATTERN);
	VecCopy(b_NP, b_NPSolver->vec());

	auto ff_O2 = std::make_shared<dolfin::File>("Results/O2 Concentration.pvd");
	ff_O2->operator<<(*O2funcs[0]);

	NPSolver->set_operator(A_NPSolver);
	NPSolver->solve(*O2funcs[1]->vector(), *b_NPSolver);

	minval = as_type<const dolfin::PETScVector>(O2funcs[1]->vector())->min();
	maxval = as_type<const dolfin::PETScVector>(O2funcs[1]->vector())->max();

	if(prcID==0) {
		std::cout<<"Solver for Nernst-Planck (O2) is set to: "<<NPSolverMethod<<std::endl;
		std::cout<<"Min O2:"<<minval<<std::endl;
		std::cout<<"Max O2:"<<maxval<<std::endl<<std::endl;
	}

	ff_O2->operator<<(*O2funcs[1]);

	//update step
	//poisson
	auto sumfunc = std::make_shared<dolfin::Function>(Vh);
	sumfunc->interpolate(dolfin::Constant(0));
	funcsLinSum({2, 1, -1, 1, -1}, {*Mgfuncs[1], *Hfuncs[1], *OHfuncs[1], *Nafuncs[1], *Clfuncs[1]}, *sumfunc);

	//Nernst-Planck
	t = dt + t;

	*(Mgfuncs[0]->vector()) = *(Mgfuncs[1]->vector());
	*(Hfuncs[0]->vector()) = *(Hfuncs[1]->vector());
	*(OHfuncs[0]->vector()) = *(OHfuncs[1]->vector());
	*(Nafuncs[0]->vector()) = *(Nafuncs[1]->vector());
	*(Clfuncs[0]->vector()) = *(Clfuncs[1]->vector());
	*(O2funcs[0]->vector()) = *(O2funcs[1]->vector());

	*cMg = *(Mgfuncs[1]->vector());
	*cH = *(Hfuncs[1]->vector());
	*cOH = *(OHfuncs[1]->vector());

	//One hour simulation
	for (std::size_t i=1; i<=totalsteps; i = i + 1) {//totalsteps
		//Poisson
		myFormAssigner(*L, {"f"}, {sumfunc});
		myLinearSystemAssembler(*L, DBCs, *b_P);
		PSolver->solve(*EFfuncs[0]->vector(), *b_P);

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
		BoundaryCurrent(DOFsSetOnMgElectrode, t, cMg->vec(), cOH->vec(), cH->vec(), 0.55, 0.1, 0.01, Ii);//A/m^2
		VecScale(Ii, 0.02);//0.0004 We have an active length of 20mm

		myFormAssigner(*A, {"Di", "zi"}, Mgconsts);
		myFormAssigner(*A, {"phi"}, {EFfuncs[0]});
		myFormAssigner(*f, {"Ii", "Ri"}, {std::make_shared<dolfin::Function>(Vh, (std::make_shared<dolfin::PETScVector>(Ii))->operator*(-0.5)), zerofunc});

		myLinearSystemAssembler(*A, {}, *A_FCT_FEM);
		myLinearSystemAssembler(*f, {}, *b1_Mg);

		FEMFCT_Lk_D_Compute(A_FCT_FEM->mat(), D_Mg, L1_Mg);
		FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D_Mg, L0_Mg, cMg->vec(), b0_Mg->vec(), dt, rij, fStar);

		MatZeroEntries(A_Mg);
		MatZeroEntries(b_Mg);
		VecSet(b_NP, 0);

		MatCopy(A_ML, A_Mg, SAME_NONZERO_PATTERN);
		MatAXPY(A_Mg, 0.5*dt, L1_Mg, SAME_NONZERO_PATTERN);

		MatCopy(A_ML, b_Mg, SAME_NONZERO_PATTERN);
		MatAXPY(b_Mg, -0.5*dt, L0_Mg, SAME_NONZERO_PATTERN);
		MatMult(b_Mg, cMg->vec(), b_NP);
		VecAXPY(b_NP, 0.5*dt, b0_Mg->vec());
		VecAXPY(b_NP, 0.5*dt, b1_Mg->vec());
		VecAXPY(b_NP, 1, fStar);

		PetscBarrier(NULL);

		MatCopy(A_Mg, A_NPSolver->mat(), SAME_NONZERO_PATTERN);
		VecCopy(b_NP, b_NPSolver->vec());

		NPSolver->set_operator(A_NPSolver);
		NPSolver->solve(*Mgfuncs[1]->vector(), *b_NPSolver);

		minval = as_type<const dolfin::PETScVector>(Mgfuncs[1]->vector())->min();
		maxval = as_type<const dolfin::PETScVector>(Mgfuncs[1]->vector())->max();

		if(prcID==0) {
			std::cout<<"Solver for Nernst-Planck (Mg) is set to: "<<NPSolverMethod<<std::endl;
			std::cout<<"Min Mg:"<<minval<<std::endl;
			std::cout<<"Max Mg:"<<maxval<<std::endl<<std::endl;
		}

		if (i%s == 0)
			ff_Mg->operator<<(*Mgfuncs[1]);

		//H
		WaterDissociation(cH->vec(), cOH->vec(), RHOH);
		VecSetOnDOFs(DOFsSetOnBoundary, RHOH, 0);//Boundaries should be removed from RHOH

		myFormAssigner(*A, {"Di", "zi"}, Hconsts);
		myFormAssigner(*f, {"Ii", "Ri"}, {std::make_shared<dolfin::Function>(Vh, (std::make_shared<dolfin::PETScVector>(Ii))->operator*(1)), std::make_shared<dolfin::Function>(Vh, (std::make_shared<dolfin::PETScVector>(RHOH))->operator*(1))});

		myLinearSystemAssembler(*A, {}, *A_FCT_FEM);
		myLinearSystemAssembler(*f, {}, *b1_H);

		FEMFCT_Lk_D_Compute(A_FCT_FEM->mat(), D_H, L1_H);
		FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D_H, L0_H, cH->vec(), b0_H->vec(), dt, rij, fStar);

		MatZeroEntries(A_H);
		MatZeroEntries(b_H);
		VecSet(b_NP, 0);

		MatCopy(A_ML, A_H, SAME_NONZERO_PATTERN);
		MatAXPY(A_H, 0.5*dt, L1_H, SAME_NONZERO_PATTERN);

		MatCopy(A_ML, b_H, SAME_NONZERO_PATTERN);
		MatAXPY(b_H, -0.5*dt, L0_H, SAME_NONZERO_PATTERN);
		MatMult(b_H, cH->vec(), b_NP);
		VecAXPY(b_NP, 0.5*dt, b0_H->vec());
		VecAXPY(b_NP, 0.5*dt, b1_H->vec());
		VecAXPY(b_NP, 1, fStar);

		PetscBarrier(NULL);

		MatCopy(A_H, A_NPSolver->mat(), SAME_NONZERO_PATTERN);
		VecCopy(b_NP, b_NPSolver->vec());

		NPSolver->set_operator(A_NPSolver);
		NPSolver->solve(*Hfuncs[1]->vector(), *b_NPSolver);

		minval = as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->min();
		maxval = as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->max();

		if(prcID==0) {
			std::cout<<"Solver for Nernst-Planck (H) is set to: "<<NPSolverMethod<<std::endl;
			std::cout<<"Min H:"<<minval<<std::endl;
			std::cout<<"Max H:"<<maxval<<std::endl<<std::endl;
		}

		if (i%s == 0)
			ff_H->operator<<(*Hfuncs[1]);

		//OH
		BoundaryCurrent(DOFsSetOnAlElectrode, t, cMg->vec(), cOH->vec(), as_type<const dolfin::PETScVector>(O2funcs[0]->vector())->vec(), 0.55, 0.1, 0.01, Ii);//A/m^2
		VecScale(Ii, 0.02);//0.0004 We have an active length of 20mm

		myFormAssigner(*A, {"Di", "zi"}, OHconsts);
		myFormAssigner(*f, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, (std::make_shared<dolfin::PETScVector>(Ii))->operator*(-1))});

		myLinearSystemAssembler(*A, {}, *A_FCT_FEM);
		myLinearSystemAssembler(*f, {}, *b1_OH);

		FEMFCT_Lk_D_Compute(A_FCT_FEM->mat(), D_OH, L1_OH);
		FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D_OH, L0_OH, cOH->vec(), b0_OH->vec(), dt, rij, fStar);

		MatZeroEntries(A_OH);
		MatZeroEntries(b_OH);
		VecSet(b_NP, 0);

		MatCopy(A_ML, A_OH, SAME_NONZERO_PATTERN);
		MatAXPY(A_OH, 0.5*dt, L1_OH, SAME_NONZERO_PATTERN);

		MatCopy(A_ML, b_OH, SAME_NONZERO_PATTERN);
		MatAXPY(b_OH, -0.5*dt, L0_OH, SAME_NONZERO_PATTERN);
		MatMult(b_OH, cOH->vec(), b_NP);
		VecAXPY(b_NP, 0.5*dt, b0_OH->vec());
		VecAXPY(b_NP, 0.5*dt, b1_OH->vec());
		VecAXPY(b_NP, 1, fStar);

		PetscBarrier(NULL);

		MatCopy(A_OH, A_NPSolver->mat(), SAME_NONZERO_PATTERN);
		VecCopy(b_NP, b_NPSolver->vec());

		NPSolver->set_operator(A_NPSolver);
		NPSolver->solve(*OHfuncs[1]->vector(), *b_NPSolver);

		minval = as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->min();
		maxval = as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->max();

		if(prcID==0) {
			std::cout<<"Solver for Nernst-Planck (OH) is set to: "<<NPSolverMethod<<std::endl;
			std::cout<<"Min OH:"<<minval<<std::endl;
			std::cout<<"Max OH:"<<maxval<<std::endl<<std::endl;
		}

		if (i%s == 0)
			ff_OH->operator<<(*OHfuncs[1]);

		//Na
		myFormAssigner(*A, {"Di", "zi"}, Naconsts);
		myFormAssigner(*f, {"Ii", "Ri"}, {zerofunc, zerofunc});

		myLinearSystemAssembler(*A, {}, *A_FCT_FEM);
		myLinearSystemAssembler(*f, {}, *b1_Na);

		FEMFCT_Lk_D_Compute(A_FCT_FEM->mat(), D_Na, L1_Na);
		FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D_Na, L0_Na, as_type<const dolfin::PETScVector>(Nafuncs[0]->vector())->vec(), b0_Na->vec(), dt, rij, fStar);

		MatZeroEntries(A_Na);
		MatZeroEntries(b_Na);
		VecSet(b_NP, 0);

		MatCopy(A_ML, A_Na, SAME_NONZERO_PATTERN);
		MatAXPY(A_Na, 0.5*dt, L1_Na, SAME_NONZERO_PATTERN);

		MatCopy(A_ML, b_Na, SAME_NONZERO_PATTERN);
		MatAXPY(b_Na, -0.5*dt, L0_Na, SAME_NONZERO_PATTERN);
		MatMult(b_Na, as_type<const dolfin::PETScVector>(Nafuncs[0]->vector())->vec(), b_NP);
		VecAXPY(b_NP, 0.5*dt, b0_Na->vec());
		VecAXPY(b_NP, 0.5*dt, b1_Na->vec());
		VecAXPY(b_NP, 1, fStar);

		PetscBarrier(NULL);

		MatCopy(A_Na, A_NPSolver->mat(), SAME_NONZERO_PATTERN);
		VecCopy(b_NP, b_NPSolver->vec());

		NPSolver->set_operator(A_NPSolver);
		NPSolver->solve(*Nafuncs[1]->vector(), *b_NPSolver);

		minval = as_type<const dolfin::PETScVector>(Nafuncs[1]->vector())->min();
		maxval = as_type<const dolfin::PETScVector>(Nafuncs[1]->vector())->max();

		if(prcID==0) {
			std::cout<<"Solver for Nernst-Planck (Na) is set to: "<<NPSolverMethod<<std::endl;
			std::cout<<"Min Na:"<<minval<<std::endl;
			std::cout<<"Max Na:"<<maxval<<std::endl<<std::endl;
		}

		if (i%s == 0)
			ff_Na->operator<<(*Nafuncs[1]);

		//Cl
		myFormAssigner(*A, {"Di", "zi"}, Clconsts);

		myLinearSystemAssembler(*A, {}, *A_FCT_FEM);
		myLinearSystemAssembler(*f, {}, *b1_Cl);

		FEMFCT_Lk_D_Compute(A_FCT_FEM->mat(), D_Cl, L1_Cl);
		FEMFCT_fStar_Compute(A_ML, A_MC->mat(), D_Cl, L0_Cl, as_type<const dolfin::PETScVector>(Clfuncs[0]->vector())->vec(), b0_Cl->vec(), dt, rij, fStar);

		MatZeroEntries(A_Cl);
		MatZeroEntries(b_Cl);
		VecSet(b_NP, 0);

		MatCopy(A_ML, A_Cl, SAME_NONZERO_PATTERN);
		MatAXPY(A_Cl, 0.5*dt, L1_Cl, SAME_NONZERO_PATTERN);

		MatCopy(A_ML, b_Cl, SAME_NONZERO_PATTERN);
		MatAXPY(b_Cl, -0.5*dt, L0_Cl, SAME_NONZERO_PATTERN);
		MatMult(b_Cl, as_type<const dolfin::PETScVector>(Clfuncs[0]->vector())->vec(), b_NP);
		VecAXPY(b_NP, 0.5*dt, b0_Cl->vec());
		VecAXPY(b_NP, 0.5*dt, b1_Cl->vec());
		VecAXPY(b_NP, 1, fStar);

		PetscBarrier(NULL);

		MatCopy(A_Cl, A_NPSolver->mat(), SAME_NONZERO_PATTERN);
		VecCopy(b_NP, b_NPSolver->vec());

		NPSolver->set_operator(A_NPSolver);
		NPSolver->solve(*Clfuncs[1]->vector(), *b_NPSolver);

		minval = as_type<const dolfin::PETScVector>(Clfuncs[1]->vector())->min();
		maxval = as_type<const dolfin::PETScVector>(Clfuncs[1]->vector())->max();

		if(prcID==0) {
			std::cout<<"Solver for Nernst-Planck (Cl) is set to: "<<NPSolverMethod<<std::endl;
			std::cout<<"Min Cl:"<<minval<<std::endl;
			std::cout<<"Max Cl:"<<maxval<<std::endl<<std::endl;
		}

		if (i%s == 0)
			ff_Cl->operator<<(*Clfuncs[1]);

		//O2
		myFormAssigner(*A, {"Di", "zi"}, O2consts);
		myFormAssigner(*f, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, (std::make_shared<dolfin::PETScVector>(Ii))->operator*(0.25))});

		myLinearSystemAssembler(*A, {}, *A_FCT_FEM);
		myLinearSystemAssembler(*f, {}, *b1_O2);

		FEMFCT_Lk_D_Compute(A_FCT_FEM->mat(), D_O2, L1_O2);

		MatZeroEntries(A_O2);
		MatZeroEntries(b_O2);
		VecSet(b_NP, 0);

		MatCopy(A_ML, A_O2, SAME_NONZERO_PATTERN);
		MatAXPY(A_O2, 0.5*dt, L1_O2, SAME_NONZERO_PATTERN);

		MatCopy(A_ML, b_O2, SAME_NONZERO_PATTERN);
		MatAXPY(b_O2, -0.5*dt, L0_O2, SAME_NONZERO_PATTERN);
		MatMult(b_O2, as_type<const dolfin::PETScVector>(O2funcs[0]->vector())->vec(), b_NP);
		VecAXPY(b_NP, 0.5*dt, b0_O2->vec());
		VecAXPY(b_NP, 0.5*dt, b1_O2->vec());
		VecAXPY(b_NP, 1, fStar);

		PetscBarrier(NULL);

		MatCopy(A_O2, A_NPSolver->mat(), SAME_NONZERO_PATTERN);
		VecCopy(b_NP, b_NPSolver->vec());

		NPSolver->set_operator(A_NPSolver);
		NPSolver->solve(*O2funcs[1]->vector(), *b_NPSolver);

		minval = as_type<const dolfin::PETScVector>(O2funcs[1]->vector())->min();
		maxval = as_type<const dolfin::PETScVector>(O2funcs[1]->vector())->max();

		if(prcID==0) {
			std::cout<<"Solver for Nernst-Planck (O2) is set to: "<<NPSolverMethod<<std::endl;
			std::cout<<"Min O2:"<<minval<<std::endl;
			std::cout<<"Max O2:"<<maxval<<std::endl<<std::endl;
		}

		if (i%s == 0)
			ff_O2->operator<<(*O2funcs[1]);

		//update step
		//poisson
		funcsLinSum({2, 1, -1, 1, -1}, {*Mgfuncs[1], *Hfuncs[1], *OHfuncs[1], *Nafuncs[1], *Clfuncs[1]}, *sumfunc);

		//Nernst-Planck
		t = dt + t;
		if (i == (10*60*100))
			dt = 1e-1;
		//if (i == ((10*60*100)+(20*60*10)))
		//	dt = 1;

		*(Mgfuncs[0]->vector()) = *(Mgfuncs[1]->vector());
		*(Hfuncs[0]->vector()) = *(Hfuncs[1]->vector());
		*(OHfuncs[0]->vector()) = *(OHfuncs[1]->vector());
		*(Nafuncs[0]->vector()) = *(Nafuncs[1]->vector());
		*(Clfuncs[0]->vector()) = *(Clfuncs[1]->vector());
		*(O2funcs[0]->vector()) = *(O2funcs[1]->vector());

		*cMg = *(Mgfuncs[1]->vector());
		*cH = *(Hfuncs[1]->vector());
		*cOH = *(OHfuncs[1]->vector());

		VecCopy(b1_Mg->vec(), b0_Mg->vec());
		VecCopy(b1_H->vec(), b0_H->vec());
		VecCopy(b1_OH->vec(), b0_OH->vec());
		VecCopy(b1_Na->vec(), b0_Na->vec());
		VecCopy(b1_Cl->vec(), b0_Cl->vec());
		VecCopy(b1_O2->vec(), b0_O2->vec());

		MatCopy(L1_Mg, L0_Mg, SAME_NONZERO_PATTERN);
		MatCopy(L1_H, L0_H, SAME_NONZERO_PATTERN);
		MatCopy(L1_OH, L0_OH, SAME_NONZERO_PATTERN);
		MatCopy(L1_Na, L0_Na, SAME_NONZERO_PATTERN);
		MatCopy(L1_Cl, L0_Cl, SAME_NONZERO_PATTERN);
		MatCopy(L1_O2, L0_O2, SAME_NONZERO_PATTERN);

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
	DOFsSetOnBoundary.clear();
	DOFsSetOnBoundary.shrink_to_fit();

	//clean the shared pointers
	ff_P.reset();
	cMg.reset();
	cOH.reset();
	cH.reset();
	A_MC.reset();
	A_FCT_FEM.reset();
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
	b0_O2.reset();
	b1_O2.reset();
	A_NPSolver.reset();
	b_NPSolver.reset();
	ff_Mg.reset();
	ff_H.reset();
	ff_OH.reset();
	ff_Na.reset();
	ff_Cl.reset();
	ff_O2.reset();
	NPSolver.reset();

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
	SharedTypeVectorDestructor(O2funcs);
	SharedTypeVectorDestructor(O2consts);

	a.reset();
	L.reset();
	MC.reset();
	A.reset();
	f.reset();
	zerofunc.reset();
	sumfunc.reset();
	Vh.reset();
	mesh.reset();

	//clean the memory of Petsc objects
	MatDestroy(&rij);
	VecDestroy(&Ii);
	VecDestroy(&fStar);
	VecDestroy(&b_NP);
	VecDestroy(&RHOH);
	//VecDestroy(&Water);

	MatDestroy(&A_ML);
	MatDestroy(&L0_Mg);
	MatDestroy(&L1_Mg);
	MatDestroy(&D_Mg);
	MatDestroy(&A_Mg);
	MatDestroy(&b_Mg);

	MatDestroy(&L0_H);
	MatDestroy(&L1_H);
	MatDestroy(&D_H);
	MatDestroy(&A_H);
	MatDestroy(&b_H);

	MatDestroy(&L0_OH);
	MatDestroy(&L1_OH);
	MatDestroy(&D_OH);
	MatDestroy(&A_OH);
	MatDestroy(&b_OH);

	MatDestroy(&L0_Na);
	MatDestroy(&L1_Na);
	MatDestroy(&D_Na);
	MatDestroy(&A_Na);
	MatDestroy(&b_Na);

	MatDestroy(&L0_Cl);
	MatDestroy(&L1_Cl);
	MatDestroy(&D_Cl);
	MatDestroy(&A_Cl);
	MatDestroy(&b_Cl);

	MatDestroy(&L0_O2);
	MatDestroy(&L1_O2);
	MatDestroy(&D_O2);
	MatDestroy(&A_O2);
	MatDestroy(&b_O2);

	PetscFinalize();

	return 0;
}
