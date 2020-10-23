#include <dolfin.h>
#include <fstream>
#include "src/FEMTools.cpp"
#include "src/GeneralTools.cpp"
#include "src/CorrosionTools.cpp"
#include "src/MeshTools.cpp"
#include "src/Poisson.h"
#include "src/MassMatrix.h"
#include "src/StiffnessMatrix.h"
#include "src/DiffusionStiffnessM.h"
#include "src/DiffusionMassM.h"

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

	//fracture
	auto aS = std::make_shared<DiffusionStiffnessM::BilinearForm>(Vh, Vh);
	auto LS = std::make_shared<DiffusionStiffnessM::LinearForm>(Vh);
	auto aM = std::make_shared<DiffusionMassM::BilinearForm>(Vh, Vh);
	auto LM = std::make_shared<DiffusionMassM::LinearForm>(Vh);

	PetscBarrier(NULL);

	//----Creating the functions that we apply to variational formulations model----
	auto zerofunc = std::make_shared<dolfin::Function>(Vh);
	(zerofunc->vector())->operator=(0);

	//Electrical field
	//auto kappa = std::make_shared<dolfin::Function>(Vh);
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

	//fracture
	//Mg
	std::vector<std::shared_ptr<dolfin::Function>> sMgfuncs;
	isconst.clear(); isconst.shrink_to_fit();
	isconst = {1, 0};
	constvalue = {10};//Molarity of Mg
	Vector_of_NonConstFunctionGenerator(Vh, sMgfuncs, isconst, constvalue);

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
	/*std::vector<std::shared_ptr<dolfin::Function>> Nafuncs;
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
	Vector_of_ConstFunctionGenerator(Vh, Clconsts, constvalue);*/

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

	//--Nodes indices and domain information is no longer needed--
	NodesOnAlElectrode.clear(); NodesOnAlElectrode.shrink_to_fit();
	NodesOnMgElectrode.clear(); NodesOnMgElectrode.shrink_to_fit();
	SharedTypeVectorDestructor(bcs);

	//----Assembling the final linear systems and solve them and storing the solution----
	//Poisson
	auto A_P = std::make_shared<dolfin::PETScMatrix>(PETSC_COMM_WORLD);
	auto b_P = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	auto BoundaryPhi = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(EFfuncs[0]->vector())));
	BoundaryPhi->zero();
	auto ff_P = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/Electric Field.pvd");
	std::string PSolverMethod = "mumps";
	auto PSolver = std::make_shared<dolfin::PETScLUSolver>(PETSC_COMM_WORLD, PSolverMethod);

	//kappa_Compute({2, -1, 1, 1, -1}, {0.71e-9, 5.27e-9, 9.31e-9, 1.33e-9, 2.03e-9}, {*Mgfuncs[0], *OHfuncs[0], *Hfuncs[0], *Nafuncs[0], *Clfuncs[0]}, *kappa);

	PetscBarrier(NULL);

	myFormAssigner(*a, {"kappa"}, {std::make_shared<dolfin::Constant>(1)});
	myFormAssigner(*L, {"kappa", "f", "g"}, {zerofunc, zerofunc, zerofunc});//there is no reaction in domain so f=\sum ziRi=0
	myLinearSystemAssembler(*a, *L, DBCs, *A_P, *b_P);

	if(prcID==0) {
		list_krylov_solver_methods();
		list_krylov_solver_preconditioners();
		list_linear_solver_methods();
		list_lu_solver_methods();
	}

	PetscBarrier(NULL);

	PSolver->set_operator(*A_P);
	PSolver->solve(*EFfuncs[0]->vector(), *b_P);

	//Nernst-Planck
	//Common part
	auto A_MC = std::make_shared<dolfin::PETScMatrix>(PETSC_COMM_WORLD);

	myLinearSystemAssembler(*MC, {}, *A_MC);

	auto A_ML = std::make_shared<dolfin::PETScMatrix>(*A_MC);
	auto A_NP = std::make_shared<dolfin::PETScMatrix>(*A_MC);
	auto rij = std::make_shared<dolfin::PETScMatrix>(*A_MC);
	auto ALinSys = std::make_shared<dolfin::PETScMatrix>(*A_MC);
	ALinSys->zero();
	auto rhsLinSys = std::make_shared<dolfin::PETScMatrix>(*A_MC);
	rhsLinSys->zero();
	auto Ii = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	Ii->zero();
	auto fStar = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	fStar->zero();
	auto bLinSys = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	bLinSys->zero();
	auto Vec_l = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	Vec_l->zero();
	auto Vec_teta = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	Vec_teta->zero();
	auto Vec_eps = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	Vec_eps->zero();

	double dt = 1e-1;
	double t = 0;
	std::size_t s = 50;
	std::size_t totalsteps = 8*3600*10;

	PetscBarrier(NULL);

	FEMFCT_ML_Compute(A_MC->mat(), A_ML->mat());

	//fracture
	auto AS_Diffusion = std::make_shared<dolfin::PETScMatrix>(PETSC_COMM_WORLD);
	auto A_Diffusion = std::make_shared<dolfin::PETScMatrix>(PETSC_COMM_WORLD);
	auto b_Diffusion = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);
	auto IMg0_Diffusion = std::make_shared<dolfin::PETScVector>(PETSC_COMM_WORLD);

	//Mg
	auto L0_Mg = std::make_shared<dolfin::PETScMatrix>(*A_MC);
	auto D0_Mg = std::make_shared<dolfin::PETScMatrix>(*A_MC);
	auto D1_Mg = std::make_shared<dolfin::PETScMatrix>(*A_MC);
	auto b0_Mg = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	b0_Mg->zero();
	auto b1_Mg = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	b1_Mg->zero();

	iMg(DOFsSetOnMgElectrode, t, dt, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), 0.55, 0.4, 1e-7, Ii->vec(), BoundaryPhi->vec(), PhiPolData, MgPolData, Vec_l->vec(), Vec_teta->vec(), Vec_eps->vec());//A/m^2

	myFormAssigner(*A, {"Di", "zi"}, Mgconsts);
	myFormAssigner(*A, {"phi"}, {EFfuncs[0]});
	myFormAssigner(*f, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, Ii->operator*(-1*5.182045e-6))});//stochimetry coefficient divided by z in equation (3)

	PetscBarrier(NULL);

	myLinearSystemAssembler(*A, {}, *A_NP);
	myLinearSystemAssembler(*f, {}, *b1_Mg);

	PetscBarrier(NULL);

	AFC_D_Compute(A_NP->mat(), D1_Mg->mat());

	MatCopy(D1_Mg->mat(), D0_Mg->mat(), DIFFERENT_NONZERO_PATTERN);//D1_Mg=D0_Mg
	MatCopy(A_NP->mat(), L0_Mg->mat(), DIFFERENT_NONZERO_PATTERN);//L0=L1
	MatAXPY(L0_Mg->mat(), 1, D0_Mg->mat(), DIFFERENT_NONZERO_PATTERN);

	AFC_EfStar_Compute(A_ML->mat(), A_MC->mat(), D1_Mg->mat(), D0_Mg->mat(), L0_Mg->mat(), as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), b0_Mg->vec(), dt, rij->mat(), fStar->vec(), false);

	PetscBarrier(NULL);

	//Linear system
	MatCopy(A_ML->mat(), ALinSys->mat(), DIFFERENT_NONZERO_PATTERN);
	MatAXPY(ALinSys->mat(), 0.5*dt, L0_Mg->mat(), DIFFERENT_NONZERO_PATTERN);//L0=L1

	MatCopy(A_ML->mat(), rhsLinSys->mat(), DIFFERENT_NONZERO_PATTERN);
	MatAXPY(rhsLinSys->mat(), -0.5*dt, L0_Mg->mat(), DIFFERENT_NONZERO_PATTERN);
	MatMult(rhsLinSys->mat(), as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), bLinSys->vec());

	VecAXPY(bLinSys->vec(), 0.5*dt, b1_Mg->vec());//b0=0
	VecAXPY(bLinSys->vec(), 1, fStar->vec());

	PetscBarrier(NULL);

	bLinSys->update_ghost_values();

	//Setting up system solver
	KSP myNPSolver;
	PC myNPConditioner;
	KSPCreate(PETSC_COMM_WORLD, &myNPSolver);
	KSPSetType(myNPSolver, KSPGMRES);
	KSPSetInitialGuessNonzero(myNPSolver, PETSC_TRUE);
	KSPGetPC(myNPSolver, &myNPConditioner);
	PCSetType(myNPConditioner, PCJACOBI);
	KSPSetUp(myNPSolver);

	auto NPSolver = std::make_shared<dolfin::PETScKrylovSolver>(myNPSolver);
	NPSolver->set_operator(*ALinSys);
	NPSolver->solve(*Mgfuncs[1]->vector(), *bLinSys);

	std::string NPSolverMethod = "gmres with jacobi";

	auto ff_Mg = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/Mg Concentration.pvd");
	ff_Mg->operator<<(*Mgfuncs[0]);

	//fracture
	myFormAssigner(*aS, {"Di"}, {Mgconsts[0]});
	myFormAssigner(*LS, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, Ii->operator*(-1*5.182045e-6))});
	myFormAssigner(*LM, {"ci0"}, {sMgfuncs[0]});
	
	PetscBarrier(NULL);

	myLinearSystemAssembler(*aS, {}, *AS_Diffusion);
	myLinearSystemAssembler(*aM, {}, *A_Diffusion);
	
	PetscBarrier(NULL);
	
	MatAXPY(A_Diffusion->mat(), 0.5*dt, AS_Diffusion->mat(), DIFFERENT_NONZERO_PATTERN);//left hand side of system (stays fixed)
	myLinearSystemAssembler(*LM, {}, *b_Diffusion);
	MatMult(AS_Diffusion->mat(), as_type<const dolfin::PETScVector>(sMgfuncs[0]->vector())->vec(), bLinSys->vec());
	bLinSys->operator*=(-0.5*dt);
	VecAXPY(bLinSys->vec(), 1, b_Diffusion->vec());
	//VecAXPY(bLinSys->vec(), 0.5*dt, IMg0_Diffusion->vec());//it is zero
	
	PetscBarrier(NULL);
	
	myLinearSystemAssembler(*LS, {}, *IMg0_Diffusion);//IMg1 which is next step IMg0
	
	PetscBarrier(NULL);
	
	VecAXPY(bLinSys->vec(), 0.5*dt, IMg0_Diffusion->vec());

	bLinSys->update_ghost_values();

	PetscBarrier(NULL);
	
	PSolver->set_operator(*A_Diffusion);
	PSolver->solve(*sMgfuncs[1]->vector(), *bLinSys);

	auto ff_fMg = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/fracture Mg Concentration.pvd");
	ff_fMg->operator<<(*sMgfuncs[0]);

	//OH
	auto L0_OH = std::make_shared<dolfin::PETScMatrix>(*A_MC);
	auto D0_OH = std::make_shared<dolfin::PETScMatrix>(*A_MC);
	auto D1_OH = std::make_shared<dolfin::PETScMatrix>(*A_MC);
	auto b0_OH = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	b0_OH->zero();
	auto b1_OH = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	b1_OH->zero();
	auto RHOH = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	RHOH->zero();

	iOH(DOFsSetOnAlElectrode, t, dt, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), 0.55, 0.4, 1e-7, Ii->vec(), BoundaryPhi->vec(), PhiPolData, AlPolData, Vec_l->vec(), Vec_teta->vec(), Vec_eps->vec());//A/m^2

	myFormAssigner(*A, {"Di", "zi"}, OHconsts);
	myFormAssigner(*f, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, Ii->operator*(-1*1.0364e-5))});

	PetscBarrier(NULL);

	myLinearSystemAssembler(*A, {}, *A_NP);
	myLinearSystemAssembler(*f, {}, *b1_OH);

	PetscBarrier(NULL);

	AFC_D_Compute(A_NP->mat(), D1_OH->mat());

	MatCopy(D1_OH->mat(), D0_OH->mat(), DIFFERENT_NONZERO_PATTERN);//D1_OH=D0_OH
	MatCopy(A_NP->mat(), L0_OH->mat(), DIFFERENT_NONZERO_PATTERN);//L0=L1
	MatAXPY(L0_OH->mat(), 1, D0_OH->mat(), DIFFERENT_NONZERO_PATTERN);

	AFC_EfStar_Compute(A_ML->mat(), A_MC->mat(), D1_OH->mat(), D0_OH->mat(), L0_OH->mat(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), b0_OH->vec(), dt, rij->mat(), fStar->vec(), false);

	PetscBarrier(NULL);

	//Linear system
	MatCopy(A_ML->mat(), ALinSys->mat(), DIFFERENT_NONZERO_PATTERN);
	MatAXPY(ALinSys->mat(), 0.5*dt, L0_OH->mat(), DIFFERENT_NONZERO_PATTERN);//L0=L1

	MatCopy(A_ML->mat(), rhsLinSys->mat(), DIFFERENT_NONZERO_PATTERN);
	MatAXPY(rhsLinSys->mat(), -0.5*dt, L0_OH->mat(), DIFFERENT_NONZERO_PATTERN);
	MatMult(rhsLinSys->mat(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), bLinSys->vec());

	VecAXPY(bLinSys->vec(), 0.5*dt, b1_OH->vec());//b0=0
	VecAXPY(bLinSys->vec(), 1, fStar->vec());

	PetscBarrier(NULL);

	bLinSys->update_ghost_values();

	//solve
	NPSolver->set_operator(*ALinSys);
	NPSolver->solve(*OHfuncs[1]->vector(), *bLinSys);

	auto ff_OH = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/OH Concentration.pvd");
	ff_OH->operator<<(*OHfuncs[0]);

	//H
	auto L0_H = std::make_shared<dolfin::PETScMatrix>(*A_MC);
	auto D0_H = std::make_shared<dolfin::PETScMatrix>(*A_MC);
	auto D1_H = std::make_shared<dolfin::PETScMatrix>(*A_MC);
	/*auto b0_H = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	b0_H->zero();
	auto b1_H = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	b1_H->zero();*/

	myFormAssigner(*A, {"Di", "zi"}, Hconsts);
	//myFormAssigner(*f, {"Ii"}, {zerofunc});

	PetscBarrier(NULL);

	myLinearSystemAssembler(*A, {}, *A_NP);
	//myLinearSystemAssembler(*f, {}, *b1_H);

	PetscBarrier(NULL);

	AFC_D_Compute(A_NP->mat(), D1_H->mat());

	MatCopy(D1_H->mat(), D0_H->mat(), DIFFERENT_NONZERO_PATTERN);//D1_H=D0_H
	MatCopy(A_NP->mat(), L0_H->mat(), DIFFERENT_NONZERO_PATTERN);//L0=L1
	MatAXPY(L0_H->mat(), 1, D0_H->mat(), DIFFERENT_NONZERO_PATTERN);

	AFC_EfStar_Compute(A_ML->mat(), A_MC->mat(), D1_H->mat(), D0_H->mat(), L0_H->mat(), as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(zerofunc->vector())->vec(), dt, rij->mat(), fStar->vec());

	PetscBarrier(NULL);

	//Linear system
	MatCopy(A_ML->mat(), ALinSys->mat(), DIFFERENT_NONZERO_PATTERN);
	MatAXPY(ALinSys->mat(), 0.5*dt, L0_H->mat(), DIFFERENT_NONZERO_PATTERN);//L0=L1

	MatCopy(A_ML->mat(), rhsLinSys->mat(), DIFFERENT_NONZERO_PATTERN);
	MatAXPY(rhsLinSys->mat(), -0.5*dt, L0_H->mat(), DIFFERENT_NONZERO_PATTERN);
	MatMult(rhsLinSys->mat(), as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), bLinSys->vec());

	//VecAXPY(bLinSys->vec(), 0.5*dt, b1_OH->vec());//b0=0
	VecAXPY(bLinSys->vec(), 1, fStar->vec());

	PetscBarrier(NULL);

	bLinSys->update_ghost_values();

	//solve
	NPSolver->set_operator(*ALinSys);
	NPSolver->solve(*Hfuncs[1]->vector(), *bLinSys);

	auto ff_H = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/H Concentration.pvd");
	ff_H->operator<<(*Hfuncs[0]);

	PetscBarrier(NULL);

	WaterDissociation(as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), RHOH->vec());
//dolfin::File ff_rhoh(PETSC_COMM_WORLD, "Results/RHOH.pvd");
//ff_rhoh.operator<<(*std::make_shared<dolfin::Function>(Vh, RHOH));
	VecSetOnLocalDOFs(DOFsSetOnAlElectrode, RHOH->vec(), 0);
	VecSetOnLocalDOFs(DOFsSetOnMgElectrode, RHOH->vec(), 0);

	VecAXPY(as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->vec(), dt, RHOH->vec());
	VecGhostUpdateBegin(as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->vec(), INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->vec(), INSERT_VALUES, SCATTER_FORWARD);
	VecAXPY(as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->vec(), dt, RHOH->vec());
	VecGhostUpdateBegin(as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->vec(), INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->vec(), INSERT_VALUES, SCATTER_FORWARD);

	//Na
	/*auto L0_Na = std::make_shared<dolfin::PETScMatrix>(*A_MC);
	auto D0_Na = std::make_shared<dolfin::PETScMatrix>(*A_MC);
	auto D1_Na = std::make_shared<dolfin::PETScMatrix>(*A_MC);

	myFormAssigner(*A, {"Di", "zi"}, Naconsts);

	PetscBarrier(NULL);

	myLinearSystemAssembler(*A, {}, *A_NP);

	PetscBarrier(NULL);

	AFC_D_Compute(A_NP->mat(), D1_Na->mat());

	MatCopy(D1_Na->mat(), D0_Na->mat(), DIFFERENT_NONZERO_PATTERN);//D1_Na=D0_Na
	MatCopy(A_NP->mat(), L0_Na->mat(), DIFFERENT_NONZERO_PATTERN);//L0=L1
	MatAXPY(L0_Na->mat(), 1, D0_Na->mat(), DIFFERENT_NONZERO_PATTERN);

	AFC_EfStar_Compute(A_ML->mat(), A_MC->mat(), D1_Na->mat(), D0_Na->mat(), L0_Na->mat(), as_type<const dolfin::PETScVector>(Nafuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(zerofunc->vector())->vec(), dt, rij->mat(), fStar->vec());

	PetscBarrier(NULL);

	//Linear system
	MatCopy(A_ML->mat(), ALinSys->mat(), DIFFERENT_NONZERO_PATTERN);
	MatAXPY(ALinSys->mat(), 0.5*dt, L0_Na->mat(), DIFFERENT_NONZERO_PATTERN);//L0=L1

	MatCopy(A_ML->mat(), rhsLinSys->mat(), DIFFERENT_NONZERO_PATTERN);
	MatAXPY(rhsLinSys->mat(), -0.5*dt, L0_Na->mat(), DIFFERENT_NONZERO_PATTERN);
	MatMult(rhsLinSys->mat(), as_type<const dolfin::PETScVector>(Nafuncs[0]->vector())->vec(), bLinSys->vec());

	VecAXPY(bLinSys->vec(), 1, fStar->vec());

	PetscBarrier(NULL);

	bLinSys->update_ghost_values();

	//solve
	NPSolver->set_operator(*ALinSys);
	NPSolver->solve(*Nafuncs[1]->vector(), *bLinSys);

	auto ff_Na = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/Na Concentration.pvd");
	ff_Na->operator<<(*Nafuncs[0]);

	//Cl
	auto L0_Cl = std::make_shared<dolfin::PETScMatrix>(*A_MC);
	auto D0_Cl = std::make_shared<dolfin::PETScMatrix>(*A_MC);
	auto D1_Cl = std::make_shared<dolfin::PETScMatrix>(*A_MC);

	myFormAssigner(*A, {"Di", "zi"}, Clconsts);

	PetscBarrier(NULL);

	myLinearSystemAssembler(*A, {}, *A_NP);

	PetscBarrier(NULL);

	AFC_D_Compute(A_NP->mat(), D1_Cl->mat());

	MatCopy(D1_Cl->mat(), D0_Cl->mat(), DIFFERENT_NONZERO_PATTERN);//D1_Na=D0_Na
	MatCopy(A_NP->mat(), L0_Cl->mat(), DIFFERENT_NONZERO_PATTERN);//L0=L1
	MatAXPY(L0_Cl->mat(), 1, D0_Cl->mat(), DIFFERENT_NONZERO_PATTERN);

	AFC_EfStar_Compute(A_ML->mat(), A_MC->mat(), D1_Cl->mat(), D0_Cl->mat(), L0_Cl->mat(), as_type<const dolfin::PETScVector>(Clfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(zerofunc->vector())->vec(), dt, rij->mat(), fStar->vec());

	PetscBarrier(NULL);

	//Linear system
	MatCopy(A_ML->mat(), ALinSys->mat(), DIFFERENT_NONZERO_PATTERN);
	MatAXPY(ALinSys->mat(), 0.5*dt, L0_Cl->mat(), DIFFERENT_NONZERO_PATTERN);//L0=L1

	MatCopy(A_ML->mat(), rhsLinSys->mat(), DIFFERENT_NONZERO_PATTERN);
	MatAXPY(rhsLinSys->mat(), -0.5*dt, L0_Cl->mat(), DIFFERENT_NONZERO_PATTERN);
	MatMult(rhsLinSys->mat(), as_type<const dolfin::PETScVector>(Clfuncs[0]->vector())->vec(), bLinSys->vec());

	VecAXPY(bLinSys->vec(), 1, fStar->vec());

	PetscBarrier(NULL);

	bLinSys->update_ghost_values();

	//solve
	NPSolver->set_operator(*ALinSys);
	NPSolver->solve(*Clfuncs[1]->vector(), *bLinSys);

	auto ff_Cl = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/Cl Concentration.pvd");
	ff_Cl->operator<<(*Clfuncs[0]);*/

	//pH
	auto pH = std::make_shared<dolfin::Function>(Vh);

	pH_Compute(*OHfuncs[0], *pH, false);

	auto ff_pH = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/pH.pvd");
	ff_pH->operator<<(*pH);

	//parameters storage
	auto ff_l = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/l.pvd");
	ff_l->operator<<(*std::make_shared<dolfin::Function>(Vh, Vec_l));
	auto ff_teta = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/teta.pvd");
	ff_teta->operator<<(*std::make_shared<dolfin::Function>(Vh, Vec_teta));
	auto ff_eps = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/eps.pvd");
	ff_eps->operator<<(*std::make_shared<dolfin::Function>(Vh, Vec_eps));

	//update step

	//poisson
	DBCs[0].set_value(std::make_shared<dolfin::Function>(Vh, BoundaryPhi));
	DBCs[1].set_value(std::make_shared<dolfin::Function>(Vh, BoundaryPhi));

	//Nernst-Planck
	*(Mgfuncs[0]->vector()) = *(Mgfuncs[1]->vector());
	//fracture
	*(sMgfuncs[0]->vector()) = *(sMgfuncs[1]->vector());
	*(OHfuncs[0]->vector()) = *(OHfuncs[1]->vector());
	*(Hfuncs[0]->vector()) = *(Hfuncs[1]->vector());
	/**(Nafuncs[0]->vector()) = *(Nafuncs[1]->vector());
	*(Clfuncs[0]->vector()) = *(Clfuncs[1]->vector());*/

	VecCopy(b1_Mg->vec(), b0_Mg->vec());
	VecCopy(b1_OH->vec(), b0_OH->vec());
	//VecCopy(b1_H->vec(), b0_H->vec());

	//Printing results
	//Poisson
	PetscScalar minval = as_type<const dolfin::PETScVector>(EFfuncs[0]->vector())->min();
	PetscScalar maxval = as_type<const dolfin::PETScVector>(EFfuncs[0]->vector())->max();

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
	
	//fracture
	minval = as_type<const dolfin::PETScVector>(sMgfuncs[1]->vector())->min();
	maxval = as_type<const dolfin::PETScVector>(sMgfuncs[1]->vector())->max();

	if(prcID==0) {
		std::cout<<"Solver for diffusion (Mg) is set to: "<<PSolverMethod<<std::endl;
		std::cout<<"Min sMg:"<<minval<<std::endl;
		std::cout<<"Max sMg:"<<maxval<<std::endl<<std::endl;
	}

	ff_fMg->operator<<(*sMgfuncs[1]);

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
	/*minval = as_type<const dolfin::PETScVector>(Nafuncs[1]->vector())->min();
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

	ff_Cl->operator<<(*Clfuncs[1]);*/

	//pH
	pH_Compute(*OHfuncs[1], *pH, false);
	ff_pH->operator<<(*pH);

	//time
	t = dt + t;
//std::cin>>s;
	//24 hour simulation
	for (std::size_t i=1; i<=totalsteps; i = i + 1) {//totalsteps

		//Poisson
		//kappa_Compute({2, -1, 1, 1, -1}, {0.71e-9, 5.27e-9, 9.31e-9, 1.33e-9, 2.03e-9}, {*Mgfuncs[0], *OHfuncs[0], *Hfuncs[0], *Nafuncs[0], *Clfuncs[0]}, *kappa);

		PetscBarrier(NULL);

		//myFormAssigner(*a, {"kappa"}, {kappa});
		//myFormAssigner(*L, {"kappa"}, {kappa});
		myLinearSystemAssembler(*a, *L, DBCs, *A_P, *b_P);

		PetscBarrier(NULL);

		PSolver->set_operator(*A_P);
		PSolver->solve(*EFfuncs[0]->vector(), *b_P);

		//Mg
		iMg(DOFsSetOnMgElectrode, t, dt, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), 0.55, 0.4, 1e-7, Ii->vec(), BoundaryPhi->vec(), PhiPolData, MgPolData, Vec_l->vec(), Vec_teta->vec(), Vec_eps->vec());//A/m^2

		myFormAssigner(*A, {"Di", "zi"}, Mgconsts);
		myFormAssigner(*A, {"phi"}, {EFfuncs[0]});
		myFormAssigner(*f, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, Ii->operator*(-1*5.182045e-6))});

		PetscBarrier(NULL);

		myLinearSystemAssembler(*A, {}, *A_NP);
		myLinearSystemAssembler(*f, {}, *b1_Mg);

		AFC_D_Compute(A_NP->mat(), D1_Mg->mat());

		AFC_EfStar_Compute(A_ML->mat(), A_MC->mat(), D1_Mg->mat(), D0_Mg->mat(), L0_Mg->mat(), as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), b0_Mg->vec(), dt, rij->mat(), fStar->vec(), false);

		PetscBarrier(NULL);

		//Linear system
		MatCopy(A_ML->mat(), ALinSys->mat(), DIFFERENT_NONZERO_PATTERN);
		//L1
		MatAXPY(ALinSys->mat(), 0.5*dt, A_NP->mat(), DIFFERENT_NONZERO_PATTERN);
		MatAXPY(ALinSys->mat(), 0.5*dt, D1_Mg->mat(), DIFFERENT_NONZERO_PATTERN);

		MatCopy(A_ML->mat(), rhsLinSys->mat(), DIFFERENT_NONZERO_PATTERN);
		MatAXPY(rhsLinSys->mat(), -0.5*dt, L0_Mg->mat(), DIFFERENT_NONZERO_PATTERN);
		MatMult(rhsLinSys->mat(), as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), bLinSys->vec());

		VecAXPY(bLinSys->vec(), 0.5*dt, b0_Mg->vec());
		VecAXPY(bLinSys->vec(), 0.5*dt, b1_Mg->vec());
		VecAXPY(bLinSys->vec(), 1, fStar->vec());

		PetscBarrier(NULL);

		bLinSys->update_ghost_values();

		//updating L0
		MatCopy(A_NP->mat(), L0_Mg->mat(), DIFFERENT_NONZERO_PATTERN);
		MatAXPY(L0_Mg->mat(), 1, D1_Mg->mat(), DIFFERENT_NONZERO_PATTERN);

		NPSolver->set_operator(*ALinSys);
		NPSolver->solve(*Mgfuncs[1]->vector(), *bLinSys);

		//fracture
		myFormAssigner(*LS, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, Ii->operator*(-1*5.182045e-6))});
		myFormAssigner(*LM, {"ci0"}, {sMgfuncs[0]});
	
		PetscBarrier(NULL);
	
		myLinearSystemAssembler(*LM, {}, *b_Diffusion);
		MatMult(AS_Diffusion->mat(), as_type<const dolfin::PETScVector>(sMgfuncs[0]->vector())->vec(), bLinSys->vec());
		bLinSys->operator*=(-0.5*dt);
		VecAXPY(bLinSys->vec(), 1, b_Diffusion->vec());
		VecAXPY(bLinSys->vec(), 0.5*dt, IMg0_Diffusion->vec());
	
		PetscBarrier(NULL);
	
		myLinearSystemAssembler(*LS, {}, *IMg0_Diffusion);//IMg1 which is next step IMg0
	
		PetscBarrier(NULL);
	
		VecAXPY(bLinSys->vec(), 0.5*dt, IMg0_Diffusion->vec());

		bLinSys->update_ghost_values();

		PetscBarrier(NULL);
	
		PSolver->set_operator(*A_Diffusion);
		PSolver->solve(*sMgfuncs[1]->vector(), *bLinSys);

		//OH
		iOH(DOFsSetOnAlElectrode, t, dt, as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), 0.55, 0.4, 1e-7, Ii->vec(), BoundaryPhi->vec(), PhiPolData, AlPolData, Vec_l->vec(), Vec_teta->vec(), Vec_eps->vec());//A/m^2

		myFormAssigner(*A, {"Di", "zi"}, OHconsts);
		myFormAssigner(*f, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, Ii->operator*(-1*1.0364e-5))});

		PetscBarrier(NULL);

		myLinearSystemAssembler(*A, {}, *A_NP);
		myLinearSystemAssembler(*f, {}, *b1_OH);

		PetscBarrier(NULL);

		AFC_D_Compute(A_NP->mat(), D1_OH->mat());

		AFC_EfStar_Compute(A_ML->mat(), A_MC->mat(), D1_OH->mat(), D0_OH->mat(), L0_OH->mat(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), b0_OH->vec(), dt, rij->mat(), fStar->vec(), false);

		PetscBarrier(NULL);

		//Linear system
		MatCopy(A_ML->mat(), ALinSys->mat(), DIFFERENT_NONZERO_PATTERN);
		//L1
		MatAXPY(ALinSys->mat(), 0.5*dt, A_NP->mat(), DIFFERENT_NONZERO_PATTERN);
		MatAXPY(ALinSys->mat(), 0.5*dt, D1_OH->mat(), DIFFERENT_NONZERO_PATTERN);

		MatCopy(A_ML->mat(), rhsLinSys->mat(), DIFFERENT_NONZERO_PATTERN);
		MatAXPY(rhsLinSys->mat(), -0.5*dt, L0_OH->mat(), DIFFERENT_NONZERO_PATTERN);//L0=L1
		MatMult(rhsLinSys->mat(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), bLinSys->vec());

		VecAXPY(bLinSys->vec(), 0.5*dt, b0_OH->vec());
		VecAXPY(bLinSys->vec(), 0.5*dt, b1_OH->vec());
		VecAXPY(bLinSys->vec(), 1, fStar->vec());

		PetscBarrier(NULL);

		bLinSys->update_ghost_values();

		//updating L0
		MatCopy(A_NP->mat(), L0_OH->mat(), DIFFERENT_NONZERO_PATTERN);
		MatAXPY(L0_OH->mat(), 1, D1_OH->mat(), DIFFERENT_NONZERO_PATTERN);

		NPSolver->set_operator(*ALinSys);
		NPSolver->solve(*OHfuncs[1]->vector(), *bLinSys);

		//H
		myFormAssigner(*A, {"Di", "zi"}, Hconsts);
		//myFormAssigner(*f, {"Ii"}, {zerofunc});

		PetscBarrier(NULL);

		myLinearSystemAssembler(*A, {}, *A_NP);
		//myLinearSystemAssembler(*f, {}, *b1_H);

		PetscBarrier(NULL);

		AFC_D_Compute(A_NP->mat(), D1_H->mat());

		AFC_EfStar_Compute(A_ML->mat(), A_MC->mat(), D1_H->mat(), D0_H->mat(), L0_H->mat(), as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(zerofunc->vector())->vec(), dt, rij->mat(), fStar->vec());

		PetscBarrier(NULL);

		//Linear system
		MatCopy(A_ML->mat(), ALinSys->mat(), DIFFERENT_NONZERO_PATTERN);
		//L1
		MatAXPY(ALinSys->mat(), 0.5*dt, A_NP->mat(), DIFFERENT_NONZERO_PATTERN);
		MatAXPY(ALinSys->mat(), 0.5*dt, D1_H->mat(), DIFFERENT_NONZERO_PATTERN);

		MatCopy(A_ML->mat(), rhsLinSys->mat(), DIFFERENT_NONZERO_PATTERN);
		MatAXPY(rhsLinSys->mat(), -0.5*dt, L0_H->mat(), DIFFERENT_NONZERO_PATTERN);//L0=L1
		MatMult(rhsLinSys->mat(), as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), bLinSys->vec());

		//VecAXPY(bLinSys->vec(), 0.5*dt, b0_OH->vec());
		//VecAXPY(bLinSys->vec(), 0.5*dt, b1_OH->vec());
		VecAXPY(bLinSys->vec(), 1, fStar->vec());

		PetscBarrier(NULL);

		bLinSys->update_ghost_values();

		//updating L0
		MatCopy(A_NP->mat(), L0_H->mat(), DIFFERENT_NONZERO_PATTERN);
		MatAXPY(L0_H->mat(), 1, D1_H->mat(), DIFFERENT_NONZERO_PATTERN);

		NPSolver->set_operator(*ALinSys);
		NPSolver->solve(*Hfuncs[1]->vector(), *bLinSys);

		WaterDissociation(as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), RHOH->vec());
		VecSetOnLocalDOFs(DOFsSetOnAlElectrode, RHOH->vec(), 0);
		VecSetOnLocalDOFs(DOFsSetOnMgElectrode, RHOH->vec(), 0);
//ff_rhoh.operator<<(*std::make_shared<dolfin::Function>(Vh, RHOH));
		VecAXPY(as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->vec(), dt, RHOH->vec());
		VecGhostUpdateBegin(as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->vec(), INSERT_VALUES, SCATTER_FORWARD);
		VecGhostUpdateEnd(as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->vec(), INSERT_VALUES, SCATTER_FORWARD);
		VecAXPY(as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->vec(), dt, RHOH->vec());
		VecGhostUpdateBegin(as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->vec(), INSERT_VALUES, SCATTER_FORWARD);
		VecGhostUpdateEnd(as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->vec(), INSERT_VALUES, SCATTER_FORWARD);

		/*if (i%s == 0) {
			//Na
			myFormAssigner(*A, {"Di", "zi"}, Naconsts);

			PetscBarrier(NULL);

			myLinearSystemAssembler(*A, {}, *A_NP);

			PetscBarrier(NULL);

			AFC_D_Compute(A_NP->mat(), D1_Na->mat());

			AFC_EfStar_Compute(A_ML->mat(), A_MC->mat(), D1_Na->mat(), D0_Na->mat(), L0_Na->mat(), as_type<const dolfin::PETScVector>(Nafuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(zerofunc->vector())->vec(), dt, rij->mat(), fStar->vec());

			PetscBarrier(NULL);

			//Linear system
			MatCopy(A_ML->mat(), ALinSys->mat(), DIFFERENT_NONZERO_PATTERN);
			//L1
			MatAXPY(ALinSys->mat(), 0.5*s*dt, A_NP->mat(), DIFFERENT_NONZERO_PATTERN);
			MatAXPY(ALinSys->mat(), 0.5*s*dt, D1_Na->mat(), DIFFERENT_NONZERO_PATTERN);

			MatCopy(A_ML->mat(), rhsLinSys->mat(), DIFFERENT_NONZERO_PATTERN);
			MatAXPY(rhsLinSys->mat(), -0.5*s*dt, L0_Na->mat(), DIFFERENT_NONZERO_PATTERN);
			MatMult(rhsLinSys->mat(), as_type<const dolfin::PETScVector>(Nafuncs[0]->vector())->vec(), bLinSys->vec());

			VecAXPY(bLinSys->vec(), 1, fStar->vec());

			PetscBarrier(NULL);

			bLinSys->update_ghost_values();

			//updating L0
			MatCopy(A_NP->mat(), L0_Na->mat(), DIFFERENT_NONZERO_PATTERN);
			MatAXPY(L0_Na->mat(), 1, D1_Na->mat(), DIFFERENT_NONZERO_PATTERN);

			NPSolver->set_operator(*ALinSys);
			NPSolver->solve(*Nafuncs[1]->vector(), *bLinSys);

			//Cl
			myFormAssigner(*A, {"Di", "zi"}, Clconsts);

			PetscBarrier(NULL);

			myLinearSystemAssembler(*A, {}, *A_NP);

			PetscBarrier(NULL);

			AFC_D_Compute(A_NP->mat(), D1_Cl->mat());

			AFC_EfStar_Compute(A_ML->mat(), A_MC->mat(), D1_Cl->mat(), D0_Cl->mat(), L0_Cl->mat(), as_type<const dolfin::PETScVector>(Clfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(zerofunc->vector())->vec(), dt, rij->mat(), fStar->vec());

			PetscBarrier(NULL);

			//Linear system
			MatCopy(A_ML->mat(), ALinSys->mat(), DIFFERENT_NONZERO_PATTERN);
			//L1
			MatAXPY(ALinSys->mat(), 0.5*s*dt, A_NP->mat(), DIFFERENT_NONZERO_PATTERN);
			MatAXPY(ALinSys->mat(), 0.5*s*dt, D1_Cl->mat(), DIFFERENT_NONZERO_PATTERN);

			MatCopy(A_ML->mat(), rhsLinSys->mat(), DIFFERENT_NONZERO_PATTERN);
			MatAXPY(rhsLinSys->mat(), -0.5*s*dt, L0_Cl->mat(), DIFFERENT_NONZERO_PATTERN);
			MatMult(rhsLinSys->mat(), as_type<const dolfin::PETScVector>(Clfuncs[0]->vector())->vec(), bLinSys->vec());

			VecAXPY(bLinSys->vec(), 1, fStar->vec());

			PetscBarrier(NULL);

			bLinSys->update_ghost_values();

			//updating L0
			MatCopy(A_NP->mat(), L0_Cl->mat(), DIFFERENT_NONZERO_PATTERN);
			MatAXPY(L0_Cl->mat(), 1, D1_Cl->mat(), DIFFERENT_NONZERO_PATTERN);

			NPSolver->set_operator(*ALinSys);
			NPSolver->solve(*Clfuncs[1]->vector(), *bLinSys);

			//update step
			*(Nafuncs[0]->vector()) = *(Nafuncs[1]->vector());
			*(Clfuncs[0]->vector()) = *(Clfuncs[1]->vector());
			MatCopy(D1_Na->mat(), D0_Na->mat(), DIFFERENT_NONZERO_PATTERN);
			MatCopy(D1_Cl->mat(), D0_Cl->mat(), DIFFERENT_NONZERO_PATTERN);
		}*/

		//update step
		//poisson
		DBCs[0].set_value(std::make_shared<dolfin::Function>(Vh, BoundaryPhi));
		DBCs[1].set_value(std::make_shared<dolfin::Function>(Vh, BoundaryPhi));

		//Nernst-Planck
		*(Mgfuncs[0]->vector()) = *(Mgfuncs[1]->vector());
		*(sMgfuncs[0]->vector()) = *(sMgfuncs[1]->vector());
		*(OHfuncs[0]->vector()) = *(OHfuncs[1]->vector());
		*(Hfuncs[0]->vector()) = *(Hfuncs[1]->vector());

		VecCopy(b1_Mg->vec(), b0_Mg->vec());
		VecCopy(b1_OH->vec(), b0_OH->vec());
		//VecCopy(b1_H->vec(), b0_H->vec());

		MatCopy(D1_Mg->mat(), D0_Mg->mat(), DIFFERENT_NONZERO_PATTERN);
		MatCopy(D1_H->mat(), D0_H->mat(), DIFFERENT_NONZERO_PATTERN);
		MatCopy(D1_OH->mat(), D0_OH->mat(), DIFFERENT_NONZERO_PATTERN);

		//Printing results
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

		//fracture
		minval = as_type<const dolfin::PETScVector>(sMgfuncs[1]->vector())->min();
		maxval = as_type<const dolfin::PETScVector>(sMgfuncs[1]->vector())->max();

		if(prcID==0) {
			std::cout<<"Solver for diffusion (Mg) is set to: "<<PSolverMethod<<std::endl;
			std::cout<<"Min sMg:"<<minval<<std::endl;
			std::cout<<"Max sMg:"<<maxval<<std::endl<<std::endl;
		}

		ff_fMg->operator<<(*sMgfuncs[1]);

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

		/*if (i%s == 0) {
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
		}*/

		if (i%s == 0) {
			//pH
			pH_Compute(*OHfuncs[0], *pH, false);
			ff_pH->operator<<(*pH);
			ff_l->operator<<(*std::make_shared<dolfin::Function>(Vh, Vec_l));
			ff_teta->operator<<(*std::make_shared<dolfin::Function>(Vh, Vec_teta));
			ff_eps->operator<<(*std::make_shared<dolfin::Function>(Vh, Vec_eps));
		}

		t = dt + t;

		//storage step
		if (i == (1*3600*10)) {
			s = 100;
		}
		if (i == (2*3600*10)) {
			s = 300;
		}
		//if (i == (12*3600*2))
			//s = 60;
//if (i==101)break;
		PetscBarrier(NULL);
	}

	if(prcID==0) {
		time(&end);
		std::cout<<"total exc time: "<<double(end-start)<<std::endl;
	}

	PetscBarrier(NULL);

	//memory release in chronological order
	//weak forms
	a.reset();L.reset();MC.reset();A.reset();f.reset();

	//vectors
	DBCs.clear(); DBCs.shrink_to_fit();
	PhiPolData.clear();
	PhiPolData.shrink_to_fit();
	MgPolData.clear();
	MgPolData.shrink_to_fit();
	AlPolData.clear();
	AlPolData.shrink_to_fit();
	DOFsSetOnAlElectrode.clear();
	DOFsSetOnAlElectrode.shrink_to_fit();
	DOFsSetOnMgElectrode.clear();
	DOFsSetOnMgElectrode.shrink_to_fit();

	//shared functions or vector of shared functions
	zerofunc.reset();
	/*kappa.reset()*/;SharedTypeVectorDestructor(EFfuncs);
	pH.reset();
	SharedTypeVectorDestructor(Mgfuncs);SharedTypeVectorDestructor(Mgconsts);
	SharedTypeVectorDestructor(OHfuncs);SharedTypeVectorDestructor(OHconsts);
	SharedTypeVectorDestructor(Hfuncs);SharedTypeVectorDestructor(Hconsts);
	/*SharedTypeVectorDestructor(Nafuncs);SharedTypeVectorDestructor(Naconsts);
	SharedTypeVectorDestructor(Clfuncs);SharedTypeVectorDestructor(Clconsts);*/

	//mesh and function space
	Vh.reset();mesh.reset();

	//shared matrices
	A_P.reset();
	A_ML.reset();A_MC.reset();A_NP.reset();rij.reset();ALinSys.reset();rhsLinSys.reset();
	L0_Mg.reset();D0_Mg.reset();D1_Mg.reset();
	L0_OH.reset();D0_OH.reset();D1_OH.reset();
	L0_H.reset();D0_H.reset();D1_H.reset();
	/*L0_Na.reset();D0_Na.reset();D1_Na.reset();
	L0_Cl.reset();D0_Cl.reset();D1_Cl.reset();*/

	//shared vectors
	b_P.reset();BoundaryPhi.reset();
	Ii.reset();fStar.reset();bLinSys.reset();Vec_l.reset();Vec_teta.reset();Vec_eps.reset();
	b0_Mg.reset();b1_Mg.reset();
	b0_OH.reset();b1_OH.reset();RHOH.reset();
	//b0_H.reset();b1_H.reset();

	//shared solvers
	PSolver.reset();
	NPSolver.reset();

	//petsc solvers
	//KSPDestroy(&myNPSolver); //NPSolver seems to take the exact pointer and destruct it on reset
	PCDestroy(&myNPConditioner);

	//shared file streams
	ff_P.reset();
	ff_Mg.reset();
	ff_OH.reset();
	ff_H.reset();
	/*ff_Na.reset();
	ff_Cl.reset();*/
	ff_pH.reset();
	ff_l.reset();
	ff_teta.reset();
	ff_eps.reset();

	PetscFinalize();

	return 0;
}
