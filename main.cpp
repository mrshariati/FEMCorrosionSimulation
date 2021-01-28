#include <dolfin.h>
#include <fstream>
#include "src/GeneralTools.cpp"
#include "src/FEMTools.cpp"
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

	//----The points that are needed in a corrosion rectangle model----
	std::vector<dolfin::Point> p_d;//_d stands for domain

	RectPointsGenerator(0.04, 0.5, p_d);
	p_d.push_back(p_d[0] + (p_d[3]-p_d[0])*0.3 + (p_d[3]-p_d[0])*0.01);
	p_d.push_back(p_d[0] + (p_d[3]-p_d[0])*0.3);

	//----Reading the rectangle mesh from points and specifying the boundaries----
	auto mesh = std::make_shared<dolfin::Mesh>(PETSC_COMM_WORLD, "mesh/Gapmesh.xml");

	std::vector<std::shared_ptr<dolfin::SubDomain>> b_d;//boundaries of domain
	b_d.push_back(std::make_shared<RectBorderLine>(p_d[0], p_d[1]));
	b_d.push_back(std::make_shared<RectBorderLine>(p_d[1], p_d[2]));
	b_d.push_back(std::make_shared<RectBorderLine>(p_d[2], p_d[3]));
	b_d.push_back(std::make_shared<RectBorderLine>(p_d[3], p_d[4]));
	b_d.push_back(std::make_shared<RectBorderLine>(p_d[5], p_d[0]));
	b_d.push_back(std::make_shared<RectBorderLine>(p_d[4], p_d[5]));//only valid for Gapmesh

	PetscBarrier(NULL);

	//--Points information are no longer needed--
	p_d.clear(); p_d.shrink_to_fit();

	//----Creating the variational formulations in the corrosion model----
	auto Vh = std::make_shared<StiffnessMatrix::FunctionSpace>(mesh);

	auto a_p = std::make_shared<Poisson::BilinearForm>(Vh, Vh);//_p stands for Poisson

	auto MC_np = std::make_shared<MassMatrix::BilinearForm>(Vh, Vh);//_np stands for Nernst-Planck
	auto A_np = std::make_shared<StiffnessMatrix::BilinearForm>(Vh, Vh);
	auto I_np = std::make_shared<StiffnessMatrix::LinearForm>(Vh);

	PetscBarrier(NULL);

	//----Creating the functions that we use in variational formulations----

	//Electrical field
	std::vector<std::shared_ptr<dolfin::Function>> ElectricFieldfuncs;//keeping all the functions for electric field in a vector
	std::vector<bool> isconst = {0};
	Vector_of_NonConstFunctionGenerator(Vh, ElectricFieldfuncs, isconst, {});

	//Mg
	std::vector<std::shared_ptr<dolfin::Function>> Mgfuncs;//keeping all the functions for Magnesium concentration in a vector
	isconst.clear(); isconst.shrink_to_fit();
	isconst = {1, 0};
	std::vector<double> constvalue = {1e-15};
	Vector_of_NonConstFunctionGenerator(Vh, Mgfuncs, isconst, constvalue);

	std::vector<std::shared_ptr<dolfin::GenericFunction>> Mgconsts;//keeping all the constants for Magnesium concentration in a vector
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {0.71e-9, 2};
	Vector_of_ConstFunctionGenerator(Vh, Mgconsts, constvalue);

	//OH
	std::vector<std::shared_ptr<dolfin::Function>> OHfuncs;//keeping all the functions for Hydroxide concentration in a vector
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {1e-4};
	Vector_of_NonConstFunctionGenerator(Vh, OHfuncs, isconst, constvalue);

	std::vector<std::shared_ptr<dolfin::GenericFunction>> OHconsts;//keeping all the constants for Hydroxide concentration in a vector
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {5.27e-9, -1};
	Vector_of_ConstFunctionGenerator(Vh, OHconsts, constvalue);

	//H
	std::vector<std::shared_ptr<dolfin::Function>> Hfuncs;//keeping all the functions for proton concentration in a vector
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {1e-4};
	Vector_of_NonConstFunctionGenerator(Vh, Hfuncs, isconst, constvalue);

	std::vector<std::shared_ptr<dolfin::GenericFunction>> Hconsts;//keeping all the constants for proton concentration in a vector
	constvalue.clear(); constvalue.shrink_to_fit();
	constvalue = {9.31e-9, 1};
	Vector_of_ConstFunctionGenerator(Vh, Hconsts, constvalue);

	//--Function generation intermediate variables are no longer needed--
	isconst.clear(); isconst.shrink_to_fit();
	constvalue.clear(); constvalue.shrink_to_fit();

	//----Deciesion on type and assignment of the boudaries (Neumann, Dirichlet, ...)----
	std::vector<dolfin::DirichletBC> DBC_p;//DBC stands for Dirichlet Boundary Condition
	myDirichletBCGenerator(Vh, {std::make_shared<dolfin::Constant>(-1.517), std::make_shared<dolfin::Constant>(-0.595)}, {b_d[3], b_d[4]}, DBC_p);//to assign values to a specific domain boundary

	std::vector<std::size_t> NodesOnAlElectrode;
	std::vector<std::size_t> NodesOnMgElectrode;
	std::vector<std::size_t> DOFsSetOnAlElectrode;
	std::vector<std::size_t> DOFsSetOnMgElectrode;
	NodesIndex_on_Subdomain(b_d[4], mesh, NodesOnAlElectrode);//to find the nodes on a specific domain boundary
	NodesIndex_on_Subdomain(b_d[3], mesh, NodesOnMgElectrode);
	NodesIndices2LocalDOFs(*Vh, *mesh, NodesOnAlElectrode, DOFsSetOnAlElectrode);//to find the dof indices of a set of nodes in mesh
	NodesIndices2LocalDOFs(*Vh, *mesh, NodesOnMgElectrode, DOFsSetOnMgElectrode);

	PetscBarrier(NULL);

	//--Nodes indices and domain information is no longer needed--
	NodesOnAlElectrode.clear(); NodesOnAlElectrode.shrink_to_fit();
	NodesOnMgElectrode.clear(); NodesOnMgElectrode.shrink_to_fit();
	SharedTypeVectorDestructor(b_d);

	//----Assembling the final linear systems, solve and storing the solution----
	//Poisson
	auto LinSysLhs_p = std::make_shared<dolfin::PETScMatrix>(PETSC_COMM_WORLD);//A in Ax=b
	auto LinSysRhs_p = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(ElectricFieldfuncs[0]->vector())));//b in Ax=b
	LinSysRhs_p->zero();
	auto Phibar = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(ElectricFieldfuncs[0]->vector())));//a vector for Electric field on the boundary
	Phibar->zero();
	std::string SolverMethod_p = "mumps";
	auto Solver_p = std::make_shared<dolfin::PETScLUSolver>(PETSC_COMM_WORLD, SolverMethod_p);

	PetscBarrier(NULL);
	
	Weak2Matrix(*a_p, DBC_p, *LinSysLhs_p, *LinSysRhs_p);

	//listing available solvers in your machine
	if(prcID==0) {
		list_krylov_solver_methods();
		list_krylov_solver_preconditioners();
		list_linear_solver_methods();
		list_lu_solver_methods();
	}

	PetscBarrier(NULL);

	Solver_p->set_operator(*LinSysLhs_p);
	Solver_p->solve(*ElectricFieldfuncs[0]->vector(), *LinSysRhs_p);

	//Nernst-Planck
	//Common part
	auto MCmatrix_np = std::make_shared<dolfin::PETScMatrix>(PETSC_COMM_WORLD);//matrix of respective weak formulation
	Weak2Matrix(*MC_np, {}, *MCmatrix_np);
	auto LinSysLhs_np = std::make_shared<dolfin::PETScMatrix>(*MCmatrix_np);
	LinSysLhs_np->zero();
	auto LinSysRhs_np = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	LinSysRhs_np->zero();
	auto tmpmatrix_np = std::make_shared<dolfin::PETScMatrix>(*MCmatrix_np);//This will assist us in calculations and subsitutions
	tmpmatrix_np->zero();
	auto MLmatrix_np = std::make_shared<dolfin::PETScMatrix>(*MCmatrix_np);//diagonal form of mass matrix
	auto Amatrix_np = std::make_shared<dolfin::PETScMatrix>(*MCmatrix_np);
	auto Ivector_np = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	Ivector_np->zero();
	auto alphamatrix_np = std::make_shared<dolfin::PETScMatrix>(*MCmatrix_np);
	auto theta = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));//affected surface
	theta->zero();
	auto epsln = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));//porosity
	epsln->zero();
	auto ldep = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));//deposit layer thickness
	ldep->zero();
	ldep->operator=(1e-7);//l0_dep

	double t = 0;
	double dt = 1e-1;
	double dt_Adaptive = dt;
	std::size_t StroringStep = 1;//every 1s
	std::size_t SimulationTime = 12*3600;//unit is seconds

	PetscBarrier(NULL);

	AFC_ML_Compute(MCmatrix_np->mat(), MLmatrix_np->mat());

	//Mg
	auto A0_Mg = std::make_shared<dolfin::PETScMatrix>(*MCmatrix_np);
	auto G0_Mg = std::make_shared<dolfin::PETScMatrix>(*MCmatrix_np);
	auto G1_Mg = std::make_shared<dolfin::PETScMatrix>(*MCmatrix_np);
	auto I0_Mg = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	I0_Mg->zero();
	auto I1_Mg = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	I1_Mg->zero();

	iMg(DOFsSetOnMgElectrode, Phibar->vec(), as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), theta->vec(), epsln->vec(), ldep->vec(), 0.4, 0.55, t, dt, Ivector_np->vec());//This is the current of elctrons and change in Mg2+ concentration is half of it

	WeakAssign(*A_np, {"Di", "zi","phi"}, {Mgconsts[0], Mgconsts[1], ElectricFieldfuncs[0]});
	WeakAssign(*I_np, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, Ivector_np->operator*(1/(2*96487.3329*2.76e-2)))});//change in Mg2+ concentration

	PetscBarrier(NULL);

	Weak2Matrix(*A_np, {}, *Amatrix_np);
	Weak2Matrix(*I_np, {}, *I1_Mg);//assumed I0_Mg=0

	PetscBarrier(NULL);

	AFC_D_Compute(Amatrix_np->mat(), G1_Mg->mat());
	MatCopy(Amatrix_np->mat(), A0_Mg->mat(), DIFFERENT_NONZERO_PATTERN);//At the begining A1_Mg=A0_Mg, we don't keep A1 independently but update the A0 at the end of calculations
	MatCopy(G1_Mg->mat(), G0_Mg->mat(), DIFFERENT_NONZERO_PATTERN);//At the begining Gamma1_Mg=Gamma0_Mg

	nonLinAFC_alpha_Compute(MLmatrix_np->mat(), MCmatrix_np->mat(), G1_Mg->mat(), G0_Mg->mat(), A0_Mg->mat(), as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), I0_Mg->vec(), dt, tmpmatrix_np->mat(), alphamatrix_np->mat());//the matrix r is not applicable for non-Linear AFC

	PetscBarrier(NULL);

	//Construction of Linear system
	nonLinAFC_LinSys_Construct(MLmatrix_np->mat(), MCmatrix_np->mat(), G1_Mg->mat(), G0_Mg->mat(), Amatrix_np->mat(), A0_Mg->mat(), alphamatrix_np->mat(), tmpmatrix_np->mat(), as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), I1_Mg->vec(), I0_Mg->vec(), dt, LinSysLhs_np->mat(), LinSysRhs_np->vec());

	//Setting up solver for _np
	KSP Solver_np;
	PC Conditioner_np;
	KSPCreate(PETSC_COMM_WORLD, &Solver_np);
	KSPSetType(Solver_np, KSPGMRES);
	KSPSetInitialGuessNonzero(Solver_np, PETSC_TRUE);
	KSPGetPC(Solver_np, &Conditioner_np);
	PCSetType(Conditioner_np, PCJACOBI);
	KSPSetUp(Solver_np);

	std::string SolverMethod_np = "gmres with jacobi";
	auto SolverWrapper_np = std::make_shared<dolfin::PETScKrylovSolver>(Solver_np);

	SolverWrapper_np->set_operator(*LinSysLhs_np);
	SolverWrapper_np->solve(*Mgfuncs[1]->vector(), *LinSysRhs_np);
	
	//to apply adaptive time step
	AFC_L_Compute(A0_Mg->mat(), G0_Mg->mat(), tmpmatrix_np->mat());
	dt_Adaptive = 0.05*AFC_dt_Compute(MLmatrix_np->mat(), tmpmatrix_np->mat());

	//OH
	auto A0_OH = std::make_shared<dolfin::PETScMatrix>(*MCmatrix_np);
	auto G0_OH = std::make_shared<dolfin::PETScMatrix>(*MCmatrix_np);
	auto G1_OH = std::make_shared<dolfin::PETScMatrix>(*MCmatrix_np);
	auto I0_OH = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	I0_OH->zero();
	auto I1_OH = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	I1_OH->zero();
	auto RHOH = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	RHOH->zero();

	Ivector_np->operator*=((96487.3329*1.2e-2)/(2*96487.3329*2.76e-2));//we want two different multiplication on Al and Mg. Because currently on Al is 0 this way this coefficient neutralizes later
	iOH(DOFsSetOnAlElectrode, Phibar->vec(), as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), theta->vec(), epsln->vec(), ldep->vec(), 0.4, 0.55, t, dt, Ivector_np->vec());//This is the current of elctrons and change in OH- concentration is the same

	WeakAssign(*A_np, {"Di", "zi"}, OHconsts);// electric field has already assigned
	WeakAssign(*I_np, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, Ivector_np->operator*(1/(96487.3329*1.2e-2)))});//change in OH- concentration

	PetscBarrier(NULL);

	Weak2Matrix(*A_np, {}, *Amatrix_np);
	Weak2Matrix(*I_np, {}, *I1_OH);//assumed I0_OH=0

	PetscBarrier(NULL);

	AFC_D_Compute(Amatrix_np->mat(), G1_OH->mat());
	MatCopy(Amatrix_np->mat(), A0_OH->mat(), DIFFERENT_NONZERO_PATTERN);//At the begining A1_OH=A0_OH, we don't keep A1 independently but update the A0 at the end of calculations
	MatCopy(G1_OH->mat(), G0_OH->mat(), DIFFERENT_NONZERO_PATTERN);//At the begining Gamma1_OH=Gamma0_OH

	nonLinAFC_alpha_Compute(MLmatrix_np->mat(), MCmatrix_np->mat(), G1_OH->mat(), G0_OH->mat(), A0_OH->mat(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), I0_OH->vec(), dt, tmpmatrix_np->mat(), alphamatrix_np->mat());//the matrix r is not applicable for non-Linear AFC

	PetscBarrier(NULL);

	//Construction of Linear system
	nonLinAFC_LinSys_Construct(MLmatrix_np->mat(), MCmatrix_np->mat(), G1_OH->mat(), G0_OH->mat(), Amatrix_np->mat(), A0_OH->mat(), alphamatrix_np->mat(), tmpmatrix_np->mat(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), I1_OH->vec(), I0_OH->vec(), dt, LinSysLhs_np->mat(), LinSysRhs_np->vec());

	PetscBarrier(NULL);

	//solve
	SolverWrapper_np->set_operator(*LinSysLhs_np);
	SolverWrapper_np->solve(*OHfuncs[1]->vector(), *LinSysRhs_np);

	//to apply adaptive time step
	AFC_L_Compute(A0_OH->mat(), G0_OH->mat(), tmpmatrix_np->mat());
	dt_Adaptive = std::min(dt_Adaptive, AFC_dt_Compute(MLmatrix_np->mat(), tmpmatrix_np->mat()));

	//H
	auto A0_H = std::make_shared<dolfin::PETScMatrix>(*MCmatrix_np);
	auto G0_H = std::make_shared<dolfin::PETScMatrix>(*MCmatrix_np);
	auto G1_H = std::make_shared<dolfin::PETScMatrix>(*MCmatrix_np);
	//H produced or consumed identical as OH

	WeakAssign(*A_np, {"Di", "zi"}, Hconsts);// electric field has already assigned

	PetscBarrier(NULL);

	Weak2Matrix(*A_np, {}, *Amatrix_np);

	PetscBarrier(NULL);

	AFC_D_Compute(Amatrix_np->mat(), G1_H->mat());
	MatCopy(Amatrix_np->mat(), A0_H->mat(), DIFFERENT_NONZERO_PATTERN);//At the begining A1_H=A0_H, we don't keep A1 independently but update the A0 at the end of calculations
	MatCopy(G1_H->mat(), G0_H->mat(), DIFFERENT_NONZERO_PATTERN);//At the begining Gamma1_H=Gamma0_H

	nonLinAFC_alpha_Compute(MLmatrix_np->mat(), MCmatrix_np->mat(), G1_H->mat(), G0_H->mat(), A0_H->mat(), as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), I0_OH->vec(), dt, tmpmatrix_np->mat(), alphamatrix_np->mat());//the matrix r is not applicable for non-Linear AFC //I0_H is identical to I0_OH

	PetscBarrier(NULL);

	//Construction of Linear system
	nonLinAFC_LinSys_Construct(MLmatrix_np->mat(), MCmatrix_np->mat(), G1_H->mat(), G0_H->mat(), Amatrix_np->mat(), A0_H->mat(), alphamatrix_np->mat(), tmpmatrix_np->mat(), as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), I1_OH->vec(), I0_OH->vec(), dt, LinSysLhs_np->mat(), LinSysRhs_np->vec());//H produced or consumed identical as OH

	//solve
	SolverWrapper_np->set_operator(*LinSysLhs_np);
	SolverWrapper_np->solve(*Hfuncs[1]->vector(), *LinSysRhs_np);

	//to apply adaptive time step
	AFC_L_Compute(A0_H->mat(), G0_H->mat(), tmpmatrix_np->mat());
	dt_Adaptive = std::min(dt_Adaptive, AFC_dt_Compute(MLmatrix_np->mat(), tmpmatrix_np->mat()));//at this point dt_Adaptive is the smallest CFD condition

	PetscBarrier(NULL);

	//Water dissociation determines the amount of hydro-ions to be produced or consumed for keeping balance
	WaterDissociation(as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), RHOH->vec());

	//electrode interface is exempt from water dissociation
	VecSetOnLocalDOFs(DOFsSetOnAlElectrode, RHOH->vec(), 0);
	VecSetOnLocalDOFs(DOFsSetOnMgElectrode, RHOH->vec(), 0);

	//adding to the current solution
	//VecAXPY(as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->vec(), dt, RHOH->vec());
	VecGhostUpdateBegin(as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->vec(), INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->vec(), INSERT_VALUES, SCATTER_FORWARD);
	//VecAXPY(as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->vec(), dt, RHOH->vec());
	VecGhostUpdateBegin(as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->vec(), INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->vec(), INSERT_VALUES, SCATTER_FORWARD);

	//pH
	auto pH = std::make_shared<dolfin::Function>(Vh);

	pH_Compute(*OHfuncs[0], *pH, false);
FunctionFilterAvg(as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->vec(), *Vh, *mesh);
	//storing
	auto StoringStream_p = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/Electric Field.pvd");
	StoringStream_p->operator<<(*ElectricFieldfuncs[0]);	
	auto StoringStream_Mg = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/Mg_Concentration.pvd");
	StoringStream_Mg->operator<<(*Mgfuncs[0]);
	StoringStream_Mg->operator<<(*Mgfuncs[1]);
	auto StoringStream_OH = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/OH_Concentration.pvd");
	StoringStream_OH->operator<<(*OHfuncs[0]);
	StoringStream_OH->operator<<(*OHfuncs[1]);
	auto StoringStream_H = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/H_Concentration.pvd");
	StoringStream_H->operator<<(*Hfuncs[0]);
	StoringStream_H->operator<<(*Hfuncs[1]);
	auto StoringStream_pH = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/pH_Concentration.pvd");
	StoringStream_pH->operator<<(*pH);
	pH_Compute(*OHfuncs[1], *pH, false);
	StoringStream_pH->operator<<(*pH);
	auto StoringStream_theta = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/theta.pvd");
	StoringStream_theta->operator<<(*std::make_shared<dolfin::Function>(Vh, theta));
	auto StoringStream_epsln = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/epsln.pvd");
	StoringStream_epsln->operator<<(*std::make_shared<dolfin::Function>(Vh, epsln));
	auto StoringStream_ldep = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/ldep.pvd");
	StoringStream_ldep->operator<<(*std::make_shared<dolfin::Function>(Vh, ldep));

	//updating
	//poisson boundary (polarization on electrode interface)
	DBC_p[0].set_value(std::make_shared<dolfin::Function>(Vh, Phibar));
	DBC_p[1].set_value(std::make_shared<dolfin::Function>(Vh, Phibar));

	//Nernst-Planck
	*(Mgfuncs[0]->vector()) = *(Mgfuncs[1]->vector());
	*(OHfuncs[0]->vector()) = *(OHfuncs[1]->vector());
	*(Hfuncs[0]->vector()) = *(Hfuncs[1]->vector());

	VecCopy(I1_Mg->vec(), I0_Mg->vec());
	VecCopy(I1_OH->vec(), I0_OH->vec());

	//time
	t = dt + t;
	if(prcID==0) {
		std::cout<<"Time step dt: "<<dt<<std::endl;
	}
	dt = dt_Adaptive;

	//Printing
	//Poisson
	PetscScalar minval = as_type<const dolfin::PETScVector>(ElectricFieldfuncs[0]->vector())->min();
	PetscScalar maxval = as_type<const dolfin::PETScVector>(ElectricFieldfuncs[0]->vector())->max();
	if(prcID==0) {
		std::cout<<"Solver for poisson is set to: "<<SolverMethod_p<<std::endl;
		std::cout<<"Min ElectricField:"<<minval<<std::endl;
		std::cout<<"Max ElectricField:"<<maxval<<std::endl<<std::endl;
		
	}
	//Mg
	minval = as_type<const dolfin::PETScVector>(Mgfuncs[1]->vector())->min();
	maxval = as_type<const dolfin::PETScVector>(Mgfuncs[1]->vector())->max();
	if(prcID==0) {
		std::cout<<"Solver for Nernst-Planck is set to: "<<SolverMethod_np<<std::endl;
		std::cout<<"Min Mg:"<<minval<<std::endl;
		std::cout<<"Max Mg:"<<maxval<<std::endl<<std::endl;
	}
	//OH
	minval = as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->min();
	maxval = as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->max();
	if(prcID==0) {
		std::cout<<"Min OH:"<<minval<<std::endl;
		std::cout<<"Max OH:"<<maxval<<std::endl<<std::endl;
	}
	//H
	minval = as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->min();
	maxval = as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->max();
	if(prcID==0) {
		std::cout<<"Min H:"<<minval<<std::endl;
		std::cout<<"Max H:"<<maxval<<std::endl<<std::endl;
	}

	while (t<=SimulationTime) {
		Weak2Matrix(*a_p, DBC_p, *LinSysLhs_p, *LinSysRhs_p);

		PetscBarrier(NULL);

		Solver_p->set_operator(*LinSysLhs_p);
		Solver_p->solve(*ElectricFieldfuncs[0]->vector(), *LinSysRhs_p);

		//Mg
		iMg(DOFsSetOnMgElectrode, Phibar->vec(), as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), theta->vec(), epsln->vec(), ldep->vec(), 0.4, 0.55, t, dt, Ivector_np->vec());

		WeakAssign(*A_np, {"Di", "zi","phi"}, {Mgconsts[0], Mgconsts[1], ElectricFieldfuncs[0]});
		WeakAssign(*I_np, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, Ivector_np->operator*(1/(2*96487.3329*2.76e-2)))});

		PetscBarrier(NULL);

		Weak2Matrix(*A_np, {}, *Amatrix_np);
		Weak2Matrix(*I_np, {}, *I1_Mg);

		AFC_D_Compute(Amatrix_np->mat(), G1_Mg->mat());

		nonLinAFC_alpha_Compute(MLmatrix_np->mat(), MCmatrix_np->mat(), G1_Mg->mat(), G0_Mg->mat(), A0_Mg->mat(), as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), I0_Mg->vec(), dt, tmpmatrix_np->mat(), alphamatrix_np->mat());//the matrix r is not applicable for non-Linear AFC

		PetscBarrier(NULL);

		//Construction of Linear system
		nonLinAFC_LinSys_Construct(MLmatrix_np->mat(), MCmatrix_np->mat(), G1_Mg->mat(), G0_Mg->mat(), Amatrix_np->mat(), A0_Mg->mat(), alphamatrix_np->mat(), tmpmatrix_np->mat(), as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), I1_Mg->vec(), I0_Mg->vec(), dt, LinSysLhs_np->mat(), LinSysRhs_np->vec());

		SolverWrapper_np->set_operator(*LinSysLhs_np);
		SolverWrapper_np->solve(*Mgfuncs[1]->vector(), *LinSysRhs_np);

		//to apply adaptive time step
		AFC_L_Compute(A0_Mg->mat(), G0_Mg->mat(), tmpmatrix_np->mat());
		dt_Adaptive = 0.05*AFC_dt_Compute(MLmatrix_np->mat(), tmpmatrix_np->mat());

		//updating A0
		MatCopy(Amatrix_np->mat(), A0_Mg->mat(), DIFFERENT_NONZERO_PATTERN);

		//OH
		Ivector_np->operator*=((96487.3329*1.2e-2)/(2*96487.3329*2.76e-2));//we want two different multiplication on Al and Mg. Because currently on Al is 0 this way this coefficient neutralizes later
		iOH(DOFsSetOnAlElectrode, Phibar->vec(), as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), theta->vec(), epsln->vec(), ldep->vec(), 0.4, 0.55, t, dt, Ivector_np->vec());

		WeakAssign(*A_np, {"Di", "zi"}, OHconsts);// electric field has already assigned
		WeakAssign(*I_np, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, Ivector_np->operator*(1/(96487.3329*1.2e-2)))});

		PetscBarrier(NULL);

		Weak2Matrix(*A_np, {}, *Amatrix_np);
		Weak2Matrix(*I_np, {}, *I1_OH);

		PetscBarrier(NULL);

		AFC_D_Compute(Amatrix_np->mat(), G1_OH->mat());

		nonLinAFC_alpha_Compute(MLmatrix_np->mat(), MCmatrix_np->mat(), G1_OH->mat(), G0_OH->mat(), A0_OH->mat(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), I0_OH->vec(), dt, tmpmatrix_np->mat(), alphamatrix_np->mat());//the matrix r is not applicable for non-Linear AFC

		PetscBarrier(NULL);

		//Construction of Linear system
		nonLinAFC_LinSys_Construct(MLmatrix_np->mat(), MCmatrix_np->mat(), G1_OH->mat(), G0_OH->mat(), Amatrix_np->mat(), A0_OH->mat(), alphamatrix_np->mat(), tmpmatrix_np->mat(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), I1_OH->vec(), I0_OH->vec(), dt, LinSysLhs_np->mat(), LinSysRhs_np->vec());

		SolverWrapper_np->set_operator(*LinSysLhs_np);
		SolverWrapper_np->solve(*OHfuncs[1]->vector(), *LinSysRhs_np);

		//to apply adaptive time step
		AFC_L_Compute(A0_OH->mat(), G0_OH->mat(), tmpmatrix_np->mat());
		dt_Adaptive = std::min(dt_Adaptive, AFC_dt_Compute(MLmatrix_np->mat(), tmpmatrix_np->mat()));

		//updating A0
		MatCopy(Amatrix_np->mat(), A0_OH->mat(), DIFFERENT_NONZERO_PATTERN);

		//H
		WeakAssign(*A_np, {"Di", "zi"}, Hconsts);// electric field has already assigned

		PetscBarrier(NULL);

		Weak2Matrix(*A_np, {}, *Amatrix_np);

		PetscBarrier(NULL);

		AFC_D_Compute(Amatrix_np->mat(), G1_H->mat());

		nonLinAFC_alpha_Compute(MLmatrix_np->mat(), MCmatrix_np->mat(), G1_H->mat(), G0_H->mat(), A0_H->mat(), as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), I0_OH->vec(), dt, tmpmatrix_np->mat(), alphamatrix_np->mat());//the matrix r is not applicable for non-Linear AFC //I0_H is identical to I0_OH

		PetscBarrier(NULL);

		//Construction of Linear system
		nonLinAFC_LinSys_Construct(MLmatrix_np->mat(), MCmatrix_np->mat(), G1_H->mat(), G0_H->mat(), Amatrix_np->mat(), A0_H->mat(), alphamatrix_np->mat(), tmpmatrix_np->mat(), as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), I1_OH->vec(), I0_OH->vec(), dt, LinSysLhs_np->mat(), LinSysRhs_np->vec());//H produced or consumed identical as OH

		SolverWrapper_np->set_operator(*LinSysLhs_np);
		SolverWrapper_np->solve(*Hfuncs[1]->vector(), *LinSysRhs_np);

		//to apply adaptive time step
		AFC_L_Compute(A0_H->mat(), G0_H->mat(), tmpmatrix_np->mat());
		dt_Adaptive = std::min(dt_Adaptive, AFC_dt_Compute(MLmatrix_np->mat(), tmpmatrix_np->mat()));//at this point dt_Adaptive is the smallest CFD condition

		//updating A0
		MatCopy(Amatrix_np->mat(), A0_H->mat(), DIFFERENT_NONZERO_PATTERN);

		PetscBarrier(NULL);

		//Water dissociation determines the amount of hydro-ions to be produced or consumed for keeping balance
		WaterDissociation(as_type<const dolfin::PETScVector>(Hfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), RHOH->vec());

		//electrode interface is exempt from water dissociation
		VecSetOnLocalDOFs(DOFsSetOnAlElectrode, RHOH->vec(), 0);
		VecSetOnLocalDOFs(DOFsSetOnMgElectrode, RHOH->vec(), 0);

		//adding to the current solution
		//VecAXPY(as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->vec(), dt, RHOH->vec());
		VecGhostUpdateBegin(as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->vec(), INSERT_VALUES, SCATTER_FORWARD);
		VecGhostUpdateEnd(as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->vec(), INSERT_VALUES, SCATTER_FORWARD);
		//VecAXPY(as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->vec(), dt, RHOH->vec());
		VecGhostUpdateBegin(as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->vec(), INSERT_VALUES, SCATTER_FORWARD);
		VecGhostUpdateEnd(as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->vec(), INSERT_VALUES, SCATTER_FORWARD);

		//pH
		if (std::fmod(std::floor(t),StroringStep) == 0.0) {//compute pH and storing data every StroringStep seconds
			pH_Compute(*OHfuncs[1], *pH, false);
			//storing
			StoringStream_p->operator<<(*ElectricFieldfuncs[0]);	
			StoringStream_Mg->operator<<(*Mgfuncs[1]);
			StoringStream_OH->operator<<(*OHfuncs[1]);
			StoringStream_H->operator<<(*Hfuncs[1]);
			StoringStream_pH->operator<<(*pH);
			StoringStream_theta->operator<<(*std::make_shared<dolfin::Function>(Vh, theta));
			StoringStream_epsln->operator<<(*std::make_shared<dolfin::Function>(Vh, epsln));
			StoringStream_ldep->operator<<(*std::make_shared<dolfin::Function>(Vh, ldep));
		}

		//updating
		//poisson
		DBC_p[0].set_value(std::make_shared<dolfin::Function>(Vh, Phibar));
		DBC_p[1].set_value(std::make_shared<dolfin::Function>(Vh, Phibar));

		//Nernst-Planck
		*(Mgfuncs[0]->vector()) = *(Mgfuncs[1]->vector());
		*(OHfuncs[0]->vector()) = *(OHfuncs[1]->vector());
		*(Hfuncs[0]->vector()) = *(Hfuncs[1]->vector());

		VecCopy(I1_Mg->vec(), I0_Mg->vec());
		VecCopy(I1_OH->vec(), I0_OH->vec());

		MatCopy(G1_Mg->mat(), G0_Mg->mat(), DIFFERENT_NONZERO_PATTERN);
		MatCopy(G1_OH->mat(), G0_OH->mat(), DIFFERENT_NONZERO_PATTERN);
		MatCopy(G1_H->mat(), G0_H->mat(), DIFFERENT_NONZERO_PATTERN);

		//time
		t = dt + t;
		if(prcID==0) {
			std::cout<<"Time step dt: "<<dt<<std::endl;
		}
		dt = dt_Adaptive;
		
		//storage step
		if (t > (10*60)) {//10 minutes
			StroringStep = 10;//seconds
		}
		if (t > (60*60)) {//1 hour
			StroringStep = 5*60;//seconds
		}

		//Printing
		//Poisson
		minval = as_type<const dolfin::PETScVector>(ElectricFieldfuncs[0]->vector())->min();
		maxval = as_type<const dolfin::PETScVector>(ElectricFieldfuncs[0]->vector())->max();
		if(prcID==0) {
			std::cout<<"Solver for poisson is set to: "<<SolverMethod_p<<std::endl;
			std::cout<<"Min ElectricField:"<<minval<<std::endl;
			std::cout<<"Max ElectricField:"<<maxval<<std::endl<<std::endl;
	
		}
		//Mg
		minval = as_type<const dolfin::PETScVector>(Mgfuncs[1]->vector())->min();
		maxval = as_type<const dolfin::PETScVector>(Mgfuncs[1]->vector())->max();
		if(prcID==0) {
			std::cout<<"Solver for Nernst-Planck is set to: "<<SolverMethod_np<<std::endl;
			std::cout<<"Min Mg:"<<minval<<std::endl;
			std::cout<<"Max Mg:"<<maxval<<std::endl<<std::endl;
		}
		//OH
		minval = as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->min();
		maxval = as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->max();
		if(prcID==0) {
			std::cout<<"Min OH:"<<minval<<std::endl;
			std::cout<<"Max OH:"<<maxval<<std::endl<<std::endl;
		}
		//H
		minval = as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->min();
		maxval = as_type<const dolfin::PETScVector>(Hfuncs[1]->vector())->max();
		if(prcID==0) {
			std::cout<<"Min H:"<<minval<<std::endl;
			std::cout<<"Max H:"<<maxval<<std::endl<<std::endl;
		}

		PetscBarrier(NULL);
getc(stdin);
	}

	if(prcID==0) {
		time(&end);
		std::cout<<"total exc time: "<<double(end-start)<<std::endl;
	}
/***here***/
	PetscBarrier(NULL);

	/*//memory release in chronological order
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
	SharedTypeVectorDestructor(EFfuncs);
	pH.reset();
	SharedTypeVectorDestructor(Mgfuncs);SharedTypeVectorDestructor(Mgconsts);
	SharedTypeVectorDestructor(OHfuncs);SharedTypeVectorDestructor(OHconsts);
	SharedTypeVectorDestructor(Hfuncs);SharedTypeVectorDestructor(Hconsts);

	//mesh and function space
	Vh.reset();mesh.reset();

	//shared matrices
	A_P.reset();
	A_ML.reset();A_MC.reset();A_NP.reset();rij.reset();ALinSys.reset();rhsLinSys.reset();
	L0_Mg.reset();D0_Mg.reset();D1_Mg.reset();
	L0_OH.reset();D0_OH.reset();D1_OH.reset();
	L0_H.reset();D0_H.reset();D1_H.reset();

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
	ff_pH.reset();
	ff_l.reset();
	ff_teta.reset();
	ff_eps.reset();*/

	PetscFinalize();

	return 0;
}
