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

	RectPointsGenerator(0.01, 0.02, p_d);//RectPointsGenerator(1, 2, ps);
	p_d.push_back(p_d[0] + (p_d[3]-p_d[0])*0.3 + (p_d[3]-p_d[0])*0.02);
	p_d.push_back(p_d[0] + (p_d[3]-p_d[0])*0.3 - (p_d[3]-p_d[0])*0.02);
	double l_Al = 0.56e-2, l_Mg = 1.36e-2;

	//----Reading the rectangle mesh from points and specifying the boundaries----
	auto mesh = std::make_shared<dolfin::Mesh>(PETSC_COMM_WORLD, "mesh/mesh.xml");

	std::vector<std::shared_ptr<dolfin::SubDomain>> b_d;//boundaries of domain
	b_d.push_back(std::make_shared<RectBorderLine>(p_d[0], p_d[1]));
	b_d.push_back(std::make_shared<RectBorderLine>(p_d[1], p_d[2]));
	b_d.push_back(std::make_shared<RectBorderLine>(p_d[2], p_d[3]));
	b_d.push_back(std::make_shared<RectBorderLine>(p_d[3], p_d[4]));//Mg
	b_d.push_back(std::make_shared<RectBorderLine>(p_d[5], p_d[0]));//Al
	b_d.push_back(std::make_shared<RectBorderLine>(p_d[4], p_d[5]));//Insulator

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

	//--Function generation intermediate variables are no longer needed--
	isconst.clear(); isconst.shrink_to_fit();
	constvalue.clear(); constvalue.shrink_to_fit();

	//----Deciesion on type and assignment of the boudaries (Neumann, Dirichlet, ...)----
	std::vector<dolfin::DirichletBC> DBC_p;//DBC stands for Dirichlet Boundary Condition
	myDirichletBCGenerator(Vh, {std::make_shared<dolfin::Constant>(-1.763), std::make_shared<dolfin::Constant>(-1.163)}, {b_d[3], b_d[4]}, DBC_p);//to assign values to a specific domain boundary

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
	Weak2Matrix(*MC_np, {}, *MCmatrix_np);//Petsc requires a constructor for vectors and matrices
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
	auto l_dep = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));//deposit layer thickness
	l_dep->zero();
	l_dep->operator=(0);//l0_dep

	std::vector<double> StoredTimeSequence;
	std::vector<double> StoredTimeStepSequence;
	double theta0 = 0.1, eps0 = 0.55, l_dep0 = 1e-7, l_max = 1e-3, i_eq = 0.5127, phi_eq = -1.463, phi_Al = -1.163, phi_Mg = -1.763;//fixed model constants
	double sqrt_theta_bar;
	double t = 0;
	double dt = 1e-2;
	double dt_Adaptive = dt;
	double scount = 60, hcount = 4*3600;
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

	iMg(DOFsSetOnMgElectrode, Phibar->vec(), as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), theta->vec(), epsln->vec(), l_dep->vec(), theta0, eps0, l_dep0, l_max, i_eq, phi_eq, phi_Mg, t, Ivector_np->vec(), sqrt_theta_bar);

	WeakAssign(*A_np, {"Di", "zi","phi"}, {Mgconsts[0], Mgconsts[1], ElectricFieldfuncs[0]});
	WeakAssign(*I_np, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, Ivector_np)});//change in Mg2+ concentration

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
	//FunctionNFilterMin(as_type<const dolfin::PETScVector>(Mgfuncs[1]->vector())->vec(), *Vh, *mesh);
	
	//to apply adaptive time step
	AFC_L_Compute(A0_Mg->mat(), G0_Mg->mat(), tmpmatrix_np->mat());
	dt_Adaptive = AFC_dt_Compute(MLmatrix_np->mat(), tmpmatrix_np->mat());

	//OH
	auto A0_OH = std::make_shared<dolfin::PETScMatrix>(*MCmatrix_np);
	auto G0_OH = std::make_shared<dolfin::PETScMatrix>(*MCmatrix_np);
	auto G1_OH = std::make_shared<dolfin::PETScMatrix>(*MCmatrix_np);
	auto I0_OH = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	I0_OH->zero();
	auto I1_OH = std::make_shared<dolfin::PETScVector>(*(as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())));
	I1_OH->zero();

	VecScale(Ivector_np->vec(), 2);//Mg->2OH

	PetscBarrier(NULL);

	iOH(DOFsSetOnAlElectrode, Phibar->vec(), l_Al, l_Mg, sqrt_theta_bar, i_eq, phi_eq, phi_Al, t, Ivector_np->vec());

	WeakAssign(*A_np, {"Di", "zi"}, OHconsts);// electric field has already assigned
	WeakAssign(*I_np, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, Ivector_np)});//change in OH- concentration

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
	//FunctionNFilterMin(as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->vec(), *Vh, *mesh);

	//to apply adaptive time step
	AFC_L_Compute(A0_OH->mat(), G0_OH->mat(), tmpmatrix_np->mat());
	dt_Adaptive = std::min(dt_Adaptive, AFC_dt_Compute(MLmatrix_np->mat(), tmpmatrix_np->mat()));

	//pH
	auto pH = std::make_shared<dolfin::Function>(Vh);
	pH_Compute(*OHfuncs[0], *pH, false);

	//storing
	auto StoringStream_p = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/Electric Field.pvd");
	StoringStream_p->operator<<(*ElectricFieldfuncs[0]);	
	auto StoringStream_Mg = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/Mg_Concentration.pvd");
	StoringStream_Mg->operator<<(*Mgfuncs[0]);
	auto StoringStream_OH = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/OH_Concentration.pvd");
	StoringStream_OH->operator<<(*OHfuncs[0]);
	auto StoringStream_pH = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/pH_Concentration.pvd");
	StoringStream_pH->operator<<(*pH);
	auto StoringStream_theta = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/theta.pvd");
	StoringStream_theta->operator<<(*std::make_shared<dolfin::Function>(Vh, theta));
	auto StoringStream_epsln = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/epsln.pvd");
	StoringStream_epsln->operator<<(*std::make_shared<dolfin::Function>(Vh, epsln));
	auto StoringStream_l_dep = std::make_shared<dolfin::File>(PETSC_COMM_WORLD, "Results/l_dep.pvd");
	StoringStream_l_dep->operator<<(*std::make_shared<dolfin::Function>(Vh, l_dep));

	//updating
	//poisson boundary (polarization on electrode interface)
	DBC_p[0].set_value(std::make_shared<dolfin::Function>(Vh, Phibar));
	DBC_p[1].set_value(std::make_shared<dolfin::Function>(Vh, Phibar));

	//Nernst-Planck
	*(Mgfuncs[0]->vector()) = *(Mgfuncs[1]->vector());
	*(OHfuncs[0]->vector()) = *(OHfuncs[1]->vector());

	VecCopy(I1_Mg->vec(), I0_Mg->vec());
	VecCopy(I1_OH->vec(), I0_OH->vec());

	//time
	t = dt + t;
	if(prcID==0) {
		std::cout<<"Time step dt: "<<dt<<std::endl;
	}
	if (dt_Adaptive<10) {
		dt = dt_Adaptive;
	}

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

	while (t<=SimulationTime) {
		Weak2Matrix(*a_p, DBC_p, *LinSysLhs_p, *LinSysRhs_p);

		PetscBarrier(NULL);

		Solver_p->set_operator(*LinSysLhs_p);
		Solver_p->solve(*ElectricFieldfuncs[0]->vector(), *LinSysRhs_p);

		//Mg
		iMg(DOFsSetOnMgElectrode, Phibar->vec(), as_type<const dolfin::PETScVector>(Mgfuncs[0]->vector())->vec(), as_type<const dolfin::PETScVector>(OHfuncs[0]->vector())->vec(), theta->vec(), epsln->vec(), l_dep->vec(), theta0, eps0, l_dep0, l_max, i_eq, phi_eq, phi_Mg, t, Ivector_np->vec(), sqrt_theta_bar);

		WeakAssign(*A_np, {"Di", "zi","phi"}, {Mgconsts[0], Mgconsts[1], ElectricFieldfuncs[0]});
		WeakAssign(*I_np, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, Ivector_np)});

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
		//FunctionNFilterMin(as_type<const dolfin::PETScVector>(Mgfuncs[1]->vector())->vec(), *Vh, *mesh);

		//to apply adaptive time step
		AFC_L_Compute(A0_Mg->mat(), G0_Mg->mat(), tmpmatrix_np->mat());
		dt_Adaptive = AFC_dt_Compute(MLmatrix_np->mat(), tmpmatrix_np->mat());

		//updating A0
		MatCopy(Amatrix_np->mat(), A0_Mg->mat(), DIFFERENT_NONZERO_PATTERN);

		//OH
		VecScale(Ivector_np->vec(), 2);//Mg->2OH

		PetscBarrier(NULL);

		iOH(DOFsSetOnAlElectrode, Phibar->vec(), l_Al, l_Mg, sqrt_theta_bar, i_eq, phi_eq, phi_Al, t, Ivector_np->vec());

		WeakAssign(*A_np, {"Di", "zi"}, OHconsts);// electric field has already assigned
		WeakAssign(*I_np, {"Ii"}, {std::make_shared<dolfin::Function>(Vh, Ivector_np)});

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
		//FunctionNFilterMin(as_type<const dolfin::PETScVector>(OHfuncs[1]->vector())->vec(), *Vh, *mesh);

		//to apply adaptive time step
		AFC_L_Compute(A0_OH->mat(), G0_OH->mat(), tmpmatrix_np->mat());
		dt_Adaptive = std::min(dt_Adaptive, AFC_dt_Compute(MLmatrix_np->mat(), tmpmatrix_np->mat()));

		//updating A0
		MatCopy(Amatrix_np->mat(), A0_OH->mat(), DIFFERENT_NONZERO_PATTERN);

		//pH
		if ((std::abs(t-scount) <= dt)) {//compute pH and storing data after scount seconds
			pH_Compute(*OHfuncs[1], *pH, false);
			//storing
			StoringStream_p->operator<<(*ElectricFieldfuncs[0]);	
			StoringStream_Mg->operator<<(*Mgfuncs[1]);
			StoringStream_OH->operator<<(*OHfuncs[1]);
			StoringStream_pH->operator<<(*pH);
			StoringStream_theta->operator<<(*std::make_shared<dolfin::Function>(Vh, theta));
			StoringStream_epsln->operator<<(*std::make_shared<dolfin::Function>(Vh, epsln));
			StoringStream_l_dep->operator<<(*std::make_shared<dolfin::Function>(Vh, l_dep));
			if (t<=3*3600+dt) {
				scount = scount + 600;
			}
			else{
				scount = hcount;
				hcount = hcount + 3600;
			}
			StoredTimeSequence.push_back(t);
			StoredTimeStepSequence.push_back(dt);
		}

		//updating
		//poisson
		DBC_p[0].set_value(std::make_shared<dolfin::Function>(Vh, Phibar));
		DBC_p[1].set_value(std::make_shared<dolfin::Function>(Vh, Phibar));

		//Nernst-Planck
		*(Mgfuncs[0]->vector()) = *(Mgfuncs[1]->vector());
		*(OHfuncs[0]->vector()) = *(OHfuncs[1]->vector());

		VecCopy(I1_Mg->vec(), I0_Mg->vec());
		VecCopy(I1_OH->vec(), I0_OH->vec());

		MatCopy(G1_Mg->mat(), G0_Mg->mat(), DIFFERENT_NONZERO_PATTERN);
		MatCopy(G1_OH->mat(), G0_OH->mat(), DIFFERENT_NONZERO_PATTERN);

		//time
		t = dt + t;
		if(prcID==0) {
			std::cout<<"Time step dt: "<<dt<<std::endl;
		}
		if (dt_Adaptive<10) {
			dt = dt_Adaptive;
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

		PetscBarrier(NULL);
	}

	if(prcID==0) {
		std::cout<<std::endl<<"Stored at t={0, ";
		for (std::size_t i=0; i<(StoredTimeSequence.size()-1); i=i+1) {
			std::cout<<StoredTimeSequence[i]<<", ";
		}
		std::cout<<StoredTimeSequence[StoredTimeSequence.size()-1]<<"}"<<std::endl;
		std::cout<<std::endl<<"Time Steps at above t dt={1e-2, ";
		for (std::size_t i=0; i<(StoredTimeStepSequence.size()-1); i=i+1) {
			std::cout<<StoredTimeStepSequence[i]<<", ";
		}
		std::cout<<StoredTimeStepSequence[StoredTimeStepSequence.size()-1]<<"}"<<std::endl;
		time(&end);
		std::cout<<"total exc time: "<<double(end-start)<<std::endl;
	}

	PetscBarrier(NULL);

	//memory release in chronological order
	//weak forms
	a_p.reset();MC_np.reset();A_np.reset();I_np.reset();

	//vectors
	DBC_p.clear(); DBC_p.shrink_to_fit();
	DOFsSetOnAlElectrode.clear();DOFsSetOnAlElectrode.shrink_to_fit();
	DOFsSetOnMgElectrode.clear();DOFsSetOnMgElectrode.shrink_to_fit();

	//shared functions or vector of shared functions
	SharedTypeVectorDestructor(ElectricFieldfuncs);
	SharedTypeVectorDestructor(Mgfuncs);SharedTypeVectorDestructor(Mgconsts);
	SharedTypeVectorDestructor(OHfuncs);SharedTypeVectorDestructor(OHconsts);
	pH.reset();

	//mesh and function space
	Vh.reset();mesh.reset();

	//shared matrices and vectors
	LinSysLhs_p.reset();LinSysRhs_p.reset();Phibar.reset();
	MCmatrix_np.reset();LinSysLhs_np.reset();LinSysRhs_np.reset();tmpmatrix_np.reset();MLmatrix_np.reset();Amatrix_np.reset();Ivector_np.reset();alphamatrix_np.reset();theta.reset();epsln.reset();l_dep.reset();
	A0_Mg.reset();G0_Mg.reset();G1_Mg.reset();I0_Mg.reset();I1_Mg.reset();
	A0_OH.reset();G0_OH.reset();G1_OH.reset();I0_OH.reset();I1_OH.reset();

	//shared solvers
	Solver_p.reset();
	SolverWrapper_np.reset();

	//petsc solvers
	KSPDestroy(&Solver_np);
	PCDestroy(&Conditioner_np);

	//shared file streams
	StoringStream_p.reset();StoringStream_Mg.reset();StoringStream_OH.reset();StoringStream_pH.reset();StoringStream_theta.reset();StoringStream_epsln.reset();StoringStream_l_dep.reset();

	PetscFinalize();

	return 0;
}
