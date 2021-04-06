#include <dolfin.h>

#include <stdio.h>

#include <cvode/cvode.h>               /* prototypes for CVODE fcts., consts.  */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sundials/sundials_types.h>   /* defs. of realtype, sunindextype      */

#define Ith(v,i)    NV_Ith_S(v,i-1)//Ith numbers components 1..3
#define IJth(A,i,j) SM_ELEMENT_D(A,i-1,j-1)//IJth numbers rows,cols 1..3
#define F 96487.3329//faraday's constant

//Functions Called by the ODE System Solver CVODE
static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);//y'=f(t,y)
static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

using namespace dolfin;

//Doi: 10.1149/2.0071501jes
//In theory presented in the mentioned paper we can calculate the exchange current as follows
//iExchange:=absolute equilibrium current
//EQ[12]==EQ[13]=>phi=-1.462456 Solved in matlab
//Matlab command:
//U_Electrolyte=-1.51:0.0001:-1.4;%-1.462456
//U=U_Electrolyte;
//iMg=(0.28.*(exp(39.31786.*0.4.*(U+1.517))-exp(-39.31786.*0.32.*(U+1.517))));
//iAl=((0.104.*exp(1e3.*(U+0.595)./500)./(0.012.*exp(1e3.*(U+0.595)./500)+1)+0.18.*10.^(-1e3.*(U+1.41)./118)));seems one has mA/cm^2 and the other A/m^2
//plot(U,iAl-iMg,'b')
//iMg=iAl=0.51928 A/(m)^2
//End Matlab command
//In the paper we used -1.463 Butler-Volmer for Mg and 0.5127 obtained

//Doi: 10.1149/2.0071501jes
//EQ[7] and EQ[8]
//Doi: 10.1016/j.electacta.2012.06.056
//EQ[39]
//f routine. Function in y'=f(t,y) for a system
static int f(realtype t, N_Vector y, N_Vector ydot, void *myData) {
	realtype* user_data = (realtype*) myData;//from void* to array {chi, eps0, l_max}
	Ith(ydot,1) = ((1-Ith(y,1))/((1-Ith(y,2))*(Ith(y,3)+user_data[2])))*3.7e-7*2.457627e-5*(user_data[0]);
	Ith(ydot,2) = -Ith(y,2)*7.4e-4*2.457627e-5*(user_data[0]);
	Ith(ydot,3) = (1/((1-user_data[1])*Ith(y,1)))*3.7e-7*2.457627e-5*(user_data[0]);//1-eps0 is directly assigned here
	return(0);
}
//Jacobian routine. J(t,y) = df/dy
static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *myData, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
	realtype* user_data = (realtype*) myData;//from void* to array {chi, eps0, l_max}
	IJth(J,1,1) = (-1/((1-Ith(y,2))*(Ith(y,3)+user_data[2])))*3.7e-7*2.457627e-5*(user_data[0]);
	IJth(J,1,2) = ((1-Ith(y,1))/((1-Ith(y,2))*(1-Ith(y,2))*(Ith(y,3)+user_data[2])))*3.7e-7*2.457627e-5*(user_data[0]);
	IJth(J,1,3) = ((1-Ith(y,1))/((1-Ith(y,2))*-1*(Ith(y,3)+user_data[2])*(Ith(y,3)+user_data[2])))*3.7e-7*2.457627e-5*(user_data[0]);

	IJth(J,2,1) = RCONST(0.0);
	IJth(J,2,2) = -7.4e-4*2.457627e-5*(user_data[0]);
	IJth(J,2,3) = RCONST(0.0);

	IJth(J,3,1) = (-1/((1-user_data[1])*Ith(y,1)*Ith(y,1)))*3.7e-7*2.457627e-5*(user_data[0]);//1-eps0 is directly assigned here
	IJth(J,3,2) = RCONST(0.0);
	IJth(J,3,3) = RCONST(0.0);
	return(0);
}

//(sqrt(theta)*i_eq)/(2*F)
int iMg(std::vector<size_t> LocalDOFSet_bar, Vec Phibar, Vec cMg, Vec cOH , Vec Vec_theta,  Vec Vec_eps, Vec Vec_l, double theta0, double eps0, double l0, double l_max, double i_eq, double phi_eq, double phi0_Mg, double t, Vec Ii, double &sqrt_theta_bar) {
	//Input: Dofset on boundary, Phibar, cMg, cOH, theta,, eps l, theta0, eps0, l0, l_max, i_eq, phi_eq, phi0_Mg, t
	//Output: Ii, iAl=bar(sqrt(theta)i_eq)

	VecSet(Ii, 0);
	sqrt_theta_bar = 0;

	PetscScalar eps;
	PetscScalar l;
	PetscScalar theta;
	PetscScalar i_Mg;
	PetscScalar phi_Mg;
	PetscScalar DOFsNum = LocalDOFSet_bar.size();

	const PetscScalar* cMg_i;
	const PetscScalar* cOH_i;

	VecGetArrayRead(cMg, &cMg_i);
	VecGetArrayRead(cOH, &cOH_i);

	realtype* myData = new realtype[3];//myData = {chi, eps0, l_max}
	realtype reltol, T = RCONST(t);
	N_Vector y, abstol, constraints;
	SUNMatrix A;
	SUNLinearSolver LS;
	void *cvode_mem;
	//allocation
	y = N_VNew_Serial(3);
	abstol = N_VNew_Serial(3);
	constraints = N_VNew_Serial(3);
	//Initialize
	myData[0] = 0;
	myData[1] = eps0;
	myData[2] = l_max;
	Ith(y,1) = RCONST(theta0);
	Ith(y,2) = RCONST(eps0);
	Ith(y,3) = RCONST(l0-l_max);//change of variable: l = l' + l_max
	//Set the scalar relative tolerance
	reltol = RCONST(1e-4);
	//Set the vector absolute tolerance
	Ith(abstol,1) = RCONST(1e-14);
	Ith(abstol,2) = RCONST(1e-14);
	Ith(abstol,3) = RCONST(1e-14);
	//Set constraints to all 1's for nonnegative solution values.
	N_VConst(RCONST(1.0), constraints);
	Ith(constraints,3) = RCONST(-1.0);//negative value
	//Call CVodeCreate to create the solver memory and specify the Backward Differentiation Formula
	cvode_mem = CVodeCreate(CV_BDF);
	//Call CVodeInit to initialize the integrator memory and specify the user's right hand side function in y'=f(t,y), the inital time T0, and the initial dependent variable vector y.
	CVodeInit(cvode_mem, f, RCONST(0.0), y);
	//Set the pointer to user-defined data
	CVodeSetUserData(cvode_mem, myData);
	//Call CVodeSVtolerances to specify the scalar relative tolerance and vector absolute tolerances
	CVodeSVtolerances(cvode_mem, reltol, abstol);
	//Create dense SUNMatrix for use in linear solves
	A = SUNDenseMatrix(3, 3);
	//Create dense SUNLinearSolver object for use by CVode
	LS = SUNLinSol_Dense(y, A);
	//Call CVodeSetLinearSolver to attach the matrix and linear solver to CVode
	CVodeSetLinearSolver(cvode_mem, LS, A);
	//Set the user-supplied Jacobian routine Jac
	CVodeSetJacFn(cvode_mem, Jac);
	//Call CVodeSetConstraints to initialize constraints
	CVodeSetConstraints(cvode_mem, constraints);

	PetscBarrier(NULL);

	//Computing the electerical current
	for (size_t j = 0; j < LocalDOFSet_bar.size(); j = j + 1) {
		//computing chi from cMg and cOH
		myData[0] = RCONST(cMg_i[LocalDOFSet_bar[j]]*cOH_i[LocalDOFSet_bar[j]]*cOH_i[LocalDOFSet_bar[j]]-0.450);
		if (myData[0]>0) {
			//call CVode to solve the system.
			CVode(cvode_mem, T, y, &T, CV_NORMAL);
			theta = Ith(y,1);
			eps = Ith(y,2);
			l = Ith(y,3)+l_max;
		}
		else {
			theta = theta0;
			eps = eps0;
			l = l0;
		}
		i_Mg = (std::sqrt(theta)*i_eq)/(2*F);
		if ((1e-15<t)&&(t<=3*3600)) {
			//extracting dirichlet condition for electric field
			//phi_Mg computation
			phi_Mg = ((phi_eq-0.001-phi0_Mg)/(std::log(3*3600/1e-15))) * std::log(t/1e-15) + phi0_Mg;
		}
		else if ((3*3600<t)&&(t<=24*3600)){
			//extracting dirichlet condition for electric field
			//phi_Mg computation
			phi_Mg = (0.001/(std::log(24*3600/(3*3600)))) * std::log(t/(3*3600)) + phi_eq-0.001;
		}
		else {
			phi_Mg = phi_eq;
		}
		sqrt_theta_bar = sqrt_theta_bar + std::sqrt(theta);//for integration
		VecSetValueLocal(Ii, LocalDOFSet_bar[j], i_Mg, INSERT_VALUES);
		VecSetValueLocal(Phibar, LocalDOFSet_bar[j], phi_Mg, INSERT_VALUES);
		VecSetValueLocal(Vec_theta, LocalDOFSet_bar[j], theta, INSERT_VALUES);
		VecSetValueLocal(Vec_eps, LocalDOFSet_bar[j], eps, INSERT_VALUES);
		VecSetValueLocal(Vec_l, LocalDOFSet_bar[j], l, INSERT_VALUES);
	}

	MPIU_Allreduce(MPI_IN_PLACE, &DOFsNum, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD);//number of dofs on all processors
	MPIU_Allreduce(MPI_IN_PLACE, &sqrt_theta_bar, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD);//sum on all processors
	sqrt_theta_bar=(sqrt_theta_bar/DOFsNum);//for estimate of integration

	VecAssemblyBegin(Ii);
	VecAssemblyEnd(Ii);
	VecGhostUpdateBegin(Ii, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Ii, INSERT_VALUES, SCATTER_FORWARD);

	VecAssemblyBegin(Phibar);
	VecAssemblyEnd(Phibar);
	VecGhostUpdateBegin(Phibar, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Phibar, INSERT_VALUES, SCATTER_FORWARD);

	VecAssemblyBegin(Vec_theta);
	VecAssemblyEnd(Vec_theta);
	VecGhostUpdateBegin(Vec_theta, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Vec_theta, INSERT_VALUES, SCATTER_FORWARD);

	VecAssemblyBegin(Vec_eps);
	VecAssemblyEnd(Vec_eps);
	VecGhostUpdateBegin(Vec_eps, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Vec_eps, INSERT_VALUES, SCATTER_FORWARD);
	
	VecAssemblyBegin(Vec_l);
	VecAssemblyEnd(Vec_l);
	VecGhostUpdateBegin(Vec_l, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Vec_l, INSERT_VALUES, SCATTER_FORWARD);

	PetscBarrier(NULL);

	//Free vectors
	N_VDestroy(y);
	N_VDestroy(abstol);
	N_VDestroy(constraints);
	delete[] myData;
	//Free integrator memory
	CVodeFree(&cvode_mem);
	//Free the linear solver memory
	SUNLinSolFree(LS);
	//Free the matrix memory
	SUNMatDestroy(A);

	VecRestoreArrayRead(cMg, &cMg_i);
	VecRestoreArrayRead(cOH, &cOH_i);

	return 0;
}

//bar{(sqrt(theta)*i_eq*l_Al)}/F
int iOH(std::vector<size_t> LocalDOFSet_bar, Vec Phibar, double l_Al, double l_Mg, double sqrt_theta_bar, double i_eq, double phi_eq, double phi0_Al, double t, Vec Ii) {
	//Input: Dofset on boundary, Phibar, l_Al, iAl, phi_eq, phi0_Al, t
	//Output: Ii

	VecSet(Ii, 0);

	PetscScalar i_Al;
	PetscScalar phi_Al;

	i_Al = (sqrt_theta_bar*i_eq*l_Mg)/(l_Al*F);
	if ((1e-15<t)&&(t<=3*3600)) {
		//extracting dirichlet condition for electric field
		//phi_Mg computation
		phi_Al = ((phi_eq+0.001-phi0_Al)/(std::log(3*3600/1e-15))) * std::log(t/1e-15) + phi0_Al;
	}
	else if ((3*3600<t)&&(t<=24*3600)){
		//extracting dirichlet condition for electric field
		//phi_Mg computation
		phi_Al = (-0.001/(std::log(24*3600/(3*3600)))) * std::log(t/(3*3600)) + phi_eq+0.001;
	}
	else {
		phi_Al = phi_eq;
	}
	for (size_t j = 0; j < LocalDOFSet_bar.size(); j = j + 1) {
		VecSetValueLocal(Ii, LocalDOFSet_bar[j], i_Al, INSERT_VALUES);//the average current of Mg electrode is applied uniformly here (note theta and epsilon describing only the Mg dynamics)
		VecSetValueLocal(Phibar, LocalDOFSet_bar[j], phi_Al, INSERT_VALUES);
	}

	VecAssemblyBegin(Ii);
	VecAssemblyEnd(Ii);
	VecGhostUpdateBegin(Ii, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Ii, INSERT_VALUES, SCATTER_FORWARD);

	VecAssemblyBegin(Phibar);
	VecAssemblyEnd(Phibar);
	VecGhostUpdateBegin(Phibar, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Phibar, INSERT_VALUES, SCATTER_FORWARD);

	PetscBarrier(NULL);

	return 0;
}

int pH_Compute(dolfin::Function func, dolfin::Function &pH, bool Hbased=true) {

	(pH.vector())->operator=(0);

	PetscInt lsize;
	VecGetLocalSize(as_type<const dolfin::PETScVector>(func.vector())->vec(), &lsize);

	const PetscScalar* c_i;
	VecGetArrayRead(as_type<const dolfin::PETScVector>(func.vector())->vec(), &c_i);

	for (PetscInt j = 0; j < lsize; j = j + 1) {
		if (Hbased) {
			if (PetscIsNormalReal(-1*std::log10(1e-3*c_i[j])))
				VecSetValueLocal(as_type<const dolfin::PETScVector>(pH.vector())->vec(), j, -1*std::log10(1e-3*c_i[j]), INSERT_VALUES);//1e-3 for converting mol/m3 to mol/L
		}
		else {
			if (PetscIsNormalReal(14+1*std::log10(1e-3*c_i[j])))
				VecSetValueLocal(as_type<const dolfin::PETScVector>(pH.vector())->vec(), j, 14+1*std::log10(1e-3*c_i[j]), INSERT_VALUES);//1e-3 for converting mol/m3 to mol/L and pOH approx 14-pH
		}
	}

	VecRestoreArrayRead(as_type<const dolfin::PETScVector>(func.vector())->vec(), &c_i);
	VecAssemblyBegin(as_type<const dolfin::PETScVector>(pH.vector())->vec());
	VecAssemblyEnd(as_type<const dolfin::PETScVector>(pH.vector())->vec());
	VecGhostUpdateBegin(as_type<const dolfin::PETScVector>(pH.vector())->vec(), INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(as_type<const dolfin::PETScVector>(pH.vector())->vec(), INSERT_VALUES, SCATTER_FORWARD);

	PetscBarrier(NULL);

	return 0;
}
