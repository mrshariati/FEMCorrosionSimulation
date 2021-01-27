#include <dolfin.h>

using namespace dolfin;

//Doi: 10.1149/2.0071501jes
//teta:=affected surface fraction
//EQ[7] and EQ[8]
double tetaODE(double cMg, double cOH, double eps, double l, double teta0, double t) {
	double konst;
	double A;
	konst = (3.7e-7*0.058) / (l*(1 - eps)*2360);
	A = cMg*cOH*cOH - 0.450;
	if(A > 0) {
		konst = konst*A;
		double teta = 1-(1-teta0)*std::exp(-konst*t);//teta=-exp(-konst*t+ln(1-teta0))+1
		return teta;
	}
	else
		return teta0;
}

//Doi: 10.1149/2.0071501jes
//eps:=porosity of the deposit
//EQ[10]
double epsODE(double cMg, double cOH, double eps0, double t) {
	double konst;
	double A;
	konst = (7.4e-4*0.058)/2360;
	A = cMg*cOH*cOH - 0.450;
	if(A > 0) {
		konst = konst*A;
		double eps = eps0*std::exp(-konst*t);//eps=exp(-konst*t+ln(eps0))
		return eps;
	}
	else
		return eps0;
}

//Doi: 10.1149/2.0071501jes
//In theory presented in the mentioned paper we can calculate the exchange current as follows
//iExchange:=absolute current
//EQ[12]==EQ[13]=>phi=-1.462456 Solved in matlab
//Matlab command:
//U_Electrolyte=-1.51:0.0001:-1.4;%-1.462456
//U=U_Electrolyte;
//iMg=(0.28.*(exp(39.31786.*0.4.*(U+1.517))-exp(-39.31786.*0.32.*(U+1.517))));
//iAl=((0.104.*exp(1e3.*(U+0.595)./500)./(0.012.*exp(1e3.*(U+0.595)./500)+1)+0.18.*10.^(-1e3.*(U+1.41)./118)));seems one has mA/cm^2 and the other A/m^2
//plot(U,iAl-iMg,'b')
//iMg=iAl=0.51928 A/(m)^2
//End Matlab command
//In the paper we used -1.463 in iMg, 0.5127 obtained for the paper

double iElectrodeFormula(double teta, double eps) {
	double iExchange;
	iExchange = 0.5127;
	return std::sqrt(teta)*iExchange;//(1 - teta + eps*teta)*iExchange //in the original model
}

//Doi: 10.1149/2.0071501jes
//Moving interface velocity uTotal
//EQ[16]
double uTotal(double iElectrode, double cMg, double cOH, double eps) {
	return -(0.024*iElectrode)/(2*96487*1735) + (7.4e-4*(cMg*cOH*cOH - 0.45)*0.058) / ((1 - eps)*2360);
}
//Cor:=Corrosion
double uCor(double iElectrode) {
	return -(0.024*iElectrode)/(2*96487*1735);
}
//Dep:=Deposit
double uDep(double cMg, double cOH, double eps) {
	double konst;
	double A;
	konst = (7.4e-4*0.058);
	A = cMg*cOH*cOH - 0.450;
	if(A > 0) {
		konst = konst*A;
		return konst/((1 - eps)*2360);
	}
	else
		return 0;
}

//In case of existence of polarization data
double Current2ElectricField(std::vector<double> Phi, std::vector<double> i, double i_t) {
	//Linear spline interpolation with free knot at the boundaries

	double ilower_bound = std::abs(i_t - i[0]);
	size_t occurance = 0;
	for (size_t k=1; k<i.size(); k=k+1) {
		if (std::abs(i_t-i[k])<ilower_bound) {
			ilower_bound = std::abs(i_t-i[k]);
			occurance = k;
		}
	}
	if (occurance==0 && i_t < i[occurance])
		return Phi[occurance]+((i_t-i[occurance])/(i[occurance+1]-i[occurance]))*(Phi[occurance+1]-Phi[occurance]);
	else if (occurance==(i.size()-1) && i_t >= i[occurance])
		return Phi[occurance]+((i_t-i[occurance])/(i[occurance]-i[occurance-1]))*(Phi[occurance]-Phi[occurance-1]);
	else {
		if (i_t < i[occurance])
			return Phi[occurance]+((i_t-i[occurance])/(i[occurance]-i[occurance-1]))*(Phi[occurance]-Phi[occurance-1]);
		else
			return Phi[occurance]+((i_t-i[occurance])/(i[occurance+1]-i[occurance]))*(Phi[occurance+1]-Phi[occurance]);
	}
}

//Doi: 10.1149/2.0071501jes
//EQ[11], EQ[12]
int iMg(std::vector<size_t> LocalDOFSet_bar, Vec Phibar, Vec cMg, Vec cOH , Vec Vec_teta,  Vec Vec_eps, Vec Vec_l, double teta0, double eps0, double t, double dt, Vec Ii) {
	//Input: Dofset on boundary, Phibar, cMg, cOH, teta,, eps l, teta0, eps0, t, dt
	//Output: Ii

	VecSet(Ii, 0);

	PetscScalar eps;
	PetscScalar l;
	PetscScalar teta;
	PetscScalar iElectrode;
	PetscScalar UElectrode;

	const PetscScalar* cMg_i;
	const PetscScalar* cOH_i;
	const PetscScalar* l_i;

	VecGetArrayRead(cMg, &cMg_i);
	VecGetArrayRead(cOH, &cOH_i);
	VecGetArrayRead(Vec_l, &l_i);

	PetscBarrier(NULL);

	//Computing the electerical current
	for (size_t j = 0; j < LocalDOFSet_bar.size(); j = j + 1) {
		eps = epsODE(cMg_i[LocalDOFSet_bar[j]], cOH_i[LocalDOFSet_bar[j]], eps0, t+dt);
		l = l_i[LocalDOFSet_bar[j]] + uDep(cMg_i[LocalDOFSet_bar[j]], cOH_i[LocalDOFSet_bar[j]], eps)*dt;//previous value + new value of l based on current other values = current value
		teta = tetaODE(cMg_i[LocalDOFSet_bar[j]], cOH_i[LocalDOFSet_bar[j]], eps, l, teta0, t+dt);
		iElectrode = std::sqrt(teta)*0.5127;//iElectrodeFormula(teta, eps) *the simpler formula has directly applied*
		VecSetValueLocal(Ii, LocalDOFSet_bar[j], iElectrode, INSERT_VALUES);
		//extracting dirichlet condition for electric field
		//phi_Mg(here UElectrode) computation
		UElectrode = ((-1.463+1.517)/(-std::log(std::sqrt(teta0)))) * std::log(std::sqrt(teta/teta0)) - 1.517;//alternative is Current2ElectricField(PolarizationData, Polarizationcurrents, iElectrode) *the simpler approach is to use formula and not experimental polarization data*
		VecSetValueLocal(Phibar, LocalDOFSet_bar[j], UElectrode, INSERT_VALUES);
		VecSetValueLocal(Vec_l, LocalDOFSet_bar[j], l, INSERT_VALUES);
		VecSetValueLocal(Vec_teta, LocalDOFSet_bar[j], teta, INSERT_VALUES);
		VecSetValueLocal(Vec_eps, LocalDOFSet_bar[j], eps, INSERT_VALUES);
	}

	VecAssemblyBegin(Ii);
	VecAssemblyEnd(Ii);
	VecGhostUpdateBegin(Ii, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Ii, INSERT_VALUES, SCATTER_FORWARD);

	VecAssemblyBegin(Phibar);
	VecAssemblyEnd(Phibar);
	VecGhostUpdateBegin(Phibar, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Phibar, INSERT_VALUES, SCATTER_FORWARD);

	VecAssemblyBegin(Vec_l);
	VecAssemblyEnd(Vec_l);
	VecGhostUpdateBegin(Vec_l, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Vec_l, INSERT_VALUES, SCATTER_FORWARD);

	VecAssemblyBegin(Vec_teta);
	VecAssemblyEnd(Vec_teta);
	VecGhostUpdateBegin(Vec_teta, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Vec_teta, INSERT_VALUES, SCATTER_FORWARD);

	VecAssemblyBegin(Vec_eps);
	VecAssemblyEnd(Vec_eps);
	VecGhostUpdateBegin(Vec_eps, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Vec_eps, INSERT_VALUES, SCATTER_FORWARD);

	PetscBarrier(NULL);

	VecRestoreArrayRead(cMg, &cMg_i);
	VecRestoreArrayRead(cOH, &cOH_i);
	VecRestoreArrayRead(Vec_l, &l_i);

	return 0;
}

//Doi: 10.1149/2.0071501jes
//EQ[11], EQ[13]
int iOH(std::vector<size_t> LocalDOFSet_bar, Vec Phibar, Vec cMg, Vec cOH , Vec Vec_teta,  Vec Vec_eps, Vec Vec_l, double teta0, double eps0, double t, double dt, Vec Ii) {
	//Input: Dofset on boundary, Phibar, cMg, cOH, teta,, eps l, teta0, eps0, t, dt
	//Output: Ii

	VecScale(Ii, 2);

	PetscScalar eps;
	PetscScalar l;
	PetscScalar teta;
	PetscScalar iElectrode;
	PetscScalar UElectrode;

	const PetscScalar* cMg_i;
	const PetscScalar* cOH_i;
	const PetscScalar* l_i;

	VecGetArrayRead(cMg, &cMg_i);
	VecGetArrayRead(cOH, &cOH_i);
	VecGetArrayRead(Vec_l, &l_i);

	PetscBarrier(NULL);

	//Computing the electerical current
	for (size_t j = 0; j < LocalDOFSet_bar.size(); j = j + 1) {
		eps = epsODE(cMg_i[LocalDOFSet_bar[j]], cOH_i[LocalDOFSet_bar[j]], eps0, t+dt);
		l = l_i[LocalDOFSet_bar[j]] + uDep(cMg_i[LocalDOFSet_bar[j]], cOH_i[LocalDOFSet_bar[j]], eps)*dt;//previous value + new value of l based on online other values = present value
		teta = tetaODE(cMg_i[LocalDOFSet_bar[j]], cOH_i[LocalDOFSet_bar[j]], eps, l, teta0, t+dt);
		iElectrode = std::sqrt(teta)*0.5127;//iElectrodeFormula(teta, eps) *the simpler formula has directly applied*
		VecSetValueLocal(Ii, LocalDOFSet_bar[j], iElectrode, INSERT_VALUES);
		//extracting dirichlet condition for electric field
		//phi_Mg(here UElectrode) computation
		UElectrode = ((-1.463+0.595)/(-std::log(std::sqrt(teta0)))) * std::log(std::sqrt(teta/teta0)) - 0.595;//alternative is Current2ElectricField(PolarizationData, Polarizationcurrents, iElectrode) *the simpler approach is to use formula and not experimental polarization data*
		VecSetValueLocal(Phibar, LocalDOFSet_bar[j], UElectrode, INSERT_VALUES);
		VecSetValueLocal(Vec_l, LocalDOFSet_bar[j], l, INSERT_VALUES);
		VecSetValueLocal(Vec_teta, LocalDOFSet_bar[j], teta, INSERT_VALUES);
		VecSetValueLocal(Vec_eps, LocalDOFSet_bar[j], eps, INSERT_VALUES);
	}

	VecAssemblyBegin(Ii);
	VecAssemblyEnd(Ii);
	VecGhostUpdateBegin(Ii, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Ii, INSERT_VALUES, SCATTER_FORWARD);

	VecAssemblyBegin(Phibar);
	VecAssemblyEnd(Phibar);
	VecGhostUpdateBegin(Phibar, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Phibar, INSERT_VALUES, SCATTER_FORWARD);

	VecAssemblyBegin(Vec_l);
	VecAssemblyEnd(Vec_l);
	VecGhostUpdateBegin(Vec_l, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Vec_l, INSERT_VALUES, SCATTER_FORWARD);

	VecAssemblyBegin(Vec_teta);
	VecAssemblyEnd(Vec_teta);
	VecGhostUpdateBegin(Vec_teta, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Vec_teta, INSERT_VALUES, SCATTER_FORWARD);

	VecAssemblyBegin(Vec_eps);
	VecAssemblyEnd(Vec_eps);
	VecGhostUpdateBegin(Vec_eps, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Vec_eps, INSERT_VALUES, SCATTER_FORWARD);

	PetscBarrier(NULL);

	VecRestoreArrayRead(cMg, &cMg_i);
	VecRestoreArrayRead(cOH, &cOH_i);
	VecRestoreArrayRead(Vec_l, &l_i);

	return 0;
}

//Doi: 10.1016/j.cis.2013.06.006
//EQ[36]
int WaterDissociation(Vec cH, Vec cOH, Vec Reaction) {
	//Input: cH, cOH
	//Output: ROH=RH

	Vec vtmp;
	VecDuplicate(cH, &vtmp);
	VecSet(vtmp, 1e-20);

	VecPointwiseMax(cH, cH, vtmp);// negative concentrations set to almost zero
	VecPointwiseMax(cOH, cOH, vtmp);// negative concentrations set to almost zero

	//EQ[5]
	VecSet(Reaction, 1.008e-8);//Kw
	VecPointwiseMult(vtmp, cH, cOH);//[cH][cOH]
	VecAXPY(Reaction, -1, vtmp);//Kw-[cH][cOH]
	VecSet(vtmp, 1.4e8);//kb
	VecPointwiseMult(Reaction, Reaction, vtmp);//kb(Kw-[cH][cOH])

	PetscBarrier(NULL);

	VecGhostUpdateBegin(Reaction, INSERT_VALUES, SCATTER_FORWARD);
	VecGhostUpdateEnd(Reaction, INSERT_VALUES, SCATTER_FORWARD);

	VecDestroy(&vtmp);

	return 0;
}

int PolarizationDataAssign(std::vector<double> &Phi, std::vector<double> &iMg, std::vector<double> &iAl, std::string Phifile, std::string iMgfile, std::string iAlfile) {

	double val;
	std::ifstream Myinputfile;
	Myinputfile.open(Phifile);
	while(Myinputfile>>val)
		Phi.push_back(val);
	Myinputfile.close();
	Myinputfile.clear();

	Myinputfile.open(iMgfile);
	while(Myinputfile>>val)
		iMg.push_back(val);
	Myinputfile.close();
	Myinputfile.clear();

	Myinputfile.open(iAlfile);
	while(Myinputfile>>val)
		iAl.push_back(val);
	Myinputfile.close();
	Myinputfile.clear();

	return 0;
}

int sigmal_Compute(std::vector<int> zi, std::vector<double> Di, std::vector<dolfin::Function> ci, dolfin::Function &sigmal) {

	sigmal = ci[0];
	(sigmal.vector())->operator*=(zi[0]*zi[0]*Di[0]);
	for (std::size_t i=1; i< zi.size(); i=i+1) {
		(sigmal.vector())->operator+=(*((ci[i].vector())->operator*(zi[i]*zi[i]*Di[i])));
	}
	(sigmal.vector())->operator*=((96487.3329*96487.3329)/(8.3145*295.15));

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
				VecSetValueLocal(as_type<const dolfin::PETScVector>(pH.vector())->vec(), j, -1*std::log10(1e-3*c_i[j]), INSERT_VALUES);
		}
		else {
			if (PetscIsNormalReal(14+1*std::log10(1e-3*c_i[j])))
				VecSetValueLocal(as_type<const dolfin::PETScVector>(pH.vector())->vec(), j, 14+1*std::log10(1e-3*c_i[j]), INSERT_VALUES);
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
