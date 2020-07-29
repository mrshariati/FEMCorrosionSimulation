#include <dolfin.h>

using namespace dolfin;

//Doi: 10.1149/2.0071501jes
//teta:=affected surface fraction
//EQ[7] and EQ[8]
double tetaODE(double t, double teta0, double l, double eps, double cMg, double cOH) {
	double konst;
	double A;
	konst = (3.7e-7*0.058) / (l*(1 - eps)*2360);
	A = cMg*cOH*cOH - 0.450;
	if(A >= 0)
		konst = konst*A;
	double teta = 1-(1-teta0)*std::exp(-konst*t);//teta=-exp(-konst*t+ln(1-teta0))+1
	return teta;
}

//Doi: 10.1149/2.0071501jes
//eps:=porosity of the deposit
//EQ[10]
double epsODE(double t, double eps0, double cMg, double cOH) {
	double konst;
	double A;
	konst = (7.4e-4*0.058)/2360;
	A = cMg*cOH*cOH - 0.450;
	if(A >= 0)
		konst = konst*A;
	double eps = eps0*std::exp(-konst*t);//eps=exp(-konst*t+ln(eps0))
	return eps;
}

//Doi: 10.1149/2.0071501jes
//iExchange:=absolute current
//EQ[12]==EQ[13]=>phi=-1.48429655 Solved in matlab
//Matlab command:
//U_Electrolyte=-1.51:0.0001:-1.4;%-1.462456
//U=U_Electrolyte;
//iMg=(0.28.*(exp(39.31786.*0.4.*(U+1.517))-exp(-39.31786.*0.32.*(U+1.517))));
//iAl=((0.104.*exp(1e3.*(U+0.595)./500)./(0.012.*exp(1e3.*(U+0.595)./500)+1)+0.18.*10.^(-1e3.*(U+1.41)./118)));seems one has mA/cm^2 and the other A/m^2
//plot(U,iAl-iMg,'b')
//iMg=iAl=0.51928 A/(m)^2
//End Matlab command

//equilibrium potential depends on combination of metals and we took E_eq=1.48
double iElectrodeFormula(double teta, double eps) {
	double iExchange;
	iExchange = 0.51928;
	return (1 - teta + eps*teta)*iExchange;
}

//Doi: 10.1149/2.0071501jes
//Moving interface velocity uTotal
//EQ[16]
double uTotal(double iElectrode, double eps, double cMg, double cOH) {
	return -(0.024*iElectrode)/(2*96487*1735) + (7.4e-4*(cMg*cOH*cOH - 0.45)*0.058) / ((1 - eps)*2360);
}
//Cor:=Corrosion
double uCor(double iElectrode) {
	return -(0.024*iElectrode)/(2*96487*1735);
}
//Dep:=Deposit
double uDep(double eps, double cMg, double cOH) {
	double konst;
	double A;
	konst = (7.4e-4*0.058);
	A = cMg*cOH*cOH - 0.450;
	if(A >= 0)
		konst = konst*A;
	return konst/((1 - eps)*2360);
}

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
int iMg(std::vector<size_t> LocalDOFSet_bar, double t, Vec cMg, Vec cOH, double eps0, double teta0, double l0, Vec &Ii, Vec &BoundaryPhi, std::vector<double> UData, std::vector<double> iMgData) {
	//Input: Dofset on boundary, t, cMg, cOH, Resource limit, eps0, teta0, l0
	//Output: Ii

	VecSet(Ii, 0);

	PetscScalar eps;
	PetscScalar l;
	PetscScalar teta;
	PetscScalar iElectrode;
	PetscScalar UElectrode;

	const PetscScalar* cMg_i;
	const PetscScalar* cOH_i;

	VecGetArrayRead(cMg, &cMg_i);
	VecGetArrayRead(cOH, &cOH_i);

	//Computing the electerical current
	for (size_t j = 0; j < LocalDOFSet_bar.size(); j = j + 1) {
		eps = epsODE(t, eps0, cMg_i[LocalDOFSet_bar[j]], cOH_i[LocalDOFSet_bar[j]]);
		l = uDep(eps, cMg_i[LocalDOFSet_bar[j]], cOH_i[LocalDOFSet_bar[j]])*t + l0;
		teta = tetaODE(t, teta0, l, eps, cMg_i[LocalDOFSet_bar[j]], cOH_i[LocalDOFSet_bar[j]]);
		iElectrode = iElectrodeFormula(teta, eps);
		VecSetValueLocal(Ii, LocalDOFSet_bar[j], iElectrode, INSERT_VALUES);
		//extracting dirichlet condition for electric field
		UElectrode = Current2ElectricField(UData, iMgData, iElectrode);
		VecSetValueLocal(BoundaryPhi, LocalDOFSet_bar[j], UElectrode, INSERT_VALUES);
	}

	VecAssemblyBegin(Ii);
	VecAssemblyEnd(Ii);

	VecAssemblyBegin(BoundaryPhi);
	VecAssemblyEnd(BoundaryPhi);

	PetscBarrier(NULL);

	VecRestoreArrayRead(cMg, &cMg_i);
	VecRestoreArrayRead(cOH, &cOH_i);

	return 0;
}

//Doi: 10.1149/2.0071501jes
//EQ[11], EQ[13]
int iOH(std::vector<size_t> LocalDOFSet_bar, double t, Vec cMg, Vec cOH, double eps0, double teta0, double l0, Vec &Ii, Vec &BoundaryPhi, std::vector<double> UData, std::vector<double> iAlData) {
	//Input: Dofset on boundary, t, cMg, cOH, Resource limit, eps0, teta0, l0
	//Output: Ii

	VecScale(Ii, 2);

	PetscScalar eps;
	PetscScalar l;
	PetscScalar teta;
	PetscScalar iElectrode;
	PetscScalar UElectrode;

	const PetscScalar* cMg_i;
	const PetscScalar* cOH_i;

	VecGetArrayRead(cMg, &cMg_i);
	VecGetArrayRead(cOH, &cOH_i);

	//Computing the electerical current
	for (size_t j = 0; j < LocalDOFSet_bar.size(); j = j + 1) {
		eps = epsODE(t, eps0, cMg_i[LocalDOFSet_bar[j]], cOH_i[LocalDOFSet_bar[j]]);
		l = uDep(eps, cMg_i[LocalDOFSet_bar[j]], cOH_i[LocalDOFSet_bar[j]])*t + l0;
		teta = tetaODE(t, teta0, l, eps, cMg_i[LocalDOFSet_bar[j]], cOH_i[LocalDOFSet_bar[j]]);
//if((1 - teta + eps*teta)>0.9||(1 - teta + eps*teta)<0.5){std::cout<<"iOH_reduced issue: "<<"teta = "<<teta<<", eps = "<<eps<<", u_dep = "<<uDep(eps, cMg_i[LocalDOFSet_bar[j]], cOH_i[LocalDOFSet_bar[j]])<<std::endl;std::cin>>iElectrode;}
		iElectrode = iElectrodeFormula(teta, eps);
		VecSetValueLocal(Ii, LocalDOFSet_bar[j], iElectrode, INSERT_VALUES);
		//extracting dirichlet condition for electric field
		UElectrode = Current2ElectricField(UData, iAlData, iElectrode);
		VecSetValueLocal(BoundaryPhi, LocalDOFSet_bar[j], UElectrode, INSERT_VALUES);
	}

	VecAssemblyBegin(Ii);
	VecAssemblyEnd(Ii);

	VecAssemblyBegin(BoundaryPhi);
	VecAssemblyEnd(BoundaryPhi);

	PetscBarrier(NULL);

	VecRestoreArrayRead(cMg, &cMg_i);
	VecRestoreArrayRead(cOH, &cOH_i);

	return 0;
}


//Doi: 10.1149/2.0071501jes
//EQ[5]
int WaterDissociation(Vec cH, Vec cOH, Vec &Reaction) {
	//Input: cH, cOH
	//Output: ROH=RH

	Vec vtmp;
	VecDuplicate(cH, &vtmp);
	VecSet(vtmp, 0);

	VecPointwiseMax(cH, cH, vtmp);// negative concentrations set to zero
	VecPointwiseMax(cOH, cOH, vtmp);// negative concentrations set to zero

	//EQ[5]
	VecSet(Reaction, 1e-8);//Kw
	VecPointwiseMult(vtmp, cH, cOH);
	VecAXPY(Reaction, -1, vtmp);
	VecScale(Reaction, 1.4e4);

	VecSet(vtmp, 55.555);//positive reaction is subjected to available water
	VecPointwiseMin(Reaction, Reaction, vtmp);

	VecCopy(cH, vtmp);//negative reaction is subjected to available cH and cOH
	VecScale(vtmp, -1);
	VecPointwiseMax(Reaction, Reaction, vtmp);
	VecCopy(cOH, vtmp);
	VecScale(vtmp, -1);
	VecPointwiseMax(Reaction, Reaction, vtmp);

	PetscBarrier(NULL);

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

int kappa_Compute(std::vector<int> zi, std::vector<double> Di, std::vector<dolfin::Function> ci, dolfin::Function &kappa) {

	kappa = ci[0];
	(kappa.vector())->operator*=(zi[0]*zi[0]*Di[0]);
	for (std::size_t i=1; i< zi.size(); i=i+1) {
		(kappa.vector())->operator+=(*((ci[i].vector())->operator*(zi[i]*zi[i]*Di[i])));
	}
	(kappa.vector())->operator*=(39.3179);

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

	return 0;
}
