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
	double teta = std::exp(-konst*t+std::log(1-teta0)) + 1;//teta=-exp(-konst*t+ln(1-teta0))+1
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
	konst = konst*A;
	double eps = std::exp(-konst*t+std::log(eps0));//eps=exp(-konst*t+ln(eps0))
	return eps;
}

//Doi: 10.1149/2.0071501jes
//iElectrode:=absolute current
//EQ[12]==EQ[13]=>phi=-1.4201232315167845415877017558844 Solved in matlab
//Matlab command:
//syms phi
//eq =(7*exp((10608857842922292*phi)/674214345399337 + 4023409336928279241/168553586349834250))/250 - (7*exp(- (8487086274337833*phi)/674214345399337 - 12874909878170492661/674214345399337000))/250 == (9*10^(phi/118 + 119/23600))/500 + (13*exp(phi/500 + 119/100000))/(125*((52*exp(phi/500 + 119/100000))/4335 + 1))
//solve(eq, phi)
//End Matlab command
//iAl=iMg=0.120312326899797
double iElectrodeFormula(double teta, double eps) {
	double iExchange;
	iExchange = 0.12031232;
	return (1 - teta + eps*teta)*iExchange*10; //mA/(cm)^2--->A/m^2
}

//Doi: 10.1149/2.0071501jes
//Moving interface velocity uTotal
//EQ[16]
double uTotal(double iElectrode, double eps, double cMg, double cOH) {
	return -(0.024*iElectrode)/(2*96487*1735) + (7.4e-4*(cMg*cOH*cOH - 0.45)*0.058) / ((1 - eps)*2450);
}
//Cor:=Corrosion
double uCor(double iElectrode) {
	return -(0.024*iElectrode)/(2*96487*1735);
}
//Dep:=Deposit
double uDep(double iElectrode, double eps, double cMg, double cOH) {
	return (7.4e-4*(cMg*cOH*cOH - 0.45)*0.058)/((1 - eps)*2450);
}

//Doi: 10.1149/2.0071501jes
//EQ[11]
int BoundaryCurrent(std::vector<size_t> GlobalDOFSet_bar, double t, Vec cMg, Vec cOH, Vec cReactionLimiter, double eps0, double teta0, double l, Vec &Ii) {
	//Input: Dofset on boundary, t, cMg, cOH, Resource limit, eps0, teta0, l
	//Output: Ii
	
	VecSet(Ii, 0);

	VecScatter par2seq;
	Vec cMg_SEQ, cOH_SEQ, cR_SEQ;

	VecScatterCreateToAll(cMg, &par2seq, &cMg_SEQ);
	VecScatterBegin(par2seq, cMg, cMg_SEQ, INSERT_VALUES, SCATTER_FORWARD);
	VecScatterEnd(par2seq, cMg, cMg_SEQ, INSERT_VALUES, SCATTER_FORWARD);

	VecScatterDestroy(&par2seq);

	VecScatterCreateToAll(cOH, &par2seq, &cOH_SEQ);
	VecScatterBegin(par2seq, cOH, cOH_SEQ, INSERT_VALUES, SCATTER_FORWARD);
	VecScatterEnd(par2seq, cOH, cOH_SEQ, INSERT_VALUES, SCATTER_FORWARD);

	VecScatterDestroy(&par2seq);

	VecScatterCreateToAll(cReactionLimiter, &par2seq, &cR_SEQ);
	VecScatterBegin(par2seq, cReactionLimiter, cR_SEQ, INSERT_VALUES, SCATTER_FORWARD);
	VecScatterEnd(par2seq, cReactionLimiter, cR_SEQ, INSERT_VALUES, SCATTER_FORWARD);

	VecScatterDestroy(&par2seq);

	PetscScalar eps;
	PetscScalar teta;
	PetscScalar iElectrode;

	const PetscScalar* cMg_i;
	const PetscScalar* cOH_i;
	const PetscScalar* cReactionLimiter_i;

	VecGetArrayRead(cMg_SEQ, &cMg_i);
	VecGetArrayRead(cOH_SEQ, &cOH_i);
	VecGetArrayRead(cR_SEQ, &cReactionLimiter_i);

	//Computing the electerical current
	for (size_t j = 0; j < GlobalDOFSet_bar.size(); j = j + 1) {
		eps = epsODE(t, eps0, cMg_i[GlobalDOFSet_bar[j]], cOH_i[GlobalDOFSet_bar[j]]);
		teta = tetaODE(t, teta0, l, eps, cMg_i[GlobalDOFSet_bar[j]], cOH_i[GlobalDOFSet_bar[j]]);
		iElectrode = iElectrodeFormula(teta, eps);
		if(iElectrode<=cReactionLimiter_i[GlobalDOFSet_bar[j]])
			VecSetValue(Ii, GlobalDOFSet_bar[j], iElectrode, INSERT_VALUES);
		else
			VecSetValue(Ii, GlobalDOFSet_bar[j], cReactionLimiter_i[GlobalDOFSet_bar[j]], INSERT_VALUES);
	}

	VecAssemblyBegin(Ii);
	VecAssemblyEnd(Ii);

	VecRestoreArrayRead(cMg, &cMg_i);
	VecRestoreArrayRead(cOH, &cOH_i);
	VecRestoreArrayRead(cReactionLimiter, &cReactionLimiter_i);

	PetscBarrier(NULL);

	VecDestroy(&cMg_SEQ);
	VecDestroy(&cOH_SEQ);
	VecDestroy(&cR_SEQ);

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
