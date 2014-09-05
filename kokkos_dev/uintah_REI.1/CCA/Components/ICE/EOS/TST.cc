#include <Packages/Uintah/CCA/Components/ICE/EOS/TST.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>
#include <iomanip>
using namespace Uintah;


TST::TST(ProblemSpecP& ps){
  // Constructor
  ps->require("a", a);
  ps->require("b", b);
  ps->require("u", u);
  ps->require("w", w);
  ps->require("Gamma", Gamma);
}


TST::~TST(){
}


void TST::outputProblemSpec(ProblemSpecP& ps){
  ProblemSpecP eos_ps = ps->appendChild("EOS");
  eos_ps->setAttribute("type","TST");
  eos_ps->appendElement("a", a);
  eos_ps->appendElement("b", b);
  eos_ps->appendElement("u", u);
  eos_ps->appendElement("w", w);
  eos_ps->appendElement("Gamma", Gamma);
}



//__________________________________
double TST::computeRhoMicro(double press, double gamma,
                            double cv, double Temp,double rho_guess){
  Pressure = press;
  Temperature = Temp;
  SpecificHeat = cv;

  /* Use a hybrid Newton-Bisection Method to compute the rho_micro.
     The solver guarantees to converge to a solution.
  */
  const double epsilon  = 1e-15;
  const double frac = 0.85;
  double rho_min = 0.0;     // Such that f(min) = -P < 0
  double rho_max = frac/b;  // Such that f(max) > 0
  if(func(rho_max) <= 0 ){
    cout<<"Pressure="<<press<<", TST EOS cannot handle such high pressure. Please choose other EOS ... "<<endl;
    exit(1);
  }
  
  IL = rho_min;
  IR = rho_max;

  double f = 0;
  double df_drho = 0;
  double delta_old, delta_new;

  double rhoM = rho_guess <= rho_max ? rho_guess : rho_max/2.0;
  double rhoM_start = rhoM;
 
  int iter = 0;  
  while(1){
    f = func(rhoM);
    setInterval(f, rhoM);
    
    if(fabs((IL-IR)/rhoM)<epsilon){
      return (IL+IR)/2.0;
    }
    
    delta_new = 1e100;
    while(1){
      df_drho = deri(rhoM);
      delta_old = delta_new;
      delta_new = -f/df_drho; 
      rhoM += delta_new;

      if(fabs(delta_new/rhoM)<epsilon){
	return rhoM;
      }      

      if(iter>=100){
	cout<<setprecision(15);
	cout<<"Not converging after 100 iterations in TST.cc."<<endl;
	cout<<"P="<<press<<" T="<<Temp<<" f="<<func(rhoM)<<" delta="<<delta_new
	    <<" rhoM="<<rhoM<<" rho_start="<<rhoM_start<<" "<<rho_max<<endl;
	exit(1);
      }
      
      if(rhoM<IL || rhoM>IR || fabs(delta_new)>fabs(delta_old*0.7)){
	break;
      }
      
      f = func(rhoM);
      setInterval(f, rhoM);      
      iter++;
    }
    
    rhoM = (IL+IR)/2.0;
    iter++;
  }
  return rhoM;
}


double TST::func(double rhoM){
  double P1 = (Gamma-1)*SpecificHeat*Temperature*rhoM/(1-b*rhoM);
  double P2 =  -a*rhoM*rhoM/((1+u*b*rhoM)*(1+w*b*rhoM));
  return P1 + P2 - Pressure;
}


double TST::deri(double rhoM){
  double denom = 1-b*rhoM;
  double P1 = (Gamma-1)*SpecificHeat*Temperature/(denom*denom);

  double d1 = 1+u*b*rhoM;
  double d2 = 1+w*b*rhoM;
  denom = d1*d2;
  double P2 = -2*a*rhoM/denom;

  double com = a*b*rhoM*rhoM/denom;
  double P3 =  com*(u/d1 + w/d2);

  return P1 + P2 + P3;
}


void TST::setInterval(double f, double rhoM){
  if(f < 0)   
    IL = rhoM;
  else if(f > 0)  
    IR = rhoM;
  else if(f ==0){    
    IL = rhoM;
    IR = rhoM; 
  } 
  return;
}


//__________________________________
// Return (1/v)*(dv/dT)  (constant pressure thermal expansivity)
double TST::getAlpha(double Temp, double sp_v, double P, double cv){
  // Cheating here a bit, computing v*(dT/dv) and returning the inverse of that
  double n1 = 2*P*sp_v + P*w*b + P*u*b;
  double n2 = sp_v-b;
  double n3 = P*sp_v*sp_v + P*sp_v*w*b + P*u*b*sp_v + P*u*b*b*w + a;

  double d1 = (sp_v+u*b) * (sp_v+w*b) * (Gamma-1) * cv;
  double d2 = sp_v + u*b;
  double d3 = sp_v + w*b;

  double dT_dv = n1*n2/d1 - n3*n2/d1*(1/d2+1/d3) + n3/d1;
  return 1/(sp_v*dT_dv);
}


//__________________________________
void TST::computeTempCC(const Patch* patch,
                        const string& comp_domain,
                        const CCVariable<double>& press, 
                        const CCVariable<double>& gamma,
                        const CCVariable<double>& cv,
                        const CCVariable<double>& rhoM, 
                        CCVariable<double>& Temp,
                        Patch::FaceType face){
  double sp_v, n1, n2, d1;  
  double P; 
  if(comp_domain == "WholeDomain") {
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      sp_v = 1.0/rhoM[c];
      P = press[c];
      n1 = sp_v-b;
      n2 = P*sp_v*sp_v + P*sp_v*w*b + P*u*b*sp_v + P*u*b*b*w + a;
      d1 = (sp_v+u*b) * (sp_v+w*b) * (Gamma-1) * cv[c];
      Temp[c]= n1*n2/d1;
    }
  } 
  // Although this isn't currently being used
  // keep it around it could be useful
  if(comp_domain == "FaceCells") {     
    for (CellIterator iter=patch->getFaceCellIterator(face);!iter.done();iter++){
      IntVector c = *iter;
      sp_v = 1.0/rhoM[c];
      P = press[c];
      n1 = sp_v-b;
      n2 = P*sp_v*sp_v + P*sp_v*w*b + P*u*b*sp_v + P*u*b*b*w + a;
      d1 = (sp_v+u*b) * (sp_v+w*b) * (Gamma-1) * cv[c];
      Temp[c]= n1*n2/d1;
    }
  }
  return;
}


//__________________________________
//
void TST::computePressEOS(double rhoM, double gamma,
                          double cv, double Temp,
                          double& press, double& dp_drho, double& dp_de){
  // Pointwise computation of thermodynamic quantities
  double sp_v = 1/rhoM;
  double p1 = (Gamma-1)*cv*Temp/(sp_v-b);
  double p2 = a/((sp_v+u*b)*(sp_v+w*b));
  double d1 = sp_v - b;
  double d2 = sp_v + u*b;
  double d3 = sp_v + w*b;

  press = p1-p2;
  dp_drho = sp_v*sp_v * (p1/d1 - p2*(1/d2+1/d3));
  dp_de   = (Gamma-1)/(sp_v-b);
  return;
}


//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void TST::hydrostaticTempAdjustment(Patch::FaceType, 
                                    const Patch*,
                                    const vector<IntVector>&,
                                    Vector&,
                                    const CCVariable<double>&,
                                    const CCVariable<double>&,
                                    const Vector&,
                                    CCVariable<double>&){ 
  throw InternalError( "ERROR:ICE:EOS:TST: hydrostaticTempAdj() \n"
                       " has not been implemented", __FILE__, __LINE__ );
}

