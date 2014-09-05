#include <Packages/Uintah/CCA/Components/ICE/EOS/JWL.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>
#include <iomanip>
using namespace Uintah;


JWL::JWL(ProblemSpecP& ps){
   // Constructor
  ps->require("A",A);
  ps->require("B",B);
  ps->require("R1",R1);
  ps->require("R2",R2);
  ps->require("om",om);
  ps->require("rho0",rho0);
}


JWL::~JWL(){
}


void JWL::outputProblemSpec(ProblemSpecP& ps){
  ProblemSpecP eos_ps = ps->appendChild("EOS");
  eos_ps->setAttribute("type","JWL");
  eos_ps->appendElement("A",A);
  eos_ps->appendElement("B",B);
  eos_ps->appendElement("R1",R1);
  eos_ps->appendElement("R2",R2);
  eos_ps->appendElement("om",om);
  eos_ps->appendElement("rho0",rho0);
}



//__________________________________
double JWL::computeRhoMicro(double press, double,
                            double cv, double Temp,double rho_guess){
  Pressure = press;
  Temperature = Temp;
  SpecificHeat = cv;
  
  /* Use a hybrid Newton-Bisection Method to compute the rho_micro.
     The solver guarantees to converge to a solution.

     Modified by:
     Changwei Xiong
     Department of Chemistry 
     University of Utah
  */
  double epsilon  = 1e-15;
  double rho_min = 0.0;                      // Such that f(min) < 0
  double rho_max = press*1.001/(om*cv*Temp); // Such that f(max) > 0
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
	cout<<"Not converging after 100 iterations in JWL.cc."<<endl;
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


double JWL::func(double rhoM){
  if(rhoM == 0){
    return -Pressure;
  }
  double V  = rho0/rhoM;
  double P1 = A*exp(-R1*V);
  double P2 = B*exp(-R2*V);
  double P3 = om*SpecificHeat*Temperature*rhoM;
  return P1 + P2 + P3 - Pressure;
}


double JWL::deri(double rhoM){
  double V  = rho0/rhoM;
  double P1 = A*exp(-R1*V);
  double P2 = B*exp(-R2*V);
  double P3 = om*SpecificHeat*Temperature*rhoM;
  return (P1*R1*V + P2*R2*V + P3)/rhoM;
}


void JWL::setInterval(double f, double rhoM){
  if(f < 0)   
    IL = rhoM;
  else if(f > 0)  
    IR = rhoM;
  else if(f ==0){    
    IL = rhoM;
    IR = rhoM; 
  } 
}


//__________________________________
// Return (1/v)*(dv/dT)  (constant pressure thermal expansivity)
double JWL::getAlpha(double, double sp_v, double P, double cv){
  // Cheating here a bit, computing v*(dT/dv) and returning the inverse of that
  double V  = rho0*sp_v;
  double P1 = A*exp(-R1*V);
  double P2 = B*exp(-R2*V);

  double alpha = om*cv/(sp_v * (P + P1*(V*R1-1)+P2*(V*R2-1)));
  return  alpha;
}


//__________________________________
void JWL::computeTempCC(const Patch* patch,
                        const string& comp_domain,
                        const CCVariable<double>& press, 
                        const CCVariable<double>&,
                        const CCVariable<double>& cv,
                        const CCVariable<double>& rhoM, 
                        CCVariable<double>& Temp,
                        Patch::FaceType face){
  if(comp_domain == "WholeDomain") {
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      double V  = rho0/rhoM[c];
      Temp[c]= (press[c]- A*exp(-R1*V) - B*exp(-R2*V)) / (om*rhoM[c]*cv[c]);
    }
  } 
  // Although this isn't currently being used
  // keep it around it could be useful
  if(comp_domain == "FaceCells") {     
   for (CellIterator iter=patch->getFaceCellIterator(face);!iter.done();iter++){
      IntVector c = *iter;
      double V  = rho0/rhoM[c];
      Temp[c]= (press[c] - A*exp(-R1*V) - B*exp(-R2*V)) / (om*rhoM[c]*cv[c]);
   }
  }
}


//__________________________________
//
void JWL::computePressEOS(double rhoM, double,
                          double cv, double Temp,
                          double& press, double& dp_drho, double& dp_de){
  // Pointwise computation of thermodynamic quantities
  double V  = rho0/rhoM;
  double P1 = A*exp(-R1*V);
  double P2 = B*exp(-R2*V);
  double P3 = om*cv*Temp*rhoM;

  press   = P1 + P2 + P3;
  dp_drho = (R1*rho0*P1 + R2*rho0*P2)/(rhoM*rhoM) + om*cv*Temp;
  dp_de   = om*rhoM;
}


//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void JWL::hydrostaticTempAdjustment(Patch::FaceType, 
                                    const Patch*,
                                    const vector<IntVector>&,
                                    Vector&,
                                    const CCVariable<double>&,
                                    const CCVariable<double>&,
                                    const Vector&,
                                    CCVariable<double>&){ 
  throw InternalError( "ERROR:ICE:EOS:JWL: hydrostaticTempAdj() \n"
                       " has not been implemented", __FILE__, __LINE__ );
}

