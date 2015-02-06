/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <CCA/Components/ICE/EOS/JWL.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>
#include <iomanip>

using namespace std;
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
  // Set up iteration variables for the solver
  IterationVariables iterVar;
  iterVar.Pressure = press;
  iterVar.Temperature = Temp;
  iterVar.SpecificHeat = cv;
  
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
  iterVar.IL = rho_min;
  iterVar.IR = rho_max;

  double f = 0;
  double df_drho = 0;
  double delta_old, delta_new;

  double rhoM = rho_guess <= rho_max ? rho_guess : rho_max/2.0;
  //double rhoM_start = rhoM;
 
  int iter = 0;
  while(1){
    f = func(rhoM, &iterVar);
    setInterval(f, rhoM, &iterVar);
    
    if(fabs((iterVar.IL-iterVar.IR)/rhoM)<epsilon){
      return (iterVar.IL+iterVar.IR)/2.0;
    }
    
    delta_new = 1e100;
    while(1){
      df_drho = deri(rhoM, &iterVar);
      delta_old = delta_new;
      delta_new = -f/df_drho; 
      rhoM += delta_new;
      
      if(fabs(delta_new/rhoM)<epsilon){
        return rhoM;
      }
      
      if(iter>=100){
        ostringstream warn;
        warn << setprecision(15);
        warn << "ERROR:ICE:JWL::computeRhoMicro not converging. \n";
        warn << "press= " << press << " temp=" << Temp << "\n";
        warn << "delta= " << delta_new << " rhoM= " << rhoM << " f = " << f 
             <<" df_drho =" << df_drho << "\n";
        throw InternalError(warn.str(), __FILE__, __LINE__);
      }
      
      if(rhoM<iterVar.IL || rhoM>iterVar.IR || fabs(delta_new)>fabs(delta_old*0.7)){
        break;
      }
      
      f = func(rhoM, &iterVar);
      setInterval(f, rhoM, &iterVar);      
      iter++;
    }
    
    rhoM = (iterVar.IL+iterVar.IR)/2.0;
    iter++;
  }
  
  return rhoM;
}


double JWL::func(double rhoM, IterationVariables *iterVar){
  if(rhoM == 0){
    return -(iterVar->Pressure);
  }
  double V  = rho0/rhoM;
  double P1 = A*exp(-R1*V);
  double P2 = B*exp(-R2*V);
  double P3 = om*iterVar->SpecificHeat*iterVar->Temperature*rhoM;
  return P1 + P2 + P3 - iterVar->Pressure;
}


double JWL::deri(double rhoM, IterationVariables *iterVar){
  double V  = rho0/rhoM;
  double P1 = A*exp(-R1*V);
  double P2 = B*exp(-R2*V);
  double P3 = om*iterVar->SpecificHeat*iterVar->Temperature*rhoM;
  return (P1*R1*V + P2*R2*V + P3)/rhoM;
}


void JWL::setInterval(double f, double rhoM, IterationVariables *iterVar){
  if(f < 0)   
    iterVar->IL = rhoM;
  else if(f > 0)  
    iterVar->IR = rhoM;
  else if(f ==0){    
    iterVar->IL = rhoM;
    iterVar->IR = rhoM; 
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
    Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;  
     
    for (CellIterator iter=patch->getFaceIterator(face,MEC);!iter.done();iter++){
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
                                    Iterator& ,
                                    Vector&,
                                    const CCVariable<double>&,
                                    const CCVariable<double>&,
                                    const Vector&,
                                    CCVariable<double>&){ 
  throw InternalError( "ERROR:ICE:EOS:JWL: hydrostaticTempAdj() \n"
                       " has not been implemented", __FILE__, __LINE__ );
}

