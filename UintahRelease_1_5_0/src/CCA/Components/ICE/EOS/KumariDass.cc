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


#include <CCA/Components/ICE/EOS/KumariDass.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace Uintah;


KumariDass::KumariDass(ProblemSpecP& ps){
   // Constructor
  ps->require("B0",      B0);
  ps->require("B0prime", B0prime);
  ps->require("lambda",  lambda);
  ps->require("rho0",    rho0);
}


KumariDass::~KumariDass(){
}


void KumariDass::outputProblemSpec(ProblemSpecP& ps){
  ProblemSpecP eos_ps = ps->appendChild("EOS");
  eos_ps->setAttribute("type","KumariDass");
  eos_ps->appendElement("B",       B0);
  eos_ps->appendElement("B0prime", B0prime);
  eos_ps->appendElement("lambda",  lambda);
  eos_ps->appendElement("rho0",    rho0);
}



//__________________________________
double KumariDass::computeRhoMicro(double press, double,
                                   double cv, double Temp,double rho_guess){

  double lambdaB0 = lambda*B0;
  double rhoM     = rho0*std::exp(-std::log(B0prime/(lambdaB0))
                                  /(lambdaB0-B0prime));
    
  
  return rhoM;
}


//__________________________________
// Return (1/v)*(dv/dT)  (constant pressure thermal expansivity)
double KumariDass::getAlpha(double, double sp_v, double P, double cv){
  // no temperature dependence
  return  0.0;
}


//__________________________________
void KumariDass::computeTempCC(const Patch* patch,
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
        
      Temp[c]= 300.0;
    }
  } 
  // Although this isn't currently being used
  // keep it around it could be useful
  if(comp_domain == "FaceCells") {   
    Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;  
     
    for (CellIterator iter=patch->getFaceIterator(face,MEC);!iter.done();iter++){
      IntVector c = *iter;
      Temp[c]= 300.0;
    }
  }
}


//__________________________________
//
void KumariDass::computePressEOS(double rhoM, double,
                                 double cv, double Temp,
                                 double& press, double& dp_drho, double& dp_de){
  // Pointwise computation of thermodynamic quantities
  double V        = rhoM/rho0;
  double lambdaB0 = lambda * B0;
  double powExpr  = std::pow(V,-lambdaB0+B0prime);
  
  // Compute returnables
  press   = (lambdaB0*powExpr+B0prime)
           /(lambda*lambdaB0+lambda*B0prime);
  dp_drho = (B0*powExpr*(-lambdaB0+B0prime))
           /(rhoM*lambdaB0+B0prime);
  dp_de   = 0.0;
}


//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void KumariDass::hydrostaticTempAdjustment(Patch::FaceType, 
                                           const Patch*,
                                           Iterator& ,
                                           Vector&,
                                           const CCVariable<double>&,
                                           const CCVariable<double>&,
                                           const Vector&,
                                           CCVariable<double>&){ 
  throw InternalError( "ERROR:ICE:EOS:KumariDass: hydrostaticTempAdj() \n"
                       " has not been implemented", __FILE__, __LINE__ );
}

