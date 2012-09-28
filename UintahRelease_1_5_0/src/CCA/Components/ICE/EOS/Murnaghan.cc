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


#include <CCA/Components/ICE/EOS/Murnaghan.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>
#include <iomanip>

using namespace Uintah;

Murnaghan::Murnaghan(ProblemSpecP& ps)
{
   // Constructor
  ps->require("n",n);
  ps->require("K",K);
  ps->require("rho0",rho0);
  ps->require("P0",P0);
}

Murnaghan::~Murnaghan()
{
}

void Murnaghan::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eos_ps = ps->appendChild("EOS");
  eos_ps->setAttribute("type","Murnaghan");
  eos_ps->appendElement("n",n);
  eos_ps->appendElement("K",K);
  eos_ps->appendElement("rho0",rho0);
  eos_ps->appendElement("P0",P0);
}


//__________________________________
double Murnaghan::computeRhoMicro(double press, double,
                                 double , double ,double)
{
  // Pointwise computation of microscopic density
  double rhoM;
  if(press>=P0){
    rhoM = rho0*pow((n*K*(press-P0)+1.),1./n);
  }
  else{
    rhoM = rho0*pow((press/P0),K*P0);
  }
  return rhoM;
}

//__________________________________
// Return (1/v)*(dv/dT)  (constant pressure thermal expansivity)
double Murnaghan::getAlpha(double, double, double, double)
{
  // No dependence on temperature
  double alpha=0.;
  return  alpha;
}

//__________________________________
void Murnaghan::computeTempCC(const Patch* patch,
                             const string& comp_domain,
                             const CCVariable<double>& /*press*/, 
                             const CCVariable<double>& /*gamma*/,
                             const CCVariable<double>& /* cv*/,
                             const CCVariable<double>& /*rhoM*/, 
                             CCVariable<double>& Temp,
                             Patch::FaceType face)
{
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
void Murnaghan::computePressEOS(double rhoM, double, double, double,
                          double& press, double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities
  if(rhoM>=rho0){
    press   = P0 + (1./(n*K))*(pow(rhoM/rho0,n)-1.);
    dp_drho = (1./(K*rho0))*pow((rhoM/rho0),n-1.);
  }
  else{
    press   = P0*pow(rhoM/rho0,(1./(K*P0)));
    dp_drho = (1./(K*rho0))*pow(rhoM/rho0,(1./(K*P0)-1.));
  }
  dp_de   = 0.0;
}

//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void Murnaghan::hydrostaticTempAdjustment(Patch::FaceType, 
                                         const Patch*,
                                         Iterator& ,
                                         Vector&,
                                         const CCVariable<double>&,
                                         const CCVariable<double>&,
                                         const Vector&,
                                         CCVariable<double>&)
{ 
//  throw InternalError( "ERROR:ICE:EOS:Murnaghan: hydrostaticTempAdj() \n"
//                       " has not been implemented", __FILE__, __LINE__ );
}
