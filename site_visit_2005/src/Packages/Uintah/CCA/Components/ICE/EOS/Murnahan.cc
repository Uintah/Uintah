#include <Packages/Uintah/CCA/Components/ICE/EOS/Murnahan.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>
#include <iomanip>

using namespace Uintah;

Murnahan::Murnahan(ProblemSpecP& ps)
{
   // Constructor
  ps->require("n",n);
  ps->require("K",K);
  ps->require("rho0",rho0);
  ps->require("P0",P0);
}

Murnahan::~Murnahan()
{
}
//__________________________________
double Murnahan::computeRhoMicro(double press, double,
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
double Murnahan::getAlpha(double, double, double, double)
{
  // No dependence on temperature
  double alpha=0.;
  return  alpha;
}

//__________________________________
void Murnahan::computeTempCC(const Patch* patch,
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
   for (CellIterator iter=patch->getFaceCellIterator(face);!iter.done();iter++){
      IntVector c = *iter;
      Temp[c]= 300.0;
   } 
  }
}

//__________________________________
void Murnahan::computePressEOS(double rhoM, double, double, double,
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
void Murnahan::hydrostaticTempAdjustment(Patch::FaceType, 
                                         const Patch*,
                                         const vector<IntVector>&,
                                         Vector&,
                                         const CCVariable<double>&,
                                         const CCVariable<double>&,
                                         const Vector&,
                                         CCVariable<double>&)
{ 
  throw InternalError( "ERROR:ICE:EOS:Murnahan: hydrostaticTempAdj() \n"
                       " has not been implemented" );
}
