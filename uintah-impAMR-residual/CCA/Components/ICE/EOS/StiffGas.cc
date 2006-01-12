#include <Packages/Uintah/CCA/Components/ICE/EOS/StiffGas.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InternalError.h>

using namespace Uintah;

StiffGas::StiffGas(ProblemSpecP& ps)
{
}

StiffGas::~StiffGas()
{
}
//__________________________________
double StiffGas::computeRhoMicro(double press, double gamma,
                                 double cv, double Temp)
{
  // Pointwise computation of microscopic density
  return  press/((gamma - 1.0)*cv*Temp);
}

//__________________________________
void StiffGas::computeTempCC(const Patch* patch,
                             const string& comp_domain,
                             const CCVariable<double>& press, 
                             const CCVariable<double>& gamma,
                             const CCVariable<double>& cv,
                             const CCVariable<double>& rho_micro, 
                             CCVariable<double>& Temp,
                             Patch::FaceType face)
{
  if(comp_domain == "WholeDomain") {
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      Temp[c]= press[c]/ ( (gamma[c] - 1.0) * cv[c] * rho_micro[c] );
    }
  } 
  // Although this isn't currently being used
  // keep it around it could be useful
  if(comp_domain == "FaceCells") {     
    for (CellIterator iter = patch->getFaceCellIterator(face);
         !iter.done();iter++) {
      IntVector c = *iter;                    
      Temp[c]= press[c]/ ( (gamma[c] - 1.0) * cv[c] * rho_micro[c] );
    }
  }
}

//__________________________________
void StiffGas::computePressEOS(double rhoM, double gamma,
                            double cv, double Temp,
                            double& press, double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities
  press   = (gamma - 1.0)*rhoM*cv*Temp;
  dp_drho = (gamma - 1.0)*cv*Temp;
  dp_de   = (gamma - 1.0)*rhoM;
}
//__________________________________
// Return (1/v)*(dv/dT)  (constant pressure thermal expansivity)
double StiffGas::getAlpha(double Temp, double , double , double )
{
  return  1.0/Temp;
}

//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void StiffGas::hydrostaticTempAdjustment(Patch::FaceType, 
                                         const Patch*,
                                         const vector<IntVector>&,
                                         Vector&,
                                         const CCVariable<double>&,
                                         const CCVariable<double>&,
                                         const Vector&,
                                         CCVariable<double>&)
{ 
  throw InternalError( "ERROR:ICE:EOS:StiffGas: hydrostaticTempAdj() \n"
                       " has not been implemented", __FILE__, __LINE__ );
}
