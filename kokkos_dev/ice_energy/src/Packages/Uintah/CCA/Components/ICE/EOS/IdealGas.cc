
#include <Packages/Uintah/CCA/Components/ICE/EOS/IdealGas.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/Thermo/ThermoInterface.h>

using namespace Uintah;

IdealGas::IdealGas(ProblemSpecP&, ICEMaterial* ice_matl )
  : EquationOfState(ice_matl)
{
   // Constructor
}

IdealGas::~IdealGas()
{
}
//__________________________________
double IdealGas::computeRhoMicro(double press, double gamma,
                                 double cv, double Temp, double)
{
  // Pointwise computation of microscopic density
  return  press/((gamma - 1.0)*cv*Temp);
}

//__________________________________
void IdealGas::computeTempCC(const Patch* patch,
                             const string& comp_domain,
                             const CCVariable<double>& press, 
                             const CCVariable<double>& gamma,
                             const CCVariable<double>& cv,
                             const CCVariable<double>& sp_vol, 
                             CCVariable<double>& Temp,
                             Patch::FaceType face)
{
  if(comp_domain == "WholeDomain") {
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      Temp[c]= press[c]/ ( (gamma[c] - 1.0) * cv[c] / sp_vol[c] );
    }
  } 
  // Although this isn't currently being used
  // keep it around it could be useful
  if(comp_domain == "FaceCells") {     
    for (CellIterator iter = patch->getFaceCellIterator(face);
         !iter.done();iter++) {
      IntVector c = *iter;                    
      Temp[c]= press[c]/ ( (gamma[c] - 1.0) * cv[c] / sp_vol[c] );
    }
  }
}

//__________________________________
void IdealGas::computePressEOS(double rhoM, double gamma,
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
double IdealGas::getAlpha(double Temp, double , double , double )
{
  return  1.0/Temp;
}

//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void IdealGas::hydrostaticTempAdjustment(Patch::FaceType face, 
                                         const Patch* patch,
                                         const vector<IntVector>& bound,
                                         Vector& gravity,
                                         const CCVariable<double>& gamma,
                                         const CCVariable<double>& cv,
                                         const Vector& cell_dx,
                                         CCVariable<double>& Temp_CC)
{ 
  IntVector axes = patch->faceAxes(face);
  int P_dir = axes[0];  // principal direction
  double plusMinusOne = patch->faceDirection(face)[P_dir];
  // On xPlus yPlus zPlus you add the increment 
  // on xminus yminus zminus you subtract the increment
  double dx_grav = gravity[P_dir] * cell_dx[P_dir];
  
   vector<IntVector>::const_iterator iter;  
   for (iter=bound.begin(); iter != bound.end(); iter++) {
     IntVector c = *iter;
     Temp_CC[c] += plusMinusOne * dx_grav/( (gamma[c] - 1.0) * cv[c] ); 
  }
}
