#include <Packages/Uintah/CCA/Components/ICE/EOS/IdealGas.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;

IdealGas::IdealGas(ProblemSpecP& ps)
{
   // Constructor
}

IdealGas::~IdealGas()
{
}

double IdealGas::computeRhoMicro(double press, double gamma,
                              double cv, double Temp)
{
  // Pointwise computation of microscopic density
  return  press/((gamma - 1.0)*cv*Temp);
}


void IdealGas::computeRhoMicro(const Patch* patch, 
                            CCVariable<double>& press, 
                            double gamma,double cv, 
                            constCCVariable<double>& Temp,
                            CCVariable<double>& rho_micro)
{

  for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++) {
    IntVector c = *iter;
    rho_micro[c] = press[c]/((gamma - 1.0)*cv*Temp[c]);
  }
}

//__________________________________
//
void IdealGas::computeTempCC(const Patch* patch,
                          const string& comp_domain,
                          const CCVariable<double>& press, 
                          const double& gamma,
                          const double& cv,
                          const CCVariable<double>& rho_micro, 
                          CCVariable<double>& Temp,
                          Patch::FaceType face)
{
  if(comp_domain == "WholeDomain") {
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++) {                     
      Temp[*iter]= press[*iter]/ ( (gamma - 1.0) * cv * rho_micro[*iter] );
    }
  } 
  // Although this isn't currently being used
  // keep it around it could be useful
  if(comp_domain == "FaceCells") {     
    for (CellIterator iter = patch->getFaceCellIterator(face);
         !iter.done();iter++) {                     
      Temp[*iter]= press[*iter]/ ( (gamma - 1.0) * cv * rho_micro[*iter] );
    }
  }
}

//__________________________________
//

void IdealGas::computePressEOS(double rhoM, double gamma,
                            double cv, double Temp,
                            double& press, double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities
  press   = (gamma - 1.0)*rhoM*cv*Temp;
  dp_drho = (gamma - 1.0)*cv*Temp;
  dp_de   = (gamma - 1.0)*rhoM;
}


void IdealGas::computePressEOS(const Patch* patch, 
                            CCVariable<double>& rhoM, 
                            double gamma, double cv, 
                            constCCVariable<double>& Temp,
                            CCVariable<double>& press, 
                            CCVariable<double>& dp_drho,
                            CCVariable<double>& dp_de)
{
  // Pointwise computation of thermodynamic quantities
  for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++) {
    IntVector c = *iter;
    press[c]   = (gamma - 1.0)*rhoM[c]*cv*Temp[c];
    dp_drho[c] = (gamma - 1.0)*cv*Temp[c];
    dp_de[c]   = (gamma - 1.0)*rhoM[c];
  }
}

//____________________________________________________________________
double IdealGas::getCompressibility(double press)
{
  return  1./press;
}

//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void IdealGas::hydrostaticTempAdjustment(Patch::FaceType face, 
                          const Patch* patch,
                          Vector& grav,
                          const double& gamma,
                          const double& cv,
                          const Vector& dx,
                          CCVariable<double>& Temp_CC)
  { 

    double delTemp_hydro;
    switch (face) {
    case Patch::xplus:
      delTemp_hydro = grav.x()*dx.x()/ ( (gamma - 1.0) * cv );
      for (CellIterator iter = patch->getFaceCellIterator(face,"plusEdgeCells");
         !iter.done();iter++) { 
        Temp_CC[*iter] += delTemp_hydro; 
      }
      break;
    case Patch::xminus:
      delTemp_hydro = grav.x()*dx.x()/ ( (gamma - 1.0) * cv );
      for (CellIterator iter = patch->getFaceCellIterator(face,"plusEdgeCells");
         !iter.done();iter++) { 
        Temp_CC[*iter] -= delTemp_hydro; 
      }
      break;
    case Patch::yplus:
      delTemp_hydro = grav.y()*dx.y()/ ( (gamma - 1.0) * cv );
      for (CellIterator iter = patch->getFaceCellIterator(face,"plusEdgeCells");
         !iter.done();iter++) { 
        Temp_CC[*iter] += delTemp_hydro; 
      }
      break;
    case Patch::yminus:
      delTemp_hydro = grav.y()*dx.y()/ ( (gamma - 1.0) * cv );
      for (CellIterator iter = patch->getFaceCellIterator(face,"plusEdgeCells");
         !iter.done();iter++) { 
        Temp_CC[*iter] -= delTemp_hydro; 
      }
      break;
    case Patch::zplus:
      delTemp_hydro = grav.z()*dx.z()/ ( (gamma - 1.0) * cv );
      for (CellIterator iter = patch->getFaceCellIterator(face,"plusEdgeCells");
         !iter.done();iter++) { 
        Temp_CC[*iter] += delTemp_hydro; 
      }
      break;
    case Patch::zminus:
      delTemp_hydro = grav.z()*dx.z()/ ( (gamma - 1.0) * cv );
      for (CellIterator iter = patch->getFaceCellIterator(face,"plusEdgeCells");
         !iter.done();iter++) { 
        Temp_CC[*iter] -= delTemp_hydro; 
      }
      break;
    case Patch::numFaces:
      break;
   case Patch::invalidFace:
      break;
    }

  }
