#include <Packages/Uintah/CCA/Components/ICE/EOS/JWL.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;

JWL::JWL(ProblemSpecP& ps)
{
   // Constructor
  A = 5.484e11;  // Pascals
  B = 9.375e9;   // Pascals
  R1 = 4.94;
  R2 = 1.21;
  om = 0.28;
  rho0 = 1630.;  // kg/m^3
  c_v = 716.;        // J/(kg K)
}

JWL::~JWL()
{
}

double JWL::computeRhoMicro(double press, double gamma,
                              double cv, double Temp)
{
  // Pointwise computation of microscopic density
  return  press/((gamma - 1.0)*cv*Temp);
}

//__________________________________
//
void JWL::computeTempCC(const Patch* patch,
                          const string& comp_domain,
                          const CCVariable<double>& press, 
                          const double& gamma,
                          const double& cv,
                          const CCVariable<double>& rho_micro, 
                          CCVariable<double>& Temp,
                          Patch::FaceType face)
{
  if(comp_domain == "WholeDomain") {
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
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

void JWL::computePressEOS(double rhoM, double gamma,
                          double cv, double Temp,
                          double& press, double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities

  press   = A*(1.-om*rhoM/(R1*rho0))*exp(-R1*rho0/rhoM) +
            B*(1.-om*rhoM/(R2*rho0))*exp(-R2*rho0/rhoM) +
            om*rhoM*c_v*Temp;

  dp_drho = A*((-om/(R1*rho0))*exp(-R1*rho0/rhoM) +
               (1.-om*rhoM/(R1*rho0))*(R1*rho0/(rhoM*rhoM))*exp(-R1*rho0/rhoM))
          + B*((-om/(R2*rho0))*exp(-R2*rho0/rhoM) +
               (1.-om*rhoM/(R2*rho0))*(R2*rho0/(rhoM*rhoM))*exp(-R2*rho0/rhoM))
          + om*c_v*Temp;

  dp_de   = om*rhoM;
}

//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void JWL::hydrostaticTempAdjustment(Patch::FaceType face, 
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
