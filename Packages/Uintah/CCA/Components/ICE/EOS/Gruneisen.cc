#include <Packages/Uintah/CCA/Components/ICE/EOS/Gruneisen.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <iostream>
#include <iomanip>

using namespace Uintah;

Gruneisen::Gruneisen(ProblemSpecP& ps)
{
   // Constructor
  ps->require("A",A);
  ps->require("B",B);
  ps->require("rho0",rho0);
  ps->require("T0",T0);
  ps->require("P0",P0);
}

Gruneisen::~Gruneisen()
{
}

double Gruneisen::computeRhoMicro(double P, double,
                                 double , double T)
{
  // Pointwise computation of microscopic density
  double rhoM = rho0*((1./A)*((P-P0) - B*(T-T0)) + 1.);

//  cout << setprecision(12);
//  cout << "cRM " << rhoM << " " << P << " " << T << " " << T0  << " " << P0 << " " << A << " " << B << endl;

  return rhoM;
}

// Return (1/v)*(dv/dT)  (constant pressure thermal expansivity)
double Gruneisen::getAlpha(double T, double, double P, double)
{
  double alpha=(B*rho0/A)/(rho0*((1./A)*((P-P0) - B*(T-T0))+1.));
  return  alpha;
}

//__________________________________
//
void Gruneisen::computeTempCC(const Patch* patch,
                        const string& comp_domain,
                        const CCVariable<double>& P, 
                        const double&,
                        const double& cv,
                        const CCVariable<double>& rhoM, 
                        CCVariable<double>& Temp,
                        Patch::FaceType face)
{
  if(comp_domain == "WholeDomain") {
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      Temp[c]= T0 + (1./B)*((P[c]-P0) - A*(rhoM[c]/rho0-1.));
    }
  } 
  // Although this isn't currently being used
  // keep it around it could be useful
  if(comp_domain == "FaceCells") {     
   for (CellIterator iter=patch->getFaceCellIterator(face);!iter.done();iter++){
      IntVector c = *iter;
      Temp[c]= T0 + (1./B)*((P[c]-P0) - A*(rhoM[c]/rho0-1.));
   }
  }
}

//__________________________________
//

void Gruneisen::computePressEOS(double rhoM, double,
                          double cv, double Temp,
                          double& press, double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities

  press   = P0 + A*(rhoM/rho0-1.) + B*(Temp-T0);

  dp_drho = A/rho0;

//  cout << setprecision(12);
//  cout << "cPEOS " << rhoM << " " << press << " " << Temp << " " << dp_drho << endl;

  dp_de   = B/cv;
}

//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void Gruneisen::hydrostaticTempAdjustment(Patch::FaceType face, 
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
