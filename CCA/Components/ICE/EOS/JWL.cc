#include <Packages/Uintah/CCA/Components/ICE/EOS/JWL.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <iostream>
#include <iomanip>

using namespace Uintah;

JWL::JWL(ProblemSpecP& ps)
{
   // Constructor
  ps->require("A",A);
  ps->require("B",B);
  ps->require("R1",R1);
  ps->require("R2",R2);
  ps->require("om",om);
  ps->require("rho0",rho0);
  // rho_micro at P=101325. and T = 300. is 1.37726724081686
}

JWL::~JWL()
{
}

double JWL::computeRhoMicro(double press, double,
                            double cv, double Temp)
{
  // Pointwise computation of microscopic density
  // P=P(rho,T) is not invertable to get rho=rho(P,T)
  // so I'm using Newton's method to find the rhoM
  // such that 
  //press - (A*exp(-R1*rho0/rhoM) +
  //         B*exp(-R2*rho0/rhoM) + om*rhoM*cv*Temp) = 0
  // First guess comes from inverting the last term of this equation

  double rhoM = min(10000.,press/(om*cv*Temp));
  double epsilon = 1.e-15;
  double delta = 1.;
  double f,df_drho,relfac=.9;
  int count = 0;

  while(fabs(delta/rhoM)>epsilon){
    f = (A*exp(-R1*rho0/rhoM) + B*exp(-R2*rho0/rhoM) + om*rhoM*cv*Temp) - press;

    df_drho = A*(R1*rho0/(rhoM*rhoM))*exp(-R1*rho0/rhoM)
            + B*(R2*rho0/(rhoM*rhoM))*exp(-R2*rho0/rhoM)
            + om*cv*Temp;

    delta = -relfac*(f/df_drho);
    rhoM+=delta;
    rhoM=fabs(rhoM);
    if(count>=100){
      cout << setprecision(15);
      cout << "JWL::computeRhoMicro not converging." << endl;
      cout << "press = " << press << " temp = " << Temp << " cv = " << cv << endl;
      cout << "delta = " << delta << " rhoM = " << rhoM << " f = " << f << " df_drho = " << df_drho << endl;


      // The following is here solely to help figure out what was going on
      // at the time the above code failed to converge.  Start over with this
      // copy and print more out.
      delta = 1.;
      rhoM = press/(om*cv*Temp);
      cout <<  rhoM << endl;;
      while(fabs(delta/rhoM)>epsilon){
       f = (A*exp(-R1*rho0/rhoM) +
            B*exp(-R2*rho0/rhoM) + om*rhoM*cv*Temp) - press;

       df_drho = A*(R1*rho0/(rhoM*rhoM))*exp(-R1*rho0/rhoM)
               + B*(R2*rho0/(rhoM*rhoM))*exp(-R2*rho0/rhoM)
               + om*cv*Temp;

       delta = -relfac*(f/df_drho);
       rhoM+=delta;
       rhoM=fabs(rhoM);
       cout <<  "f = " << f << " df_drho = " << df_drho << " delta = " << delta << endl;
       cout <<  rhoM << endl;;
       if(count>=120){
         exit(1);
       }
       count++;
      }

      exit(1);
    }
    count++;
  }
  return rhoM;
  
}

// Return (1/v)*(dv/dT)  (constant pressure thermal expansivity)
double JWL::getAlpha(double, double sp_v, double P, double cv)
{
  // Cheating here a bit, computing v*(dT/dv) and returning the inverse of that
  double alpha;
  alpha = 1.0/((sp_v*rho0/(om*cv))*
          (P - A*exp(-R1*sp_v*rho0) - B*exp(-R2*sp_v*rho0)) +
           sp_v*(A*R1*rho0*exp(-R1*sp_v*rho0) + B*R2*sp_v*exp(-R2*sp_v*rho0)));

  return  alpha;
}

//__________________________________
//
void JWL::computeTempCC(const Patch* patch,
                        const string& comp_domain,
                        const CCVariable<double>& press, 
                        const double&,
                        const double& cv,
                        const CCVariable<double>& rhoM, 
                        CCVariable<double>& Temp,
                        Patch::FaceType face)
{
  if(comp_domain == "WholeDomain") {
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      Temp[c]= (press[c] - A*exp(-R1*rho0/rhoM[c])
                         - B*exp(-R2*rho0/rhoM[c])) / (om*rhoM[c]*cv);
    }
  } 
  // Although this isn't currently being used
  // keep it around it could be useful
  if(comp_domain == "FaceCells") {     
   for (CellIterator iter=patch->getFaceCellIterator(face);!iter.done();iter++){
      IntVector c = *iter;
      Temp[c]= (press[c] - A*exp(-R1*rho0/rhoM[c])
                         - B*exp(-R2*rho0/rhoM[c])) / (om*rhoM[c]*cv);
   }
  }
}

//__________________________________
//

void JWL::computePressEOS(double rhoM, double,
                          double cv, double Temp,
                          double& press, double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities

  press   = A*exp(-R1*rho0/rhoM) +
            B*exp(-R2*rho0/rhoM) + om*rhoM*cv*Temp;

  dp_drho = A*(exp(-R1*rho0/rhoM))
          + B*(exp(-R2*rho0/rhoM)) + om*cv*Temp;

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
