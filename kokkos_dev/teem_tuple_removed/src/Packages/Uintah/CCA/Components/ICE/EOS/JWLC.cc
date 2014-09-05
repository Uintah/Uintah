#include <Packages/Uintah/CCA/Components/ICE/EOS/JWLC.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>
#include <iomanip>

using namespace Uintah;

JWLC::JWLC(ProblemSpecP& ps)
{
   // Constructor
  ps->require("A",A);
  ps->require("B",B);
  ps->require("C",C);
  ps->require("R1",R1);
  ps->require("R2",R2);
  ps->require("om",om);
  ps->require("rho0",rho0);
}

JWLC::~JWLC()
{
}
//__________________________________
double JWLC::computeRhoMicro(double press, double,
                             double cv, double Temp)
{
  // Pointwise computation of microscopic density
  // P=P(rho,T) is not invertable to get rho=rho(P,T)
  // so I'm using Newton's method to find the rhoM
  // such that 
  //press - (A*exp(-R1*rho0/rhoM) +
  //         B*exp(-R2*rho0/rhoM) + C*pow((rhoM/rho0),1+om)) = 0

  double rhoM = rho0;

  double epsilon = 1.e-15;
  double delta = 1.;
  double f,df_drho,relfac=.9;
  int count = 0;

  while(fabs(delta/rhoM)>epsilon){
    f = (A*exp(-R1*rho0/rhoM) + B*exp(-R2*rho0/rhoM)
                              + C*pow((rhoM/rho0),1+om)) - press;

    df_drho = A*(R1*rho0/(rhoM*rhoM))*exp(-R1*rho0/rhoM)
            + B*(R2*rho0/(rhoM*rhoM))*exp(-R2*rho0/rhoM)
            + C*((1.+om)/pow(rho0,1.+om))*pow(rhoM,om);

    delta = -relfac*(f/df_drho);
    rhoM+=delta;
    rhoM=fabs(rhoM);
    if(count>=100){
      cout << setprecision(15);
      cout << "JWLC::computeRhoMicro not converging." << endl;
      cout << "press= " << press << " temp= " << Temp << " cv= " << cv << endl;
      cout << "delta= " << delta << " rhoM= " << rhoM << " f = " << f <<
              " df_drho = " << df_drho << endl;


      // The following is here solely to help figure out what was going on
      // at the time the above code failed to converge.  Start over with this
      // copy and print more out.
      delta = 1.;
      rhoM = 2.*rho0;
      cout <<  rhoM << endl;;
      while(fabs(delta/rhoM)>epsilon){
        f = (A*exp(-R1*rho0/rhoM) + B*exp(-R2*rho0/rhoM)
                                  + C*pow((rhoM/rho0),1+om)) - press;

        df_drho = A*(R1*rho0/(rhoM*rhoM))*exp(-R1*rho0/rhoM)
                + B*(R2*rho0/(rhoM*rhoM))*exp(-R2*rho0/rhoM)
                + C*((1.+om)/pow(rho0,1.+om))*pow(rhoM,om);

       delta = -relfac*(f/df_drho);
       rhoM+=delta;
       rhoM=fabs(rhoM);
       cout <<  "f = " << f << " df_drho = " << df_drho <<
                " delta = " << delta << " rhoM = " << rhoM << endl;
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
//__________________________________
// Return (1/v)*(dv/dT)  (constant pressure thermal expansivity)
double JWLC::getAlpha(double, double , double , double )
{
  // Cheating here a bit, computing v*(dT/dv) and returning the inverse of that
  double alpha=0.;
  return  alpha;
}

//__________________________________
void JWLC::computeTempCC(const Patch* patch,
                         const string& comp_domain,
                         const CCVariable<double>&, 
                         const CCVariable<double>&,
                         const CCVariable<double>&,
                         const CCVariable<double>&, 
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
void JWLC::computePressEOS(double rhoM, double, double, double,
                          double& press, double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities

  press   = A*exp(-R1*rho0/rhoM) +
            B*exp(-R2*rho0/rhoM) + C*pow((rhoM/rho0),1+om);

  dp_drho = (A*R1*rho0/(rhoM*rhoM))*(exp(-R1*rho0/rhoM))
          + (B*R2*rho0/(rhoM*rhoM))*(exp(-R2*rho0/rhoM))
          + C*((1.+om)/pow(rho0,1.+om))*pow(rhoM,om);

  dp_de   = 0.0;
}

//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void JWLC::hydrostaticTempAdjustment(Patch::FaceType, 
                                    const Patch*,
                                    const vector<IntVector>&,
                                    Vector&,
                                    const CCVariable<double>&,
                                    const CCVariable<double>&,
                                    const Vector&,
                                    CCVariable<double>&)
{ 
  throw InternalError( "ERROR:ICE:EOS:JWLC: hydrostaticTempAdj() \n"
                       " has not been implemented" );
}
