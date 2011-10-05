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


#include <CCA/Components/ICE/EOS/JWLC.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>
#include <iomanip>

using namespace std;
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

void JWLC::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eos_ps = ps->appendChild("EOS");
  eos_ps->setAttribute("type","JWLC");

  eos_ps->appendElement("A",A);
  eos_ps->appendElement("B",B);
  eos_ps->appendElement("C",C);
  eos_ps->appendElement("R1",R1);
  eos_ps->appendElement("R2",R2);
  eos_ps->appendElement("om",om);
  eos_ps->appendElement("rho0",rho0);
}

//__________________________________
double JWLC::computeRhoMicro(double press, double,
                             double cv, double Temp,double rho_guess)
{
  // Pointwise computation of microscopic density
  // P=P(rho,T) is not invertable to get rho=rho(P,T)
  // so I'm using Newton's method to find the rhoM
  // such that 
  //press - (A*exp(-R1*rho0/rhoM) +
  //         B*exp(-R2*rho0/rhoM) + C*pow((rhoM/rho0),1+om)) = 0

  double rhoM = min(rho_guess,rho0);

  double epsilon = 1.e-15;
  double delta = 1.;
  double f,df_drho,relfac=.9;
  int count = 0;

  double one_plus_omega = 1.+om;

  while(fabs(delta/rhoM)>epsilon){
    double inv_rho_rat=rho0/rhoM;
    double rho_rat=rhoM/rho0;
    double A_e_to_the_R1_rho0_over_rhoM=A*exp(-R1*inv_rho_rat);
    double B_e_to_the_R2_rho0_over_rhoM=B*exp(-R2*inv_rho_rat);
    double C_rho_rat_tothe_one_plus_omega=C*pow(rho_rat,one_plus_omega);

    f = (A_e_to_the_R1_rho0_over_rhoM +
         B_e_to_the_R2_rho0_over_rhoM + C_rho_rat_tothe_one_plus_omega) - press;

    double rho0_rhoMsqrd = rho0/(rhoM*rhoM);
    df_drho = R1*rho0_rhoMsqrd*A_e_to_the_R1_rho0_over_rhoM
            + R2*rho0_rhoMsqrd*B_e_to_the_R2_rho0_over_rhoM
            + (one_plus_omega/rhoM)*C_rho_rat_tothe_one_plus_omega;

    delta = -relfac*(f/df_drho);
    rhoM+=delta;
    rhoM=fabs(rhoM);
    if(count>=100){

      // The following is here solely to help figure out what was going on
      // at the time the above code failed to converge.  Start over with this
      // copy and print more out.
      delta = 1.;
      rhoM = 2.*rho0;
      while(fabs(delta/rhoM)>epsilon){
       double inv_rho_rat=rho0/rhoM;
       double rho_rat=rhoM/rho0;
       double A_e_to_the_R1_rho0_over_rhoM=A*exp(-R1*inv_rho_rat);
       double B_e_to_the_R2_rho0_over_rhoM=B*exp(-R2*inv_rho_rat);
       double C_rho_rat_tothe_one_plus_omega=C*pow(rho_rat,one_plus_omega);

       f = (A_e_to_the_R1_rho0_over_rhoM +
            B_e_to_the_R2_rho0_over_rhoM +
            C_rho_rat_tothe_one_plus_omega) - press;

       double rho0_rhoMsqrd = rho0/(rhoM*rhoM);
       df_drho = R1*rho0_rhoMsqrd*A_e_to_the_R1_rho0_over_rhoM
                + R2*rho0_rhoMsqrd*B_e_to_the_R2_rho0_over_rhoM
                + (one_plus_omega/rhoM)*C_rho_rat_tothe_one_plus_omega;
  
       delta = -relfac*(f/df_drho);
       rhoM+=delta;
       rhoM=fabs(rhoM);
       if(count>=150){
         ostringstream warn;
         warn << setprecision(15);
         warn << "ERROR:ICE:JWLC::computeRhoMicro not converging. \n";
         warn << "press= " << press << " temp=" << Temp << " cv=" << cv << "\n";
         warn << "delta= " << delta << " rhoM= " << rhoM << " f = " << f 
              <<" df_drho =" << df_drho << " rho_guess =" << rho_guess << "\n";
         throw InternalError(warn.str(), __FILE__, __LINE__);
         
       }
       count++;
      }
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
   Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;
   
   for (CellIterator iter=patch->getFaceIterator(face,MEC);!iter.done();iter++){
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
  // This looked like the following before optimization
//  double pressold   = A*exp(-R1*rho0/rhoM) +
//            B*exp(-R2*rho0/rhoM) + C*pow((rhoM/rho0),1+om);
  
//  double dp_drhoold = (A*R1*rho0/(rhoM*rhoM))*(exp(-R1*rho0/rhoM))
//          + (B*R2*rho0/(rhoM*rhoM))*(exp(-R2*rho0/rhoM))
//          + C*((1.+om)/pow(rho0,1.+om))*pow(rhoM,om);

  double one_plus_omega = 1.+om;
  double inv_rho_rat=rho0/rhoM;
  double rho_rat=rhoM/rho0;
  double A_e_to_the_R1_rho0_over_rhoM=A*exp(-R1*inv_rho_rat);
  double B_e_to_the_R2_rho0_over_rhoM=B*exp(-R2*inv_rho_rat);
  double C_rho_rat_tothe_one_plus_omega=C*pow(rho_rat,one_plus_omega);

  press   = A_e_to_the_R1_rho0_over_rhoM +
            B_e_to_the_R2_rho0_over_rhoM + C_rho_rat_tothe_one_plus_omega;

  double rho0_rhoMsqrd = rho0/(rhoM*rhoM);
  dp_drho = R1*rho0_rhoMsqrd*A_e_to_the_R1_rho0_over_rhoM
          + R2*rho0_rhoMsqrd*B_e_to_the_R2_rho0_over_rhoM
          + (one_plus_omega/rhoM)*C_rho_rat_tothe_one_plus_omega;

  dp_de   = 0.0;
}

//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void JWLC::hydrostaticTempAdjustment(Patch::FaceType, 
                                    const Patch*,
                                     Iterator&,
                                    Vector&,
                                    const CCVariable<double>&,
                                    const CCVariable<double>&,
                                    const Vector&,
                                    CCVariable<double>&)
{ 
//  IntVector axes = patch->getFaceAxes(face);
//  int P_dir = axes[0];  // principal direction
//  double plusMinusOne = patch->faceDirection(face)[P_dir];
  // On xPlus yPlus zPlus you add the increment
  // on xminus yminus zminus you subtract the increment
//  double dx_grav = gravity[P_dir] * cell_dx[P_dir];
                                                                                
//  The following is commented out because this EOS is not temperature
//  dependent, so I'm not adjusting the temperature.

//   vector<IntVector>::const_iterator iter;
//   for (iter=bound.begin(); iter != bound.end(); iter++) {
//     IntVector c = *iter;
//     Temp_CC[c] += plusMinusOne * dx_grav/( (gamma[c] - 1.0) * cv[c] );
//  }

}
