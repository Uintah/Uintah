#include <Packages/Uintah/CCA/Components/ICE/EOS/SFEOS.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>
#include <iomanip>

using namespace Uintah;

SFEOS::SFEOS(ProblemSpecP& ps)
{
   // Constructor
  ps->require("eps0",eps[0]);
  ps->require("eps1",eps[1]);
  ps->require("eps2",eps[2]);
  ps->require("eps3",eps[3]);
  ps->require("eps4",eps[4]);
  ps->require("eps5",eps[5]);
  ps->require("eps6",eps[6]);
  ps->require("eps7",eps[7]);
  ps->require("eps8",eps[8]);
  ps->require("eps9",eps[9]);
  ps->require("p0",p[0]);
  ps->require("p1",p[1]);
  ps->require("p2",p[2]);
  ps->require("p3",p[3]);
  ps->require("p4",p[4]);
  ps->require("p5",p[5]);
  ps->require("p6",p[6]);
  ps->require("p7",p[7]);
  ps->require("p8",p[8]);
  ps->require("p9",p[9]);
  ps->require("rho0",rho0);

  int i;
  for(i=0; i<10; i++){
    if(p[i]>0.0) lnp[i] = log(p[i]);
  }
  for(i=0; i<9; i++){
    if(eps[i+1]<eps[i]) slope[i] = (lnp[i+1] - lnp[i])/(eps[i+1] - eps[i]);
  }
}

SFEOS::~SFEOS()
{
}
//_________________________________
void SFEOS::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eos_ps = ps->appendChild("EOS");
  eos_ps->setAttribute("type","SFEOS");
  eos_ps->appendElement("eps0",eps[0]);
  eos_ps->appendElement("eps1",eps[1]);
  eos_ps->appendElement("eps2",eps[2]);
  eos_ps->appendElement("eps3",eps[3]);
  eos_ps->appendElement("eps4",eps[4]);
  eos_ps->appendElement("eps5",eps[5]);
  eos_ps->appendElement("eps6",eps[6]);
  eos_ps->appendElement("eps7",eps[7]);
  eos_ps->appendElement("eps8",eps[8]);
  eos_ps->appendElement("eps9",eps[9]);
  eos_ps->appendElement("p0",p[0]);
  eos_ps->appendElement("p1",p[1]);
  eos_ps->appendElement("p2",p[2]);
  eos_ps->appendElement("p3",p[3]);
  eos_ps->appendElement("p4",p[4]);
  eos_ps->appendElement("p5",p[5]);
  eos_ps->appendElement("p6",p[6]);
  eos_ps->appendElement("p7",p[7]);
  eos_ps->appendElement("p8",p[8]);
  eos_ps->appendElement("p9",p[9]);
  eos_ps->appendElement("rho0",rho0);
}

//__________________________________
double SFEOS::computeRhoMicro(double press, double,
                             double cv, double Temp,double rho_guess)
{
  int i1 = 0, i;
  for(i=1; i<9; i++){
    if(eps[i+1]<eps[i])
      if(press>p[i]) i1 = i;
  }
  double vol_strain = (log(press) - lnp[i1])/slope[i1] + eps[i1];
  double rho_cur= rho0/exp(vol_strain);

  return rho_cur;

}

//__________________________________
void SFEOS::computePressEOS(double rho, double, double, double,
                          double& press, double& dp_drho, double& dp_de)
{
  double vol_strain = log(rho0/rho);

  int i1 = 0, i;
  for(i=1; i<9; i++){
    if(eps[i+1]<eps[i])
      if(vol_strain<eps[i]) i1 = i;
  }
  press = lnp[i1] + slope[i1]*(vol_strain - eps[i1]);
  press = exp(press);
  
  dp_drho = -press*slope[i1]/rho;
  dp_de = 0.0;
}

//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void SFEOS::hydrostaticTempAdjustment(Patch::FaceType, 
                                    const Patch*,
                                    const vector<IntVector>&,
                                    Vector&,
                                    const CCVariable<double>&,
                                    const CCVariable<double>&,
                                    const Vector&,
                                    CCVariable<double>&)
{ 
//  IntVector axes = patch->faceAxes(face);
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

// Return (1/v)*(dv/dT)  (constant pressure thermal expansivity)
double SFEOS::getAlpha(double, double , double , double )
{
  // Cheating here a bit, computing v*(dT/dv) and returning the inverse of that
  double alpha=0.;
  return  alpha;
}
                                                                                
//__________________________________
void SFEOS::computeTempCC(const Patch* patch,
                         const string& comp_domain,
                         const CCVariable<double>&,
                         const CCVariable<double>&,
                         const CCVariable<double>&,
                         const CCVariable<double>&,
                         CCVariable<double>& Temp,
                         Patch::FaceType face)
{
  if(comp_domain == "WholeDomain") {
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){      IntVector c = *iter;
      Temp[c]= 300.0;
    }
  }
  // Although this isn't currently being used
  // keep it around it could be useful
  if(comp_domain == "FaceCells") {
   for (CellIterator iter=patch->getFaceCellIterator(face);!iter.done();iter++){      IntVector c = *iter;
      Temp[c]= 300.0;
   }
  }
}
