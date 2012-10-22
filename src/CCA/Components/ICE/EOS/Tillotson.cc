/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/ICE/EOS/Tillotson.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>
#include <iomanip>

using namespace Uintah;

Tillotson::Tillotson(ProblemSpecP& ps)
{
   // Constructor
  ps->require("a",a);
  ps->require("b",b);
  ps->require("A",A);
  ps->require("B",B);
  ps->require("E0",E0);
  ps->require("Es",Es);
  ps->require("Esp",Esp);
  ps->require("alpha",alpha);
  ps->require("beta",beta);
  ps->require("rho0",rho0);
}

Tillotson::~Tillotson()
{
}
//_________________________________
void Tillotson::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eos_ps = ps->appendChild("EOS");
  eos_ps->setAttribute("type","Tillotson");
  eos_ps->appendElement("a",a);
  eos_ps->appendElement("b",b);
  eos_ps->appendElement("A",A);
  eos_ps->appendElement("B",B);
  eos_ps->appendElement("E0",E0);
  eos_ps->appendElement("Es",Es);
  eos_ps->appendElement("Esp",Esp);
  eos_ps->appendElement("alpha",alpha);
  eos_ps->appendElement("beta",beta);
  eos_ps->appendElement("rho0",rho0);
}

//__________________________________
double Tillotson::computeRhoMicro(double press, double,
                             double cv, double Temp,double rho_guess)
{
  // Pointwise computation of microscopic density
  // P=P(rho,T) is not invertable to get rho=rho(P,T)
  // so I'm using Newton's method to find the rho
  // such that 
  // P-P(rho,T) = 0;

  double rho=rho_guess;

//  cout << setprecision(12);
//  cout << "rhoin = " << rho-rho0 << endl;
//  cout << "pressin = " << press << endl;

  double epsilon = 1.e-15;
  double delta = 1.;
  double f,df_drho=1.,relfac=0.5;
  double flow, fhigh,pofrho;
  int count = 0;
  double delrho=1.e-7;
  double E = .00001034777*Es;

//  if(press<2.*delrho){
//    delrho=.25*press;
//  }

//  double press_test = press;
//  double press_ref = (E*rho0)*(a + b/(E/E0+1.));

   while((fabs(delta/rho)>epsilon && !(fabs(delta)<epsilon)) && count<100
                                  &&  (press>10100 && press<10200)){

    pofrho=Pofrho(rho);
    f = pofrho - press;
    flow =Pofrho(rho-delrho);
    fhigh=Pofrho(rho+delrho);
    df_drho=(fhigh-flow)/(2.*delrho);

    if((rho+delrho>rho0 && rho-delrho>rho0) ||
       (rho+delrho<rho0 && rho-delrho<rho0)){
       relfac=1.0;
    }
    else{
       relfac=0.7;
    }

    delta = -relfac*(f/df_drho);
    rho+=delta;
//    cout << "delta = " << delta << endl;
//    cout << "fhigh = " << fhigh << endl;
//    cout << "flow = " << flow << endl;
//    cout << "delrho = " << delrho << endl;
    count++;
  }
//  cout << "count_first = " << count << endl;
//  cout << "rhoout_first = " << rho-rho0 << endl;

  delta = 1., count = 0;
                                                                                
  while((fabs(delta/rho)>epsilon && !(fabs(delta)<epsilon)) && count<100){
    double eta=rho/rho0;
    double mu=(eta-1.);
    double rhosq=rho*rho;
    double etasq=eta*eta;
    double musq=mu*mu;
    double rho0sq=rho0*rho0;

    if(eta>=1){
      f = (E*rho)*(a + b/(E/(E0*etasq)+1.)) + A*mu + B*musq - press;
                                                                                
      df_drho = a*E + b*E*rhosq*((3.*(E*rho0sq/E0 + rhosq) - 2.*rhosq)/
                      ((E*rho0sq/E0 + rhosq)*(E*rho0sq/E0 + rhosq)))
                   + A/rho0 + (2.*B/rho0)*mu;
    }
    else{
        double AA=A*0.;
        double expterm=exp(-alpha*((1./eta-1.)*(1./eta-1.)));
                                                                                
        f = a*E*rho
          + (b*E*rho/(E/(E0*etasq)+1)+AA*mu*exp(-beta*(1./eta-1.)))*expterm
          - press;
                                                                                
        df_drho = a*E
                + b*E*rhosq*((3.*(E*rho0sq/E0 + rhosq) - 2.*rhosq)/
                  ((E*rho0sq/E0 + rhosq)*(E*rho0sq/E0 + rhosq)))*expterm
                + b*E*rho/(E/(E0*etasq)+1.)*(2.*alpha*(rho0/rho - 1)*
                  (rho0/rhosq)*expterm);
    }
                                                                                
    delta = -relfac*(f/df_drho);
    rho+=delta;
//    cout << "delta = " << delta << endl;
    count++;
  }
  
//  cout << "rhoout_last = " << rho << endl;
//  cout << "df_drho = " << df_drho << endl;
//  cout << "count_last = " << count << endl;

  return rho;
}

//__________________________________
void Tillotson::computePressEOS(double rho, double, double, double,
                          double& press, double& dp_drho, double& dp_de)
{
//  cout << setprecision(12);
//  cout << "RHOin = " << rho-rho0 << endl;

#if 0
  double delrho=1.e-5;
  press = Pofrho(rho);
  double plow  = Pofrho(rho-delrho);
  double phigh = Pofrho(rho+delrho);
  dp_drho=(phigh-plow)/(2.*delrho);
#endif
#if 1
  double eta=rho/rho0;
  double mu=(eta-1.);
  double E = .00001034777*Es;
  double rhosq=rho*rho;
  double rho0sq=rho0*rho0;
  double etasq=eta*eta;
  double musq=mu*mu;
  if(eta>=1.){
     press   = (E*rho)*(a + b/(E/(E0*etasq)+1.)) + A*mu + B*musq;

     dp_drho = a*E
             + b*E*rhosq*((3.*(E*rho0sq/E0 + rhosq) - 2.*rhosq)/
                          ((E*rho0sq/E0 + rhosq)*(E*rho0sq/E0 + rhosq)))
             + A/rho0 + (2.*B/rho0)*mu;
//     cout << "eta>=1" << endl;
   }
   else{
     double AA=A*0.;
     double expterm=exp(-alpha*((1./eta-1.)*(1./eta-1.)));

     press = a*E*rho
           + (b*E*rho/(E/(E0*etasq)+1)+AA*mu*exp(-beta*(1./eta-1.)))*expterm;

     dp_drho = a*E
             + b*E*rhosq*((3.*(E*rho0sq/E0 + rhosq) - 2.*rhosq)/
               ((E*rho0sq/E0 + rhosq)*(E*rho0sq/E0 + rhosq)))*expterm
             + b*E*rho/(E/(E0*etasq)+1.)*(2.*alpha*(rho0/rho - 1)*
               (rho0/rhosq)*expterm);

//     cout << "eta<1" << endl;
   }
#endif

  dp_de   = 0.0;

//  cout << "press_out = " << press << endl;
//  cout << "dp_drho_out = " << dp_drho << endl;
}

//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void Tillotson::hydrostaticTempAdjustment(Patch::FaceType, 
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

// Return (1/v)*(dv/dT)  (constant pressure thermal expansivity)
double Tillotson::getAlpha(double, double , double , double )
{
  // Cheating here a bit, computing v*(dT/dv) and returning the inverse of that
  double alpha=0.;
  return  alpha;
}
                                                                                
//__________________________________
void Tillotson::computeTempCC(const Patch* patch,
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
