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


#include <CCA/Components/ICE/EOS/BirchMurnaghan.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <iostream>
#include <iomanip>

using namespace Uintah;

BirchMurnaghan::BirchMurnaghan(ProblemSpecP& ps)
{
  // Constructor
  // eos model inputs
  ps->require("n",n);
  ps->require("K",K);
  ps->require("rho0",rho0);
  ps->require("P0",P0);
  ps->getWithDefault("useSpecificHeatModel",useSpecificHeatModel,false);

  // specific heat model inputs
  if(useSpecificHeatModel)
  {
    ps->require("a", a);
    ps->require("b", b);
    ps->require("c0",c0);
    ps->require("c1",c1);
    ps->require("c2",c2);
    ps->require("c3",c3);

    // make sure we can compute the specific heat
    if(c0==0 && c1==0 && c2==0 && c3==0)
      throw new ProblemSetupException("ERROR BirchMurnaghan: at least one of the specific heat coefficients (c0,c1,c2,c3) must be nonzero.", __FILE__, __LINE__);
  }
}

BirchMurnaghan::~BirchMurnaghan()
{
}

void BirchMurnaghan::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eos_ps = ps->appendChild("EOS");
  // eos model outputs
  eos_ps->setAttribute("type","BirchMurnaghan");
  eos_ps->appendElement("n",   n);
  eos_ps->appendElement("K",   K);
  eos_ps->appendElement("rho0",rho0);
  eos_ps->appendElement("P0",  P0);
  // specific heat model outputs
  eos_ps->appendElement("useSpecificHeatModel",useSpecificHeatModel);
  if(useSpecificHeatModel)
  {
    eos_ps->appendElement("a",  a);
    eos_ps->appendElement("b",  b);
    eos_ps->appendElement("c0", c0);
    eos_ps->appendElement("c1", c1);
    eos_ps->appendElement("c2", c2);
    eos_ps->appendElement("c3", c3);
  }
}


//__________________________________
double BirchMurnaghan::computeRhoMicro(double press, double,
                                       double cv, double temp, double rho_guess)
{
  // Pointwise computation of microscopic density
  double rhoM = rho_guess;
  if(press>=P0){
    if(useSpecificHeatModel)
    {
       // NEEDS TO BE IMPLEMENTED
       throw new InternalError("ERROR BirchMurnaghan: Specific Heat EOS is not implemented in computeRhoMicro(...) yet.", __FILE__, __LINE__);       
    } else {
      // Use normal Birch-Murnaghan EOS
      //  Solved using Newton Method code adapted from JWLC.cc
      double f;                // difference between current and previous function value
      double df_drho;          // rate of change of function value
      double epsilon = 1.e-15; // convergence limit 
      double delta   = 1.0;    // change in rhoM each step
      double relfac  = 0.9;    
      int count      = 0;      // counter of total iterations

      while(fabs(delta/rhoM)>epsilon){  // Main Iterative loop
        // Compute the difference between the previous pressure and the new pressure
        f       = computeP(rho0/rhoM) - press;

        // Compute the new pressure derivative
        df_drho = computedPdrho(rho0/rhoM);

        // factor by which to adjust rhoM
        delta = -relfac*(f/df_drho);
        rhoM +=  delta;
        rhoM  =  fabs(rhoM);

        if(count>=100){
          // The following is here solely to help figure out what was going on
          // at the time the above code failed to converge.  Start over with this
          // copy and print more out.
          delta = 1.0;
          rhoM  = 2.0*rho0;

          while(fabs(delta/rhoM)>epsilon){
            f       = computeP(rho0/rhoM) - press; 
            df_drho = computedPdrho(rho0/rhoM);
 
            // determine by how much to change
            delta = -relfac*(f/df_drho);
            rhoM +=  delta;
            rhoM  =  fabs(rhoM);

            // After 50 more iterations finally quit out
            if(count>=150){
              ostringstream warn;
              warn << std::setprecision(15);
              warn << "ERROR:ICE:BirchMurnaghan::computeRhoMicro(...) not converging. \n";
              warn << "press= " << press << " temp=" << temp << " cv=" << cv << "\n";
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
  }
  else{
    // This is default expansion beyond the initial pressure--to prevent negative pressures.
    //  This is holdover from Murnahan.cc
    rhoM = rho0*pow((press/P0),K*P0);
  }
  return rhoM;
}

//__________________________________
// Return (1/v)*(dv/dT)  (constant pressure thermal expansivity)
double BirchMurnaghan::getAlpha(double temp, double sp_v, double press, double cv)
{
  // No dependence on temperature w/o specific heat model
  if(!useSpecificHeatModel)
  {
    return 0.0;
  } else {
    // From: Thibault, P. A Review of Equation of State Models, Chemical Equilibrium Calculations and CERV 
    //         Code Requirements for SHS Detonation Modelling.  TimeScales Scientific Ltd: 
    //         Subcontract W7702-08R183/001/EDM, October 2009.
    // 
    //   "Under the quasi-harmonic assumption that the frequency
    //    modes are independent of temperature, the thermal Gruneisen coefficient is equal to the Debye
    //    Gruneisen coefficient, Gamma = alpha*V*Kt/Cv"
    //
    // where: 
    //    alpha - Thermal Expansion Coefficeint
    //    V     - Volume
    //    Kt    - Bulk Modulus at constant temperature (Taken to be K0 here)
    //    Cv    - Specific heat
    // 
    // Using:
    //    Gamma(v)=a+b*(V/V0) from Menikoff's paper (see header file)
    //
    //    alpha = (a+b*V/V0) * Cv(T)/(K*V)
     
    double cv = temp*temp*temp/(c0 + c1*temp + c2*temp*temp + c3*temp*temp*temp);
    double alpha = (a+b*rho0*sp_v)*cv*(K+n*(press-P0))*rho0;

    return alpha;
  }
}

//__________________________________
void BirchMurnaghan::computeTempCC(const Patch* patch,
                             const string& comp_domain,
                             const CCVariable<double>& /*press*/, 
                             const CCVariable<double>& /*gamma*/,
                             const CCVariable<double>& /* cv*/,
                             const CCVariable<double>& /*rhoM*/, 
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
void BirchMurnaghan::computePressEOS(double rhoM, double, double, double temperature,
                          double& press, double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities
  if(rhoM>=rho0){
    if(useSpecificHeatModel)
    {
      throw new InternalError("ERROR BirchMurnaghan: Temperature contribution to the pressure is not yet implemented in computePressEOS(...).", __FILE__, __LINE__);       
      double v     = rho0/rhoM; // reduced volume
      // compute the specific heat term
      /* Comment out for now to prevent "unused variable warnings"
      double Cv    = temperature*temperature*temperature/(c0
                     +c1*temperature
                     +c2*temperature*temperature
                     +c3*temperature*temperature*temperature);

      // compute the Debye temperature term
      double debye = pow(1.0/v,a)*exp(b*(1.0/v - 1.0));
      */

      // compute the volumetric portion of pressure
      double p1    = computeP(v);

      // compute the temperature portion of the pressure
      double p2    = 0.0; // NEEDS TO BE IMPLEMENTED

      press        = p1+p2; // overall pressure


      // compute the pressure derivative
      double dpdrho1 = computedPdrho(v);

      double dpdrho2 = 0.0; // NEEDS TO BE IMPLEMENTED

      dp_drho        = dpdrho1 + dpdrho2;
      
    } else {
      double v = rho0/rhoM; // reduced volume
      press    = computeP(v);
      dp_drho  = computedPdrho(v);
    } 
  }
  else{
    // This is default expansion beyond the initial pressure--to prevent negative pressures.
    //  This is holdover from Murnahan.cc
    press   = P0*pow(rhoM/rho0,(1./(K*P0)));
    dp_drho = (1./(K*rho0))*pow(rhoM/rho0,(1./(K*P0)-1.));
  }
  dp_de   = 0.0; // Note: this likely should not be zero
}

//______________________________________________________________________
// Update temperature boundary conditions due to hydrostatic pressure gradient
// call this after set Dirchlet and Neuman BC
void BirchMurnaghan::hydrostaticTempAdjustment(Patch::FaceType, 
                                         const Patch*,
                                         Iterator& ,
                                         Vector&,
                                         const CCVariable<double>&,
                                         const CCVariable<double>&,
                                         const Vector&,
                                         CCVariable<double>&)
{ 
//  throw InternalError( "ERROR:ICE:EOS:BirchMurnaghan: hydrostaticTempAdj() \n"
//                       " has not been implemented", __FILE__, __LINE__ );
}

//____________________________________________
// Private Internal Functions
/// @param v the current relative volume
double BirchMurnaghan::computeP(double v)
{
  return 3.0/(2.0*K) * (pow(v,-7.0/3.0) - pow(v,-5.0/3.0))
                             * (1.0 + 0.75*(n-4.0)*(pow(v,-2.0/3.0)-1.0));
}

/// @param v the current relative volume
double BirchMurnaghan::computedPdrho(double v)
{
  return 3.0/(2.0*K) * (-7.0*rho0/(3.0*pow(v,10.0/3.0)) + 5.0*rho0/(3.0*pow(v,8.0/3.0)))
                             * (1.0 + (0.75*n-3.0)*(1.0/(pow(v,2.0/3.0))-1.0))
                             - (1.0/K * (1.0/pow(v,7.0/3.0)-1.0/pow(v,5.0/3.0))*(0.75*n-3.0)*rho0/pow(v,5.0/3.0));
}


