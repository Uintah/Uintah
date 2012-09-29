/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/ICE/SpecificHeatModel/SpecificHeat.h>
#include <CCA/Components/ICE/SpecificHeatModel/Component.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <cmath>
#include <iostream>

using namespace Uintah;

const double kb = 1.3806503e-23; // Boltzmann constant (m^2*kg/s^2*K)
const double R  = 8.3144621;     // Gas Constant (J/mol*K)
const double upperbound = 1.0 + 1.0e-12;
const double lowerbound = 1.0 - 1.0e-12;

ComponentCv::ComponentCv(ProblemSpecP& ps)
 : SpecificHeat(ps)
{
  ps->getWithDefault("XCO2",d_fractionCO2,0.0);
  ps->getWithDefault("XH2O",d_fractionH2O,0.0);
  ps->getWithDefault("XCO", d_fractionCO, 0.0);
  ps->getWithDefault("XH2", d_fractionH2, 0.0);
  ps->getWithDefault("XO2", d_fractionO2, 0.0);
  ps->getWithDefault("XN2", d_fractionN2, 0.0);
  ps->getWithDefault("XOH", d_fractionOH, 0.0);
  ps->getWithDefault("XNO", d_fractionNO, 0.0);
  ps->getWithDefault("XO",  d_fractionO,  0.0);
  ps->getWithDefault("XH",  d_fractionH,  0.0);

  // Sum mole fractions and check for ~1.0
  d_sum = d_fractionCO2 + d_fractionH2O + d_fractionCO + d_fractionH2 + d_fractionO2 + d_fractionN2 + d_fractionOH + d_fractionNO + d_fractionO + d_fractionH;
  if(d_sum > upperbound || d_sum < lowerbound  ) 
    throw new ProblemSetupException("Sum of fractions of constituents must add to 1.", __FILE__, __LINE__);

  // compute some saved values
  d_gamma       = 1.6*(d_fractionO + d_fractionH) + 1.4*(d_fractionNO+d_fractionOH+d_fractionO2+d_fractionN2+d_fractionH2+d_fractionCO) 
                + 1.3*(d_fractionCO2+d_fractionH2O);
  d_massPerMole = d_fractionCO2*44.01 + d_fractionH2O*18.01428 + d_fractionCO*28.01 + d_fractionH2*1.00794 + d_fractionO2*31.9988 
                + d_fractionN2 * 28.0134 + d_fractionOH*17.00734 + d_fractionNO*30.0061 + d_fractionO*15.9994 + d_fractionH*1.00794;

  if(d_gamma <= 0.0)
    throw new ProblemSetupException("Gamma must be >= 0.", __FILE__, __LINE__);

  if(d_massPerMole <= 0.0)
    throw new ProblemSetupException("Mass Per Mole of Atmosphere must be >= 0.", __FILE__, __LINE__);
}

ComponentCv::~ComponentCv()
{
}

void ComponentCv::outputProblemSpec(ProblemSpecP& ice_ps)
{
  ProblemSpecP cvmodel = ice_ps->appendChild("SpecificHeatModel");
  cvmodel->setAttribute("type", "Component");
  cvmodel->appendElement("XCO2", d_fractionCO2);
  cvmodel->appendElement("XH2O", d_fractionH2O);
  cvmodel->appendElement("XCO", d_fractionCO);
  cvmodel->appendElement("XH2", d_fractionH2);
  cvmodel->appendElement("XO2", d_fractionO2);
  cvmodel->appendElement("XN2", d_fractionN2);
  cvmodel->appendElement("XOH", d_fractionOH);
  cvmodel->appendElement("XNO", d_fractionNO);
  cvmodel->appendElement("XO", d_fractionO);
  cvmodel->appendElement("XH", d_fractionH);

}


double ComponentCv::getSpecificHeat(double T)
{
  double cpMolar = 0.0;

  // Clamp minimum to 300 because of fitting forms
  if(T < 300.0)
    T = 300.0;
  // Clamp maximum to 5000 because of fitting form
  if(T > 5000.0)
    T = 5000.0;

  // reused temperatures
  double T2 = T*T;
  double T3 = T2*T;
  double T4 = T3*T;



  // Add contributions of each gas
  if(T<=1000.0) {
  // Low temperature constants //
    // Constants from: Heywood, J.B. Internal Combustion Engine Fundamentals, 
    //                 McGraw-Hill Publishing, 1988, p. 131.
    cpMolar += d_fractionCO2*(0.24008e1+0.87351e-2*T-0.66071e-5*T2+0.20022e-8*T3+0.63274e-15*T4);
    cpMolar += d_fractionH2O*(0.40701e1-0.11084e-2*T+0.41521e-5*T2-0.29637e-8*T3+0.80702e-12*T4);
    cpMolar += d_fractionCO *(0.37101e1-0.16191e-2*T+0.36924e-5*T2-0.20320e-8*T3+0.23953e-12*T4);
    cpMolar += d_fractionH2 *(0.30574e1+0.26765e-2*T-0.58099e-5*T2+0.55210e-8*T3-0.18123e-11*T4);
    cpMolar += d_fractionO2 *(0.36256e1-0.18782e-2*T+0.70555e-5*T2-0.67635e-8*T3+0.21556e-11*T4);
    cpMolar += d_fractionN2 *(0.36748e1-0.12082e-2*T+0.23240e-5*T2-0.63218e-9*T3-0.22577e-12*T4);
  } else { // High temperature constants // 
    // Constants from: Heywood, J.B. Internal Combustion Engine Fundamentals, 
    //                 McGraw-Hill Publishing, 1988, p. 131.
    cpMolar += d_fractionCO2*(0.44608e1+0.30982e-2*T-0.12393e-5*T2+0.22741e-9*T3-0.15526e-13*T4);
    cpMolar += d_fractionH2O*(0.27168e1+0.29451e-2*T-0.80224e-6*T2+0.10227e-9*T3-0.48472e-14*T4);
    cpMolar += d_fractionCO *(0.29841e1+0.14891e-2*T-0.57900e-6*T2+0.10365e-9*T3-0.69354e-14*T4);
    cpMolar += d_fractionH2 *(0.31002e1+0.51119e-3*T+0.52644e-7*T2-0.34910e-10*T3+0.36945e-14*T4);
    cpMolar += d_fractionO2 *(0.36220e1+0.73618e-3*T-0.19652e-6*T2+0.36202e-10*T3-0.28946e-14*T4);
    cpMolar += d_fractionN2 *(0.28963e1+0.15155e-2*T-0.57235e-6*T2+0.99807e-10*T3-0.65224e-14*T4);
  }

  // Same over entire temperature range
  cpMolar += d_fractionOH *(0.29106e1+0.95932e-3*T-0.19442e-6*T2+0.13757e-10*T3+0.14225e-15*T4);
  cpMolar += d_fractionNO *(0.31890e1+0.13382e-2*T-0.52899e-6*T2+0.95919e-10*T3-0.64848e-14*T4);
  cpMolar += d_fractionO  *(0.25421e1-0.27551e-4*T-0.31028e-8*T2+0.45511e-11*T3-0.43681e-15*T4);
  cpMolar += d_fractionH  *(0.25);


  // Here gamma is approximated with monotomics == 1.6, diatomics == 1.4 and triatomics == 1.3
  double cvMolar = cpMolar / d_gamma;

  // the factor of 10^3 if to convert g->kg
  return 1000.0 * (cvMolar * R)/ d_massPerMole;
}

double ComponentCv::getGamma(double T)
{
  return d_gamma;
}

double ComponentCv::getInternalEnergy(double T)
{
  double hMolar = 0.0;

  // Clamp minimum to 300 because of fitting forms
  if(T < 300.0)
    T = 300.0;
  // Clamp maximum to 5000 because of fitting form
  if(T > 5000.0)
    T = 5000.0;

  // reused temperatures
  double T2 = T*T;
  double T3 = T2*T;
  double T4 = T3*T;



  // Add contributions of each gas
  if(T<=1000.0) {
  // Low temperature constants //
    // Constants from: Heywood, J.B. Internal Combustion Engine Fundamentals,
    //                 McGraw-Hill Publishing, 1988, p. 131.
    hMolar += T * d_fractionCO2*(0.24008e1+0.87351e-2*T/2.0-0.66071e-5*T2/3.0+0.20022e-8*T3/4.0+0.63274e-15*T4/5.0 - 0.48378e5/T);
    hMolar += T * d_fractionH2O*(0.40701e1-0.11084e-2*T/2.0+0.41521e-5*T2/3.0-0.29637e-8*T3/4.0+0.80702e-12*T4/5.0 - 0.30280e5/T);
    hMolar += T * d_fractionCO *(0.37101e1-0.16191e-2*T/2.0+0.36924e-5*T2/3.0-0.20320e-8*T3/4.0+0.23953e-12*T4/5.0 - 0.14356e5/T);
    hMolar += T * d_fractionH2 *(0.30574e1+0.26765e-2*T/2.0-0.58099e-5*T2/3.0+0.55210e-8*T3/4.0-0.18123e-11*T4/5.0 - 0.98890e3/T);
    hMolar += T * d_fractionO2 *(0.36256e1-0.18782e-2*T/2.0+0.70555e-5*T2/3.0-0.67635e-8*T3/4.0+0.21556e-11*T4/5.0 - 0.10475e4/T);
    hMolar += T * d_fractionN2 *(0.36748e1-0.12082e-2*T/2.0+0.23240e-5*T2/3.0-0.63218e-9*T3/4.0-0.22577e-12*T4/5.0 - 0.10612e4/T);
  } else { // High temperature constants //
    // Constants from: Heywood, J.B. Internal Combustion Engine Fundamentals,
    //                 McGraw-Hill Publishing, 1988, p. 131.
    hMolar += T * d_fractionCO2*(0.44608e1+0.30982e-2*T/2.0-0.12393e-5*T2/3.0+0.22741e-9*T3/4.0-0.15526e-13*T4/5.0 - 0.48961e5/T);
    hMolar += T * d_fractionH2O*(0.27168e1+0.29451e-2*T/2.0-0.80224e-6*T2/3.0+0.10227e-9*T3/4.0-0.48472e-14*T4/5.0 - 0.29906e5/T);
    hMolar += T * d_fractionCO *(0.29841e1+0.14891e-2*T/2.0-0.57900e-6*T2/3.0+0.10365e-9*T3/4.0-0.69354e-14*T4/5.0 - 0.14245e5/T);
    hMolar += T * d_fractionH2 *(0.31002e1+0.51119e-3*T/2.0+0.52644e-7*T2/3.0-0.34910e-10*T3/4.0+0.36945e-14*T4/5.0 - 0.87738e3/T);
    hMolar += T * d_fractionO2 *(0.36220e1+0.73618e-3*T/2.0-0.19652e-6*T2/3.0+0.36202e-10*T3/4.0-0.28946e-14*T4/5.0 - 0.1202e4/T);
    hMolar += T * d_fractionN2 *(0.28963e1+0.15155e-2*T/2.0-0.57235e-6*T2/3.0+0.99807e-10*T3/4.0-0.65224e-14*T4/5.0 - 0.90586e3/T);
  }

  // Same over entire temperature range
  hMolar += T * d_fractionOH *(0.29106e1+0.95932e-3*T/2.0-0.19442e-6*T2/3.0+0.13757e-10*T3/4.0+0.14225e-15*T4/5.0 + 0.39354e4/T);
  hMolar += T * d_fractionNO *(0.31890e1+0.13382e-2*T/2.0-0.52899e-6*T2/3.0+0.95919e-10*T3/4.0-0.64848e-14*T4/5.0 + 0.98283e4/T);
  hMolar += T * d_fractionO  *(0.25421e1-0.27551e-4*T/2.0-0.31028e-8*T2/3.0+0.45511e-11*T3/4.0-0.43681e-15*T4/5.0 + 0.29231e5/T);
  hMolar += T * d_fractionH  *(0.25) + 0.25472e5;



  // Here gamma is approximated with monotomics == 1.6, diatomics == 1.4 and triatomics == 1.3
  double uMolar = hMolar / d_gamma;

  // the factor of 10^3 if to convert g->kg
  return 1000.0 * (uMolar * R)/ d_massPerMole;
  
}

