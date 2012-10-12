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

#include "CubicCp.h"

using namespace Uintah;
         
// Construct a specific heat model.  
CubicCp::CubicCp()
{
}

CubicCp::CubicCp(ProblemSpecP& ps)
{
  ps->getWithDefault("a",    d_a,    1.0);
  ps->getWithDefault("b",    d_b,    1.0);
  ps->getWithDefault("beta", d_beta, 0.0);
  ps->require(       "c0",   d_c0);
  ps->require(       "c1",   d_c1);
  ps->require(       "c2",   d_c2);
  ps->require(       "c3",   d_c3);
}

// Construct a copy of a specific heat model.  
CubicCp::CubicCp(const CubicCp* lhs)
{
    this->d_a    = lhs->d_a;
    this->d_b    = lhs->d_b;
    this->d_beta = lhs->d_beta;
    this->d_c0   = lhs->d_c0;
    this->d_c1   = lhs->d_c1;
    this->d_c2   = lhs->d_c2;
    this->d_c3   = lhs->d_c3;
}

// Destructor of specific heat model.  
CubicCp::~CubicCp()
{
}
         
void CubicCp::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP cm_ps = ps->appendChild("specific_heat_model");
  cm_ps->setAttribute("type","cubic_Cp");
    
  cm_ps->appendElement("a",    d_a);
  cm_ps->appendElement("b",    d_b);
  cm_ps->appendElement("beta", d_beta);
  cm_ps->appendElement("c0",   d_c0);
  cm_ps->appendElement("c1",   d_c1);
  cm_ps->appendElement("c2",   d_c2);
  cm_ps->appendElement("c3",   d_c3);
}

// Compute the specific heat
double 
CubicCp::computeSpecificHeat(const PlasticityState* state)
{
  // From: Sewell, T.D., Menikoff, R. Complete Equation of State for Beta-HMX 
  //   and Implications for Initiation.  Shock Compression of Condensed Matter, 2003.
  double reducedT = state->temperature / computeDebyeT(state->density,state->initialDensity);
  double T2 = reducedT*reducedT;
  double T3 = reducedT*T2;
    
  return T3/(d_c0+d_c1*reducedT+d_c2*T2+d_c3*T3)                 // Cv
         // Correction term (B^2*V*T*K_T) on the order of a few percent (defaults to 0 because B = 0)
         // From: Menikoff, R., Sewell, T.D. Constitutent Properties of HMX Needed for 
         //   Meso-Scale Simulations LA-UR-003804-rev, 2001.
         + (d_beta*d_beta*state->temperature*state->bulkModulus   // Correction term (B^2*V*T*K_T)
           / state->density);                                       
                                                                   
}

/*! A helper function to compute the Debye Temperature */
double 
CubicCp::computeDebyeT(double rho, double rho0)
{
    // From: Sewell, T.D., Menikoff, R. Complete Equation of State for Beta-HMX 
    //   and Implications for Initiation.  Shock Compression of Condensed Matter, 2003.
    return pow(rho/rho0,d_a)*exp(d_b*(rho/rho0-1.0));
}

/*! A helper function to compute the Gruneisen coefficient */
double 
CubicCp::computeGamma(double rho, double rho0)
{
    // From: Sewell, T.D., Menikoff, R. Complete Equation of State for Beta-HMX 
    //   and Implications for Initiation.  Shock Compression of Condensed Matter, 2003.
    return d_a+d_b*(rho0/rho);
}

