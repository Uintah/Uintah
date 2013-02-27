/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

#include <CCA/Components/ICE/SpecificHeatModel/SpecificHeat.h>
#include <CCA/Components/ICE/SpecificHeatModel/Polynomial.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <cmath>
#include <iostream>

using namespace Uintah;

const double kb = 1.3806503e-23; // Boltzmann constant (m^2*kg/s^2*K)

PolynomialCv::PolynomialCv(ProblemSpecP& ps)
 : SpecificHeat(ps)
{
  ps->require("MaxOrder",    d_maxOrder);
  ps->getWithDefault("Tmax", d_Tmax, 1.0e6);
  ps->getWithDefault("Tmin", d_Tmin, 0.0);


  // check order and find that many blocks
  if(d_maxOrder < 0) {
    throw new ProblemSetupException("Polynomial order must be >= 0", __FILE__, __LINE__);
  }

  ps->require("coefficient", d_coefficient);
}

PolynomialCv::~PolynomialCv()
{
}

void PolynomialCv::outputProblemSpec(ProblemSpecP &ice_ps)
{
  ProblemSpecP cvmodel = ice_ps->appendChild("SpecificHeatModel");
  cvmodel->setAttribute("type", "Polynomial");
  cvmodel->appendElement("MaxOrder", d_maxOrder);
  cvmodel->appendElement("Tmin", d_Tmin);
  cvmodel->appendElement("Tmax", d_Tmax);
  cvmodel->appendElement("coefficient", d_coefficient);

}

double PolynomialCv::getSpecificHeat(double T)
{
  // clamp values
  double t = T;
  if(T < d_Tmin)
    t = d_Tmin;
  if(T > d_Tmax)
    t = d_Tmax;

  // Calculate contributions
  double sum = d_coefficient[0];
  double x = 1.0;
  for(int i = 1; i <= d_maxOrder; i++) {
    x   *= t;                     // add in a factor of x
    sum += d_coefficient[i] * x;  // a_i * x**i
  }

  // do the final divide
  return sum/x;
}

double PolynomialCv::getGamma(double T)
{
  return 1.4;  // this should be the input file value
}

double PolynomialCv::getInternalEnergy(double T)
{

  throw new InternalError("No 'getInternalEnergy' function defined for PolynomialCv model.", __FILE__, __LINE__);

  return -999.0;
}

