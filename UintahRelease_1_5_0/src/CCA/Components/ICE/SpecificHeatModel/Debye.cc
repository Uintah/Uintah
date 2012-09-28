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

#include <CCA/Components/ICE/SpecificHeatModel/SpecificHeat.h>
#include <CCA/Components/ICE/SpecificHeatModel/Debye.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <cmath>
#include <iostream>

using namespace Uintah;

const double kb = 1.3806503e-23; // Boltzmann constant (m^2*kg/s^2*K)

DebyeCv::DebyeCv(ProblemSpecP& ps)
 : SpecificHeat(ps)
{
  ps->require("Atoms",d_N);
  ps->require("DebyeTemperature",d_T_D);
}

DebyeCv::~DebyeCv()
{
}

void DebyeCv::outputProblemSpec(ProblemSpecP& ice_ps)
{
  ProblemSpecP cvmodel = ice_ps->appendChild("SpecificHeatModel");
  cvmodel->setAttribute("type", "Debye");
  cvmodel->appendElement("Atoms", d_N);
  cvmodel->appendElement("DebyeTemperature", d_T_D);
}

double DebyeCv::getSpecificHeat(double T)
{
  double Trat = T/d_T_D;
  double invTrat = d_T_D/T;
  double preIntegralFactor = 9.0*kb*d_N*Trat*Trat*Trat;

  // Riemann sum on integral (for now... this should be replaced with a Riemann Zeta Function w/ n=1)
  double dx = invTrat / 1000.0;
  double integralFactor = 0.0;
  for(double x = 1.0e-12; x <= invTrat; x += dx)
  {
    double eToX = std::exp(x);
    double toAdd = dx * (x*x*x*x*eToX)/((eToX-1.0)*(eToX-1.0));
    if(std::isnan(toAdd) || std::isinf(toAdd)) {
       std::cerr << "Integral portion of Debye Cv was inf or nan.  Ignoring and moving to next iteration..." << std::endl;
       continue;
    }
    integralFactor += toAdd;
  }

  return preIntegralFactor * integralFactor;
}

double DebyeCv::getGamma(double T)
{
  return 1.4;  // this should be the input file value
}

double DebyeCv::getInternalEnergy(double T)
{
  double Trat = T/d_T_D;
  double invTrat = d_T_D/T;
  double preIntegralFactor = 9.0*kb*d_N*Trat*Trat*Trat;

  // Riemann sum on integral (for now... this should be replaced with a Riemann Zeta Function w/ n=1)
  double dx = invTrat / 1000.0;
  double integralFactor = 0.0;
  for(double x = 1.0e-12; x <= invTrat; x += dx)
  {
    double eToX = std::exp(x);
    double toAdd = dx * (x*x*x)/((eToX-1.0));
    if(std::isnan(toAdd) || std::isinf(toAdd)) {
       std::cerr << "Integral portion of Debye U was inf or nan.  Ignoring and moving to next iteration..." << std::endl;
       continue;
    }
    integralFactor += toAdd;
  }

  return preIntegralFactor * integralFactor;
}

