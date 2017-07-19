/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <CCA/Components/MPM/Diffusion/ConductivityModels/BinaryEquation.h>

#include <iostream>

using namespace Uintah;

BinaryEquation::BinaryEquation(ProblemSpecP& ps) :
  ConductivityEquation(ps)
{
  ps->require("min_conc", d_min_conc);
  ps->require("max_conc", d_max_conc);
  ps->require("min_conductivity", d_min_conductivity);
  ps->require("max_conductivity", d_max_conductivity);

  d_slope = (d_max_conductivity - d_min_conductivity)/(d_max_conc - d_min_conc);
}

BinaryEquation::~BinaryEquation()
{

}

double BinaryEquation::computeConductivity(double concentration)
{
  if(concentration > d_min_conc && concentration <= d_max_conc){
    return (d_min_conductivity + d_slope * concentration);
  }else if(concentration > d_max_conc){
    return d_max_conductivity;
  }else{
    return d_min_conductivity;
  }

  //return (d_min_conductivity + d_slope * concentration);
}

void BinaryEquation::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eq_ps;
  eq_ps = ps->appendChild("conductivity_equation");
  eq_ps->setAttribute("type", "binary");
  eq_ps->appendElement("min_conc", d_min_conc);
  eq_ps->appendElement("max_conc", d_max_conc);
  eq_ps->appendElement("min_conductivity", d_min_conductivity);
  eq_ps->appendElement("max_conductivity", d_max_conductivity);
}
