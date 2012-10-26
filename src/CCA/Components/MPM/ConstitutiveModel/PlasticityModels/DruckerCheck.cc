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

#include "DruckerCheck.h"
#include <Core/ProblemSpec/ProblemSpec.h>
#include <cmath>
#include <vector>


using namespace Uintah;
using namespace std;

DruckerCheck::DruckerCheck(ProblemSpecP& )
{
}

DruckerCheck::DruckerCheck(const DruckerCheck*)
{
}

DruckerCheck::~DruckerCheck()
{
}

void DruckerCheck::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP stability_ps = ps->appendChild("stability_check");
  stability_ps->setAttribute("type","drucker");
}
         
bool 
DruckerCheck::checkStability(const Matrix3& ,
                             const Matrix3& deformRate ,
                             const TangentModulusTensor& Cep ,
                             Vector& )
{
  // Calculate the stress rate
  Matrix3 stressRate(0.0);
  Cep.contract(deformRate, stressRate);

  //cout << "Deform Rate = \n" << deformRate << endl;
  //cout << "Cep = \n" << Cep ;
  //cout << "Stress Rate = \n" << stressRate << endl;

  double val = stressRate.Contract(deformRate);
  //cout << "val = " << val << endl << endl;
  if (val > 0.0) return false;
  return true;
}

