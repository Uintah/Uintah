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

#include <cmath>
#include <CCA/Components/Models/SolidReactionModel/ModifiedArrhenius.h>

using namespace Uintah;
using namespace std;



ModifiedArrhenius::ModifiedArrhenius(ProblemSpecP& params)
{
    params->require("Ea", d_Ea);
    params->require("A",  d_A);
    params->require("b",  d_b);
}

void ModifiedArrhenius::outputProblemSpec(ProblemSpecP& ps)
{
   ProblemSpecP model_ps = ps->appendChild("RateConstantModel");
   model_ps->setAttribute("type","ModifiedArrhenius");

   model_ps->appendElement("A",  d_A);
   model_ps->appendElement("Ea", d_Ea);
   model_ps->appendElement("b",  d_b);
}

/// @brief Gets a rate constant given a temperature
/// @param T Temperature in Kelvin at which to get constant
/// @return rate Rate at given temperature
double ModifiedArrhenius::getConstant(double T)
{
    double R = 8.314462175;                      // J/molK
    return d_A * pow(T, d_b) * exp(-d_Ea/(R*T));  // s^-1
}

