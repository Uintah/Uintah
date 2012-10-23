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

#include <cmath>
#include <CCA/Components/Models/SolidReactionModel/ProutTompkinsModel.h>
#include <Core/Exceptions/ProblemSetupException.h>

using namespace Uintah;
using namespace std;

ProutTompkinsModel::ProutTompkinsModel(ProblemSpecP &params)
{
    params->require("q", d_q);
    params->getWithDefault("p", d_p, -log10(1.0-d_q));
    params->require("n", d_n);
    params->require("m", d_m);
}

void ProutTompkinsModel::outputProblemSpec(ProblemSpecP& ps)
{
   ProblemSpecP model_ps = ps->appendChild("RateModel");
   model_ps->setAttribute("type","ProutTompkins");

   model_ps->appendElement("q", d_q);
   model_ps->appendElement("p", d_p);
   model_ps->appendElement("m", d_m);
   model_ps->appendElement("n", d_n);
}

/// @brief Get the contribution to the rate from the fraction reacted
/// @param fractionReactant The fraction in the volume that is reactant, i.e. m_r/(m_r+m_p)
/// @return a scalar for the extent of reaction
double ProutTompkinsModel::getDifferentialFractionChange(double fractionReactant)
{
    return pow(fractionReactant, d_n) 
    * pow(1.0-d_q*fractionReactant, d_m);
}
