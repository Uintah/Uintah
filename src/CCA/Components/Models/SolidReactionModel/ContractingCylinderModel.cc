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
#include <CCA/Components/Models/SolidReactionModel/ContractingCylinderModel.h>

using namespace Uintah;
using namespace std;

ContractingCylinderModel::ContractingCylinderModel(ProblemSpecP &params)
{
}

void ContractingCylinderModel::outputProblemSpec(ProblemSpecP& ps)
{
   ProblemSpecP model_ps = ps->appendChild("RateModel");
   model_ps->setAttribute("type","ContractingCylinder");
}

/// @brief Get the contribution to the rate from the fraction reacted
/// @param fractionReactant The fraction in the volume that is reactant, i.e. m_r/(m_r+m_p)
/// @return a scalar for the extent of reaction
double ContractingCylinderModel::getDifferentialFractionChange(double fractionReactant)
{
    return 2.0 * sqrt(1.0-fractionReactant);
}
