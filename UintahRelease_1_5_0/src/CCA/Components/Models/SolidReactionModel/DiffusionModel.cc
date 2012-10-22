/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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
#include <CCA/Components/Models/SolidReactionModel/DiffusionModel.h>
#include <Core/Exceptions/ProblemSetupException.h>

using namespace Uintah;
using namespace std;

DiffusionModel::DiffusionModel(ProblemSpecP &params)
{
    params->require("dimension", d_dimension);
    if(d_dimension > 4 || d_dimension < 1)
        throw new ProblemSetupException("ERROR: Diffusion reaction model must be 1, 2, 3 or 4 dimensional.", __FILE__, __LINE__);
}

void DiffusionModel::outputProblemSpec(ProblemSpecP& ps)
{
   ProblemSpecP model_ps = ps->appendChild("RateModel");
   model_ps->setAttribute("type","Diffusion");

   model_ps->appendElement("dimension", d_dimension);
}

/// @brief Get the contribution to the rate from the fraction reacted
/// @param fractionReactant The fraction in the volume that is reactant, i.e. m_r/(m_r+m_p)
/// @return a scalar for the extent of reaction
double DiffusionModel::getDifferentialFractionChange(double fractionReactant)
{
    double oneMinus = 1.0-fractionReactant;
    switch(d_dimension)
    {
        case 1:  // 1-Dimensional
            return 0.5*fractionReactant;
        break;
        case 2:  // 2-Dimensional
            return 1.0/(-log(oneMinus));
        break;
        case 3:  // Jander Equation
            return 3.0*pow(oneMinus, 2.0/3.0)/(2.0*(1.0-pow(oneMinus, 1.0/3.0)));
        break;
        case 4:  // Ginstling-Brounshtein
            
        break;
            
        default:
            throw new ProblemSetupException("ERROR: Diffusion reaction model must be 1, 2 or 3 dimensional.", __FILE__, __LINE__);
    }
    return -999.0;
}
