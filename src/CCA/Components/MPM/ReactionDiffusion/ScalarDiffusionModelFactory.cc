/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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
#include <CCA/Components/MPM/ReactionDiffusion/ScalarDiffusionModelFactory.h>
#include <CCA/Components/MPM/ReactionDiffusion/ScalarDiffusionModel.h>
#include <CCA/Components/MPM/ReactionDiffusion/JGConcentrationDiffusion.h>
#include <CCA/Components/MPM/ReactionDiffusion/RFConcDiffusion1MPM.h>
#include <CCA/Components/MPM/ReactionDiffusion/GaoDiffusion.h>

#include <sci_defs/uintah_defs.h> // For NO_FORTRAN

#include <CCA/Components/MPM/MPMFlags.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>

#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace Uintah;

ScalarDiffusionModel* ScalarDiffusionModelFactory::create(ProblemSpecP& ps,
                                                          SimulationStateP& ss,
                                                          MPMFlags* flags)
{
  ProblemSpecP child = ps->findBlock("diffusion_model");
  if(!child)
    throw ProblemSetupException("Cannot find scalar_diffusion_model tag", __FILE__, __LINE__);
  string diffusion_type;
  if(!child->getAttribute("type", diffusion_type))
    throw ProblemSetupException("No type for scalar_diffusion_model", __FILE__, __LINE__);

  if (diffusion_type == "linear")
    return(scinew ScalarDiffusionModel(child, ss, flags, diffusion_type));
  if (diffusion_type == "jg")
    return(scinew JGConcentrationDiffusion(child, ss, flags, diffusion_type));
  else if (diffusion_type == "rf1")
    return(scinew RFConcDiffusion1MPM(child, ss, flags, diffusion_type));
  else if (diffusion_type == "gao_diffusion")
    return(scinew GaoDiffusion(child, ss, flags, diffusion_type));

  else
    throw ProblemSetupException("Unknown Scalar Diffusion Type ("+diffusion_type+")", __FILE__, __LINE__);

  return 0;
}
