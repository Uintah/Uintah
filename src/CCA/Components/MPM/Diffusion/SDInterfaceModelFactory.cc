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
#include <CCA/Components/MPM/Diffusion/SDInterfaceModelFactory.h>

#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>

#include <string>
#include <CCA/Components/MPM/Diffusion/DiffusionInterfaces/CommonIFConcDiff.h>
#include <CCA/Components/MPM/Diffusion/DiffusionInterfaces/SDInterfaceModel.h>

using namespace std;
using namespace Uintah;

SDInterfaceModel* SDInterfaceModelFactory::create(ProblemSpecP& ps,
                                                  SimulationStateP& ss,
                                                  MPMFlags* flags,
                                                  MPMLabel* mpm_lb)
{
  ProblemSpecP mpm_ps = 
     ps->findBlockWithOutAttribute("MaterialProperties")->findBlock("MPM");
	if(!mpm_ps)
    throw ProblemSetupException("Cannot find scalar_diffuion_model tag", __FILE__, __LINE__);

  ProblemSpecP child = mpm_ps->findBlock("diffusion_interface");

  if(!child)
    throw ProblemSetupException("Cannot find diffusion_interface tag", __FILE__, __LINE__);
	
  string diff_interface_type;
  if(!child->getWithDefault("type",diff_interface_type, "null"))
    throw ProblemSetupException("No type for diffusion_interface", __FILE__, __LINE__);

  if (flags->d_integrator_type != "implicit" &&
      flags->d_integrator_type != "explicit"){
    string txt="MPM: time integrator [explicit or implicit] hasn't been set.";
    throw ProblemSetupException(txt, __FILE__, __LINE__);
  }

  if(flags->d_integrator_type == "implicit"){
    string txt="MPM:  Implicit Scalar Diffusion is not working yet!";
    throw ProblemSetupException(txt, __FILE__, __LINE__);
  }

  if (diff_interface_type == "common"){
    return(scinew CommonIFConcDiff(child, ss, flags, mpm_lb));
  }else if (diff_interface_type == "null"){
    return(scinew SDInterfaceModel(child, ss, flags, mpm_lb));
  }else{
    throw ProblemSetupException("Unknown Scalar Interface Type ("+diff_interface_type+")", __FILE__, __LINE__);
  }

  return 0;
}
