/*
 * FunctionInterpolatorFactory.cc
 *
 *  Created on: Feb 5, 2019
 *      Author: jbhooper
 */
#include <CCA/Components/MPM/Diffusion/FunctionInterpolators/FunctionInterpolatorFactory.h>

#include <Core/Exceptions/ProblemSetupException.h>

#include <Core/ProblemSpec/ProblemSpec.h>



namespace Uintah {

  FunctionInterpolator* FunctionInterpolatorFactory::create(  ProblemSpecP      & probSpec
                                                           ,  SimulationStateP  & simState
                                                           ,  MPMFlags          * flags)
  {
    std::string interpType;
    if (!probSpec->getAttribute("interp",interpType)) {
      throw ProblemSetupException("No function interpolator type given.",
                                  __FILE__, __LINE__);
    }

    if (interpType == "linear") {
      return(scinew LinearFnInterpolator(probSpec, simState, flags, interpType));
    }
    if (interpType == "heaviside") {
      return(scinew HeavisideInterpolator(probSpec, simState, flags, interpType));
    }
    if (interpType == "heaviside_global") {
      return(scinew GloballyGatedHeavisideInterpolator(probSpec, simState, flags, interpType));
    }

    if (interpType == "heaviside_local") {
      return(scinew LocallyGatedHeavisideInterpolator(probSpec, simState, flags, interpType));
    }

    throw ProblemSetupException("Unknown function interpolator:  \""+interpType
                                +"\"", __FILE__, __LINE__);
  }
}


