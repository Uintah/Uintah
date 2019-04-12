/*
 * FunctionInterpolatorFactory.h
 *
 *  Created on: Feb 5, 2019
 *      Author: jbhooper
 */

#ifndef CCA_COMPONENTS_MPM_DIFFUSION_FUNCTIONINTERPOLATORS_FUNCTIONINTERPOLATORFACTORY_H_
#define CCA_COMPONENTS_MPM_DIFFUSION_FUNCTIONINTERPOLATORS_FUNCTIONINTERPOLATORFACTORY_H_

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/MPM/Diffusion/FunctionInterpolators/FunctionInterpolator.h>
#include <CCA/Components/MPM/Diffusion/FunctionInterpolators/HeavisideInterpolator.h>
#include <CCA/Components/MPM/Diffusion/FunctionInterpolators/LinearFnInterpolator.h>
#include <CCA/Components/MPM/Diffusion/FunctionInterpolators/GloballyGatedHeavisideInterpolator.h>
#include <CCA/Components/MPM/Diffusion/FunctionInterpolators/LocallyGatedHeavisideInterpolator.h>

#include <string>

namespace Uintah {
  class MPMLabel;
  class MPMFlags;

  class FunctionInterpolatorFactory
  {
    public:
      static FunctionInterpolator* create(  ProblemSpecP      & probSpec
                                         ,  SimulationStateP  & simState
                                         ,  MPMFlags          * flags     );
  };
}



#endif /* CCA_COMPONENTS_MPM_DIFFUSION_FUNCTIONINTERPOLATORS_FUNCTIONINTERPOLATORFACTORY_H_ */
