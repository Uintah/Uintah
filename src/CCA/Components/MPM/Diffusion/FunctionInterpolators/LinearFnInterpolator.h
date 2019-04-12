/*
 * LinearInterpolator.h
 *
 *  Created on: Feb 5, 2019
 *      Author: jbhooper
 */

#ifndef CCA_COMPONENTS_MPM_DIFFUSION_FUNCTIONINTERPOLATORS_LINEARFNINTERPOLATOR_H_
#define CCA_COMPONENTS_MPM_DIFFUSION_FUNCTIONINTERPOLATORS_LINEARFNINTERPOLATOR_H_

#include <CCA/Components/MPM/Diffusion/FunctionInterpolators/FunctionInterpolator.h>

namespace Uintah {
  class LinearFnInterpolator : public FunctionInterpolator {
    public:
               LinearFnInterpolator(  ProblemSpecP      & probSpec
                                   ,  SimulationStateP  & simState
                                   ,  MPMFlags          * mFlags  );

      virtual ~LinearFnInterpolator();

      virtual functionPoint interpolate(  const functionPoint left
                                       ,  const functionPoint right
                                       ,  const double        x_in
                                       ,  const Vector        /* gradient */
                                       ,  const bool          /* input flag */
                                       ) const;

  };
}



#endif /* CCA_COMPONENTS_MPM_DIFFUSION_FUNCTIONINTERPOLATORS_LINEARFNINTERPOLATOR_H_ */
