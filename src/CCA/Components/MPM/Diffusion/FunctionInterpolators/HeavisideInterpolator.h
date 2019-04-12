/*
 * HeavisideInterpolator.h
 *
 *  Created on: Feb 5, 2019
 *      Author: jbhooper
 */

#ifndef CCA_COMPONENTS_MPM_DIFFUSION_FUNCTIONINTERPOLATORS_HEAVISIDEINTERPOLATOR_H_
#define CCA_COMPONENTS_MPM_DIFFUSION_FUNCTIONINTERPOLATORS_HEAVISIDEINTERPOLATOR_H_

#include <CCA/Components/MPM/Diffusion/FunctionInterpolators/FunctionInterpolator.h>

namespace Uintah {
  class HeavisideInterpolator : public FunctionInterpolator {
    public:
               HeavisideInterpolator( ProblemSpecP      & probSpec
                                    , SimulationStateP  & simState
                                    , MPMFlags          * mFlags   );

      virtual ~HeavisideInterpolator();

      virtual functionPoint interpolate( const functionPoint left
                                       , const functionPoint right
                                       , const double        x_in
                                       , const Vector        /* gradient */
                                       , const bool          /* interp flag */
                                       ) const;

    private:
      double m_switchLocation;
  };
}



#endif /* CCA_COMPONENTS_MPM_DIFFUSION_FUNCTIONINTERPOLATORS_HEAVISIDEINTERPOLATOR_H_ */
