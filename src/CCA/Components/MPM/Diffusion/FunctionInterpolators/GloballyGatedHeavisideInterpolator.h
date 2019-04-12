/*
 * GloballyGatedHeavisideInterpolator.h
 *
 *  Created on: April 10, 2019
 *      Author: jbhooper
 */

#ifndef CCA_COMPONENTS_MPM_DIFFUSION_FUNCTIONINTERPOLATORS_GLOBALLYGATEDHEAVISIDEINTERPOLATOR_H_
#define CCA_COMPONENTS_MPM_DIFFUSION_FUNCTIONINTERPOLATORS_GLOBALLYGATEDHEAVISIDEINTERPOLATOR_H_

#include <CCA/Components/MPM/Diffusion/FunctionInterpolators/FunctionInterpolator.h>

namespace Uintah {
  class GloballyGatedHeavisideInterpolator : public FunctionInterpolator {
    public:
               GloballyGatedHeavisideInterpolator( ProblemSpecP      & probSpec
                                                 , SimulationStateP  & simState
                                                 , MPMFlags          * mFlags   );

      virtual ~GloballyGatedHeavisideInterpolator();

      virtual functionPoint interpolate( const functionPoint left
                                       , const functionPoint right
                                       , const double        x_in
                                       , const Vector        /* gradient */
                                       , const bool          minConcSaturation
                                       ) const;

    private:
      double m_switchLocation;
  };
}



#endif /* CCA_COMPONENTS_MPM_DIFFUSION_FUNCTIONINTERPOLATORS_GLOBALLYGATEDHEAVISIDEINTERPOLATOR_H_ */
