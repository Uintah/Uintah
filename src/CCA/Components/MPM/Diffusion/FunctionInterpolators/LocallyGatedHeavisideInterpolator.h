/*
 * LocallyGatedHeavisideInterpolator.h
 *
 *  Created on: April 10, 2019
 *      Author: jbhooper
 */

#ifndef CCA_COMPONENTS_MPM_DIFFUSION_FUNCTIONINTERPOLATORS_LOCALLYGATEDHEAVISIDEINTERPOLATOR_H_
#define CCA_COMPONENTS_MPM_DIFFUSION_FUNCTIONINTERPOLATORS_LOCALLYGATEDHEAVISIDEINTERPOLATOR_H_

#include <CCA/Components/MPM/Diffusion/FunctionInterpolators/FunctionInterpolator.h>

namespace Uintah {
  class LocallyGatedHeavisideInterpolator : public FunctionInterpolator {
    public:
               LocallyGatedHeavisideInterpolator( ProblemSpecP      & probSpec
                                                 , SimulationStateP  & simState
                                                 , MPMFlags          * mFlags   );

      virtual ~LocallyGatedHeavisideInterpolator();

      virtual functionPoint interpolate( const functionPoint left
                                       , const functionPoint right
                                       , const double        x_in
                                       , const Vector        grad_in
                                       , const bool          /* minConcSaturation */
                                       ) const;

    private:
      double m_switchLocation;
      double m_minGradNorm;
  };
}



#endif /* CCA_COMPONENTS_MPM_DIFFUSION_FUNCTIONINTERPOLATORS_LOCALLYGATEDHEAVISIDEINTERPOLATOR_H_ */
