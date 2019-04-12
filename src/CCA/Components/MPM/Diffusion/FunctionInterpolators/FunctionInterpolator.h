/*
 * FunctionInterpolator.h
 *
 *  Created on: Jan 31, 2019
 *      Author: jbhooper
 */

#ifndef CCA_COMPONENTS_MPM_DIFFUSION_FUNCTIONINTERPOLATORS_FUNCTIONINTERPOLATOR_H_
#define CCA_COMPONENTS_MPM_DIFFUSION_FUNCTIONINTERPOLATORS_FUNCTIONINTERPOLATOR_H_

#include <Core/Grid/SimulationStateP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/MPM/MPMFlags.h>

#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah {
  class FunctionInterpolator {

    public:
      typedef std::tuple<double,double> functionPoint;

               FunctionInterpolator();

      virtual ~FunctionInterpolator();

      virtual functionPoint interpolate( const functionPoint left
                                       , const functionPoint right
                                       , const double        x_in
                                       , const Vector        gradient = Vector(0)
                                       , const bool          externalFlag = false
                                       ) const = 0;
  };
}



#endif /* CCA_COMPONENTS_MPM_DIFFUSION_FUNCTIONINTERPOLATORS_FUNCTIONINTERPOLATOR_H_ */
