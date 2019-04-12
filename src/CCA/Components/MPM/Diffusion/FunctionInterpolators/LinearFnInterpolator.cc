/*
 * LinearInterpolator.cc
 *
 *  Created on: Feb 5, 2019
 *      Author: jbhooper
 */

#include <CCA/Components/MPM/Diffusion/FunctionInterpolators/LinearFnInterpolator.h>

namespace Uintah {

  typedef FunctionInterpolator::functionPoint functionPoint;

  LinearFnInterpolator::LinearFnInterpolator( ProblemSpecP      & probSpec
                                            , SimulationStateP  & simState
                                            , MPMFlags          * mFlags    )
  {

  }

  LinearFnInterpolator::~LinearFnInterpolator() {

  }

  functionPoint LinearFnInterpolator::interpolate(  const functionPoint left
                                                 ,  const functionPoint right
                                                 ,  const double        x_in
                                                 ,  const Vector        /* gradient */
                                                 ,  const bool          /* interp flag */
                                                 ) const
  {
    double x_l, x_r, y_l, y_r;
    std::tie(x_l, y_l) = left;
    std::tie(x_r, y_r) = right;
    double delta_x = (x_r - x_l);
    double delta_y = (y_r - y_l);

    double y_out = y_l + ((x_in - x_l)/delta_x)*delta_y;
    return (std::make_tuple(x_in,y_out));
  }
}


