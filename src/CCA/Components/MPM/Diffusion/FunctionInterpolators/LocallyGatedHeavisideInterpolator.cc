/*
 * LOCALLYGatedHeavisideInterpolator.cc
 *
 *  Created on: Apr 10, 2019
 *      Author: jbhooper
 */

#include <CCA/Components/MPM/Diffusion/FunctionInterpolators/LocallyGatedHeavisideInterpolator.h>

namespace Uintah {

  typedef FunctionInterpolator::functionPoint functionPoint;

  LocallyGatedHeavisideInterpolator::LocallyGatedHeavisideInterpolator( ProblemSpecP      & probSpec
                                                           , SimulationStateP  & simState
                                                           , MPMFlags          * mFlags
                                                           , std::string         interpType
                                                           ):FunctionInterpolator(interpType)
  {
    probSpec->require("switch_point",m_switchLocation);
    if (m_switchLocation < 0.0 || m_switchLocation > 1.0) {
      std::string errorMsg = "ERROR:  Heaviside functional interpolator requires switch";
      errorMsg += " point between 0.0 and 1.0.";
      throw ProblemSetupException(errorMsg.c_str(), __FILE__,__LINE__);
    }
    // Value to require the gradient norm to be below to allow solidification.
    probSpec->getWithDefault("min_grad_norm",m_minGradNorm,0.01);

  }

  LocallyGatedHeavisideInterpolator::~LocallyGatedHeavisideInterpolator() {

  }

  functionPoint LocallyGatedHeavisideInterpolator::interpolate( const functionPoint left
                                                  , const functionPoint right
                                                  , const double        x_in
                                                  , const Vector        gradC
                                                  , const bool          /* minConcReached */
                                                  ) const
  {
    double xl, xr;
    double yl, yr;
    std::tie(xl,yl) = left;
    if (gradC.length() > m_minGradNorm) {
      return std::make_tuple(x_in,yl);
      // If gradient is not shallow enough, then solidification should not be allowed and the
      //   transport coefficient should be pinned to the interpolated value.  In this case, that's
      //   simply the liquid value since this is a Heaviside interpolator.
    }
    // If our minimum system concentration is not yet at the desired concentration,
    //   don't allow solidification yet, and dropping to the solid diffusion rate.

    std::tie(xr,yr) = right;
    double xRangeInv = 1.0/(xr - xl);

    // Function is 1.0 until x_switch, then 0.0 after that.
    double y_out = (((x_in-xl)*xRangeInv < m_switchLocation) ? yl : yr);

    return (std::make_tuple(x_in,y_out));


  }

  void LocallyGatedHeavisideInterpolator::outputProblemSpec(ProblemSpecP & ps
                                                           ,bool           doOutput) const
  {
    ProblemSpecP interpPS = ps;
    if (doOutput) {
      interpPS=ps->appendChild("function_interp");
      interpPS->setAttribute("interp",d_interpType);
      interpPS->appendElement("switch_point",m_switchLocation);
      interpPS->appendElement("min_grad_norm", m_minGradNorm);
    }
  }
}


