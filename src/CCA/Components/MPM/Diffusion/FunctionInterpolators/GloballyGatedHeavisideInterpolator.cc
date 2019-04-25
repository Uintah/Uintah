/*
 * GloballyGatedHeavisideInterpolator.cc
 *
 *  Created on: Apr 10, 2019
 *      Author: jbhooper
 */

#include <CCA/Components/MPM/Diffusion/FunctionInterpolators/GloballyGatedHeavisideInterpolator.h>

namespace Uintah {

  typedef FunctionInterpolator::functionPoint functionPoint;

  GloballyGatedHeavisideInterpolator::GloballyGatedHeavisideInterpolator( ProblemSpecP      & probSpec
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

  }

  GloballyGatedHeavisideInterpolator::~GloballyGatedHeavisideInterpolator() {

  }

  functionPoint GloballyGatedHeavisideInterpolator::interpolate( const functionPoint left
                                                  , const functionPoint right
                                                  , const double        x_in
                                                  , const Vector        /* gradient */
                                                  , const bool          minConcReached
                                                  ) const
  {
    double xl, xr;
    double yl, yr;
    std::tie(xl,yl) = left;
    if (!minConcReached) return std::make_tuple(x_in,yl);
    // If our minimum system concentration is not yet at the desired concentration,
    //   don't allow solidification yet, and dropping to the solid diffusion rate.

    std::tie(xr,yr) = right;
    double xRangeInv = 1.0/(xr - xl);

    // Function is 1.0 until x_switch, then 0.0 after that.
    double y_out = (((x_in-xl)*xRangeInv < m_switchLocation) ? yl : yr);

    return (std::make_tuple(x_in,y_out));


  }

  void GloballyGatedHeavisideInterpolator::outputProblemSpec(ProblemSpecP & ps
                                                            ,bool           doOutput ) const
  {
    ProblemSpecP interpPS = ps;
    if (doOutput) {
      interpPS=ps->appendChild("function_interp");
      interpPS->setAttribute("interp",d_interpType);
      interpPS->appendElement("switch_point",m_switchLocation);
    }
  }
}


