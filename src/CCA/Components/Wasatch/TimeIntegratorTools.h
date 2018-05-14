/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef Wasatch_TimeIntegratorTools_h
#define Wasatch_TimeIntegratorTools_h

namespace WasatchCore{

  /**
   *  \ingroup WasatchCore
   *  \enum  TimeIntegratorEnum
   *  \author Tony Saad
   *  \date   July 2013
   *
   *  \brief Enum that defines the currently supported time integrators in Wasatch.
   */
  enum TimeIntegratorEnum {
    FE,      // Forward-Euler
    RK2SSP,  // Runge-Kutta 2nd order strong stability preserving
    RK3SSP   // Runge-Kutta 3rd order strong stability preserving
  };

  /**
   *  \ingroup WasatchCore
   *  \struct  TimeIntegrator
   *  \author  Tony Saad
   *  \date    July 2013
   *
   *  \brief Defines coefficients for Runge-Kutta type integrators only two level
   storage requirements (i.e. old, and new).
   */
  struct TimeIntegrator {
    TimeIntegratorEnum timeIntEnum;
    std::string name;
    int nStages;
    double alpha[3];
    double beta[3];
    double timeCorrection[3];
    bool hasDualTime;

    TimeIntegrator( const TimeIntegratorEnum theTimeIntEnum )
    : timeIntEnum( theTimeIntEnum )
    {
      initialize();
    }

    TimeIntegrator(const std::string& timeIntName)
    : timeIntEnum( (timeIntName == "RK2SSP") ? RK2SSP : ( (timeIntName == "RK3SSP") ? RK3SSP : FE ) ),
      name( timeIntName )
    {
      initialize();
    }
    
    void initialize()
    {
      switch (timeIntEnum) {
        default:
          
        case FE:
          nStages = 1;
          alpha[0] = 1.0; beta[0]  = 1.0;
          alpha[1] = 0.0; beta[1]  = 0.0;
          alpha[2] = 0.0; beta[2]  = 0.0;
          break;
          
        case RK2SSP:
          nStages = 2;
          alpha[0] = 1.0; beta[0]  = 1.0;
          alpha[1] = 0.5; beta[1]  = 0.5;
          alpha[2] = 0.0; beta[2]  = 0.0;
          break;
          
        case RK3SSP:
          nStages = 3;
          alpha[0] = 1.0;     beta[0]  = 1.0;
          alpha[1] = 0.75;    beta[1]  = 0.25;
          alpha[2] = 1.0/3.0; beta[2]  = 2.0/3.0;
          break;
      }
      hasDualTime = false;
      
      timeCorrection[0] = 0.0; // for the first rk stage, the time is t0
      timeCorrection[1] = 1.0; // for the second rk stage, the time is t0 + dt
      timeCorrection[2] = 0.5; // for the third rk stage, the time is t0 + 0.5*dt
    }
    
    inline void has_dual_time( const bool hasDT ) { hasDualTime = hasDT; }
    inline bool has_dual_time(){ return hasDualTime; }
  };

} // namespace WasatchCore

#endif // Wasatch_TimeIntegratorTools_h
