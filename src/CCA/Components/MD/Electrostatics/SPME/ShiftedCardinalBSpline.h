/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#ifndef UINTAH_MD_SHIFTEDCARDINALBSPLINE_H_
#define UINTAH_MD_SHIFTEDCARDINALBSPLINE_H_

#include <vector>
#include <cmath>
#include <Core/Geometry/Vector.h>

namespace Uintah {

  /*
   * @class ShiftedCardinalBSpline
   * @ingroup MD
   * @author Justin Hooper
   * @date June, 2013
   *
   * @brief
   *
   * @param
   */

  class ShiftedCardinalBSpline {
    public:
      /*
       *
       */
      ShiftedCardinalBSpline();

      /*
       *
       */
      ~ShiftedCardinalBSpline();

      /*
       * @brief Construct from known spline order
       * @param const int order:  Order of the spline to be constructed
       */
      ShiftedCardinalBSpline(const int order);

      /*
       * @brief Construct from already built spline
       * @param const ShiftedCardinalBSpline&:  Already constructed spline
       */
      ShiftedCardinalBSpline(const ShiftedCardinalBSpline& spline);

      /*
       * @brief Evaluate the spline at all points on a regular nearby grid with the
       *        initial point of x
       * @param x:  The reference point from which to build the spline
       * @return std::vector<double>: Uniformly spaced points (spacing of 1.0) along the spline.
       *                              First point at x.
       */
      std::vector<double> evaluateGridAligned(const double x) const;

      /*
       * @brief Find the derivative of the spline at point x.
       * @param x:  Point at which the derivative is to be evaluated
       * @return double:  The value of the derivative
       */
      double derivative(const double x) const;

      /*
       * @brief Evaluate the derivative of the spline at all points on a regular interval across
       *        the entire spline's support.
       * @param x:  The reference point from which to evaluate derivatives.
       * @return std::vector<double>:  Uniformly spaced points (spacing of 1.0) containing the derivative
       *                               values of the spline.  First point at x.
       */
      std::vector<double> derivativeGridAligned(const double x) const;
      /*
       * @brief  Generate spline, 1st, and 2nd derivatives at Vector 0 <= S <= 1
       * @param  splineValues:  Vector with offsets in R^3 for spline in each vector direction
       * @return base:   vector<Vector> for all related grid points (0..p+1) of spline of order p with values of splineValues
       * @return first:  vector<Vector> for all related grid points of 1st order spline derivative
       * @return second: vector<Vector> for all related grid points of 2nd order spline derivative
       */
      void evaluateThroughSecondDerivative(const SCIRun::Vector& splineValues,
                                           std::vector<SCIRun::Vector>& base,
                                           std::vector<SCIRun::Vector>& first,
                                           std::vector<SCIRun::Vector>& second) const;

      /*
       * @brief The maximum support range (number of grid points) the spline will occupy.
       * @param None
       * @return int:  Number of evenly spaced grid points (spacing 1.0) over which the spline
       *                  is non-zero.
       */
      inline int getSupport() const
      {
        return (d_splineOrder + 1);
      }

      /*
       * @brief Returns the spline's order
       * @param None
       * @return int:  The order of the spline
       */
      inline int getOrder() const
      {
        return (d_splineOrder);
      }

      inline ShiftedCardinalBSpline& operator=(const ShiftedCardinalBSpline& spline)
      {
        d_splineOrder = spline.d_splineOrder;

        return *this;
      }

    private:
      int d_splineOrder;

      inline double S2(const double x) const
      {
        if (x < 0 || x > 2)
          return 0;
        return (1.0 - std::abs(x - 1.0));
      }

      inline SCIRun::Vector S2(const SCIRun::Vector In) const
      {
    	  double xVal = In.x();
    	  double yVal = In.y();
    	  double zVal = In.z();

    	  if (xVal < 0 || xVal > 2) {
    		  xVal = 0;
    	  }
    	  else {
    		  xVal = (1.0 - std::abs(xVal - 1.0));
    	  }
    	  if (yVal < 0 || yVal > 2) {
    		  yVal = 0;
    	  }
    	  else {
    		  yVal = (1.0 - std::abs(yVal - 1.0));
    	  }
    	  if (zVal < 0 || zVal > 2) {
    		  zVal = 0;
    	  }
    	  else {
    		  zVal = (1.0 - std::abs(zVal - 1.0));
    	  }
    	  return SCIRun::Vector(xVal,yVal,zVal);
      }
      std::vector<double> evaluateInternal(const double,
                                           const int) const;
      void evaluateVectorizedInternalInPlace(const SCIRun::Vector&,
                                             const int,
                                             std::vector<SCIRun::Vector>&) const;

  };
}

#endif /* SHIFTEDCARDINALBSPLINE_H_ */
