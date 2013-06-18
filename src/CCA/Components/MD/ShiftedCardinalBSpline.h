/*
 * The MIT License
 *
 * Copyright (c) 1997-2013 The University of Utah
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

      std::vector<double> evaluateInternal(const double,
                                           const int) const;

  };
}

#endif /* SHIFTEDCARDINALBSPLINE_H_ */
