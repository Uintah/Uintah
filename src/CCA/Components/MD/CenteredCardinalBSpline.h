/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifndef UINTAH_MD_CENTEREDCARDINALBSPLINE_H
#define UINTAH_MD_CENTEREDCARDINALBSPLINE_H

#include <vector>
#include <cmath>

namespace Uintah {

/**
 *  @class CenteredCardinalBSpline
 *  @ingroup MD
 *  @author Justin Hooper & Alan Humphrey
 *  @date   January, 2013
 *
 *  @brief
 *
 *  @param
 */

class CenteredCardinalBSpline {

  public:
    /**
     * @brief
     * @param
     */
    CenteredCardinalBSpline();

    /**
     * @brief
     * @param
     */
    ~CenteredCardinalBSpline();

    /**
     * @brief Single argument constructor.
     * @param splineOrder The order of the spline.
     */
    CenteredCardinalBSpline(const int order);

    /**
     * @brief Constructor for non-null spline.
     * @param splineOrder - int, order of the spline.
     */
    CenteredCardinalBSpline(const CenteredCardinalBSpline& spline);

    /**
     * @brief Evaluate the spline across the entire support range.
     * @param x  Value of the point for which spline is calculated.
     * @return std::vector<double> - Contains the spline values on evenly spaced points with spacing 1.0
     *           over the entire support range of the spline.
     */
    std::vector<double> evaluate(const double x) const;

    /**
     * @brief Generate the derivative of the spline for the entire support range
     * @param x - double, Value of the point for which the spline derivative is calculated
     * @return std::vector<double> - Contains the spline derivatives on evenly spaced points with spacing 1.0
     *           over the entire support range of the spline.
     */
    std::vector<double> derivative(const double x) const;

    /*
     * @brief
     * @param
     * @return
     */
    inline double maxMagnitude(const int order) const
    {
      return (static_cast<double>(order + 1) / 2.0);
    }

    /*
     * @brief
     * @param
     * @param
     * @return
     */
    inline int leftSupport(const double x,
                           const int order) const
    {
      if (x < -maxMagnitude(order) || x > maxMagnitude(order)) {
        return 0;
      } else {
        return static_cast<int>(ceil(-maxMagnitude(order) - x));
      }
    }

    /*
     * @brief
     * @param
     * @param
     * @return
     */
    inline int rightSupport(const double x,
                            const int order) const
    {
      if (x < -maxMagnitude(order) || x > maxMagnitude(order)) {
        return 0;
      } else {
        return static_cast<int>(floor(maxMagnitude(order) - x));
      }
    }

    /*
     * @brief Return the order of the current spline.
     * @param None
     * @return int The order of the current spline.
     */
    inline int getOrder() const
    {
      return this->splineOrder;
    }

    /*
     * @brief Return the max support of this spline.
     * @param None
     * @return int The max support of this spline.
     */
    inline int getMaxSupport() const
    {
      return (this->splineOrder + 1) / 2;
    }

    /*
     * @brief Return half of the max support of this spline.
     * @param None
     * @return int Half of the max support of this spline
     */
    inline int getHalfMaxSupport() const
    {
      return (this->splineOrder + 1) / 4;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline CenteredCardinalBSpline& operator=(const CenteredCardinalBSpline& spline)
    {
      splineOrder = spline.splineOrder;
      basisShifts = spline.basisShifts;
      prefactorValues = spline.prefactorValues;
      prefactorMap = spline.prefactorMap;

      return *this;
    }

    /**
     * @brief Selection function on [-0.5,0.5)
     * @param x Input value for selection function
     * @return 1 if -0.5 <= X < 0.5, 0 otherwise
     */
    inline double S0(const double x) const
    {
      if (x < -0.5) {
        return 0;
      }
      if (x >= 0.5) {
        return 0;
      }
      return 1;
    }

    /**
     * @brief Selection function on [-1.0,1.0)
     * @param x Input value for selection function
     * @return 1 if -1.0 <= X < 1.0, 0 otherwise
     */
    inline double S1(const double x) const
    {
      if (x <= -1.0) {
        return 0;
      }
      if (x > 1.0) {
        return 0;
      }
      return (1.0 - std::abs(x));
    }

  private:

    int splineOrder;

    // For calculating values of the spline
    std::vector<int> basisShifts;             //!<
    std::vector<double> prefactorValues;      //!<
    std::vector<int> prefactorMap;            //!<

    std::vector<int> generateBasisShifts(const int);

    std::vector<double> generatePrefactorValues(const int);

    std::vector<int> generatePrefactorMap(const int,
                                          const std::vector<int>&);

    std::vector<double> evaluateInternal(const double x,
                                         const int order,
                                         const std::vector<int>& shifts,
                                         const std::vector<int>& map,
                                         const std::vector<double>& values) const;
};

}  // End namespace Uintah

#endif
