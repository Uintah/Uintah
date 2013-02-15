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

/**
 * @brief Selection function on [-0.5,0.5)
 * @param X - double, input value for selection function
 * @return 1 if -0.5 <= X < 0.5, 0 otherwise
 */
double s0(const double& X);

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
    CenteredCardinalBSpline(int splineOrder);

    /**
     * @brief Constructor for non-null spline.
     * @param splineOrder - int, order of the spline.
     */
    CenteredCardinalBSpline(const CenteredCardinalBSpline& spline);

    /**
     * @brief
     * @param
     */
    CenteredCardinalBSpline& operator=(const CenteredCardinalBSpline& spline);

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
     * @brief Return the support range of the current spline.
     * @param None
     * @return int The support range (maximum number of grid points) over which the spline has non-zero values.
     */
    inline int getSupport() const
    {
      return this->splineSupport;
    }

    /*
     * @brief Returns half of the support range (rounded down) for the current spline.
     * @param None
     * @return int Half the support range over which the spline is defined. In a 0-centric language, this is
     *             also the array index of the principle value in the arrays returned by evaluate and derivative.
     */
    inline int getHalfSupport() const
    {
      return this->splineHalfSupport;
    }

  private:

    int splineOrder;
    int splineSupport;
    int splineHalfSupport;

    // Stores values necessary for calculating the spline
    std::vector<double> prefactorValues;
    std::vector<int> prefactorMap;
    std::vector<int> basisOffsets;

    // Stores values necessary for calculating the spline derivatives
    std::vector<double> derivativeValues;
    std::vector<int> derivativeMap;
    std::vector<int> derivativeOffsets;

    // Internal functions involved in setting up the spline
    std::vector<int> generateBasisOffsets(const int order);

    std::vector<int> generatePrefactorMap(const int order,
                                          const std::vector<int>& offset);

    std::vector<double> generatePrefactorValues(const int order);

    std::vector<double> evaluateInternal(const double x,
                                         const int order,
                                         const std::vector<int>& map,
                                         const std::vector<int>& offsets,
                                         const std::vector<double>& values) const;
};

}  // End namespace Uintah

#endif
