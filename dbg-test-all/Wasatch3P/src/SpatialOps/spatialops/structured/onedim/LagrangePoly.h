/*
 * Copyright (c) 2014 The University of Utah
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
#ifndef LagrangePoly_h
#define LagrangePoly_h

#include <spatialops/SpatialOpsConfigure.h>

#include <vector>
#include <iterator>

//====================================================================


/**
 *  \class LagrangeCoefficients
 *  \author James C. Sutherland
 *  \date July, 2008
 *  \brief Provides Lagrange polynomial coefficients for interpolant
 *         and first derivatives.
 *
 *  This class is primarily intended for use by the
 *  LagrangeInterpolant and LagrangeDerivative classes.
 *
 *  \par Interpolants
 *  The Lagrange interpolating polynomial of order \f$n\f$ is given
 *  as, \f$f(x)=\sum_{k=0}^{n} y_k L_{k}(x),\f$ with
 *  \f[
 *    L_{k}(x)=\prod_{{i=0}\atop{i\ne k}}^{n}\frac{x-x_{i}}{x_{k}-x_{i}}
 *  \f]
 *
 *  \par First Derivatives
 *  For the derivatives, \f$f(x)=\sum_{k=0}^{n} y_k
 *  L_{k}^{\prime}(x),\f$ with \f$L_{k}^{\prime}(x)\f$ is given as
 * \f[
 *    L_{k}^{\prime}(x) = \left[ \sum_{{j=0}\atop{j\ne k}}^{n} \left(
 *   \prod_{ {i=0}\atop{i \ne j,k} }^n (x-x_i) \right) \right] \left[
 *   \prod_{{i=0}\atop{i\ne k}}^{n} (x_k-x_i) \right]^{-1}.
 *  \f]
 *  Note that \f$n\f$ is the polynomial order.  The derivative order
 *  is one less than this.
 */
class LagrangeCoefficients
{
  typedef std::vector<double>::iterator       VecIter;
  typedef std::vector<double>::const_iterator ConstVecIter;

  const std::vector<double> xpts_;
  std::vector<double> coefs_;

  void get_bounds( const double x, const int polynomialOrder,
                   int& ilo, int& nlo,
                   int& nhi ) const;

public:

  LagrangeCoefficients( const std::vector<double> xpts );
  LagrangeCoefficients( const double* const xbegin,
                        const double* const xend );

  ~LagrangeCoefficients();

  void get_interp_coefs_indices( const double x,
                                 const int polyOrder,
                                 std::vector<double>& coefs,
                                 std::vector<int>& indices ) const;

  void get_derivative_coefs_indices( const double x,
                                     const int polyOrder,
                                     std::vector<double>& coefs,
                                     std::vector<int>& indices ) const;

  const std::vector<double>& get_x() const{ return xpts_; }
};


//====================================================================


/**
 *  \class LagrangeInterpolant
 *  \author James C. Sutherland
 *  \date   July, 2008
 *  \brief Implements interpolants based on Lagrange polynomials.
 */
class LagrangeInterpolant
{
public:

  /**
   *  Constructs a LagrangeInterpolant from the set of data.
   *
   *  \param xpts The independent values.
   *  \param ypts The dependent (function) values.
   *  \param order OPTIONAL.  If supplied at construction, this will
   *         be used as the default polynomial order for the
   *         interpolant.  The polynomial order can also be specified
   *         when the interpolated value is requested.  This value
   *         defaults to 2 (second order).
   *
   *  NOTE: xpts and ypts must be of the same length.
   */
  LagrangeInterpolant( const std::vector<double>& xpts,
                       const std::vector<double>& ypts,
                       const int order = 2 );

  ~LagrangeInterpolant();

  /**
   *  \brief Obtain the interpolated function value at the specified point.
   *
   *  \param x The value at which we want the interpolated function.
   *         It will use an interpolant of the polynomial order
   *         specified at construction (defaults to 2).
   */
  inline double value( const double x ) const{ return value(x,order_); }

  /**
   *  \brief Obtain the interpolated function value at the specified point.
   *  \param x The value at which we want the interpolated function.
   *  \param order The polynomial order of interpolant to use.
   */
  double value( const double x, const int order ) const;

  /**
   *  \brief Obtain the coefficient values and the indices on the
   *         original x vector that are used to form the interpolated
   *         value.
   *  \param x The value at which we want to evaluate the coefficients
   *  \param order The polynomial order of the interpolant.
   *  \param coefs The coefficients to be used in weighting the
   *         original function values to form the interpolated value.
   *  \param indices The indices of the original function values to be
   *         used with the above coefficients.
   */
  void get_coefs_indices( const double x,
                          const int order,
                          std::vector<double>& coefs,
                          std::vector<int>& indices ) const;

private:

  const LagrangeCoefficients coefs_;
  const std::vector<double> ypts_;
  const int order_;
  mutable std::vector<double> coefVals_;
  mutable std::vector<int> indices_;
};


//====================================================================


/**
 *  \class  LagrangeDerivative
 *  \author James C. Sutherland
 *  \date   July, 2008
 *  \brief  Implements derivatives based on Lagrange polynomials.
 */
class LagrangeDerivative
{
public:

  /**
   *  Constructs a LagrangeDerivative from the set of data.
   *
   *  \param x The independent values.
   *  \param y The dependent (function) values.
   *  \param order OPTIONAL.  If supplied at construction, this will
   *         be used as the default polynomial order for the
   *         interpolant.  The polynomial order can also be specified
   *         when the interpolated value is requested.  This value
   *         defaults to 2 (second order).
   *
   *  NOTE: xpts and ypts must be of the same length.
   */
  LagrangeDerivative( const std::vector<double>& x,
                      const std::vector<double>& y,
                      const int order = 2 );

  ~LagrangeDerivative();

  /**
   *  \brief Obtain the interpolated function value at the specified point.
   *
   *  \param x The value at which we want the interpolated function.
   *         It will use an interpolant of the polynomial order
   *         specified at construction (defaults to 2).  The
   *         derivative order will be one less than this.
   */
  inline double value( const double x ) const{ return value(x,order_); }

  /**
   *  \brief Obtain the interpolated function value at the specified point.
   *  \param x The value at which we want the interpolated function.
   *  \param order The polynomial order to use.  Derivative order will
   *         be one less than this.
   */
  double value( const double x, const int order ) const;

  /**
   *  \brief Obtain the coefficient values and the indices on the
   *         original x vector that are used to form the interpolated
   *         value.
   *  \param x The value at which we want to evaluate the coefficients
   *  \param order The polynomial order.  The derivative order of
   *         accuracy will be one less than this.
   *  \param coefs The coefficients to be used in weighting the
   *         original function values to form the interpolated value.
   *  \param indices The indices of the original function values to be
   *         used with the above coefficients.
   */
  void get_coefs_indices( const double x,
                          const int order,
                          std::vector<double>& coefs,
                          std::vector<int>& indices ) const;

private:
  const LagrangeCoefficients coefs_;
  const std::vector<double> ypts_;
  const int order_;
  mutable std::vector<double> coefVals_;
  mutable std::vector<int> indices_;
};


//====================================================================


#endif
