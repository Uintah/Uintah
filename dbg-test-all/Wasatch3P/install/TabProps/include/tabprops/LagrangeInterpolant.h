/**
 *  \file   LagrangeInterpolant.h
 *  \date   Jun 24, 2013
 *  \author "James C. Sutherland"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013 The University of Utah
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
 *
 */

#ifndef LAGRANGEINTERPOLANT_H_
#define LAGRANGEINTERPOLANT_H_

#include <vector>
#include <utility>
#include <cassert>

#include <tabprops/TabPropsConfig.h>

#include <boost/serialization/export.hpp>

/**
 *  \class  LagrangeInterpolant
 *  \date   Jun 24, 2013
 *  \author "James C. Sutherland"
 */
class LagrangeInterpolant
{
protected:
  unsigned order_;
  bool allowClipping_;
public:
  LagrangeInterpolant( const unsigned order, const bool clip ) : order_(order), allowClipping_(clip) {}

  /**
   *  Given the independent variables, obtain the function value.
   */
  virtual double value( const double* const x ) const = 0;

  /**
   *  Given the independent variables, obtain the function value.
   */
  inline double value( const std::vector<double>& x ) const{
    assert( x.size() == get_dimension() );
    return value(&x[0]);
  }

  /**
   * Obtain the function derivative in the requested dimension
   * @param x the independent variable(s)
   * @param dim the dimension that the derivative is requested in (0-based)
   * @return the function derivative with respect to the given independent variable dimension.
   */
  virtual double derivative( const double* const x, const int dim ) const = 0;

  /**
   * Obtain the function derivative in the requested dimension
   * @param x the independent variable(s)
   * @param dim the dimension that the derivative is requested in (0-based)
   * @return the function derivative with respect to the given independent variable dimension.
   */
  inline double derivative( const std::vector<double>& x, const int dim ) const{
    assert( x.size() == get_dimension() );
    return derivative(&x[0],dim);
  }

  /**
   * Obtain the function derivative in the requested dimension
   * @param x the independent variable(s)
   * @param dim1 the dimension that the derivative is requested in (0-based)
   * @param dim2 the dimension that the derivative is requested in (0-based)
   * @return the function derivative with respect to the given independent variable dimension.
   */
  virtual double second_derivative( const double* const x, const int dim1, const int dim2 ) const = 0;

  /**
   * Obtain the function derivative in the requested dimension
   * @param x the independent variable(s)
   * @param dim1 the dimension that the derivative is requested in (0-based)
   * @param dim2 the dimension that the derivative is requested in (0-based)
   * @return the function derivative with respect to the given independent variable dimension.
   */
  inline double second_derivative( const std::vector<double>& x, const int dim1, const int dim2 ) const{
    assert( x.size() == get_dimension() );
    return second_derivative(&x[0],dim1,dim2);
  }

  /**
   *  \brief Obtain the number of independent variables.
   */
  virtual unsigned get_dimension() const = 0;

  /**
   *  \brief Query the order of interpolant.
   */
  unsigned int get_order() const{ return order_; }

  bool clipping() const{ return allowClipping_; }

  /**
   * Obtain the max/min values supported in each dimension for the interpolant.
   */
  virtual std::vector<std::pair<double,double> > get_bounds() const = 0;

  /**
   *  Copy this object and return a LagrangeInterpolant pointer. This
   *  facilitates polymorphic copying given a base-class pointer or reference.
   */
  virtual LagrangeInterpolant* clone() const = 0;

  virtual bool operator==( const LagrangeInterpolant& ) const = 0;
  inline bool operator != ( const LagrangeInterpolant& a ) const{ return !( *this==a ); }

  virtual ~LagrangeInterpolant(){}

  template<typename Archive> void serialize( Archive&, const unsigned int );
};


/**
 * \class LagrangeInterpolant1D
 * \date June, 2013
 * \author James C. Sutherland
 *
 * \brief Provides one-dimensional lagrange polynomial interpolation
 */
class LagrangeInterpolant1D : public LagrangeInterpolant
{
  std::pair<double,double> bounds_;
  std::vector<double> xvals_, fvals_;
  bool isUniform_;
public:
  /**
   * @brief Construct a LagrangeInterpolant1D object.
   * @param order the interpolant order (>= 1)
   * @param xvals the independent variable values
   * @param fvals the dependent variable values
   * @param clipValues if true, then queries outside the range of xvals will be clipped.
   */
  LagrangeInterpolant1D( const unsigned order,
                         const std::vector<double>& xvals,
                         const std::vector<double>& fvals,
                         const bool clipValues=true );

  LagrangeInterpolant1D( const LagrangeInterpolant1D& );

  LagrangeInterpolant1D();

  double value( const double x ) const{ return value(&x); }
  double value( const double* const x ) const;

  double derivative( const double* const x, const int dim ) const;
  double second_derivative( const double* const x, const int dim1, const int dim2 ) const;

  unsigned get_dimension() const{ return 1; }

  std::vector<std::pair<double,double> > get_bounds() const;

  LagrangeInterpolant* clone() const{ return new LagrangeInterpolant1D(*this); }

  bool operator==( const LagrangeInterpolant& ) const;

  ~LagrangeInterpolant1D();

  template<typename Archive> void serialize( Archive&, const unsigned int );
};


/**
 * \class LagrangeInterpolant2D
 * \date June, 2013
 * \author James C. Sutherland
 *
 * \brief Provides two-dimensional lagrange polynomial interpolation
 */
class LagrangeInterpolant2D : public LagrangeInterpolant
{
  std::pair<double,double> xbounds_, ybounds_;
  std::vector<double> xvals_, yvals_, fvals_;
  mutable std::vector<double> xcoef1d_, ycoef1d_;
  bool isUniformX_, isUniformY_;
public:
  /**
   * @brief Construct a LagrangeInterpolant2D object.
   * @param order the interpolant order (>= 1)
   * @param xvals the first independent variable values
   * @param yvals the second independent variable values
   * @param fvals the dependent variable values.  It is assumed that this is
   * stored in a structured arrangement so that it varies in "x" and then in "y"
   * @param clipValues if true, then queries outside the range of xvals will be clipped.
   */
  LagrangeInterpolant2D( const unsigned order,
                         const std::vector<double>& xvals,
                         const std::vector<double>& yvals,
                         const std::vector<double>& fvals,
                         const bool clipValues=true );

  LagrangeInterpolant2D( const LagrangeInterpolant2D& );

  LagrangeInterpolant2D();

  double value( const double* const x ) const;

  double derivative( const double* const x, const int dim ) const;
  double second_derivative( const double* const x, const int dim1, const int dim2 ) const;

  unsigned get_dimension() const{ return 2; }

  std::vector<std::pair<double,double> > get_bounds() const;

  LagrangeInterpolant* clone() const{ return new LagrangeInterpolant2D(*this); }

  bool operator==( const LagrangeInterpolant& ) const;

  ~LagrangeInterpolant2D();

  template<typename Archive> void serialize( Archive&, const unsigned int );
};


/**
 * \class LagrangeInterpolant3D
 * \date June, 2013
 * \author James C. Sutherland
 *
 * \brief Provides three-dimensional lagrange polynomial interpolation
 */
class LagrangeInterpolant3D : public LagrangeInterpolant
{
  std::pair<double,double> xbounds_, ybounds_, zbounds_;
  std::vector<double> xvals_, yvals_, zvals_, fvals_;
  mutable std::vector<double> xcoef1d_, ycoef1d_, zcoef1d_;
  bool isUniformX_, isUniformY_, isUniformZ_;
public:
  /**
   * @brief Construct a LagrangeInterpolant3D object.
   * @param order the interpolant order (>= 1)
   * @param xvals the first independent variable values
   * @param yvals the second independent variable values
   * @param zvals the second independent variable values
   * @param fvals the dependent variable values.  It is assumed that this is
   * stored in a structured arrangement so that it varies in x then y and z.
   * @param clipValues if true, then queries outside the range of xvals will be clipped.
   */
  LagrangeInterpolant3D( const unsigned order,
                         const std::vector<double>& xvals,
                         const std::vector<double>& yvals,
                         const std::vector<double>& zvals,
                         const std::vector<double>& fvals,
                         const bool clipValues=true );

  LagrangeInterpolant3D( const LagrangeInterpolant3D& );

  LagrangeInterpolant3D();

  double value( const double* const x ) const;

  double derivative( const double* const x, const int dim ) const;
  double second_derivative( const double* const x, const int dim1, const int dim2 ) const;

  unsigned get_dimension() const{ return 3; }

  std::vector<std::pair<double,double> > get_bounds() const;

  LagrangeInterpolant* clone() const{ return new LagrangeInterpolant3D(*this); }

  bool operator==( const LagrangeInterpolant& ) const;

  ~LagrangeInterpolant3D();

  template<typename Archive> void serialize( Archive&, const unsigned int );
};


/**
 * \class LagrangeInterpolant4D
 * \date June, 2013
 * \author James C. Sutherland
 *
 * \brief Provides four-dimensional lagrange polynomial interpolation
 */
class LagrangeInterpolant4D : public LagrangeInterpolant
{
  std::vector< std::pair<double,double> > bounds_;
  std::vector<double> x1vals_, x2vals_, x3vals_, x4vals_, fvals_;
  mutable std::vector<double> x1coef1d_, x2coef1d_, x3coef1d_, x4coef1d_;
  bool isUniform_[4];
public:
  /**
   * @brief Construct a LagrangeInterpolant4D object.
   * @param order the interpolant order (>= 1)
   * @param x1vals the first independent variable values
   * @param x2vals the second independent variable values
   * @param x3vals the third independent variable values
   * @param x4vals the fourth independent variable values
   * @param fvals the dependent variable values.  It is assumed that this is
   * stored in a structured arrangement so that it varies in x1 then x2 then x3 then x4.
   * @param clipValues if true, then queries outside the range of xvals will be clipped.
   */
  LagrangeInterpolant4D( const unsigned order,
                         const std::vector<double>& x1vals,
                         const std::vector<double>& x2vals,
                         const std::vector<double>& x3vals,
                         const std::vector<double>& x4vals,
                         const std::vector<double>& fvals,
                         const bool clipValues=true );

  LagrangeInterpolant4D( const LagrangeInterpolant4D& );

  LagrangeInterpolant4D();

  double value( const double* const x ) const;

  double derivative( const double* const x, const int dim ) const;
  double second_derivative( const double* const x, const int dim1, const int dim2 ) const;

  unsigned get_dimension() const{ return 4; }

  std::vector<std::pair<double,double> > get_bounds() const;

  LagrangeInterpolant* clone() const{ return new LagrangeInterpolant4D(*this); }

  bool operator==( const LagrangeInterpolant& ) const;

 ~LagrangeInterpolant4D();

  template<typename Archive> void serialize( Archive&, const unsigned int );
};


/**
 * \class LagrangeInterpolant5D
 * \date June, 2013
 * \author James C. Sutherland
 *
 * \brief Provides four-dimensional lagrange polynomial interpolation
 */
class LagrangeInterpolant5D : public LagrangeInterpolant
{
  std::vector< std::pair<double,double> > bounds_;
  std::vector<double> x1vals_, x2vals_, x3vals_, x4vals_, x5vals_, fvals_;
  mutable std::vector<double> x1coef1d_, x2coef1d_, x3coef1d_, x4coef1d_, x5coef1d_;
  bool isUniform_[5];
public:
  /**
   * @brief Construct a LagrangeInterpolant4D object.
   * @param order the interpolant order (>= 1)
   * @param x1vals the first independent variable values
   * @param x2vals the second independent variable values
   * @param x3vals the third independent variable values
   * @param x4vals the fourth independent variable values
   * @param x5vals the fifth independent variable values
   * @param fvals the dependent variable values.  It is assumed that this is
   * stored in a structured arrangement so that it varies in x1 then x2 then x3 then x4 then x5.
   * @param clipValues if true, then queries outside the range of xvals will be clipped.
   */
  LagrangeInterpolant5D( const unsigned order,
                         const std::vector<double>& x1vals,
                         const std::vector<double>& x2vals,
                         const std::vector<double>& x3vals,
                         const std::vector<double>& x4vals,
                         const std::vector<double>& x5vals,
                         const std::vector<double>& fvals,
                         const bool clipValues=true );

  LagrangeInterpolant5D( const LagrangeInterpolant5D& );

  LagrangeInterpolant5D();

  double value( const double* const x ) const;

  double derivative( const double* const x, const int dim ) const;
  double second_derivative( const double* const x, const int dim1, const int dim2 ) const;

  unsigned get_dimension() const{ return 5; }

  std::vector<std::pair<double,double> > get_bounds() const;

  LagrangeInterpolant* clone() const{ return new LagrangeInterpolant5D(*this); }

  bool operator==( const LagrangeInterpolant& ) const;

 ~LagrangeInterpolant5D();

  template<typename Archive> void serialize( Archive&, const unsigned int );
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT( LagrangeInterpolant )
BOOST_CLASS_EXPORT_KEY( LagrangeInterpolant1D )
BOOST_CLASS_EXPORT_KEY( LagrangeInterpolant2D )
BOOST_CLASS_EXPORT_KEY( LagrangeInterpolant3D )
BOOST_CLASS_EXPORT_KEY( LagrangeInterpolant4D )
BOOST_CLASS_EXPORT_KEY( LagrangeInterpolant5D )

#endif /* LAGRANGEINTERPOLANT_H_ */
