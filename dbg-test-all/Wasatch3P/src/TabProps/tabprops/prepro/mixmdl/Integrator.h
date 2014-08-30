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

#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <set>
#include <cmath>
#include <limits>
#include <string>

#include <tabprops/prepro/mixmdl/MixMdlFunctor.h>

//====================================================================

/** @class Integrator
 *  @brief Base class for 1-D integrators of single functions.
 *
 *  @author  James C. Sutherland
 *  @date    February, 2005
 *
 *  These integrators use FUNCTORS to represent the function to be
 *  evaluated.  This allows non-static methods of various objects
 *  to be integrated.
 */
class Integrator
{

 public:
  /** @brief Construct an integrator object.
   *  @arg lo : lower limit on integral (defaults to 0.0)
   *  @arg hi : upper limit on integral (defaults to 1.0)
   */
  Integrator( const double lo = 0.0,
	      const double hi = 1.0 );

  /*  destructor  */
  virtual ~Integrator(){ /* does nothing */ }

  //-- set functions for error tolerances
  void set_abs_tol( double err ){ absErrTol_ = err; }
  void set_rel_tol( double err ){ relErrTol_ = err; }

  /**
   *  Set the FUNCTOR that represents the function to be evaluated.
   *  This must be set prior to the integration being performed!
   */
  void set_integrand_func( FunctorDoubleBase * functor )
    { func_ = functor; }

  /**  set limits on the integral  */
  void set_bounds( const double lo, const double hi ){
    loBound_=lo;
    hiBound_=hi;
  }

  /** set a point where a singularity exists in the integrand */
  void add_singul( const double pt ){ singul_.insert(pt); }

  /** @brief Carry out the integration.
   *
   *  Derived classes should ensure that the FUNCTOR has been set
   *  prior to this method being called!
   */
  virtual double integrate() = 0;

  virtual void interrogate(){}
  virtual void set_outfile_name( const std::string & n ){}

  /*  access functions  */
  int get_n_singul() const { return singul_.size(); }

  double get_lobound() const { return loBound_; }
  double get_hibound() const { return hiBound_; }

  virtual int get_n_intervals() const = 0;

  //====================================
  // utility functions
  // used for testing integration of PDF.
  //====================================
  // the integrand function
  double f( const double x);
  double f3(const double x);
  // analytic integral
  double result( const double lo, const double hi );
  double result3( const double lo, const double hi );

 protected:

  //==============================================
  //
  //           Protected member data
  //
  //==============================================

  double loBound_;        // lower limit of integration
  double hiBound_;        // upper limit of integration
  double absErrTol_;      // absolute error tolerance for integrator
  double relErrTol_;      // relative error tolerance for integrator

  std::set<double> singul_;   // list of singularities

  FunctorDoubleBase * func_;  // function to integrate

};

//====================================================================

/**
 *  @author  James C. Sutherland
 *  @date    February, 2005
 *
 *  A basic, adaptive Simpson's integrator.
 *  Does not (yet) deal with singularities...
 */
class BasicIntegrator : public Integrator{

 public:
  BasicIntegrator( double lo = 0.0,
		   double hi = 1.0,
		   int nCoarsePts = 50 )
    : Integrator( lo, hi ),
      nCoarse( nCoarsePts ),
      maxLevels( 10 )
      {
	nMaxEvals = 0;
	npts = 0;
      }

  ~BasicIntegrator(){}

  /** Carry out the integral */
  double integrate();

  int get_n_intervals() const{ return npts; }

 private:

  const int nCoarse;
  const int maxLevels;
  int nMaxEvals;
  int npts;

  double refine( const double a,  const double b,
		 const double fa, const double fb,
		 const double fmid,
		 const double S_whole,
		 int lev );

  inline double simpson( const double f1, const double f2,
			 const double f3, const double dx )
  { return dx/3.0 * (f1 + 4.0*f2 + f3); }

};

//==============================================================================

#endif
