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

#ifndef PRES_PDF_MIXMDL_H
#define PRES_PDF_MIXMDL_H

#include <cassert>
#include <cmath>

#include <tabprops/prepro/mixmdl/Integrator.h>
#include <tabprops/prepro/mixmdl/MixMdlFunctor.h>

/** @class PresumedPDFMixMdl
 *  @brief Abstract base class for presumed-PDF mixing models.
 *
 *  @author  James C. Sutherland
 *  @date    February, 2005
 *
 *  Base class to support Presumed PDF mixing models.
 *
 *  Examples of such models include:
 *     Beta-PDF
 *     Clipped Gaussian PDF
 *     Log-normal PDF
 *
 *  The presumed PDF mixing models have a pointer to an integrator
 *  object, which must be set.
 *
 *  Furthermore, FUNCTORS are used to facilitate passing non-static
 *  function pointers to other objects and using them to construct
 *  the integrand.
 *
 *  The integrand is nominally constructed by multiplying:
 *       F(x) * P(x)
 *  where P(x) is the probability of "x" occuring as dictated by
 *  the PDF, and F(x) is an arbitrary function of x.
 *
 *  The user must "hook up" the FUNCTOR representing F(x) to the
 *  mixing model.
 *
 *  See BetaMixMdl.cpp and ClippedGaussMixMdl.cpp
 *  for examples.
 */
class PresumedPDFMixMdl
{
 public:

  /** @brief Construct a PresumedPDFMixMdl object
   *  @param theIntegrator : pointer to an integrator object
   */
  PresumedPDFMixMdl( Integrator * const theIntegrator = NULL )
    : integrator_( theIntegrator ),
      convoluteFunction_( NULL ),
      integrandFunction_( NULL )
  {
    mean_     = 0.0;
    variance_ = 0.0;
  }

  virtual ~PresumedPDFMixMdl()
  {
    delete integrandFunction_;
  }

  /** @brief Drive the integration.
   *
   *  Before applying this, be sure to set:
   *   \li The function to be convoluted.
   *   \li The integrand function.
   *       This can be set here or on the integrator directly.
   *   \li The integrator.
   *
   *  This function drives the calculation of
   *   \f$ \int_0^1 F(\phi) P(\phi) d \phi \f$
   *  where \f$ F(\phi) \f$ is the function to be convoluted and
   *  \f$ P(\phi) \f$ is the PDF of \f$ \phi \f$.
   */
  virtual double integrate() = 0;

  /** @brief Compute the integrand
   *
   *  This is typically a function convoluted with the
   *  presumed PDF.  If we end up needing more generality,
   *  this could be turned into a virtual method.
   */
  inline double integrand( const double x ){
    assert( NULL != convoluteFunction_ );
    return ( (*convoluteFunction_)(x) * get_pdf(x) );
  }

  //====================================
  //         "set" functions
  //====================================

  /** @brief Set the function that will be convoluted with the PDF.
   *
   *  Since this is evaluated using a functor, it may be a
   *  non-static member function of another class.
   */
  void set_convolution_func( FunctorDoubleBase * const functor ){
    assert( NULL != functor );
    convoluteFunction_ = functor;
    set_integrand_func();
  }


  /** @brief Set the integrator.
   *  @param theIntegrator : Pointer to the integator object to be used.
   */
  void set_integrator( Integrator * const theIntegrator ){
    assert( NULL != theIntegrator );
    integrator_ = theIntegrator;
  }

  void set_mean( const double mn ){
    assert( mn <= 1.0 && mn >= 0.0 );
    mean_ = mn;
    varMax_ = mn*(1.0-mn);   // assumes that the mean ranges from 0 to 1
    assert( varMax_ <= 1.0 && varMax_ >= 0.0 );
  }

  /** @brief Set the true (unscaled) variance for the presumed pdf */
  void set_variance( const double var ){
    variance_ = var;
    if( varMax_ > 0 )
      scaleVar_ = var/varMax_;
    else
      scaleVar_ = 0.0;
    assert( variance_ >= 0.0 && variance_ <= varMax_ );
    assert( scaleVar_ >= 0.0 && scaleVar_ <= 1.0 );
  }

  /** Set the scaled variance */
  void set_scaled_variance( const double scalvar ){
    scaleVar_ = scalvar;
    variance_ = scalvar*varMax_;
  }

  //====================================
  //        "get" functions
  //====================================

  inline double get_mean() const{ return mean_; }
  inline double get_variance() const{ return variance_; }
  inline double get_scaled_variance() const{ return scaleVar_; }

  Integrator* const get_integrator() const{ return integrator_; }

  /**
   *  function to get the PDF.
   *
   *  The mean and variance must be set via the appropriate
   *  "set" methods.  Then, given a point in the domain of the PDF,
   *  this function returns the probability of that value occuring.
   */
  virtual double get_pdf(const double eta) = 0;

  //====================================
  // utility functions
  // used for testing integration of PDF.
  //====================================

  // integrates to 1
  inline double test_pdf( const double x )
    { return 1.0; }

  // integrates to the mean
  inline double test_mean( const double x )
    { return x; }

  // integrates to the (unscaled) variance
  inline double test_var( const double x )
    { double d=x-mean_;  return d*d; }


 protected:

  void set_integrand_func(){
    assert( NULL != integrator_ );
    if( NULL == integrandFunction_ )
      integrandFunction_ = new FunctorDouble<PresumedPDFMixMdl>( this, &PresumedPDFMixMdl::integrand );
    integrator_->set_integrand_func( integrandFunction_ );
  }

  //==============================================
  //
  //               Protected Data
  //
  //==============================================

  double mean_;      //      mean        for presumed PDF
  double variance_;  //    variance      for presumed PDF
  double scaleVar_;  // scaled variance  for presumed PDF
  double varMax_;    // maximum variance for the given mean

  //-- pointer to integrator
  //   NOTE: we must also hook up the appropriate integrand functor
  //         in the integrator as a pointer to the integrand function
  //         in this class
  Integrator * integrator_;

  /** functor for the function which will be convoluted with the PDF. */
  FunctorDoubleBase * convoluteFunction_;

  /** functor for the integrand. */
  FunctorDoubleBase * integrandFunction_;

};

//==============================================================================

#endif
