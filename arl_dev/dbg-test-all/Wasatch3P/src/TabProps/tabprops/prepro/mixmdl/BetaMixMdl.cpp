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

#include <cmath>  // for log, exp

#include <tabprops/prepro/mixmdl/BetaMixMdl.h>
#include <tabprops/prepro/mixmdl/Integrator.h>

//------------------------------------------------------------------------------
//  Function to compute the natural logarithm of the gamma function
//  See Numerical Recipes for algorithm
static double lgGamma( const double xx )
{
  static const double SQRT_2_PI = 2.50662827463100;
  static const double cf[] = {
     1.000000000190015,
     76.18009172947146,
    -86.50532032941677,
     24.01409824083091,
    -1.231739572460166,
     1.208650973866179e-3,
    -5.395239384953e-6
  };
  double tmp = xx+5.5;
  double leadingTerms = (xx+0.5)*std::log(tmp) - (tmp);
  double sumterm = cf[0];
  for( int j=1; j<7; j++ )
    sumterm += cf[j]/( xx+double(j) );
  return leadingTerms + std::log(SQRT_2_PI * sumterm / xx);
};
//------------------------------------------------------------------------------
BetaMixMdl::BetaMixMdl()
  : PresumedPDFMixMdl(),
    betaCutoffVariance_( 0.9 )
{
}
//------------------------------------------------------------------------------
double
BetaMixMdl::integrate()
{
  //
  //  Integrate the Beta-PDF.  Treat the beta-pdf carefully at
  //  low and high variances because even singular integrators
  //  fail in this region.
  //

  // WHAT ABOUT CONVERTING BETWEEN FAVRE AND RANS VALUES???

  // jcs:
  //    1. consider storing off the cutoff integrated value and
  //       tagging it with the mean.  Then we may be able to re-use
  //       it for the next calculation if we are varying in the
  //       variance direction fastest.
  //

  assert( NULL != convoluteFunction_ );

  static const double SMALL = 1.0e-8;
  //  static const double ONE_MINUS_SMALL = 1.0-SMALL;

  double ans = 0.0;

  if( variance_ < SMALL )  // use mean values
    {
      ans = (*convoluteFunction_)( mean_ );
    }
  else if( scaleVar_ > 0.99 ) // use unmixed values
    {
      double x = 0.0;
      const double t0 = (*convoluteFunction_)( x );
      x = 1.0;
      const double t1 = (*convoluteFunction_)(x);
      ans = t1*mean_ + t0*(1.0-mean_);
    }
  else if( scaleVar_ > betaCutoffVariance_ ) // interpolate between good value and unmixed
    {
      // for high variances, the BetaPDF cannot be integrated because
      // it becomes singular.  Thus, for such points, we will interpolate
      // between previous values and the unmixed (double-delta function)
      // values to get points where the variance is really high.

      // integrate at the cutoff variance
      const double varCopy     = variance_;
      const double varScalCopy = scaleVar_;
      variance_ = betaCutoffVariance_ * varMax_;
      scaleVar_ = betaCutoffVariance_;
      const double cutoff = integrate();

      // now get the unmixed values
      variance_ = varMax_;
      scaleVar_ = 1.0;
      const double unmixed = integrate();

      // reset variance
      variance_ = varCopy;
      scaleVar_ = varScalCopy;

      // interpolate these to get the answer
      ans = cutoff +
	(unmixed-cutoff)/(varMax_-betaCutoffVariance_*varMax_) *
	(variance_-betaCutoffVariance_*varMax_);
    }
  else // do the integral
    {
      // set break points mean and at extrema
      integrator_->add_singul( mean_ );
      integrator_->add_singul( 0.0 );
      integrator_->add_singul( 1.0e-4 );
      integrator_->add_singul( 1.0-1.0e-4 );
      integrator_->add_singul( 1.0e-8 );
      integrator_->add_singul( 1.0-1.0e-8 );
      integrator_->add_singul( 1.0 );

      ans = integrator_->integrate();
    }

  return ans;
};
//------------------------------------------------------------------------------
double
BetaMixMdl::get_pdf( const double x )
  //
  //  This function computes the beta PDF for a given expectation
  //  value (x) at a given mean and variance
  //
  //  Function(s) called:
  //     lgGamma  - returns the natural log of the gamma function
  //
{
  static const  double SMALL = 1.0e-16;
  static const  double ONEMINUSSMALL = 1.0 - SMALL;

   double pdf = 0.0;

   if (variance_ < SMALL)  // deal with small variances...

    if ( (x <= (mean_ + 100.0*SMALL)) &&
	 (x >= (mean_ - 100.0*SMALL)) )
      pdf = 1.0e9;
    else
      pdf = 0.0;

  else{ // deal with other variances

    double xx = x;
    if ( xx < SMALL )
      xx = SMALL;
    else if ( xx > ONEMINUSSMALL )
      xx = 1.0-SMALL;

    pdf = bpdf(xx);

  }

  return pdf;
};
//--------------------------------------------------------------------
double
BetaMixMdl::bpdf( const double xx )
{
    const double alpha = mean_*( varMax_/variance_ - 1.0 );
    const double beta  = (1.0-mean_)/mean_ * alpha;

    const double lnP =
      (alpha-1.0)*std::log(xx) + (beta-1.0)*std::log(1.0-xx)
      + lgGamma(alpha + beta) - lgGamma(alpha) - lgGamma(beta);

    return std::exp(lnP);
}
//------------------------------------------------------------------------------

/*
#include <iostream>
#include <fstream>
int main()
{
  using namespace std;

  double mean = 0.6;
  double scvar = 0.99;

  cout << "enter the mean: ";
  cin >> mean;
  cout << "enter the scaled variance: ";
  cin >> scvar;

  BetaMixMdl beta;
  beta.set_mean( mean );
  beta.set_scaled_variance( scvar );

  ofstream fout("beta.dat");
  fout << "#  mean=" << mean
       << ",   variance=" << scvar*mean*(1.0-mean)
       << ",   scaled variance=" << scvar << endl;
  fout << "#  x         p"  << endl;

  const int n=501;
  for( int i=1; i<n-1; i++ ){
    const double x = double(i)/double(n-1);
    const double p = beta.get_pdf(x);
    fout << x << "   " << p << endl;
  }
  fout.close();
}
*/
