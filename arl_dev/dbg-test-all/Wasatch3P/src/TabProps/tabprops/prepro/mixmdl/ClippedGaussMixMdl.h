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

#ifndef CLIP_GAUSS_MIXMDL
#define CLIP_GAUSS_MIXMDL

#include <cassert>
#include <vector>
#include <string>
#include <cmath>

#include <tabprops/prepro/mixmdl/PresumedPDFMixMdl.h>

class ClipGaussParams;
template <typename T> class BracketHelper;

//====================================================================

//--------------------------------------------------------------------
/**
 *  @class ClipGauss
 *  @brief Implement the Clipped-gaussian mixing model.
 *
 *  @author  James C. Sutherland
 *  @date    February, 2005
 *
 *  Implement the clipped gaussian mixing model.  This implementation is restricted to a
 *  range [0,1].  The clipped gaussian model may be derived for an arbitrary range.
 *  However, I have not done it yet.
 *
 *  For a complete description of the Clipped Gaussian PDF, see
 *     "Evaluation of Mixing and * Reaction Models for Large-Eddy Simulation of
 *     Nonpremixed Turbulent Combustion Using Direct Numerical Simulation"
 *     James C. Sutherland, 2004 (Ph.D. Thesis)
 *
 *  The ClipGauss class is derived from the PresumedPDFMixMdl class.
 */
class ClipGauss : public PresumedPDFMixMdl
{
 public:

  /** @brief Construct a ClipGauss object */
  ClipGauss( const int nfpts=201,
	     const int ngpts=201 );

  ~ClipGauss();

  /**
   *  @brief Integrate the CG pdf.  See notes on the virtual method from the parent class.
   */
  double integrate();

  /**
   *  @brief Obtain the clipped-gaussian PDF for the given mean and variance
   */
  double get_pdf( const double x );

 private:
  ClipGaussParams * cgParams_;
};

//====================================================================

/** Structure to hold clipped gaussian parameters */
class CGParams{
public:
/*  CGParams(){ FF=GG=0.0; converged=false;}

  // copy constructor
  CGParams( const CGParams & s )
  {
    FF = s.FF;
    GG = s.GG;
    converged = s.converged;
  }
*/
  double FF, GG;
  bool converged;
};

//====================================================================

/**
 *  @class ClipGaussParams
 *  @brief Obtain parameters required for the clipped-gaussian PDF.
 *
 *  @author  James C. Sutherland
 *  @date    February, 2005
 *
 *  The ClipGaussParams class computes the parameters required for the clipped-gaussian
 *  function.  The parameters are functions of the mean and variance.
 *
 *  This assumes that the Gaussian is clipped at 0 and 1.
 *
 *  It generates an internal table of coefficients and interpolates those to find a "good"
 *  guess, and then solves the appropriate system of nonlinear equations to obtain the
 *  best possible set of parameters.
 */
class ClipGaussParams{

 public:

  /**
   *  @brief Construct a ClipGaussParams object
   *  @param nMeanPts : Number of points in the mean dimension to precompute.
   *  @param nVarPts  : Number of points in the variance dimension to precompute.
   *
   *  The parameters for the clipped gaussian PDF are obtained via solution to highly
   *  nonlinear equations and often require a very good initial guess.  Building a good
   *  lookup table provides a good starting point for obtaining the clipped gaussian
   *  parameters.
   */
  ClipGaussParams( const int nMeanPts=101,
		   const int nVarPts =201 );

  ~ClipGaussParams();

  /** @brief Set the mean values to be used in the clipped gaussian table */
  void set_means( std::vector<double> & mean );

  /**
   *  @brief Set the scaled variances to be used in the clipped gaussian table
   *
   *   The scaled variance, \f$\sigma^2\f$, is given in terms of the mean,
   *     \f$\bar{\phi}\f$, by
   *     \f$ \sigma^2 = \bar{\phi}(1-\bar{\phi})\f$.
   */
  void set_scaled_variances( std::vector<double> & vars );

  /** Build the clipped gaussian parameter table. */
  void build_table();

  /**
   *  @brief Retrieve the CG parameters for the given mean and UNSCALED variance
   *
   *  @param mean : mean
   *  @param var  : unscaled variance
   */
  CGParams lookup( const double mean,
		   const double var ){
     double varmax = mean*(1.0-mean);
     assert( varmax <= 1.0 );
     return lookup_scal( mean, var/varmax );
  };

  /**
   *  @brief Retrieve the CG parameters for the given mean and SCALED variance
   *
   *  @param mean     : mean
   *  @param scaleVar : scaled variance
   */
  CGParams lookup_scal( const double mean,
			const double scaleVar );

  /** @brief Read the parameter table from disk */
  void read_from_disk( const std::string & fname );

  /** @brief Write the parameter table to disk */
  void write_to_disk(  const std::string & fname );

  int get_div_count() const {return divergeCount_;};

  /** @brief Get the number of nonlinear iterations taken for the last solve */
  int get_n_iter() const {return nonLinIter_;};

  /** @brief Reset all solver statistics */
  void reset_stats(){iterTally_=0; solverTally_=0;};

  /** @brief Write solver statistics to screen */
  void dump_stats();

 private:

  //====================================
  //            Private Data
  //====================================

  bool haveTable_;
  int nMean_, nVariance_, divergeCount_;

  // counters for solver statistics
  int iterTally_, solverTally_;
  int nonLinIter_;

  // these variables facilitate faster lookups
  double lastMeanLookup_, lastVarLookup_;
  CGParams lastParams_;

  // information for the CG parameters table
  std::vector<double> meanPts_, scaleVarPts_;
  std::vector<CGParams> params_;

  BracketHelper<double>* meanBracketer_;
  BracketHelper<double>* varBracketer_;

  //====================================
  //        Private Methods
  //====================================

  void finalize_setup();

  /**
   *  given the mean, scaled variance, and a guess, solve for the CG parameters
   */
  CGParams solve( const CGParams & guess,
		  const double & mean,
		  const double & var );

  void eval_err( const double mean, const double var,
		 const double FF,   const double GG,
		 double & ff_err,   double & gg_err );

  inline double interpolate( const double x1, const double x2,
			     const double y1, const double y2,
			     const double x ){
    assert( x2 != x1 );
    return ( y1 + (y2-y1)/(x2-x1) * (x-x1) );
  }

  inline double interpolate_log( const double x1, const double x2,
				 const double y1, const double y2,
				 const double x )
  {
    using std::exp;   using std::log;  using std::abs;

    double fac1 = (y1==abs(y1)) ? 1.0 : -1.0;
    double fac2 = (y2==abs(y2)) ? 1.0 : -1.0;
    return exp( interpolate( x1, x2, fac1*log(abs(y1)), fac2*log(abs(y2)), x ) );
  }

};


#endif
