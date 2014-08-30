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

#include <iostream>   // for debug and testing io
#include <fstream>
#include <iomanip>
#include <cmath>
#include <math.h>     // for erf
#include <algorithm>  // for std::max()

#include <tabprops/prepro/mixmdl/ClippedGaussMixMdl.h>
#include <tabprops/prepro/mixmdl/BracketHelper.h>
#include <tabprops/prepro/mixmdl/Integrator.h>

using std::max;
using std::vector;
using std::string;

static const double PI = 3.14159265358979;

//#define CGDEBUG

//====================================================================

ClipGauss::ClipGauss( const int nfpts,
		      const int ngpts )
  : PresumedPDFMixMdl(),
    cgParams_( NULL )
{
  cgParams_ = new ClipGaussParams(nfpts,ngpts);
  //  cgParams_->read_from_disk("cgparams.dat");
  cgParams_->build_table();
  cgParams_->write_to_disk("cgparams.dat");
}
//--------------------------------------------------------------------
ClipGauss::~ClipGauss()
{
  delete cgParams_;
}
//--------------------------------------------------------------------
double
ClipGauss::integrate( )
{
  static const double SMALL = 1.0e-5;
  //  cgParams_->reset_stats();

  // set bounds on the integral from 0+ to 1- because we
  // deal with the endpoints using the intermittancies below
  integrator_->set_bounds( SMALL, 1.0-SMALL );

  // deal with very low (completely mixed) and
  // very high (completely unmixed) variances
  if( scaleVar_ == 0.0  || variance_ < SMALL ){
    return (*convoluteFunction_)(mean_);
  }
  else if( scaleVar_ == 1.0 || variance_ > 1.0-SMALL ){
    const double t0 = (*convoluteFunction_)(0.0);
    const double t1 = (*convoluteFunction_)(1.0);
    return mean_*t1 + (1.0-mean_)*t0;
  }

  // do the integration, computing the intermittancies
  // and adding them as appropriate.
  const double alpha0 = (*convoluteFunction_)(0.0) * get_pdf(0.0);
  const double alpha1 = (*convoluteFunction_)(1.0) * get_pdf(1.0);

  double integral = 0.0;
  if( scaleVar_ < 1.0 )
    integral = integrator_->integrate();

  //cgParams_->dump_stats();

  return alpha0 + alpha1 + integral;
}
//--------------------------------------------------------------------
double
ClipGauss::get_pdf( const double x )
{
  using std::sqrt;  using std::pow;
  using std::abs;   using std::exp;

  //-- Get FF and GG parameters from the data structure
  const CGParams param = cgParams_->lookup_scal( mean_, scaleVar_ );

  const double & FF = param.FF;
  const double & GG = param.GG;

  // compute the pdf
  //-- return intermittancies if we are at 0 or 1
  if ( x == 0.0 )        // just alpha1
    return ( 0.5 * ( erf(-FF/sqrt(abs(2.0*GG))) + 1.0 ) );

  else if ( x == 1.0 )    // just alpha2
    return ( 0.5 * ( 1.0 - erf((1.0-FF)/sqrt(abs(2.0*GG))) ) );

  else
    return ( exp(-pow(x-FF,2.0)/(2.0*GG)) / sqrt(abs(2.0*PI*GG)) );
}

//====================================================================
//====================================================================

ClipGaussParams::ClipGaussParams( const int nMeanPts,
				  const int nVarPts )
  : meanBracketer_( NULL ),
    varBracketer_( NULL )
{
  haveTable_ = false;
  divergeCount_ = 0;
  iterTally_ = 0;
  solverTally_ = 0;
  nMean_ = nMeanPts;
  nVariance_ = nVarPts;

  lastMeanLookup_ = -1.0;
  lastVarLookup_  = -1.0;

  params_.resize(nMean_*nVariance_);
  meanPts_.resize(nMean_);
  scaleVarPts_.resize(nVariance_);

  double dx = 1.0 / double(nMean_-1);
  for( int i=0; i<nMean_; i++ )
    meanPts_[i] = double(i)*dx;

  dx = 1.0 / double(nVariance_-1);
  for( int i=0; i<nVariance_; i++ )
    scaleVarPts_[i] = double(i)*dx;
}
//--------------------------------------------------------------------
ClipGaussParams::~ClipGaussParams()
{
  delete meanBracketer_;
  delete varBracketer_;
}
//--------------------------------------------------------------------
void
ClipGaussParams::set_means( vector<double> & mean )
{
  for( int i=0; i<(int)mean.size(); i++ ){
    if( mean[i] < 0.0  ||  mean[i] > 1.0 ) return;
  }
  meanPts_ = mean;
  nMean_ = meanPts_.size();
}
//--------------------------------------------------------------------
void
ClipGaussParams::set_scaled_variances( vector<double> & vars )
{
  for( int i=0; i<(int)vars.size(); i++ ){
    if( vars[i] < 0.0  ||  vars[i] > 1.0 ) return;
  }
  scaleVarPts_ = vars;
  nVariance_ = vars.size();
}
//--------------------------------------------------------------------
CGParams
ClipGaussParams::lookup_scal( const double mean,
			      const double var )
{
  //
  //  keep track of the last lookup we made, and store the results
  //  if we are doing the same lookup, just return the stored value
  //  rather than redoing it.
  //
  //  This will often be the case, since in an integration over the
  //  pdf, we hold mean and variance fixed (i.e. fixed FF and GG) and
  //  then integrate over the range of the PDF.  In the coarse of the
  //  integration, this routine may be called MANY times!
  if( lastMeanLookup_ == mean  && lastVarLookup_  == var ){
    if( lastParams_.converged ) return lastParams_;
  }
  //
  // bracket the mean and variance in the table.
  //
  int loMn, hiMn, loVr, hiVr;
  meanBracketer_->bracket( mean, loMn, hiMn );
  varBracketer_->bracket( var, loVr, hiVr );

  //
  //  ROUGHLY SPEAKING:
  //    FF varies   linearly    with mean and linearly with var
  //    GG varies exponentially with var  but linearly with mean
  //  So we interpolate accordingly.
  //

  int ix1, ix2;

  //
  // step 1: interpolate in the variance direction...
  //
  CGParams pmn1, pmn2;
  ix1 = loMn*nVariance_ + loVr;
  ix2 = loMn*nVariance_ + hiVr;

  if(      ! params_[ix1].converged ) pmn1 = params_[ix2];
  else if( ! params_[ix2].converged ) pmn1 = params_[ix1];

  if( loVr==0 )
    pmn1 = params_[ix2];
  else if( hiVr == nVariance_-1 )
    pmn1 = params_[ix1];
  else{
    pmn1.FF = interpolate( scaleVarPts_[loVr], scaleVarPts_[hiVr],
			   params_[ix1].FF, params_[ix2].FF, var );

    pmn1.GG = interpolate_log( scaleVarPts_[loVr], scaleVarPts_[hiVr],
			       params_[ix1].GG, params_[ix2].GG, var );

    pmn1.converged = params_[ix1].converged && params_[ix2].converged && true;
  }
  //
  // second interpolation to finish out variance direction.
  //
  ix1 = hiMn*nVariance_ + loVr;
  ix2 = hiMn*nVariance_ + hiVr;

  if(      ! params_[ix1].converged ) pmn2 = params_[ix2];
  else if( ! params_[ix2].converged ) pmn2 = params_[ix1];

  if( loVr == 0 )
    pmn2 = params_[ix2];
  else if( hiVr == nVariance_-1 )
    pmn2 = params_[ix1];
  else{
    pmn2.FF = interpolate( scaleVarPts_[loVr], scaleVarPts_[hiVr],
			   params_[ix1].FF, params_[ix2].FF, var );

    pmn2.GG = interpolate_log( scaleVarPts_[loVr], scaleVarPts_[hiVr],
			       params_[ix1].GG, params_[ix2].GG, var );

    pmn2.converged = params_[ix1].converged && params_[ix2].converged && true;
  }
  //
  // step 2: interpolate in the mean direction...
  //
  CGParams p;
  if( loMn == 0 )
    p = pmn2;
  else if( hiMn == nMean_-1 )
    p = pmn1;
  else{
    p.FF = interpolate( meanPts_[loMn], meanPts_[hiMn],
			pmn1.FF, pmn2.FF, mean );

    p.GG = interpolate( meanPts_[loMn], meanPts_[hiMn],
			pmn1.GG, pmn2.GG, mean );

    p.converged = pmn1.converged && pmn2.converged;
  }

  // Solve for this point to get a better answer,
  // rather than returning the interpolated value.
  // Then store it for potential re-use.
  lastParams_ = solve( p, mean, var );
  lastMeanLookup_ = mean;
  lastVarLookup_ = var;

  if( !lastParams_.converged )
    std::cout << "Did not converge for [" << mean << "," << var << "]" << std::endl;

  return lastParams_;
}
//--------------------------------------------------------------------
void
ClipGaussParams::read_from_disk( const string & fname )
{
  using namespace std;

  if( haveTable_ ){
    delete meanBracketer_;  meanBracketer_=NULL;
    delete varBracketer_;   varBracketer_ =NULL;
  }

  ifstream fin( fname.c_str(), ios::in );
  assert( fin );

  string stmp;
  char ctmp;

  getline(fin,stmp); //chew up first 2 lines
  getline(fin,stmp);
  fin.get(ctmp);

  fin >> nMean_ >> nVariance_;

  // set up storage
  params_.resize(nMean_*nVariance_);
  meanPts_.resize(nMean_);
  scaleVarPts_.resize(nVariance_);

  for( int i=0; i<4; i++ ) getline(fin,stmp);

  assert(fin);

  for( int imn=0; imn<nMean_; imn++ ){
    for( int ivar=0; ivar<nVariance_; ivar++ ){
      int ix = imn*nVariance_ + ivar;
      double dtmp;
      fin >> meanPts_[imn] >> scaleVarPts_[ivar] >> dtmp
	  >> params_[ix].FF >> params_[ix].GG;
      getline(fin,stmp);
      int si = stmp.find("failed",0);
      if( si >= 0  && si < (int)stmp.length() ){
	params_[ix].converged = false;
	divergeCount_++;
	stmp.clear();
      }
      else
	params_[ix].converged = true;
      assert( fin );
    }
  }
  fin.close();
  finalize_setup();
}
//--------------------------------------------------------------------
void ClipGaussParams::write_to_disk( const string & fname )
{
  if( !haveTable_ ) return;

  using namespace std;
  ofstream fout( fname.c_str(), ios::out );

  fout << "#--------------------------------------------------------------------" << endl
       << "# N mean  N Var" << endl
       << "# " << setw(4) << nMean_ << setw(8) << nVariance_ << endl;

  fout << setiosflags(ios::left)
       << setw(12) << "# mean"
       << setw(12) << "scaled"
       << setw(12) << "variance"
       << setw(18) << "FF"
       << setw(18) << "GG" << endl << "#"
       << setiosflags(ios::right) << setw(19) << "variance" << endl
       << resetiosflags( ios::right );

  fout << "#--------------------------------------------------------------------" << endl;

  for( int imean=0; imean<nMean_; imean++ ){
    double varmax = meanPts_[imean]*( 1.0 - meanPts_[imean] );
    for( int ivar=0; ivar<nVariance_; ivar++ ){
      int ix = imean*nVariance_ + ivar;
      fout << setiosflags(ios::fixed | ios::left | ios::showpoint)
	   << setiosflags( ios::left )
	   << setprecision(6)
	   << setw(12) << meanPts_[imean]
	   << setw(12) << scaleVarPts_[ivar]
	   << setw(12) << scaleVarPts_[ivar]*varmax;

      fout << resetiosflags( ios::fixed | ios::right )
	   << setiosflags(ios::scientific)
	   << setprecision(8)
	   << setw(18) << params_[ix].FF
	   << setw(18) << params_[ix].GG;
      if( ! params_[ix].converged )  fout << "  convergence failed!";
      fout << endl;
    }
    fout << endl;
  }
  fout.close();
}
//--------------------------------------------------------------------
void
ClipGaussParams::build_table()
{
  if( haveTable_ ){
    delete meanBracketer_;  meanBracketer_=NULL;
    delete varBracketer_;   varBracketer_ =NULL;
  }

  divergeCount_=0;
  //
  // NOTE:
  //  good guesses are CRUCIAL for this thing to converge!  At very
  //  high variances, the system becomes very poorly conditioned!
  //
  CGParams guess;
  guess.FF=meanPts_[1];

  double varmax = meanPts_[1]*(1.0-meanPts_[1]);
  guess.GG=scaleVarPts_[1];

  for( int imean=0; imean<nMean_; imean++ ){
    varmax = meanPts_[imean]*(1.0-meanPts_[imean]);
    for( int ivar=0; ivar<nVariance_; ivar++ ){
      const int ix = imean*nVariance_ + ivar;
      params_[ix] = solve( guess, meanPts_[imean], scaleVarPts_[ivar] );

      if( ! params_[ix].converged )
	{ divergeCount_++; }

      guess = params_[ix];
      if( ivar >= nVariance_-3 ){
	guess.FF = guess.FF * 1.75;
	guess.GG = guess.GG * 4.5;
      }
      else if( ivar >= nVariance_-4 ){
	guess.GG = guess.GG * 2.0;
	guess.FF = guess.FF * 1.25;
      }
      else if( ivar >= nVariance_-5 ){
	guess.GG = guess.GG * 1.5;
      }
    }
    // set guess to previous mean entry with zero variance
    guess = params_[imean*nVariance_];
  }
  finalize_setup();
#ifdef CGDEBUG
  dump_stats();
#endif
}
//--------------------------------------------------------------------
CGParams
ClipGaussParams::solve( const CGParams & guess,
			const double & mean,
			const double & scalVar )
{
  using std::abs;

#ifdef CGDEBUG
  using std::cout; using std::endl;
#endif
  static const int    MAXIT = 2000;
  static const double
    SMALL = 1.0e-8,
    LARGE = 1.0e9,
    ABS_TOL = 1.0e-9,     // Linf norm relative error tolerance for solver
    REL_TOL = 1.0e-7,     // Linf norm absolute error tolerance
    RELAX_FACT = 1.00,    // relaxation factor for nonlinear solve
    INCREMENT = 1.0e-6;   // increment for determining derivatives

  const double var = mean*(1.0-mean) * scalVar;

  // copy guesses into solution space to start solution...
  CGParams prm = guess;
  prm.converged = false;
  double & FF = prm.FF;
  double & GG = prm.GG;

  if( GG == 0.0 ) GG = var;

  if( var <= SMALL ){
    // assume mean values (delta function at mean)
    FF = mean;
    GG = 0.0;
    prm.converged = true;
    return prm;
  }
  if( scalVar > 1.0-SMALL ){
    FF = mean;
    GG = LARGE;
    prm.converged = true;
    return prm;
  }

  solverTally_++;

  // solve the nonlinear system for the modified mean (FF) and variance (GG)
  nonLinIter_ = 0;
  bool converged = false;
  while( !converged ){
    double meanErr, varErr;
    eval_err( mean, var, FF, GG, meanErr, varErr );
    const double relErr = max(abs(meanErr/mean), abs(varErr/var));
    const double absErr = max(abs(meanErr), abs(varErr));

    if( nonLinIter_ >= MAXIT ){
      prm.converged = false;
#ifdef CGDEBUG
      cout << "solving for mean=" << mean << ", scaled var=" << scalVar << ", var="<<var << endl
	   << "   with guesses  FF=" << guess.FF << ",  GG="<< guess.GG
	   << endl
	   << "  *did not converge in " << nonLinIter_ << " iterations" << endl
	   << "    relative error norm: " << relErr << endl
	   << "    absolute error norm: " << absErr << endl
	   << "  current values:  FF=" << FF << ", GG=" << GG
	   << endl << endl;
#endif
      //    break;
      return prm;
    }

    if( absErr <= ABS_TOL  || relErr <= REL_TOL ){
      converged=true;
      prm.converged=true;
      continue;
    }

    //---------------------------------------------------------//
    // evaluate the jacobian in the following format           //
    //                      __                          __     //
    //                     |                              |    //
    //                     | del(errMean)    del(errMean) |    //
    //                     | ------------    ------------ |    //
    //                     |   del(FF)         del(GG)    |    //
    //     jacobian(i,j) = |                              |    //
    //                     | del(errVar)    del(g_err)    |    //
    //                     | -----------    ----------    |    //
    //                     |   del(FF)       del(GG)      |    //
    //                     |__                          __|    //
    //---------------------------------------------------------//

    double m1, m2, v1, v2;
    double jacobian[2][2];

    //-- info for jacobian(:,1)
    eval_err( mean, var, FF*(1-INCREMENT), GG, m1, v1 );
    eval_err( mean, var, FF*(1+INCREMENT), GG, m2, v2 );
    jacobian[0][0] = (m2 - m1)/(2*FF*INCREMENT);
    jacobian[1][0] = (v2 - v1)/(2*FF*INCREMENT);

    //-- info for jacobian(:,2)
    eval_err( mean, var, FF, GG*(1-INCREMENT), m1, v1 );
    eval_err( mean, var, FF, GG*(1+INCREMENT), m2, v2 );
    jacobian[0][1] = (m2 - m1)/(2*GG*INCREMENT);
    jacobian[1][1] = (v2 - v1)/(2*GG*INCREMENT);

    // Now solve the linear system for the correction vector
    //   (dx):  [J]*(dx)=(-r(x))
    // where r(x) is the residual vector.
    //
    // solve directly using analytical solution for 2x2 system
    double dx[2];
    double tmp = jacobian[0][0]*jacobian[1][1] - jacobian[1][0]*jacobian[0][1];
    if( tmp == 0.0 ){
      prm.converged = false;
      return prm;
    }
    dx[0] =  (jacobian[0][1]*varErr - jacobian[1][1]*meanErr)/tmp;
    dx[1] = -(jacobian[0][0]*varErr - jacobian[1][0]*meanErr)/tmp;

    //-- check for convergence on displacement vector
    m1 = abs(dx[0]/FF);
    v1 = abs(dx[1]/GG);
    if( max( dx[0],dx[1] ) < ABS_TOL  &&  max( m1,v1 ) < REL_TOL ){
      prm.converged = true;
      converged=true;
    }

    //-- relax the solution trajectory if necessary
    int count = 0;
    while( count < 10 ) {
      count++;

      //-- check lower bound on GG (ensure positiveness)
      while( GG+dx[1] < SMALL ){
	dx[0] *= 0.5;
	dx[1] *= 0.5;
      }

      //-- check to see what the residual would be if we took this step
      //   and ensure that the residual will be decreased by taking this step.
      eval_err( mean, var, FF+dx[0], GG+dx[1], m1, v1 );

      tmp = max( abs(abs(m1)-abs(meanErr)), abs(abs(v1)-abs(varErr)) ) / absErr;
      if (tmp > 0.99999){
	// relaxation factor here is chosen based on trial and error
	dx[0] *= 0.8;
	dx[1] *= 0.8;
      }
      else
	break;
    }

    //-- update guesses for FF and GG  x_new = x + dx

    FF += RELAX_FACT * dx[0];
    GG += RELAX_FACT * dx[1];

    assert( GG >= 0.0 );
    nonLinIter_++;
    iterTally_++;

  }
  return prm;
}
//--------------------------------------------------------------------
void
ClipGaussParams::eval_err( const double mean, const double var,
			   const double FF,   const double GG,
			   double & f_err,    double & g_err )
{
  using std::sqrt;  using std::pow;
  using std::exp;

  const double SQRT2PI = sqrt(2.0*PI);
  const double sqrt2piGG = sqrt(2.0*PI*GG);
  const double invsqrt2piGG = 1.0/sqrt2piGG;
  //
  // alpha1, beta1, and gamma1 are symbolic evaluations of various integrals
  //
  assert( GG > 0.0 );
  const double alpha1 =
    SQRT2PI/2.0 * (1.0 - erf(0.5*sqrt(2.0/GG)*(1.0-FF)));
  const double beta1  =
    GG * (  exp(-FF*FF/(2.0*GG)) - exp(-pow(FF-1.0,2.0) / (2.0*GG)) );
  const double gamma1 = sqrt2piGG/2.0 *
    ( erf(0.5*sqrt(2.0/GG)*FF) - erf(0.5*sqrt(2.0/GG)*(FF-1.0)) );

  const double fnew =
    alpha1/SQRT2PI + invsqrt2piGG * (  beta1 + FF*gamma1 );

  const double gnew = -mean*mean + alpha1/SQRT2PI +
    invsqrt2piGG * ( -GG*exp(-pow(FF-1,2.0)/(2.0*GG)) +
		     FF*beta1 + FF*FF*gamma1 + GG*gamma1 );

  f_err = mean - fnew;
  g_err = var - gnew;
}
//--------------------------------------------------------------------
void
ClipGaussParams::dump_stats()
{
  std::cout << "Solved " << solverTally_ << " points using "
	    << iterTally_ << " iterations." << std::endl
	    << "average number of iterations per point: "
	    << iterTally_/(nMean_*nVariance_)
	    << std::endl;
}
//--------------------------------------------------------------------
void
ClipGaussParams::finalize_setup()
{
  // set up bracketers
  assert( NULL == meanBracketer_ );
  assert( NULL == varBracketer_ );
  meanBracketer_ = new BracketHelper<double>(0,meanPts_);
  varBracketer_  = new BracketHelper<double>(0,scaleVarPts_);

  haveTable_ = true;
}
//--------------------------------------------------------------------



//====================================================================
//====================================================================
//
//    TEST UTILITIES FOR <ClipGauss> and <ClipGaussParams> objects
//
//====================================================================
//====================================================================


void test_cg_table()
{
  using namespace std;

  ClipGaussParams cg( 51, 21 );
  cg.build_table();
  cg.write_to_disk( "cgtbl.dat" );
  cg.read_from_disk("cgtbl.dat");

  cout << endl << cg.get_div_count() << " points diverged!" << endl;
  /*
	double f, g;
	cout << "enter the mean:"; cin>>f;
	cout << "enter the variance (scaled): "; cin>>g;
	CGParams p = cg.lookup_scal( f,g );
	cout << "FF="<<p.FF << " GG=" << p.GG << endl;
  */

  ClipGaussParams cg2;
  cg2.read_from_disk( "cgtbl.dat" );
  cg2.write_to_disk( "cgtbl2.dat" );

  return;
}

/*
int main()
{
  test_cg();
//  test_cg_table();
}

*/
