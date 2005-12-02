/*
 *  LikelihoodPart.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

extern "C" {
#include <cvode/llnltyps.h>
#include <cvode/cvode.h>
#include <cvode/cvdense.h>
#include <cvode/nvector.h>
#include <cvode/cvdense.h>
}


#include <math.h>

#include <Packages/MIT/Dataflow/Modules/Metropolis/Bayer.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/LikelihoodPart.h>
#include <Core/Containers/Array1.h>

namespace MIT {

using namespace SCIRun;

static double sqr( double x ) { return x*x; }

LikelihoodPart::LikelihoodPart( PartInterface *parent, const string &name )
  : Part( parent, name )
{
  bayer_ = new Bayer;
}

LikelihoodPart::~LikelihoodPart()
{
}
  

void
LikelihoodPart::measurements( MeasurementsHandle &handle)
{
  measurements_ = handle;
  neq_ = measurements_->t.size();
}
double
LikelihoodPart::compute( vector<double> &theta )
{
  Array1 <double> ym1(neq_);
  Array1 <double> ym2(neq_);
  double dt = exp(theta[4]);
  
  double reltol = 1e-6;   // should be input from the UI
  double abstol = 1e-6;
  double f = 0.1335;

  double p[6];

  p[0] = theta[0];
  p[1] = theta[1];
  p[2] = theta[2];
  p[3] = log(0.0249);
  p[4] = log(0.0842);
  p[5] = theta[3];


  bayer_->set_p( p );
    
  // get y0 
  N_Vector y = N_VNew( neq_, 0 );
  bayer_->get_y0( y );
  
  long int iopt[OPT_SIZE];
  
  for (int i=0; i<OPT_SIZE; i++) iopt[i] = 0;
  iopt[MXSTEP] = 2000;
  
  // init integrator
  void *cvode_mem = CVodeMalloc( neq_, bayer_->f(),     // size and function
				 0, y,                  // t0 and y
				 ADAMS, NEWTON,         // methods
				 SS, &reltol, &abstol,  // tolerances
				 bayer_,                // f_data
				 0,                     // err file
				 TRUE, iopt, 0, 0       // optIn, iopt, ropt 
				 ); 
  
  if ( !cvode_mem ) {
    cerr << "CVodeMalloc failed\n";
    return -1; // should exit!!
  }
  
  // select linear solver
  CVDense( cvode_mem, 0, 0 );  // use internal jacobian
  
  
  for (int i=0; i<neq_; i++) { 
    double tf = measurements_->t[i]+dt;
    double t;
    
    int status = CVode( cvode_mem, tf, y, &t, NORMAL );
    
    if ( status != SUCCESS ) {
      cerr << "CVode fail. status = " << status << endl;
      //Thread::niceAbort();
      return -1; // should exit or report error!!
    }
    
    ym1[i] = N_VIth(y,0);
    ym2[i] = N_VIth(y,neq_-2);
  }
  
  N_VFree( y );
  
  double l1 = 0.05*0.05;
  double l2 = 0.015*0.015;

  double logl1 = log(l1);
  double logl2 = log(l2);
  
  double sum = 0;
  for (int i=0; i< neq_; i++) 
    sum += sqr(measurements_->concentration(0,i) - ym1[i]/f);

  
  double like = -0.5*( 
		      7*logl1  +
		      sum/l1  +
		      3*logl2 +
		      (sqr( measurements_->concentration(1,0) - ym2[1])+
		       sqr( measurements_->concentration(1,1) - ym2[3])+
		       sqr( measurements_->concentration(1,2) - ym2[6])
		       ) / l2
		      );
  return like;
}

} // End namespace MIT


