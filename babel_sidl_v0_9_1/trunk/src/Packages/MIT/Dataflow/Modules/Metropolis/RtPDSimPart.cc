/*
 *  RtPDSimPart.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

#include <Core/Malloc/Allocator.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/Sampler.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/RtPDSimPart.h>

extern "C" {
  void dgemm_( char &, char &, 
	       int &, int &, int &, double &, 
	       double *, int &, double *, int &, 
	       double &, 
	       double *, int &);
  float genchi_( float & );
  void mvtpdf_( double *, double *, double *, float &, double &, int &);
}
	       
namespace MIT {

using namespace SCIRun;

RtPDSimPart::RtPDSimPart( Sampler *sampler, 
			  PartInterface *parent,
			  const string &name )
  : PDSimPart( sampler, parent, name )
{
  UNUR_DISTR *distr = unur_distr_normal(0,0);
  UNUR_PAR *par = unur_arou_new(distr);
  gen_ = unur_init(par);
  if ( ! gen_ ) 
    cerr << "Error, cannot create generator object\n";
  unur_distr_free(distr);

  df_ = 4.0;
}

RtPDSimPart::~RtPDSimPart()
{
}
  
void
RtPDSimPart::compute( vector<double> &theta, 
		      vector<double> &star )
{
  int n = theta.size();

  vector<double> r(n);

  for (int i=0; i<n; i++)
    r[i] = unur_sample_cont(gen_);
  
  Array2<double> &lkappa = sampler_->get_lkappa();
    
  double v = sqrt( df_ / genchi_(df_) );
  for (int i=0; i<n; i++) {
    double mult = 0;
    for (int j=0; j<n; j++)
      mult += lkappa(j,i)*r[j]; 
    star[i] = mult*v + theta[i];
  }
}

double
RtPDSimPart::lpr( vector<double> & )
{
  return 0;
}


} // End namespace MIT





