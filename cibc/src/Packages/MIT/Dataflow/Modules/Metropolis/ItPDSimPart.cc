/*
 *  ItPDSimPart.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

#include <Core/Malloc/Allocator.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/Sampler.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/ItPDSimPart.h>

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

ItPDSimPart::ItPDSimPart( Sampler *sampler, 
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

  mean_.push_back(-3.14);
  mean_.push_back(-3.73);
  mean_.push_back(-8.82);
  mean_.push_back(-6.07);
  mean_.push_back( 1.85);

  df_ = 5.0;
}

ItPDSimPart::~ItPDSimPart()
{
}
  
void
ItPDSimPart::compute( vector<double> &, 
		      vector<double> &star )
{
  int n = star.size();

  vector<double> r(n);

  for (int i=0; i<n; i++)
    r[i] = unur_sample_cont(gen_);
  
  Array2<double> &lkappa = sampler_->get_lkappa();
    
  double v = sqrt( df_ / genchi_(df_) );
  for (int i=0; i<n; i++) {
    double mult = 0;
    for (int j=0; j<n; j++)
      mult += lkappa(j,i)*r[j]; 
    star[i] = mult*v + mean_[i];
  }
}

double
ItPDSimPart::lpr( vector<double> &theta )
{
  int n = theta.size();
  Array2<double> &lkappa = sampler_->get_lkappa();
  double value;
  mvtpdf_( &theta[0], &mean_[0], *(lkappa.get_dataptr()), 
	   df_, value, n );
  return value;
}

void
ItPDSimPart::means( const vector<double> &values )
{
  mean_ = values;
}

void
ItPDSimPart::mean( int i, double v ) 
{
  mean_[i] = v;
}

} // End namespace MIT





