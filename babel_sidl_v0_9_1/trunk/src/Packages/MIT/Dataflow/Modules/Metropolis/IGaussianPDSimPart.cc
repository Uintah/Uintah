/*
 *  IGaussianPDSimPart.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

#include <Core/Malloc/Allocator.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/Sampler.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/IGaussianPDSimPart.h>

extern "C" {
  void dgemm_( char &, char &, 
	       int &, int &, int &, double &, 
	       double *, int &, double *, int &, 
	       double &, 
	       double *, int &);
  float genchi_( float & );
  void mnormpdf_( double *, double *, double *, double &, int &);
}

namespace MIT {

using namespace SCIRun;

IGaussianPDSimPart::IGaussianPDSimPart( Sampler *sampler, 
					PartInterface *parent,
					const string &name )
  :PDSimPart( sampler, parent, name )
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

}

IGaussianPDSimPart::~IGaussianPDSimPart()
{
}
  
void
IGaussianPDSimPart::compute( vector<double> &theta, 
			     vector<double> &star )
{
  int n = star.size();

  vector<double> r(n);

  for (int i=0; i<n; i++)
    r[i] = unur_sample_cont(gen_);
  
  Array2<double> &lkappa = sampler_->get_lkappa();
	 
  for (int i=0; i<n; i++) {
    double result = 0;
    for (int j=0; j<n; j++)
      result += lkappa(j,i)*r[j]; 
    star[i] = theta[i] + result;
  }

}

double
IGaussianPDSimPart::lpr( vector<double> &theta)
{
  int n = theta.size();
  Array2<double> &lkappa = sampler_->get_lkappa();
  double value;
  mnormpdf_( &theta[0], &mean_[0], *(lkappa.get_dataptr()), value, n );
  return value;
}

void
IGaussianPDSimPart::means( const vector<double> &values )
{
  mean_ = values;
}

void
IGaussianPDSimPart::mean( int i, double v ) 
{
  mean_[i] = v;
}

} // End namespace MIT




