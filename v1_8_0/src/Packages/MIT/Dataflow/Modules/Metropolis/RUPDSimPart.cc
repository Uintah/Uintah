/*
 *  RUPDSimPart.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

#include <Core/Malloc/Allocator.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/Sampler.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/RUPDSimPart.h>

namespace MIT {

using namespace SCIRun;

RUPDSimPart::RUPDSimPart( Sampler *sampler, 
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

  par_.push_back(0.05);
  par_.push_back(0.02);
  par_.push_back(0.5);
  par_.push_back(0.2);
  par_.push_back(0.1);
}

RUPDSimPart::~RUPDSimPart()
{
}
  
void
RUPDSimPart::compute( vector<double> &theta, 
		      vector<double> &star )
{
  int n = theta.size();

  vector<double> r(n);

  for (int i=0; i<n; i++)
    r[i] = unur_sample_cont(gen_);
  
  for (int i=0; i<n; i++) 
    star[i] = (r[i]-0.5) * par_[i] + theta[i];
}

double
RUPDSimPart::lpr( vector<double> &theta )
{
  return 0;
}

void
RUPDSimPart::pars( const vector<double> &values )
{
  par_ = values;
}

void
RUPDSimPart::par( int i, double v ) 
{
  par_[i] = v;
}

} // End namespace MIT






