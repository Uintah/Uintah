/*
 *  IUniformPDSimPart.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

#include <Packages/MIT/Dataflow/Modules/Metropolis/Sampler.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/IUniformPDSimPart.h>

namespace MIT {

using namespace SCIRun;

IUniformPDSimPart::IUniformPDSimPart( Sampler *sampler, 
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

  par_.resize(5);
  par_[0] = 0.1;
  par_[1] = 0.1;
  par_[2] = 1.0;
  par_[3] = 0.5;
  par_[4] = 0.5;

  center_.resize(5);
  center_[0] = -3.16;
  center_[1] = -3.74;
  center_[2] = -9.0;
  center_[3] = -6.0;
  center_[4] = -1.85;
}

IUniformPDSimPart::~IUniformPDSimPart()
{
}
  
void
IUniformPDSimPart::compute( vector<double> &, 
			    vector<double> &star )
{
  int n = star.size();

  vector<double> r(n);

  for (int i=0; i<n; i++)
    r[i] = unur_sample_cont(gen_);

  for (int i=0; i<n; i++) 
    star[i] = (r[i]-0.5) * par_[i] + center_[i];;
}

double
IUniformPDSimPart::lpr( vector<double> &)
{
  return 0;
}

void
IUniformPDSimPart::pars( const vector<double> &values )
{
  par_ = values;
}

void
IUniformPDSimPart::par( int i, double v ) 
{
  par_[i] = v;
}
void
IUniformPDSimPart::centers( const vector<double> &values )
{
  center_ = values;
}

void
IUniformPDSimPart::center( int i, double v ) 
{
  center_[i] = v;
}
} // End namespace MIT



