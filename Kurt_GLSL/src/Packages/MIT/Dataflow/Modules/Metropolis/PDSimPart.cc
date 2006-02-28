/*
 *  PDSimPart.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

#include <Packages/MIT/Dataflow/Modules/Metropolis/Sampler.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/PDSimPart.h>

namespace MIT {

using namespace SCIRun;

PDSimPart::PDSimPart( Sampler *sampler, 
		      PartInterface *parent, 
		      const string &name )
  : Part( parent, name, this ), 
    PartInterface( this, parent, name, false ),
    sampler_(sampler)
{
  UNUR_DISTR *distr = unur_distr_normal(0,0);
  UNUR_PAR *par = unur_arou_new(distr);
  gen_ = unur_init(par);
  if ( ! gen_ ) 
    cerr << "Error, cannot create generator object\n";
  unur_distr_free(distr);
}

PDSimPart::~PDSimPart()
{
}
  
void
PDSimPart::compute( vector<double> &theta, vector<double> &star )
{
  int n = theta.size();

  vector<double> r(n);

  for (int i=0; i<n; i++)
    r[i] = unur_sample_cont(gen_);

  Array2<double> &lkappa = sampler_->get_lkappa();

  for (int i=0; i<n; i++) {
    star[i] = theta[i];
    for (int j=0; j<n; j++ )
      star[i] += lkappa(i,j) * r[j];
  }
}

double
PDSimPart::lpr( vector<double> &)
{
  return 0;
}

} // End namespace MIT


