/*
 *  ItPDSimPart.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

#include <Packages/MIT/Dataflow/Modules/Metropolis/Sampler.h>
#include <Packages/MIT/Dataflow/Modules/Metropolis/ItPDSimPart.h>

extern "C" {
  void dgemm_( char &, char &, 
	       int &, int &, int &, double &, 
	       double *, int &, double *, int &, 
	       double &, 
	       double *, int &);
  double genchi_( double & );
}
	       
namespace MIT {

using namespace SCIRun;

static double sqr( double x ) { return x*x; }

ItPDSimPart::ItPDSimPart( Sampler *sampler, 
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
  mean_.push_back(-7.07);
  mean_.push_back(-1.85);

  df_ = 5.0;
}

ItPDSimPart::~ItPDSimPart()
{
}
  
void
ItPDSimPart::compute( vector<double> &, 
		      vector<double> &star )
{
  int n = theta.size();

  vector<double> r(n);

  for (int i=0; i<n; i++)
    r[i] = unur_sample_cont(gen_);
  
  Array2<double> &lkappa = sampler_->get_lkappa();
  double *result = scinew double[n];

  char a='N';
  char b='T';
  int  cols=1;
  double alpha=1;
  double beta=0;
  
  dgemm( a, b, n, cols, n, alpha, 
	 *(lkappa.get_dataptr()), n, r, cols, 
	 beta, result, n );
	 
  double v = sqrt( df / genchi(df) );

  for (int i=0; i<n; i++) 
    star[i] += result[i]*v + mean_[i];
  
}

double
ItPDSimPart::lpr( vector<double> &)
{
  return 0;
}

} // End namespace MIT





