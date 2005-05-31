/*
 *  PriorPart.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

#include <Packages/MIT/Dataflow/Modules/Metropolis/PriorPart.h>

namespace MIT {

using namespace SCIRun;

static double sqr( double x ) { return x*x; }

PriorPart::PriorPart( PartInterface *parent, const string &name )
  : Part( parent, name )
{
  mu0 = -5;
  sigma_mu = 2;

  eta0 = -5;
  sigma_eta = 2;

  vp1 = -20;
  vp2 = 10;
  vp3 = 1.5;

  sp1 = -5;
  sp2 = 2;
  sp3 = 2;
}

PriorPart::~PriorPart()
{
}
  

double
PriorPart::compute( vector<double> &theta )
{
  double mu = theta[0];
  double eta = theta[1];
  double l1 = theta[2];
  double l2 = theta[3];
  double l3 = theta[4];

  double prior = -0.5*( (sqr((mu-mu0)/sigma_mu)) +
			(sqr((eta-eta0)/sigma_eta)) +
			(sqr((vp1-l1)/sp1)) +
			(sqr((vp2-l2)/sp2)) +
			(sqr((vp3-l3)/sp3)));
  return prior;
}

} // End namespace MIT


