/*
 *  MultivariateNormalDSimPart.h:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

#ifndef MultivariateNormalDSimPart_h
#define MultivariateNormalDSimPart_h

#include <Packages/MIT/Dataflow/Modules/Metropolis/PDSimPart.h>

namespace MIT {

using namespace SCIRun;

class Sampler;
class MultivariateNormalDSimPart : public PDSimPart {
public:
  MultivariateNormalDSimPart( Sampler *, PartInterface *parent, 
			      const string &name="MultivariateNormalDSim");
  virtual ~MultivariateNormalDSimPart();

  virtual void compute( vector<double> &, vector<double> & );
  virtual double lpr( vector<double> &);

};
  

} // End namespace MIT


#endif // MultivariateNormalDSimPart_h


