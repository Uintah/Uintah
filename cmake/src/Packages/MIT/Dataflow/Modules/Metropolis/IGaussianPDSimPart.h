/*
 *  IGaussianPDSimPart.h:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

#ifndef IGaussianPDSimPart_h
#define IGaussianPDSimPart_h

#include <Packages/MIT/Dataflow/Modules/Metropolis/PDSimPart.h>

namespace MIT {

using namespace SCIRun;

class Sampler;
class IGaussianPDSimPart : public PDSimPart {
private:
  vector<double> mean_;

public:
  IGaussianPDSimPart( Sampler *, PartInterface *parent, 
			      const string &name="IGaussianPDSim");
  virtual ~IGaussianPDSimPart();

  virtual void compute( vector<double> &, vector<double> & );
  virtual double lpr( vector<double> &);

  const vector<double> &mean() { return mean_; }

  // set values
  void means( const vector<double> &);
  void mean( int, double );

  // Signals
  Signal1< const vector<double> &> mean_changed;
};
  

} // End namespace MIT


#endif // IGaussianPDSimPart_h


