/*
 *  ItPDSimPart.h:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

#ifndef ItPDSimPart_h
#define ItPDSimPart_h

#include <Packages/MIT/Dataflow/Modules/Metropolis/PDSimPart.h>

namespace MIT {

using namespace SCIRun;

class Sampler;
class ItPDSimPart : public PDSimPart {
private:
  vector<double> mean_;
  float df_;

public:
  ItPDSimPart( Sampler *, PartInterface *parent, 
	       const string &name="ItPDSim");
  virtual ~ItPDSimPart();

  virtual void compute( vector<double> &, vector<double> & );
  virtual double lpr( vector<double> &);

  // get values
  float df() { return df_; }

  // set values
  void df( float v ) { df_ = v; }
};
  

} // End namespace MIT


#endif // ItPDSimPart_h


