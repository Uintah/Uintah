/*
 *  RtPDSimPart.h:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

#ifndef RtPDSimPart_h
#define RtPDSimPart_h

#include <Packages/MIT/Dataflow/Modules/Metropolis/PDSimPart.h>

namespace MIT {

using namespace SCIRun;

class Sampler;

class RtPDSimPart : public PDSimPart {
private:
  float df_;

public:
  RtPDSimPart( Sampler *, PartInterface *parent, 
	       const string &name="RtPDSim");
  virtual ~RtPDSimPart();

  virtual void compute( vector<double> &, vector<double> & );
  virtual double lpr( vector<double> &);

  // get values
  float df() { return df_; }

  // set values
  void df( float v ) { df_ = v; df_changed(df_); }

  // Signals
  Signal1<float> df_changed;
};
  

} // End namespace MIT


#endif // RtPDSimPart_h


