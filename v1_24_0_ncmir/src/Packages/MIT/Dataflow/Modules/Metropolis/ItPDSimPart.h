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
  const vector<double> &mean() { return mean_; }

  // set values
  void df( float v ) { df_ = v; df_changed(df_); }
  void means( const vector<double> &);
  void mean( int, double );

  // Signals
  Signal1<float> df_changed;
  Signal1< const vector<double> &> mean_changed;
};
  

} // End namespace MIT


#endif // ItPDSimPart_h


