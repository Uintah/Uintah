/*
 *  IUPDSimPart.h:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

#ifndef IUPDSimPart_h
#define IUPDSimPart_h

#include <Packages/MIT/Dataflow/Modules/Metropolis/PDSimPart.h>

namespace MIT {

using namespace SCIRun;

class Sampler;
class RUPDSimPart : public PDSimPart {
private:
  vector<double> par_;

public:
  RUPDSimPart( Sampler *, PartInterface *parent, 
	       const string &name="IUPDSim");
  virtual ~RUPDSimPart();

  virtual void compute( vector<double> &, vector<double> & );
  virtual double lpr( vector<double> &);

  // get values
  const vector<double> &par() { return par_; }

  // set values
  void pars( const vector<double> &);
  void par( int, double );

  // Signals
  Signal1< const vector<double> &> par_changed;
};
  

} // End namespace MIT


#endif // IUPDSimPart_h


