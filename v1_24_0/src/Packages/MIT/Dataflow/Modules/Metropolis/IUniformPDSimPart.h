/*
 *  IUniformPDSimPart.h:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

#ifndef IUniformPDSimPart_h
#define IUniformPDSimPart_h

#include <Packages/MIT/Dataflow/Modules/Metropolis/PDSimPart.h>

namespace MIT {

using namespace SCIRun;

class Sampler;

class IUniformPDSimPart : public PDSimPart {
protected:
  vector<double> par_;
  vector<double> center_;

public:
  IUniformPDSimPart( Sampler *, PartInterface *parent, 
			      const string &name="IUniformPDSim");
  virtual ~IUniformPDSimPart();

  virtual void compute( vector<double> &, vector<double> & );
  virtual double lpr( vector<double> &);

  // get values
  const vector<double> &par() { return par_; }
  const vector<double> &center() { return center_; }

  // set values
  void pars( const vector<double> &);
  void par( int, double );
  void centers( const vector<double> &);
  void center( int, double );

  // Signals
  Signal1< const vector<double> &> par_changed;
  Signal1< const vector<double> &> center_changed;
};
  

} // End namespace MIT


#endif // IUniformPDSimPart_h


