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

};
  

} // End namespace MIT


#endif // IUniformPDSimPart_h


