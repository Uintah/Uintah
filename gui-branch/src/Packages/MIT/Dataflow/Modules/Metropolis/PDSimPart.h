/*
 *  PDSimPart.h:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

#ifndef PDSimPart_h
#define PDSimPart_h

extern "C" {
#include <unuran.h>
};

#include <Core/Parts/Part.h>
#include <Core/Parts/PartInterface.h>

#include <Packages/MIT/Core/Datatypes/MetropolisData.h>


namespace MIT {

using namespace SCIRun;

class Sampler;
class PDSimPart : public Part, PartInterface {
protected:
  Sampler *sampler_;
  UNUR_GEN *gen_;
public:
  PDSimPart( Sampler *, PartInterface *parent, const string &name="PDSim");
  virtual ~PDSimPart();

  virtual void compute( vector<double> &, vector<double> & );
  virtual double lpr( vector<double> &);

};
  

} // End namespace MIT


#endif // PDSimPart_h


