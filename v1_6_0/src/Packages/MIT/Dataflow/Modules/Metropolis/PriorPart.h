/*
 *  PriorPart.h:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

#ifndef PriorPart_h
#define PriorPart_h

#include <Core/Parts/Part.h>

#include <Packages/MIT/Core/Datatypes/MetropolisData.h>

namespace MIT {

using namespace SCIRun;

class PriorPart : public Part {
private:
  double mu0, eta0;
  double sigma_mu, sigma_eta;
  double vp1, vp2, vp3, sp1, sp2, sp3;

public:
  PriorPart( PartInterface *parent, const string &name="Prior");
  virtual ~PriorPart();

  double compute( vector<double> & );
};
  

} // End namespace MIT


#endif // PriorPart_h


