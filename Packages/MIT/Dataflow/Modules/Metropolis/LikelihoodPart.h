/*
 *  LikelihoodPart.h:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

#ifndef LikelihoodPart_h
#define LikelihoodPart_h

#include <Core/Parts/Part.h>

#include <Packages/MIT/Core/Datatypes/MetropolisData.h>
#include <Packages/MIT/Dataflow/Ports/MetropolisPorts.h>

class Bayer;

namespace MIT {

using namespace SCIRun;


class  LikelihoodPart : public Part {
private:
  MeasurementsHandle measurements_;
  Bayer *bayer_;

  int neq_;

public:
  LikelihoodPart( PartInterface *parent, const string &name="Prior");
  virtual ~LikelihoodPart();

  void measurements( MeasurementsHandle & );
  double compute( vector<double> & );
};
  

} // End namespace MIT

#endif // LikelihoodPart_h




