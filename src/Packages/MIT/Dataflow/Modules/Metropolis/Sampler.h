/*
 *  Sampler.h:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

extern "C" {
#include </usr/local/include/unuran.h>
};

#include <Dataflow/Network/Module.h>
#include <Core/Parts/Part.h>
#include <Dataflow/Network/Module.h>

#include <Packages/MIT/Core/Datatypes/MetropolisData.h>
#include <Packages/MIT/Dataflow/Ports/MetropolisPorts.h>

namespace MIT {

using namespace SCIRun;

class SamplerInterface;
class PriorPart;
class LikelihoodPart;

class Sampler : public Module, public Part {
private:
  // input ports
  DistributionIPort *d_port_;
  DistributionHandle distribution_;

  MeasurementsIPort *m_port_;
  MeasurementsHandle measurements_;

  // output ports
  ResultsOPort *r_port_;
  ResultsHandle results;

  // internal Parts
  SamplerInterface *interface_;

  PriorPart *prior_;
  LikelihoodPart *likelihood_;

  // control
  bool user_ready;

  int seed;
  double kappa;
  int burning;
  int monitor;
  int thin;
  int iterations;

  int nparms;

  double *theta;
  double *star;

  Array2<double> lkappa;

  UNUR_GEN *gen;

public:
  Sampler(const string& id);

  virtual ~Sampler();

  virtual void execute();

  void init();
  void reset();
  void metropolis();
  void pdsim( double [], double [] );
  double logpost( double [] );
  void go();

  void tcl_command( TCLArgs &args, void *data);
};
  

} // End namespace MIT


