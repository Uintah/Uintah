/*
 *  Sampler.h:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep-2001
 *
 */

extern "C" {
#include <unuran.h>
};

#include <Dataflow/Network/Module.h>
#include <Core/Parts/Part.h>
#include <Core/Parts/PartManager.h>
#include <Core/Parts/GraphPart.h>
#include <Dataflow/Network/Module.h>

#include <Packages/MIT/Core/Datatypes/MetropolisData.h>
#include <Packages/MIT/Dataflow/Ports/MetropolisPorts.h>

namespace MIT {

using namespace SCIRun;

class SamplerInterface;
class PriorPart;
class LikelihoodPart;
class PDSimPart;

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
  PartManagerOf<PDSimPart> pdsim_;

  GraphPart *graph_;

  // control
  bool user_ready_;
  bool has_lkappa_;

  bool stop_;
  bool paused_;
  int current_iter_;

  int seed;
  double kappa;
  int iterations;

  int nparms;

  double post_;
  double lpr_;

  vector< vector<double> > theta;
  int old, star;

  Array2<double> lkappa;

  UNUR_GEN *gen;

public:
  Sampler(const string& id);

  virtual ~Sampler();

  virtual void execute();

  void init();
  void reset();
  void metropolis();
  Array2<double> &get_lkappa();
  double logpost( vector<double> & );
  void go( int );

  void tcl_command( TCLArgs &args, void *data);
};
  

} // End namespace MIT


