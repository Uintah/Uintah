#ifndef __DERIVE_PARTICLEEIGENEVALUATOR_H__
#define __DERIVE_PARTICLEEIGENEVALUATOR_H__

#include <Dataflow/Network/Module.h>
#include <Uintah/Dataflow/Ports/TensorParticlesPort.h>
#include <Uintah/Dataflow/Ports/ScalarParticlesPort.h>
#include <Uintah/Dataflow/Ports/VectorParticlesPort.h>
#include <string>

using std::string;
using namespace SCIRun;

namespace Uintah {

  class ParticleEigenEvaluator: public Module {
  public:
    ParticleEigenEvaluator(const string& id);
    virtual ~ParticleEigenEvaluator() {}
    
    virtual void execute(void);
    
  private:
    //    TCLstring tcl_status;
    GuiInt guiEigenSelect;
    TensorParticlesIPort *in;

    ScalarParticlesOPort *spout; // for eigen values
    VectorParticlesOPort *vpout; // for eigen vectors
  };
}

#endif // __DERIVE_EIGENEVALUATOR_H__




