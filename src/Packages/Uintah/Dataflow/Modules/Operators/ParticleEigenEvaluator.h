#ifndef __DERIVE_PARTICLEEIGENEVALUATOR_H__
#define __DERIVE_PARTICLEEIGENEVALUATOR_H__

#include <Dataflow/Network/Module.h>
#include <Uintah/Core/Datatypes/TensorParticlesPort.h>
#include <Uintah/Core/Datatypes/ScalarParticlesPort.h>
#include <Uintah/Core/Datatypes/VectorParticlesPort.h>
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




