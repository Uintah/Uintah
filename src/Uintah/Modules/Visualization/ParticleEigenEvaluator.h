#ifndef __VISUALIZATION_PARTICLEEIGENEVALUATOR_H__
#define __VISUALIZATION_PARTICLEEIGENEVALUATOR_H__

#include <PSECore/Dataflow/Module.h>
#include <SCICore/Containers/String.h>
#include <Uintah/Datatypes/TensorParticlesPort.h>
#include <Uintah/Datatypes/ScalarParticlesPort.h>
#include <Uintah/Datatypes/VectorParticlesPort.h>

namespace Uintah {
namespace Modules {
  using namespace SCICore::Containers;
  using namespace PSECore::Dataflow;
  using namespace SCICore::Datatypes;
  using namespace PSECore::Datatypes;

  class ParticleEigenEvaluator: public Module {
  public:
    ParticleEigenEvaluator(const clString& id);
    virtual ~ParticleEigenEvaluator() {}
    
    virtual void execute(void);
    
  private:
    //    TCLstring tcl_status;
    TCLint tclEigenSelect;
    TensorParticlesIPort *in;

    ScalarParticlesOPort *spout; // for eigen values
    VectorParticlesOPort *vpout; // for eigen vectors
  };
}
}
#endif // __VISUALIZATION_EIGENEVALUATOR_H__




