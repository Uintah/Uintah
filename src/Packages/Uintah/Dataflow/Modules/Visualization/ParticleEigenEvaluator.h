#ifndef __VISUALIZATION_PARTICLEEIGENEVALUATOR_H__
#define __VISUALIZATION_PARTICLEEIGENEVALUATOR_H__

#include <Dataflow/Network/Module.h>
#include <Core/Containers/String.h>
#include <Packages/Uintah/Core/Datatypes/TensorParticlesPort.h>
#include <Packages/Uintah/Core/Datatypes/ScalarParticlesPort.h>
#include <Packages/Uintah/Core/Datatypes/VectorParticlesPort.h>

namespace Uintah {
using namespace SCIRun;

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
} // End namespace Uintah
  };
#endif // __VISUALIZATION_EIGENEVALUATOR_H__




