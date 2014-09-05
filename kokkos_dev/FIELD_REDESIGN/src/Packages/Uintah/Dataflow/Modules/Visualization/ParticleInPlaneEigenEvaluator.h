#ifndef __VISUALIZATION_PARTICLEINPLANEEIGENEVALUATOR_H__
#define __VISUALIZATION_PARTICLEINPLANEEIGENEVALUATOR_H__

#include <SCICore/TclInterface/TCLvar.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Containers/String.h>
#include <Uintah/Datatypes/TensorParticlesPort.h>
#include <Uintah/Datatypes/ScalarParticlesPort.h>

namespace Uintah {
namespace Modules {
  using namespace SCICore::Containers;
  using namespace PSECore::Dataflow;
  using namespace SCICore::Datatypes;
  using namespace PSECore::Datatypes;
  using namespace SCICore::TclInterface;

  class ParticleInPlaneEigenEvaluator: public Module {
  public:
    ParticleInPlaneEigenEvaluator(const clString& id);
    virtual ~ParticleInPlaneEigenEvaluator() {}
    
    virtual void execute(void);
    
  private:
    //    TCLstring tcl_status;
    TCLint tclPlaneSelect;
    TCLint tclCalculationType;
    TCLdouble tclDelta;
    
    TensorParticlesIPort *in;

    ScalarParticlesOPort *spout; // for eigen value data
  };
}
}
#endif // __VISUALIZATION_PARTICLEINPLANEEIGENEVALUATOR_H__

