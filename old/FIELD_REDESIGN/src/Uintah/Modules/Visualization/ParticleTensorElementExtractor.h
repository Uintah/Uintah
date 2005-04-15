#ifndef __VISUALIZATION_PARTICLE_TENSORELEMENTEXTRACTOR_H__
#define __VISUALIZATION_PARTICLE_TENSORELEMENTEXTRACTOR_H__

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

  class ParticleTensorElementExtractor: public Module {
  public:
    ParticleTensorElementExtractor(const clString& id);
    virtual ~ParticleTensorElementExtractor() {}
    
    virtual void execute(void);
    
  private:
    //    TCLstring tcl_status;
    TCLint tclRow;
    TCLint tclColumn;
    TensorParticlesIPort *in;

    ScalarParticlesOPort *spout; // output elements
  };
}
}
#endif // __VISUALIZATION_PARTICLE_TENSORELEMENTEXTRACTOR_H__

