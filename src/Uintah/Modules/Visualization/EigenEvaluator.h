#ifndef __VISUALIZATION_EIGENEVALUATOR_H__
#define __VISUALIZATION_EIGENEVALUATOR_H__

#include <PSECore/Dataflow/Module.h>
#include <SCICore/Containers/String.h>
#include <Uintah/Datatypes/TensorFieldPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/VectorFieldPort.h>

namespace Uintah {
namespace Modules {
  using namespace SCICore::Containers;
  using namespace PSECore::Dataflow;
  using namespace SCICore::Datatypes;
  using namespace PSECore::Datatypes;

  class EigenEvaluator: public Module {
  public:
    EigenEvaluator(const clString& id);
    virtual ~EigenEvaluator() {}
    
    virtual void execute(void);
    
  private:
    //    TCLstring tcl_status;
    TCLint tclEigenSelect;
    TensorFieldIPort *in;

    ScalarFieldOPort *sfout; // for eigen values
    VectorFieldOPort *vfout; // for eigen vectors
  };
}
}
#endif // __VISUALIZATION_EIGENEVALUATOR_H__

