#ifndef __VISUALIZATION_INPLANEEIGENEVALUATOR_H__
#define __VISUALIZATION_INPLANEEIGENEVALUATOR_H__

#include <SCICore/TclInterface/TCLvar.h>
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
  using namespace SCICore::TclInterface;

  class InPlaneEigenEvaluator: public Module {
  public:
    InPlaneEigenEvaluator(const clString& id);
    virtual ~InPlaneEigenEvaluator() {}
    
    virtual void execute(void);
    
  private:
    //    TCLstring tcl_status;
    TCLint tclPlaneSelect;
    TCLint tclCalculationType;
    TCLdouble tclDelta;
    
    TensorFieldIPort *in;

    ScalarFieldOPort *sfout; // for eigen value data
  };
}
}
#endif // __VISUALIZATION_INPLANEEIGENEVALUATOR_H__

