#ifndef __VISUALIZATION_TENSORPARTICLESOPERATOR_H__
#define __VISUALIZATION_TENSORPARTICLESOPERATOR_H__

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

  class TensorParticlesOperator: public Module {
  public:
    TensorParticlesOperator(const clString& id);
    virtual ~TensorParticlesOperator() {}
    
    virtual void execute(void);
    
  private:
    //    TCLstring tcl_status;
    TCLint tclOperation;

    // element extractor operation
    TCLint tclRow;
    TCLint tclColumn;
    
    // eigen value/vector operation
    //TCLint tclEigenSelect;

    // eigen 2D operation
    TCLint tclPlaneSelect;
    TCLdouble tclDelta;
    TCLint tclEigen2DCalcType;
    
    TensorParticlesIPort *in;

    ScalarParticlesOPort *spout;
  };
}
}
#endif // __VISUALIZATION_TENSORPARTICLESOPERATOR_H__

