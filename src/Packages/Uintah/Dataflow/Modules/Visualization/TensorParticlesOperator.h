#ifndef __VISUALIZATION_TENSORPARTICLESOPERATOR_H__
#define __VISUALIZATION_TENSORPARTICLESOPERATOR_H__

#include <Core/TclInterface/TCLvar.h>
#include <Dataflow/Network/Module.h>
#include <Core/Containers/String.h>
#include <Packages/Uintah/Core/Datatypes/TensorParticlesPort.h>
#include <Packages/Uintah/Core/Datatypes/ScalarParticlesPort.h>

namespace Uintah {
using namespace SCIRun;

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
} // End namespace Uintah
  };
#endif // __VISUALIZATION_TENSORPARTICLESOPERATOR_H__

