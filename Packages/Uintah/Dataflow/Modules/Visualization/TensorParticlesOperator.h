#ifndef __VISUALIZATION_TENSORPARTICLESOPERATOR_H__
#define __VISUALIZATION_TENSORPARTICLESOPERATOR_H__

#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Module.h>
#include <Core/Containers/String.h>
#include <Packages/Uintah/Core/Datatypes/TensorParticlesPort.h>
#include <Packages/Uintah/Core/Datatypes/ScalarParticlesPort.h>

namespace Uintah {

using namespace SCIRun;

  class TensorParticlesOperator: public Module {
  public:
    TensorParticlesOperator(const string& id);
    virtual ~TensorParticlesOperator() {}
    
    virtual void execute(void);
    
  private:
    //    GuiString tcl_status;
    GuiInt tclOperation;

    // element extractor operation
    GuiInt tclRow;
    GuiInt tclColumn;
    
    // eigen value/vector operation
    //GuiInt tclEigenSelect;

    // eigen 2D operation
    GuiInt tclPlaneSelect;
    GuiDouble tclDelta;
    GuiInt tclEigen2DCalcType;
    
    TensorParticlesIPort *in;

    ScalarParticlesOPort *spout;
  };

} // End namespace Uintah

#endif // __VISUALIZATION_TENSORPARTICLESOPERATOR_H__

