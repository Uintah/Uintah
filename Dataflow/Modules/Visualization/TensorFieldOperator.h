#ifndef __VISUALIZATION_TENSORFIELDOPERATOR_H__
#define __VISUALIZATION_TENSORFIELDOPERATOR_H__

#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Module.h>
#include <Core/Containers/String.h>
#include <Packages/Uintah/Core/Datatypes/TensorFieldPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Dataflow/Ports/VectorFieldPort.h>

namespace Uintah {

using namespace SCIRun;

  class TensorFieldOperator: public Module {
  public:
    TensorFieldOperator(const clString& id);
    virtual ~TensorFieldOperator() {}
    
    virtual void execute(void);
    
  private:
    template<class TensorField, class ScalarField>
    void performOperation(TensorField* tensorField, ScalarField* scalarField);

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
    
    TensorFieldIPort *in;

    ScalarFieldOPort *sfout;
    //VectorFieldOPort *vfout;
  };

} // End namespace Uintah

#endif // __VISUALIZATION_TENSORFIELDOPERATOR_H__

