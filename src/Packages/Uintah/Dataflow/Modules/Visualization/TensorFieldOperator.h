#ifndef __VISUALIZATION_TENSORFIELDOPERATOR_H__
#define __VISUALIZATION_TENSORFIELDOPERATOR_H__

#include <Core/TclInterface/TCLvar.h>
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
    
    TensorFieldIPort *in;

    ScalarFieldOPort *sfout;
    //VectorFieldOPort *vfout;
  };

} // End namespace Uintah

#endif // __VISUALIZATION_TENSORFIELDOPERATOR_H__

