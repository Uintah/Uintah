#ifndef __VISUALIZATION_TENSORFIELDOPERATOR_H__
#define __VISUALIZATION_TENSORFIELDOPERATOR_H__

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
}
}
#endif // __VISUALIZATION_TENSORFIELDOPERATOR_H__

