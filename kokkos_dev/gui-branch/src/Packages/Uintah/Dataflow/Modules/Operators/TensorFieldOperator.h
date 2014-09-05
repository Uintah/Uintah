#ifndef __DERIVE_TENSORFIELDOPERATOR_H__
#define __DERIVE_TENSORFIELDOPERATOR_H__

#include "TensorOperatorFunctors.h"
#include "UnaryFieldOperator.h"
#include "OperatorThread.h"
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Packages/Uintah/Core/Grid/ShareAssignArray3.h>
#include <string>
#include <iostream>
using std::string;
using std::cerr;
using std::endl;

namespace Uintah {
using namespace SCIRun;
class TensorFieldOperator: public Module, public UnaryFieldOperator {
public:
  TensorFieldOperator(const string& id);
  virtual ~TensorFieldOperator() {}
    
  virtual void execute(void);
    
private:
  template<class TensorField, class ScalarField>
    void performOperation(TensorField* tensorField, ScalarField* scalarField);

  //    TCLstring tcl_status;
  GuiInt guiOperation;

  // element extractor operation
  GuiInt guiRow;
  GuiInt guiColumn;
    
    // eigen value/vector operation
    //GuiInt guiEigenSelect;

    // eigen 2D operation
  GuiInt guiPlaneSelect;
  GuiDouble guiDelta;
  GuiInt guiEigen2DCalcType;
    
  FieldIPort *in;

  FieldOPort *sfout;
  //VectorFieldOPort *vfout;
    
};

template<class TensorField, class ScalarField>
void TensorFieldOperator::performOperation(TensorField* tensorField,
					   ScalarField* scalarField)
{
  initField(tensorField, scalarField);

  switch(guiOperation.get()) {
  case 0: // extract element
    computeScalars(tensorField, scalarField,
		   TensorElementExtractionOp(guiRow.get(), guiColumn.get()));
    break;
  case 1: // 2D eigen-value/vector
    if (guiEigen2DCalcType.get() == 0) {
      // e1 - e2
      int plane = guiPlaneSelect.get();
      if (plane == 2)
	computeScalars(tensorField, scalarField, Eigen2DXYOp());
      else if (plane == 1)
	computeScalars(tensorField, scalarField, Eigen2DXZOp());
      else
	computeScalars(tensorField, scalarField, Eigen2DYZOp());
    }
    else {
      // cos(e1 - e2) / delta
      int plane = guiPlaneSelect.get();
      double delta = guiDelta.get();
      if (plane == 2)
	computeScalars(tensorField, scalarField, Eigen2DXYCosOp(delta));
      else if (plane == 1)
	computeScalars(tensorField, scalarField, Eigen2DXZCosOp(delta));
      else
	computeScalars(tensorField, scalarField, Eigen2DYZCosOp(delta));
    }
    break;
  case 2: // pressure
    computeScalars(tensorField, scalarField, PressureOp());
    break;
  case 3: // equivalent stress 
    computeScalars(tensorField, scalarField, EquivalentStressOp());
    break;
  default:
    std::cerr << "TensorFieldOperator::performOperation: "
	      << "Unexpected Operation Type #: " << guiOperation.get() << "\n";
  }
}

} // end namespace Uintah 

#endif // __DERIVE_TENSORFIELDOPERATOR_H__

