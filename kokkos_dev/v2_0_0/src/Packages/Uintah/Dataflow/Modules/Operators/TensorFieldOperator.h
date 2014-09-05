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
#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>


/*************************************************************
 *  How to add a new function:
 *
 * 1. Add a new struct in TensorOperatorFunctors.h which has the desired
 *    operation in it.  Follow the template in the file.
 * 2. Add a new case statement in TensorFieldOperator::performOperation
 *    using the new functor that you created in #1.
 * 3. Add a corresponding case statement to TensorParticlesOperator.cc
 * 4. Edit ../../GUI/TensorOperator.tcl to include a new radiobutton for
 *    your new operator.  Make sure the value is the same as the one
 *    you selected for the case statement in #2.
 *    Be sure to change the name of the radiobutton to $w.calc.myoperation .
 *    and create a new method for your operation.  Add $w.calc.myoperation
 *    to the list of radiobuttons that are packed at the end.
 * 5. Add your new tcl method to the nested if statements following the
 *    radio buttons in the code.
 * 6. Add the body of the new method following the pattern of the others.
 * 7. Add a new ui method corresponding to your new function.  This is
 *    referenced in the method created in 6.  This will allow you to show
 *    in the user interface what operation you are calculating.
 */
namespace Uintah {
  using std::string;
  using std::cerr;
  using std::endl;
  using namespace SCIRun;
class TensorFieldOperator: public Module, public UnaryFieldOperator {
public:
  TensorFieldOperator(GuiContext* ctx);
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
  case 4: // Octahedral shear stress
    computeScalars(tensorField, scalarField, OctShearStressOp());
    break;
  default:
    std::cerr << "TensorFieldOperator::performOperation: "
	      << "Unexpected Operation Type #: " << guiOperation.get() << "\n";
  }
}

} // end namespace Uintah 

#endif // __DERIVE_TENSORFIELDOPERATOR_H__

