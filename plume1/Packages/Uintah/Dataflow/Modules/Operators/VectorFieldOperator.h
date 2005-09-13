#ifndef __OPERATORS_VECTORFIELDOPERATOR_H__
#define __OPERATORS_VECTORFIELDOPERATOR_H__

#include "VectorOperatorFunctors.h"
#include "UnaryFieldOperator.h"
#include "OperatorThread.h"
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Module.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Mutex.h>
#include <Dataflow/Ports/FieldPort.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>


namespace Uintah {
using std::string;
using std::cerr;
using std::endl;
using namespace SCIRun;

  class VectorFieldOperator: public Module, public UnaryFieldOperator {
  public:
    VectorFieldOperator(GuiContext* ctx);
    virtual ~VectorFieldOperator() {}
    
    virtual void execute(void);
    
  private:
    template<class VectorField, class ScalarField>
     void performOperation(VectorField* vectorField, ScalarField* scalarField);

    //    TCLstring tcl_status;
    GuiInt guiOperation;

    FieldIPort *in;

    FieldOPort *sfout;
    //VectorFieldOPort *vfout;
  };

template<class VectorField, class ScalarField>
void VectorFieldOperator::performOperation(VectorField* vectorField,
					   ScalarField* scalarField)
{
  initField(vectorField, scalarField);

  switch(guiOperation.get()) {
  case 0: // extract element 1
  case 1: // extract element 2
  case 2: // extract element 3
    computeScalars(vectorField, scalarField,
		   VectorElementExtractionOp(guiOperation.get()));
    break;
  case 3: // Vector length
    computeScalars(vectorField, scalarField, LengthOp());
    break;
  case 4: // Vector curvature
    computeScalars(vectorField, scalarField, VorticityOp());
    break;
  default:
    std::cerr << "VectorFieldOperator::performOperation: "
	      << "Unexpected Operation Type #: " << guiOperation.get() << "\n";
  }
}

}

#endif // __OPERATORS_VECTORFIELDOPERATOR_H__

