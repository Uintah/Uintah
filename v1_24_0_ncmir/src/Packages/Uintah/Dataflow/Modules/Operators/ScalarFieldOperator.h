#ifndef __OPERATORS_SCALARFIELDOPERATOR_H__
#define __OPERATORS_SCALARFIELDOPERATOR_H__

#include "ScalarOperatorFunctors.h"
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

  class ScalarFieldOperator: public Module, public UnaryFieldOperator {
  public:
    ScalarFieldOperator(GuiContext* ctx);
    virtual ~ScalarFieldOperator() {}
    
    virtual void execute(void);
    
  protected:
    template<class Field, class ScalarField>      
     void performOperation(Field* field,
			   ScalarField* scalarField);
    
  private:
    //    TCLstring tcl_status;
    GuiInt guiOperation;

    FieldIPort *in;

    FieldOPort *sfout;
    //ScalarFieldOPort *vfout;
  };

template<class Field, class ScalarField>
void ScalarFieldOperator::performOperation(Field* field,
					   ScalarField* scalarField)
{
  initField(field, scalarField);

  switch(guiOperation.get()) {
  case 0: // extract element 1
    computeScalars(field, scalarField,
		   NaturalLogOp());
    break;
  case 1:
    computeScalars(field, scalarField,
		   ExponentialOp());
    break;
  case 20:
    computeScalars(field, scalarField,
		   NoOp());
    break;
  default:
    std::cerr << "ScalarFieldOperator::performOperation: "
	      << "Unexpected Operation Type #: " << guiOperation.get() << "\n";
  }
}

}

#endif // __OPERATORS_SCALARFIELDOPERATOR_H__

