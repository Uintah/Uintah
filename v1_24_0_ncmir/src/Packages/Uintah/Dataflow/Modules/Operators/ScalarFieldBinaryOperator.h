#ifndef __Uintah_Package_Dataflow_Modules_Operators_ScalarBinaryOperator_H__
#define __Uintah_Package_Dataflow_Modules_Operators_ScalarBinaryOperator_H__

/**************************************
CLASS
  ScalarFieldBinaryOperator
      Operator for binary operations on scalar fields


GENERAL INFORMATION

  ScalarFieldBinaryOperator
  
  Author:  James Bigler (bigler@cs.utah.edu)
           
           Department of Computer Science
           
           University of Utah
  
  Date:    April 2002
  
  C-SAFE
  
  Copyright <C> 2002 SCI Group

KEYWORDS
  Operator, Binary, scalar_field

DESCRIPTION
  Module to help apply binary operations to scalar fields.

WARNING
  None



****************************************/

#include "ScalarOperatorFunctors.h"
#include "BinaryFieldOperator.h"
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

  class ScalarFieldBinaryOperator: public Module, public BinaryFieldOperator {
  public:
    ScalarFieldBinaryOperator(GuiContext* ctx);
    virtual ~ScalarFieldBinaryOperator() {}
    
    virtual void execute(void);
    
  protected:
    template<class FieldLeft, class FieldRight, class ScalarField>      
    void performOperation(FieldLeft* left_field, FieldRight *right_field,
			  ScalarField* scalarField);
    
  private:
    //    TCLstring tcl_status;
    GuiInt guiOperation;

    FieldIPort *in_left;
    FieldIPort *in_right;

    FieldOPort *sfout;
    //ScalarFieldOPort *vfout;
  };

template<class FieldLeft, class FieldRight, class ScalarField>
void ScalarFieldBinaryOperator::performOperation(FieldLeft* left_field,
					   FieldRight *right_field,
					   ScalarField* scalarField)
{
  //  cout << "ScalarFieldBinaryOperator::performOperation:start\n";
  bool successful = initField(left_field, right_field, scalarField);
  if (!successful) {
    std::cerr << "ScalarFieldBinaryOperator::performOperation: Error - initField was not successful\n";
    return;
  }

  //  cout << "ScalarFieldBinaryOperator::performOperation:fields have been initialized.\n";
  
  switch(guiOperation.get()) {
  case 0: // Add
    computeScalars(left_field, right_field, scalarField,
		   AddOp());
    break;
  case 1: // Subtract
    computeScalars(left_field, right_field, scalarField,
		   SubOp());
    break;
  case 2: // Multiplication
    computeScalars(left_field, right_field, scalarField,
		   MultOp());
    break;
  case 3: // Division
    computeScalars(left_field, right_field, scalarField,
		   DivOp());
    break;
  case 4: // Average
    computeScalars(left_field, right_field, scalarField,
		   AverageOp());
    break;
  default:
    std::cerr << "ScalarFieldBinaryOperator::performOperation: "
	      << "Unexpected Operation Type #: " << guiOperation.get() << "\n";
  }
  //  cout << "ScalarFieldBinaryOperator::performOperation:end\n";
}

}

#endif // __OPERATORS_SCALARFIELDOPERATOR_H__

