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
#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Geometry/IntVector.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using std::string;
using std::cerr;
using std::endl;
using namespace SCIRun;

class ScalarFieldBinaryOperatorAlgo:  public DynamicAlgoBase, public BinaryFieldOperator 
{
public:
  virtual FieldHandle execute( FieldHandle left,
                               FieldHandle right, GuiInt op) = 0;
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const SCIRun::TypeDescription *ftd);
                               
protected:
    template<class FieldLeft, class FieldRight, class ScalarField>      
    void performOperation(FieldLeft* left_field, FieldRight *right_field,
			  ScalarField* scalarField, GuiInt op );
};

template< class IFIELD >
class ScalarFieldBinaryOperatorAlgoT : public ScalarFieldBinaryOperatorAlgo
{
  virtual FieldHandle execute( FieldHandle left,
                               FieldHandle right, GuiInt op);
};

template< class IFIELD >
FieldHandle
ScalarFieldBinaryOperatorAlgoT<IFIELD>::execute( FieldHandle left,
                                                 FieldHandle right,
                                                 GuiInt op)
{
  IFIELD *lf = (IFIELD*)(left.get_rep());
  IFIELD *rf = (IFIELD*)(right.get_rep());
  typename IFIELD::mesh_handle_type mh = left->get_typed_mesh();
  mh.detach();
  typename IFIELD::mesh_type *mesh = mh.get_rep();

  IFIELD *sf = scinew IFIELD( mesh );
  peformOperation( lf, rf, sf, op ); 
  return sf;
}


template<class FieldLeft, class FieldRight, class ScalarField>
void 
ScalarFieldBinaryOperatorAlgo::performOperation(FieldLeft* left_field,
					   FieldRight *right_field,
					   ScalarField* scalarField,
                                           GuiInt op)
{
  //  cout << "ScalarFieldBinaryOperator::performOperation:start\n";
  bool successful = initField(left_field, right_field, scalarField);
  if (!successful) {
    std::cerr << "ScalarFieldBinaryOperator::performOperation: Error - initField was not successful\n";
    return;
  }

  //  cout << "ScalarFieldBinaryOperator::performOperation:fields have been initialized.\n";
  
  switch(op.get()) {
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
	      << "Unexpected Operation Type #: " << op.get() << "\n";
  }
  //  cout << "ScalarFieldBinaryOperator::performOperation:end\n";
}

}

#endif // __OPERATORS_SCALARFIELDOPERATOR_H__

