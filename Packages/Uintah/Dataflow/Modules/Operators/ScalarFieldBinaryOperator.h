/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


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

#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Geometry/IntVector.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

#include "ScalarOperatorFunctors.h"
#include "BinaryFieldOperator.h"
#include "OperatorThread.h"

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
                               FieldHandle right, int op) = 0;
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const SCIRun::TypeDescription *ftd);
                               
};

template< class FIELD >
class ScalarFieldBinaryOperatorAlgoT : public ScalarFieldBinaryOperatorAlgo
{
  virtual FieldHandle execute( FieldHandle left,
                               FieldHandle right, int op);
protected:
    void performOperation(FIELD* left_field, FIELD *right_field,
			  FIELD* scalarField, int op );
};

template< class FIELD >
FieldHandle
ScalarFieldBinaryOperatorAlgoT<FIELD>::execute( FieldHandle left, 
                                                FieldHandle right,
                                                 int op)
{
  FIELD *lf = (FIELD*)(left.get_rep());
  FIELD *rf = (FIELD*)(right.get_rep());
  typename FIELD::mesh_handle_type mh = lf->get_typed_mesh();
  mh.detach();
  typename FIELD::mesh_type *mesh = mh.get_rep();

  FIELD *sf = scinew FIELD( mesh );
  performOperation( lf, rf, sf, op ); 
  return sf;
}


template<class FIELD >
void 
ScalarFieldBinaryOperatorAlgoT<FIELD>::performOperation(FIELD* left_field,
					   FIELD *right_field,
					   FIELD* scalarField,
                                           int op)
{
  //  cout << "ScalarFieldBinaryOperator::performOperation:start\n";
  bool successful = initField(left_field, right_field, scalarField);
  if (!successful) {
    std::cerr << "ScalarFieldBinaryOperator::performOperation: Error - initField was not successful\n";
    return;
  }

  //  cout << "ScalarFieldBinaryOperator::performOperation:fields have been initialized.\n";
  
  switch(op) {
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
	      << "Unexpected Operation Type #: " << op << "\n";
  }
  //  cout << "ScalarFieldBinaryOperator::performOperation:end\n";
}

}

#endif // __OPERATORS_SCALARFIELDOPERATOR_H__

