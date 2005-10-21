#ifndef __OPERATORS_SCALARFIELDOPERATOR_H__
#define __OPERATORS_SCALARFIELDOPERATOR_H__

#include "ScalarOperatorFunctors.h"
#include "UnaryFieldOperator.h"
#include "OperatorThread.h"
#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Geometry/IntVector.h>
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

class ScalarFieldOperatorAlgo: public DynamicAlgoBase, public UnaryFieldOperator 

{
public:
  virtual FieldHandle execute(FieldHandle field, GuiInt op) = 0;
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const SCIRun::TypeDescription *ftd);
protected:
  template<class Field, class ScalarField>      
  void performOperation(Field* field,
                        ScalarField* scalarField,
                        GuiInt op);
  template<class T1, class T2> 
  void set_properties( T1* sf1, T2* sf2);
};

template< class IFIELD >
class ScalarFieldOperatorAlgoT : public ScalarFieldOperatorAlgo
{
public:
  //! virtual interface
  virtual FieldHandle execute(FieldHandle fh, GuiInt op);
};

template< class IFIELD >
FieldHandle
ScalarFieldOperatorAlgoT<IFIELD>::execute( FieldHandle fh, GuiInt op)
{
  
  IFIELD *scalarField1 = (IFIELD *)(fh.get_rep());
  typename IFIELD::mesh_handle_type mh = scalarField1->get_typed_mesh();
  mh.detach();
  typename IFIELD::mesh_type *mesh = mh.get_rep();
  
  IFIELD *scalarField2 = 0;
  scalarField2 = scinew IFIELD(mesh);
  performOperation( scalarField1, scalarField2, op );
  set_properties(scalarField1, scalarField2 );

  return scalarField2;

}

template<class Field, class ScalarField>
void 
ScalarFieldOperatorAlgo::performOperation(Field* field,
					   ScalarField* scalarField,
                                           GuiInt op)
{
  initField(field, scalarField);

  switch(op.get()) {
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
	      << "Unexpected Operation Type #: " << op.get() << "\n";
  }
}

}

#endif // __OPERATORS_SCALARFIELDOPERATOR_H__

