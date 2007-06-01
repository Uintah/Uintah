#ifndef __OPERATORS_SCALARFIELDOPERATOR_H__
#define __OPERATORS_SCALARFIELDOPERATOR_H__

#include <SCIRun/Core/Basis/Constant.h>
#include <SCIRun/Core/Basis/HexTrilinearLgn.h>
#include <SCIRun/Core/Containers/FData.h>
#include <SCIRun/Core/Datatypes/GenericField.h>
#include <SCIRun/Dataflow/GuiInterface/GuiVar.h>
#include <SCIRun/Core/Geometry/IntVector.h>
#include <SCIRun/Core/Util/TypeDescription.h>
#include <SCIRun/Core/Util/DynamicLoader.h>

#include "ScalarOperatorFunctors.h"
#include "UnaryFieldOperator.h"
#include "OperatorThread.h"

#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using std::string;
using std::cerr;
using std::endl;

class ScalarFieldOperatorAlgo: public DynamicAlgoBase, public UnaryFieldOperator 

{
public:
  virtual FieldHandle execute(FieldHandle field, int op) = 0;
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const SCIRun::TypeDescription *ftd);
protected:
  template<class Field, class ScalarField>      
  void performOperation(Field* field,
                        ScalarField* scalarField,
                        int op);
  template<class T1, class T2> 
  void set_properties( T1* sf1, T2* sf2, int op);
};

template< class IFIELD >
class ScalarFieldOperatorAlgoT : public ScalarFieldOperatorAlgo
{
public:
  //! virtual interface
  virtual FieldHandle execute(FieldHandle fh, int op);
};

template< class IFIELD >
FieldHandle
ScalarFieldOperatorAlgoT<IFIELD>::execute( FieldHandle fh, int op)
{
  
  IFIELD *scalarField1 = (IFIELD *)(fh.get_rep());
  typename IFIELD::mesh_handle_type mh = scalarField1->get_typed_mesh();
  mh.detach();
  typename IFIELD::mesh_type *mesh = mh.get_rep();
  
  IFIELD *scalarField2 = 0;
  scalarField2 = scinew IFIELD(mesh);
  performOperation( scalarField1, scalarField2, op );
  set_properties(scalarField1, scalarField2, op );

  return scalarField2;

}

template<class Field, class ScalarField>
void 
ScalarFieldOperatorAlgo::performOperation(Field* field,
					   ScalarField* scalarField,
                                           int op)
{
  initField(field, scalarField);

  switch(op) {
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
	      << "Unexpected Operation Type #: " << op << "\n";
  }
}
template<class T1, class T2> 
void
ScalarFieldOperatorAlgo::set_properties( T1* sf1, T2* sf2, int op)
{

  sf2->copy_properties( sf1 );
  string name;
  if( sf1->get_property("name", name) ){
      switch(op) {
      case 0: // extract element 1
        sf2->set_property("name",
                          string(name +":ln"), false);
        break;
      case 1: // extract element 2
        sf2->set_property("name", 
                          string(name +":e"), false);
        break;
      default:
        sf2->set_property("name", name, false);
      }
  }
}

}

#endif // __OPERATORS_SCALARFIELDOPERATOR_H__

