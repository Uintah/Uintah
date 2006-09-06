#ifndef __OPERATORS_SCALARFIELDOPERATOR_H__
#define __OPERATORS_SCALARFIELDOPERATOR_H__

#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

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
  for(size_t i = 0; i < sf1->nproperties(); i++){
    string prop_name(sf1->get_property_name( i ));
    if(prop_name == "name"){
      string prop_component;
      sf1->get_property( prop_name, prop_component);
      switch(op) {
      case 0: // extract element 1
        sf2->set_property("name",
                          string(prop_component +":ln"), true);
        break;
      case 1: // extract element 2
        sf2->set_property("name", 
                          string(prop_component +":e"), true);
        break;
      default:
        sf2->set_property("name",
                          string(prop_component.c_str()), true);
      }
    } else if( prop_name == "generation") {
      int generation;
      sf1->get_property( prop_name, generation);
      sf2->set_property(prop_name.c_str(), generation , true);
    } else if( prop_name == "timestep" ) {
      int timestep;
      sf1->get_property( prop_name, timestep);
      sf2->set_property(prop_name.c_str(), timestep , true);
    } else if( prop_name == "offset" ){
      IntVector offset(0,0,0);        
      sf1->get_property( prop_name, offset);
      sf2->set_property(prop_name.c_str(), IntVector(offset) , true);
    } else if( prop_name == "delta_t" ){
      double dt;
      sf1->get_property( prop_name, dt);
      sf2->set_property(prop_name.c_str(), dt , true);
    } else if( prop_name == "vartype" ){
      int vartype;
      sf1->get_property( prop_name, vartype);
      sf2->set_property(prop_name.c_str(), vartype , true);
    } else {
      cerr<<"Unknown field property: "<<prop_name<<", not transferred.\n";
    }
  }
}

}

#endif // __OPERATORS_SCALARFIELDOPERATOR_H__

