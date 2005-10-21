#ifndef __OPERATORS_VECTORFIELDOPERATOR_H__
#define __OPERATORS_VECTORFIELDOPERATOR_H__

#include "VectorOperatorFunctors.h"
#include "UnaryFieldOperator.h"
#include "OperatorThread.h"
#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Containers/FData.h>
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


class VectorFieldOperatorAlgo: 
    public DynamicAlgoBase, public UnaryFieldOperator 
{
public:
  typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;
  typedef LVMesh::handle_type                 LVMeshHandle;
  typedef HexTrilinearLgn<double>             FDdoubleBasis;
  typedef ConstantBasis<double>               CFDdoubleBasis;

  typedef GenericField<LVMesh, CFDdoubleBasis, FData3d<double, LVMesh> > CDField;
  typedef GenericField<LVMesh, FDdoubleBasis,  FData3d<double, LVMesh> > LDField;

  virtual FieldHandle execute(FieldHandle tensorfh, GuiInt op) = 0;
  static CompileInfoHandle get_compile_info(const SCIRun::TypeDescription *ftd);

protected:
 template<class VectorField, class ScalarField>
  void performOperation(VectorField* vectorField, ScalarField* scalarField,
                        GuiInt op);


};

template<class VectorField >
class VectorFieldOperatorAlgoT: public VectorFieldOperatorAlgo
{
public:
  virtual FieldHandle execute(FieldHandle vectorfh, GuiInt op);
};

template<class VectorField >
FieldHandle
VectorFieldOperatorAlgoT<VectorField>::execute(FieldHandle vectorfh, GuiInt op)
{
  VectorField *vectorField = (VectorField *)(vectorfh.get_rep());
  typename VectorField::mesh_handle_type mh = vectorfh->get_typed_mesh();
  mh.detach();
  typename VectorField::mesh_type *mesh = mh.get_rep();

  FieldHandle scalarField;
  if( vectorField->basis_order == 0 ){
    CDField *sf = scinew CDField( mesh );
    performOperation( vectorField, sf, op );
    scalarField = sf;
  } else {
    LDField *sf = scinew LDField( mesh );
    performOperation( vectorField, sf, op );
    scalarField = sf;
  }

  for(unsigned int i = 0; i < vectorField->nproperties(); i++){
    string prop_name(vectorField->get_property_name( i ));
    if(prop_name == "varname"){
      string prop_component;
      vectorField->get_property( prop_name, prop_component);
      switch(guiOperation.get()) {
      case 0: // extract element 1
        scalarField->set_property("varname",
                                  string(prop_component +":1"), true);
        break;
      case 1: // extract element 2
        scalarField->set_property("varname", 
                                  string(prop_component +":2"), true);
        break;
      case 2: // extract element 3
        scalarField->set_property("varname", 
                                  string(prop_component +":3"), true);
        break;
      case 3: // Vector length
        scalarField->set_property("varname", 
                                  string(prop_component +":length"), true);
        break;
      case 4: // Vector curvature
        scalarField->set_property("varname",
                                  string(prop_component +":vorticity"), true);
        break;
      default:
        scalarField->set_property("varname",
                                  string(prop_component.c_str()), true);
      }
    } else if( prop_name == "generation") {
      int generation;
      vectorField->get_property( prop_name, generation);
      scalarField->set_property(prop_name.c_str(), generation , true);
    } else if( prop_name == "timestep" ) {
      int timestep;
      vectorField->get_property( prop_name, timestep);
      scalarField->set_property(prop_name.c_str(), timestep , true);
    } else if( prop_name == "offset" ){
      IntVector offset(0,0,0);        
      vectorField->get_property( prop_name, offset);
      scalarField->set_property(prop_name.c_str(), IntVector(offset) , true);
      cerr<<"vector offset is "<< offset <<"n";
    } else if( prop_name == "delta_t" ){
      double dt;
      vectorField->get_property( prop_name, dt);
      scalarField->set_property(prop_name.c_str(), dt , true);
    } else if( prop_name == "vartype" ){
      int vartype;
      vectorField->get_property( prop_name, vartype);
      scalarField->set_property(prop_name.c_str(), vartype , true);
    } else {
      warning( "Unknown field property, not transferred.");
    }
  }
  return scalarField;
}   

template<class VectorField, class ScalarField>
void VectorFieldOperatorAlgo::performOperation(VectorField* vectorField,
                                               ScalarField* scalarField,
                                               GuiInt op)
{
  initField(vectorField, scalarField);

  switch(op.get()) {
  case 0: // extract element 1
  case 1: // extract element 2
  case 2: // extract element 3
    computeScalars(vectorField, scalarField,
		   VectorElementExtractionOp(op.get()));
    break;
  case 3: // Vector length
    computeScalars(vectorField, scalarField, LengthOp());
    break;
  case 4: // Vector curvature
    computeScalars(vectorField, scalarField, VorticityOp());
    break;
  default:
    std::cerr << "VectorFieldOperator::performOperation: "
	      << "Unexpected Operation Type #: " << op.get() << "\n";
  }
}

}

#endif // __OPERATORS_VECTORFIELDOPERATOR_H__

