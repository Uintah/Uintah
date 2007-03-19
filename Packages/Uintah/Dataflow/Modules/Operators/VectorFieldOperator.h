#ifndef __OPERATORS_VECTORFIELDOPERATOR_H__
#define __OPERATORS_VECTORFIELDOPERATOR_H__

#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

#include "VectorOperatorFunctors.h"
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

  virtual FieldHandle execute(FieldHandle tensorfh, int op) = 0;
  static CompileInfoHandle get_compile_info(const SCIRun::TypeDescription *ftd);

protected:
 template<class VectorField, class ScalarField>
  void performOperation(VectorField* vectorField, ScalarField* scalarField,
                        int op);


};

template<class VectorField >
class VectorFieldOperatorAlgoT: public VectorFieldOperatorAlgo
{
public:
  virtual FieldHandle execute(FieldHandle vectorfh, int op);
};

template<class VectorField >
FieldHandle
VectorFieldOperatorAlgoT<VectorField>::execute(FieldHandle vectorfh, int op)
{
  VectorField *vectorField = (VectorField *)(vectorfh.get_rep());
  typename VectorField::mesh_handle_type mh = vectorField->get_typed_mesh();
  mh.detach();
  typename VectorField::mesh_type *mesh = mh.get_rep();

  FieldHandle scalarField;
  if( vectorField->basis_order() == 0 ){
    CDField *sf = scinew CDField( mesh );
    performOperation( vectorField, sf, op );
    scalarField = sf;
  } else {
    LDField *sf = scinew LDField( mesh );
    performOperation( vectorField, sf, op );
    scalarField = sf;
  }

  scalarField->copy_properties( vectorField );
  string name;
  if( vectorField->get_property("name", name) ){
      switch(op) {
      case 0: // extract element 1
        scalarField->set_property("name",
                                  string(name +":1"), false);
        break;
      case 1: // extract element 2
        scalarField->set_property("name", 
                                  string(name +":2"), false);
        break;
      case 2: // extract element 3
        scalarField->set_property("name", 
                                  string(name +":3"), false);
        break;
      case 3: // Vector length
        scalarField->set_property("name", 
                                  string(name +":length"), false);
        break;
      case 4: // Vector curvature
        scalarField->set_property("name",
                                  string(name +":vorticity"), false);
        break;
      default:
        scalarField->set_property("name", name, false);
      }
  }    

  return scalarField;
}   

template<class VectorField, class ScalarField>
void VectorFieldOperatorAlgo::performOperation(VectorField* vectorField,
                                               ScalarField* scalarField,
                                               int op)
{
  initField(vectorField, scalarField);

  switch(op) {
  case 0: // extract element 1
  case 1: // extract element 2
  case 2: // extract element 3
    computeScalars(vectorField, scalarField,
		   VectorElementExtractionOp(op));
    break;
  case 3: // Vector length
    computeScalars(vectorField, scalarField, LengthOp());
    break;
  case 4: // Vector curvature
    computeScalars(vectorField, scalarField, VorticityOp());
    break;
  default:
    std::cerr << "VectorFieldOperator::performOperation: "
	      << "Unexpected Operation Type #: " << op << "\n";
  }
}

}

#endif // __OPERATORS_VECTORFIELDOPERATOR_H__

