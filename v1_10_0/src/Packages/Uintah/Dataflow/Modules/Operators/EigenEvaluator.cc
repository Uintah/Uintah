#include "EigenEvaluator.h"
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/LatVolField.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>

namespace Uintah {

  DECLARE_MAKER(EigenEvaluator)

template<class TensorField, class VectorField, class ScalarField>
void computeGridEigens(TensorField* tensorField,
		       ScalarField* eValueField, VectorField* eVectorField,
		       int chosenEValue);
 

EigenEvaluator::EigenEvaluator(GuiContext* ctx)
  : Module("EigenEvaluator",ctx,Source, "Operators", "Uintah"),
    guiEigenSelect(ctx->subVar("eigenSelect"))
    //    tcl_status(ctx->subVar("tcl_status")),
{
}
  
void EigenEvaluator::execute(void) {
  //  tcl_status.set("Calling EigenEvaluator!"); 

  in = (FieldIPort *) get_iport("Tensor Field");
  sfout = (FieldOPort *) get_oport("Eigenvalue Field");
  vfout = (FieldOPort *) get_oport("Eigenvector Field");
  

  FieldHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"EigenEvaluator::execute(void) Didn't get a handle\n";
    return;
  } else if ( hTF->get_type_name(1) != "Matrix3" ){
    std::cerr<<"Input is not a Tensor field\n";
    return;
  }

  LatVolField<double> *eValueField = 0;
  LatVolField<Vector> *eVectorField = 0;

  if( LatVolField<Matrix3> *tensorField =
      dynamic_cast<LatVolField<Matrix3>*>(hTF.get_rep())) {

    eValueField = scinew LatVolField<double>(hTF->data_at());
    eVectorField = scinew LatVolField<Vector>(hTF->data_at());
    computeGridEigens(tensorField, eValueField,
		      eVectorField, guiEigenSelect.get());
  }

  if( eValueField )
    sfout->send(eValueField);
  if( eVectorField )
    vfout->send(eVectorField);  
}

template<class TensorField, class VectorField, class ScalarField>
void computeGridEigens(TensorField* tensorField,
		       ScalarField* eValueField, VectorField* eVectorField,
		       int chosenEValue)
{
  ASSERT( tensorField->data_at() == Field::CELL ||
	  tensorField->data_at() == Field::NODE );
  typename TensorField::mesh_handle_type tmh = tensorField->get_typed_mesh();
  typename ScalarField::mesh_handle_type smh = eValueField->get_typed_mesh();
  typename VectorField::mesh_handle_type vmh = eVectorField->get_typed_mesh();

  BBox box;
  box = tmh->get_bounding_box();

  //resize the geometry
  smh->set_ni(tmh->get_ni());
  smh->set_nj(tmh->get_nj());
  smh->set_nk(tmh->get_nk());
  smh->set_transform(tmh->get_transform());
  vmh->set_ni(tmh->get_ni());
  vmh->set_nj(tmh->get_nj());
  vmh->set_nk(tmh->get_nk());
  smh->set_transform(tmh->get_transform());
  //resize the data storage
  eValueField->resize_fdata();
  eVectorField->resize_fdata();


  int num_eigen_values;
  Matrix3 M;
  double e[3];
  std::vector<Vector> eigenVectors;
  
  if( tensorField->data_at() == Field::CELL){
    typename TensorField::mesh_type::Cell::iterator t_it; tmh->begin(t_it);
    typename ScalarField::mesh_type::Cell::iterator s_it; smh->begin(s_it);
    typename VectorField::mesh_type::Cell::iterator v_it; vmh->begin(v_it);
    typename TensorField::mesh_type::Cell::iterator t_end; tmh->end(t_end);
    
    for( ; t_it != t_end; ++t_it, ++s_it, ++v_it){
      M = tensorField->fdata()[*t_it];
      num_eigen_values = M.getEigenValues(e[0], e[1], e[2]);
      if (num_eigen_values <= chosenEValue) {
	eValueField->fdata()[*s_it] = 0;
	eVectorField->fdata()[*v_it] = Vector(0,0,0);
      } else {
	eValueField->fdata()[*s_it] = e[chosenEValue];
	eigenVectors = M.getEigenVectors(e[chosenEValue], e[0]);
	if (eigenVectors.size() != 1) {
	  eVectorField->fdata()[*v_it] = Vector(0, 0, 0);
	} else {
	  eVectorField->fdata()[*v_it] = eigenVectors[0].normal();
	}
      }
    }
  } else {
    typename TensorField::mesh_type::Node::iterator t_it; tmh->begin(t_it);
    typename ScalarField::mesh_type::Node::iterator s_it; smh->begin(s_it);
    typename VectorField::mesh_type::Node::iterator v_it; vmh->begin(v_it);
    typename TensorField::mesh_type::Node::iterator t_end; tmh->end(t_end);
    
    for( ; t_it != t_end; ++t_it, ++s_it, ++v_it){
      M = tensorField->fdata()[*t_it];
      num_eigen_values = M.getEigenValues(e[0], e[1], e[2]);
      if (num_eigen_values <= chosenEValue) {
	eValueField->fdata()[*s_it] = 0;
	eVectorField->fdata()[*v_it] = Vector(0,0,0);
      } else {
	eValueField->fdata()[*s_it] = e[chosenEValue];
	eigenVectors = M.getEigenVectors(e[chosenEValue], e[0]);
	if (eigenVectors.size() != 1) {
	  eVectorField->fdata()[*v_it] = Vector(0, 0, 0);
	} else {
	  eVectorField->fdata()[*v_it] = eigenVectors[0].normal();
	}
      }
    }
  }    
}
  
}



