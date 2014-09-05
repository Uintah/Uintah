#ifndef __Uintah_Package_Dataflow_Modules_Operators_BinaryFieldOperator_H__
#define __Uintah_Package_Dataflow_Modules_Operators_BinaryFieldOperator_H__

/**************************************
CLASS
  BinaryFieldOperator
      Operator for binary operations on fields


GENERAL INFORMATION

  BinaryFieldOperator
  
  Author:  James Bigler (bigler@cs.utah.edu)
           
           Department of Computer Science
           
           University of Utah
  
  Date:    April 2002
  
  C-SAFE
  
  Copyright <C> 2002 SCI Group

KEYWORDS
  Operator, Binary, scalar_field

DESCRIPTION
  Class to help apply binary operations to scalar fields.

WARNING
  None



****************************************/

#include "OperatorThread.h"
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using namespace std;
using namespace SCIRun;

  class BinaryFieldOperator {
  public:
    BinaryFieldOperator(){};
    virtual ~BinaryFieldOperator() {}
    
  protected:
    // This function takes two fields, determines if they are compatable
    // (i.e. they have the same topology), and initializes the scalar field
    // True is returned if the task was successful, false otherwise.
    template<class FieldLeft, class FieldRight, class ScalarField>
    bool initField(FieldLeft* left_field, FieldRight * right_field,
		   ScalarField* scalarField);
    
    template<class FieldLeft, class FieldRight, class ScalarField,
	     class ScalarOp >
    void computeScalars(FieldLeft* left_field, FieldRight * right_field,
			ScalarField* scalarField,
			ScalarOp op /* ScalarOp should be a functor for
				       modiyfying scalars */ );
  };

template<class FieldLeft, class FieldRight, class ScalarField>
bool BinaryFieldOperator::initField(FieldLeft* left_field,
				    FieldRight * right_field,
				    ScalarField* scalarField) {
  // We need to make sure that the data for the two fields are the same
  if ( left_field->data_at() != right_field->data_at() )
    return false;

  // Now we can check if we have code to support the type of field coming in.
  // Currently we only have support for Node and Cell centered data.
  if ( left_field->data_at() != FieldLeft::CELL &&
       left_field->data_at() != FieldLeft::NODE )
    return false;

  // We need to get the mesh dimensions and make sure they are the same for
  // both incoming fields.  Once we determing that we can, initialize the
  // the outgoing field.
  
  typename FieldLeft::mesh_handle_type mhl = left_field->get_typed_mesh();
  typename FieldRight::mesh_handle_type mhr = right_field->get_typed_mesh();
  // Check the data dimensions
  if (mhl->get_ni() != mhr->get_ni() ||
      mhl->get_nj() != mhr->get_nj() ||
      mhl->get_nk() != mhr->get_nk())
    return false;

  // Now here's a sticky situation.  We've determined that the data dimensions
  // match.  That's good, because we know we are dealing with the same size
  // data.  However should we be concerned with the actuall data location?
  // I'm going to take a leap and say that it should matter, but I won't
  // enforce it.  I will spit out some kind of warning though, so that the
  // user will be appraised of the situtation.
  
  // Check bounding boxes.  This takes into consideration the transformation
  // of the geometry.
  BBox left_box = mhl->get_bounding_box();
  BBox right_box = mhr->get_bounding_box();
  if (left_box.min() != right_box.min() || left_box.max() != right_box.max()) {
    cerr << "BinaryFieldOperator::initField: BBox for left hand operator does not match that of the right hand operator.\n";
  }

  // Now we can initialize the outgoing field to match those coming in.
  typename ScalarField::mesh_handle_type smh = scalarField->get_typed_mesh();
  //resize the geometry
  smh->set_ni(mhl->get_ni());
  smh->set_nj(mhl->get_nj());
  smh->set_nk(mhl->get_nk());
  smh->set_transform(mhl->get_transform());
  //resize the data storage
  scalarField->resize_fdata();

  return true;
}
    
template<class FieldLeft, class FieldRight, class ScalarField, class ScalarOp >
void BinaryFieldOperator::computeScalars(FieldLeft* left_field,
					 FieldRight * right_field,
					 ScalarField* scalarField,
					 ScalarOp op) {

  // so far only node and cell centered data
  ASSERT( left_field->data_at() == FieldLeft::CELL ||
	  left_field->data_at() == FieldRight::NODE );


  typename FieldLeft::mesh_handle_type mhl = left_field->get_typed_mesh();
  typename FieldRight::mesh_handle_type mhr = right_field->get_typed_mesh();
  typename ScalarField::mesh_handle_type smh =
    scalarField->get_typed_mesh();
 
  if( left_field->data_at() == Field::CELL){
    typename FieldLeft::mesh_type::Cell::iterator it; mhl->begin(it);
    typename FieldLeft::mesh_type::Cell::iterator end; mhl->end(end);
    typename FieldRight::mesh_type::Cell::iterator it_r; mhr->begin(it_r);
    typename ScalarField::mesh_type::Cell::iterator s_it; smh->begin(s_it);
    for( ; it != end; ++it, ++s_it, ++it_r){
      scalarField->fdata()[*s_it] = op(left_field->fdata()[*it],
				       right_field->fdata()[*it_r]);
    }
  } else {
    typename FieldLeft::mesh_type::Node::iterator it; mhl->begin(it);
    typename FieldLeft::mesh_type::Node::iterator end; mhl->end(end);
    typename FieldRight::mesh_type::Node::iterator it_r; mhr->begin(it_r);
    typename ScalarField::mesh_type::Node::iterator s_it; smh->begin(s_it);
    for( ; it != end; ++it, ++s_it, ++it_r){
      scalarField->fdata()[*s_it] = op(left_field->fdata()[*it],
				       right_field->fdata()[*it_r]);
    }
  }  
}
  

} // end namespace Uintah


#endif // __Uintah_Package_Dataflow_Modules_Operators_BinaryFieldOperator_H__
