#ifndef __OPERATORS_UNARYFIELDOPERATOR_H__
#define __OPERATORS_UNARYFIELDOPERATOR_H__

#include "OperatorThread.h"
#include <Core/Geometry/IntVector.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Semaphore.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>

namespace Uintah {
using namespace SCIRun;

  class UnaryFieldOperator {
  public:
    UnaryFieldOperator(){};
    virtual ~UnaryFieldOperator() {}
    
  protected:
    template<class Field, class ScalarField>
     void initField(Field* field,
		    ScalarField* scalarField);
    
    template<class Field, class ScalarField, class ScalarOp >
     void computeScalars(Field* field,
			 ScalarField* scalarField,
			 ScalarOp op /* ScalarOp should be a functor for
					modiyfying scalars */ );
};

template<class Field, class ScalarField>
void UnaryFieldOperator::initField(Field* field,
				    ScalarField* scalarField)
{
  ASSERT( field->data_at() == Field::CELL ||
	  field->data_at() == Field::NODE );

  typename Field::mesh_handle_type mh = field->get_typed_mesh();
  typename ScalarField::mesh_handle_type smh = scalarField->get_typed_mesh();
  BBox box;
  box = smh->get_bounding_box();
  //resize the geometry
  smh->set_ni(mh->get_ni());
  smh->set_nj(mh->get_nj());
  smh->set_nk(mh->get_nk());
  smh->set_transform(mh->get_transform());
  //resize the data storage
  scalarField->resize_fdata();

}

template<class Field, class ScalarField, class Op>
void UnaryFieldOperator::computeScalars(Field* field,
					 ScalarField* scalarField,
					 Op op)
{
  // so far only node and cell centered data
  ASSERT( field->data_at() == Field::CELL ||
	  field->data_at() == Field::NODE );


  typename Field::mesh_handle_type mh =
    field->get_typed_mesh();
  typename ScalarField::mesh_handle_type smh =
    scalarField->get_typed_mesh();
 
  if( field->data_at() == Field::CELL){
    typename Field::mesh_type::Cell::iterator it; mh->begin(it);
    typename Field::mesh_type::Cell::iterator end; mh->end(end);
    typename ScalarField::mesh_type::Cell::iterator s_it; smh->begin(s_it);
    for( ; it != end; ++it, ++s_it){
      scalarField->fdata()[*s_it] = op(field->fdata()[*it]);
    }
  } else {
    typename Field::mesh_type::Node::iterator it; mh->begin(it);
    typename Field::mesh_type::Node::iterator end; mh->end(end);
    typename ScalarField::mesh_type::Node::iterator s_it; smh->begin(s_it);
    
    for( ; it != end; ++it, ++s_it){
      scalarField->fdata()[*s_it] = op(field->fdata()[*it]);
    }
  }  
}

} // End namespace Uintah
#endif // __OPERATORS_UNARYFIELDOPERATOR_H__


