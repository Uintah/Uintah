#ifndef __OPERATORS_UNARYFIELDOPERATOR_H__
#define __OPERATORS_UNARYFIELDOPERATOR_H__

#include "OperatorThread.h"
#include <Core/Geometry/IntVector.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Semaphore.h>
#include <Packages/Uintah/Core/Grid/ShareAssignArray3.h>


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
  smh->set_nx(mh->get_nx());
  smh->set_ny(mh->get_ny());
  smh->set_nz(mh->get_nz());
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
 
  if( field->get_type_name(0) != "LevelField"){
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
  } else {
    int max_workers = Max(Thread::numProcessors()/3, 4);
    Semaphore* thread_sema = scinew Semaphore( "scalar operator semaphore",
					       max_workers); 
    typedef typename Field::value_type Data;
    vector<ShareAssignArray3<Data> >& data = field->fdata();
    vector<ShareAssignArray3<Data> >::iterator it = data.begin();
    vector<ShareAssignArray3<Data> >::iterator end = data.end();
    IntVector offset( (*it).getLowIndex() );
    for(;it != end; ++it) {
      thread_sema->down();
      Thread *thrd = 
	scinew Thread(
		      scinew OperatorThread< Data, ScalarField, Op >
		      ( *it, scalarField, offset, op, thread_sema ),
		      "scalar operator worker");
      thrd->detach();
    }
    thread_sema->down(max_workers);
    if(thread_sema) delete thread_sema;
  }
}

}

#endif // __OPERATORS_UNARYFIELDOPERATOR_H__


