#ifndef __OPERATORS_VECTORFIELDOPERATOR_H__
#define __OPERATORS_VECTORFIELDOPERATOR_H__

#include "VectorOperatorFunctors.h"
#include "OperatorThread.h"
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Module.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Mutex.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Packages/Uintah/Core/Grid/ShareAssignArray3.h>
#include <string>
#include <iostream>
using std::string;
using std::cerr;
using std::endl;


namespace Uintah {
using namespace SCIRun;

  class VectorFieldOperator: public Module {
  public:
    VectorFieldOperator(const string& id);
    virtual ~VectorFieldOperator() {}
    
    virtual void execute(void);
    
  private:
    template<class VectorField, class ScalarField>
     void performOperation(VectorField* vectorField, ScalarField* scalarField);
    template<class VectorField, class Field>
     void initField(VectorField* vectorField, Field* field);
    
    template<class VectorField, class ScalarField, class VectorOp >
     void computeScalars(VectorField* vectorField, ScalarField* scalarField,
			 VectorOp op /* VectorOp should be a functor for
					converting tensors scalars */ );

    //    TCLstring tcl_status;
    GuiInt guiOperation;

    FieldIPort *in;

    FieldOPort *sfout;
    //VectorFieldOPort *vfout;
  };

template<class VectorField, class ScalarField>
void VectorFieldOperator::performOperation(VectorField* vectorField,
					   ScalarField* scalarField)
{
  initField(vectorField, scalarField);

  switch(guiOperation.get()) {
  case 0: // extract element 1
  case 1: // extract element 2
  case 2: // extract element 3
    computeScalars(vectorField, scalarField,
		   VectorElementExtractionOp(guiOperation.get()));
    break;
  case 3: // Vector length
    computeScalars(vectorField, scalarField, LengthOp());
    break;
  default:
    std::cerr << "VectorFieldOperator::performOperation: "
	      << "Unexpected Operation Type #: " << guiOperation.get() << "\n";
  }
}

template<class VectorField, class ScalarField>
void VectorFieldOperator::initField(VectorField* vectorField,
				    ScalarField* field)
{
  ASSERT( vectorField->data_at() == Field::CELL ||
	  vectorField->data_at() == Field::NODE );

  typename VectorField::mesh_handle_type vmh = vectorField->get_typed_mesh();
  typename ScalarField::mesh_handle_type fmh = field->get_typed_mesh();
  BBox box;
  box = vmh->get_bounding_box();
  //resize the geometry
  fmh->set_nx(vmh->get_nx());
  fmh->set_ny(vmh->get_ny());
  fmh->set_nz(vmh->get_nz());
  fmh->set_transform(vmh->get_transform());
  //resize the data storage
  field->resize_fdata();

}

template<class VectorField, class ScalarField, class VectorOp >
void VectorFieldOperator::computeScalars(VectorField* vectorField,
					 ScalarField* scalarField,
					 VectorOp op 
					 /* VectorOp should be a functor for
					    converting vectors scalars */ )
{
  // so far only node and cell centered data
  ASSERT( vectorField->data_at() == Field::CELL ||
	  vectorField->data_at() == Field::NODE );


  typename VectorField::mesh_handle_type vmh = vectorField->get_typed_mesh();
  typename ScalarField::mesh_handle_type smh = scalarField->get_typed_mesh();
  if( vectorField->get_type_name(0) != "LevelField"){
    if( vectorField->data_at() == Field::CELL){
      typename VectorField::mesh_type::Cell::iterator v_it; vmh->begin(v_it);
      typename VectorField::mesh_type::Cell::iterator v_end; vmh->end(v_end);
      typename ScalarField::mesh_type::Cell::iterator s_it; smh->begin(s_it);
    
      /*     cerr<<"v_it = ("<<(*v_it).i_<<","<<(*v_it).j_<<","<<(*v_it).k_<< */
      /*       "), v_end = ("<<(*v_end).i_<<","<<(*v_end).j_<<","<<(*v_end).k_<<")\n"; */
      for( ; v_it != v_end; ++v_it, ++s_it){
	scalarField->fdata()[*s_it] = op(vectorField->fdata()[*v_it]);
      }
    } else {
      typename VectorField::mesh_type::Node::iterator v_it; vmh->begin(v_it);
      typename VectorField::mesh_type::Node::iterator v_end; vmh->end(v_end);
      typename ScalarField::mesh_type::Node::iterator s_it; smh->begin(s_it);
  
      for( ; v_it != v_end; ++v_it, ++s_it){
	scalarField->fdata()[*s_it] = op(vectorField->fdata()[*v_it]);
      }
    }  
  } else {
    int max_workers = Max(Thread::numProcessors()/3, 4);
    Semaphore* thread_sema = scinew Semaphore( "vector operator semaphore",
					       max_workers); 
    vector<ShareAssignArray3<Vector> > tdata = vectorField->fdata();
    vector<ShareAssignArray3<Vector> >::iterator vit = tdata.begin();
    vector<ShareAssignArray3<Vector> >::iterator vit_end = tdata.end();

#if 0
    float minx,maxx,miny,maxy,minz,maxz,minl,maxl;
    Array3<Vector>::iterator it((*vit).begin());
    Array3<Vector>::iterator it_end((*vit).end());
    minx = maxx = (*it)[0];
    miny = maxy = (*it)[1];
    minz = maxz = (*it)[2];
    minl = maxl = (*it).length();
    for(;vit != vit_end; ++vit) {
      it = (*vit).begin();
      it_end = (*vit).end();
      for(;it != it_end; ++it) {
	minx = Min(minx,(*it)[0]);
	maxx = Max(maxx,(*it)[0]);
	miny = Min(miny,(*it)[1]);
	maxy = Max(maxy,(*it)[1]);
	minz = Min(minz,(*it)[2]);
	maxz = Max(maxz,(*it)[2]);
	minl = Min(minl,(*it).length());
	maxl = Max(maxl,(*it).length());
      }
    }
    cout << "minx = " << minx << ", maxx = " << maxx << endl;
    cout << "miny = " << miny << ", maxy = " << maxy << endl;
    cout << "minz = " << minz << ", maxz = " << maxz << endl;
    cout << "minl = " << minl << ", maxl = " << maxl << endl;
    vit = tdata.begin();
#endif
    IntVector offset( (*vit).getLowIndex() );
    for(;vit != vit_end; ++vit) {
      thread_sema->down();
      Thread *thrd = 
	scinew Thread(
	  scinew OperatorThread< Vector, ScalarField, VectorOp >
	  ( *vit, scalarField, offset, op, thread_sema ),
	  "vector operator worker");
      thrd->detach();
    }
    thread_sema->down(max_workers);
    if(thread_sema) delete thread_sema;
  }
    
}

}

#endif // __OPERATORS_VECTORFIELDOPERATOR_H__

