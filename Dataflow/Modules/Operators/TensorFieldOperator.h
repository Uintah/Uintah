#ifndef __DERIVE_TENSORFIELDOPERATOR_H__
#define __DERIVE_TENSORFIELDOPERATOR_H__

#include "TensorOperatorFunctors.h"
#include "OperatorThread.h"
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Packages/Uintah/Core/Grid/ShareAssignArray3.h>
#include <string>
#include <iostream>
using std::string;
using std::cerr;
using std::endl;

namespace Uintah {
using namespace SCIRun;
class TensorFieldOperator: public Module {
public:
  TensorFieldOperator(const string& id);
  virtual ~TensorFieldOperator() {}
    
  virtual void execute(void);
    
private:
  template<class TensorField, class ScalarField>
    void performOperation(TensorField* tensorField, ScalarField* scalarField);

  template<class TensorField, class Field>
    void initField(TensorField* tensorField, Field* field);

  template<class TensorField, class ScalarField, class TensorOp >
    void computeScalars(TensorField* tensorField, ScalarField* scalarField,
			TensorOp op /* TensorOp should be a functor for
				       converting tensors scalars */ );    
  //    TCLstring tcl_status;
  GuiInt guiOperation;

  // element extractor operation
  GuiInt guiRow;
  GuiInt guiColumn;
    
    // eigen value/vector operation
    //GuiInt guiEigenSelect;

    // eigen 2D operation
  GuiInt guiPlaneSelect;
  GuiDouble guiDelta;
  GuiInt guiEigen2DCalcType;
    
  FieldIPort *in;

  FieldOPort *sfout;
  //VectorFieldOPort *vfout;
    
};

template<class TensorField, class ScalarField>
void TensorFieldOperator::performOperation(TensorField* tensorField,
					   ScalarField* scalarField)
{
  initField(tensorField, scalarField);

  switch(guiOperation.get()) {
  case 0: // extract element
    computeScalars(tensorField, scalarField,
		   TensorElementExtractionOp(guiRow.get(), guiColumn.get()));
    break;
  case 1: // 2D eigen-value/vector
    if (guiEigen2DCalcType.get() == 0) {
      // e1 - e2
      int plane = guiPlaneSelect.get();
      if (plane == 2)
	computeScalars(tensorField, scalarField, Eigen2DXYOp());
      else if (plane == 1)
	computeScalars(tensorField, scalarField, Eigen2DXZOp());
      else
	computeScalars(tensorField, scalarField, Eigen2DYZOp());
    }
    else {
      // cos(e1 - e2) / delta
      int plane = guiPlaneSelect.get();
      double delta = guiDelta.get();
      if (plane == 2)
	computeScalars(tensorField, scalarField, Eigen2DXYCosOp(delta));
      else if (plane == 1)
	computeScalars(tensorField, scalarField, Eigen2DXZCosOp(delta));
      else
	computeScalars(tensorField, scalarField, Eigen2DYZCosOp(delta));
    }
    break;
  case 2: // pressure
    computeScalars(tensorField, scalarField, PressureOp());
    break;
  case 3: // equivalent stress 
    computeScalars(tensorField, scalarField, EquivalentStressOp());
    break;
  default:
    std::cerr << "TensorFieldOperator::performOperation: "
	      << "Unexpected Operation Type #: " << guiOperation.get() << "\n";
  }
}

template<class TensorField, class ScalarField>
void TensorFieldOperator::initField(TensorField* tensorField,
				    ScalarField* field)
{
  ASSERT( tensorField->data_at() == Field::CELL ||
	  tensorField->data_at() == Field::NODE );

  typename TensorField::mesh_handle_type tmh = tensorField->get_typed_mesh();
  typename ScalarField::mesh_handle_type fmh = field->get_typed_mesh();
  BBox box;
  box = tmh->get_bounding_box();
  //resize the geometry
  fmh->set_nx(tmh->get_nx());
  fmh->set_ny(tmh->get_ny());
  fmh->set_nz(tmh->get_nz());
  fmh->set_min( box.min() );
  fmh->set_max( box.max() );
  //resize the data storage
  field->resize_fdata();

}

template<class TensorField, class ScalarField, class TensorOp >
void TensorFieldOperator::computeScalars(TensorField* tensorField,
					 ScalarField* scalarField,
					 TensorOp op
					 /* TensorOp should be a functor for
					    converting tensors scalars */ )
{
  // so far only node and cell centered data
  ASSERT( tensorField->data_at() == Field::CELL ||
	  tensorField->data_at() == Field::NODE );

  Matrix3 M;
  typename TensorField::mesh_handle_type tmh = tensorField->get_typed_mesh();
  typename ScalarField::mesh_handle_type smh = scalarField->get_typed_mesh();

  if( tensorField->get_type_name(0) != "LevelField"){
    if( tensorField->data_at() == Field::CELL){
      typename TensorField::mesh_type::Cell::iterator t_it; tmh->begin(t_it);
      typename ScalarField::mesh_type::Cell::iterator s_it; smh->begin(s_it);
      typename TensorField::mesh_type::Cell::iterator t_end; tmh->end(t_end);
      
      for( ; t_it != t_end; ++t_it, ++s_it){
	scalarField->fdata()[*s_it] = op(tensorField->fdata()[*t_it]);
      }
    } else {
      typename TensorField::mesh_type::Node::iterator t_it; tmh->begin(t_it);
      typename ScalarField::mesh_type::Node::iterator s_it; smh->begin(s_it);
      typename TensorField::mesh_type::Node::iterator t_end; tmh->end(t_end);
      
      for( ; t_it != t_end; ++t_it, ++s_it){
	scalarField->fdata()[*s_it] = op(tensorField->fdata()[*t_it]);
      }
    }  
  } else {
    int max_workers = Max(Thread::numProcessors()/3, 4);
    Semaphore* thread_sema = scinew Semaphore( "tensor operator semaphore",
					       max_workers); 
    vector<ShareAssignArray3<Matrix3> > tdata = tensorField->fdata();
    vector<ShareAssignArray3<Matrix3> >::iterator vit = tdata.begin();
    vector<ShareAssignArray3<Matrix3> >::iterator vit_end = tdata.end();

    IntVector offset( (*vit).getLowIndex() );
    for(;vit != vit_end; ++vit) {
      thread_sema->down();
      Thread *thrd = 
	scinew Thread(
	  scinew OperatorThread< Matrix3, ScalarField, TensorOp >
	  ( *vit, scalarField, offset, op, thread_sema ),
	  "tensor operator worker");
      thrd->detach();
    }
    thread_sema->down(max_workers);
    if(thread_sema) delete thread_sema;
  }
}
} // end namespace Uintah 

#endif // __DERIVE_TENSORFIELDOPERATOR_H__

