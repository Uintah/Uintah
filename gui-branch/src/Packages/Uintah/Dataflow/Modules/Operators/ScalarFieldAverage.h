#ifndef __OPERATORS_SCALARFIELDAVERAGE_H__
#define __OPERATORS_SCALARFIELDAVERAGE_H__

#include "OperatorThread.h"
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCL.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Packages/Uintah/Core/Grid/ShareAssignArray3.h>
#include <string>
#include <iostream>
using std::string;
using std::cerr;
using std::endl;

using namespace SCIRun;

namespace Uintah {
  
class ScalarFieldAverage: public Module {
public:
  ScalarFieldAverage(const string& id);
  virtual ~ScalarFieldAverage() {}
    
  virtual void execute(void);
private:
    
  GuiDouble t0_;
  GuiDouble t1_;
  GuiInt tsteps_;
    
  FieldIPort *in;
  FieldOPort *sfout;
    
  LatticeVol<double> *aveField;
  FieldHandle aveFieldH;
  string varname;
  double time;
  //ScalarFieldOPort *vfout;

  void fillField( FieldHandle f );
  void averageField( FieldHandle f );

  template<class ScalarField1, class ScalarField2>
    void initField(ScalarField1* scalarField1,
		   ScalarField2* scalarField2);
  
  template<class ScalarField1, class ScalarField2, class ScalarOp>
    void computeScalars(ScalarField1* scalarField1,
			ScalarField2* scalarField2,
			ScalarOp op);
  
  template<class ScalarField1, class ScalarField2>
    void computeAverages(ScalarField1* scalarField1,
			 ScalarField2* scalarField2);
};

template<class ScalarField1, class ScalarField2>
void ScalarFieldAverage::initField(ScalarField1* scalarField1,
				    ScalarField2* scalarField2)
{
  ASSERT( scalarField1->data_at() == Field::CELL ||
	  scalarField1->data_at() == Field::NODE );

  typename ScalarField1::mesh_handle_type smh = scalarField1->get_typed_mesh();
  typename ScalarField2::mesh_handle_type fmh = scalarField2->get_typed_mesh();
  BBox box;
  box = smh->get_bounding_box();
  //resize the geometry
  fmh->set_nx(smh->get_nx());
  fmh->set_ny(smh->get_ny());
  fmh->set_nz(smh->get_nz());
  fmh->set_transform(smh->get_transform());
  //resize the data storage
  scalarField2->resize_fdata();

}


template<class ScalarField1, class ScalarField2, class ScalarOp>
void ScalarFieldAverage::computeScalars(ScalarField1* scalarField1,
					 ScalarField2* scalarField2,
					 ScalarOp op)
{
  // so far only node and cell centered data
  ASSERT( scalarField1->data_at() == Field::CELL ||
	  scalarField1->data_at() == Field::NODE );


  typename ScalarField1::mesh_handle_type s1mh =
    scalarField1->get_typed_mesh();
  typename ScalarField2::mesh_handle_type s2mh =
    scalarField2->get_typed_mesh();
 
  if( scalarField1->get_type_name(0) != "LevelField"){
    if( scalarField1->data_at() == Field::CELL){
      typename ScalarField1::mesh_type::Cell::iterator v_it; s1mh->begin(v_it);
      typename ScalarField1::mesh_type::Cell::iterator v_end; s1mh->end(v_end);
      typename ScalarField2::mesh_type::Cell::iterator s_it; s2mh->begin(s_it);
      for( ; v_it != v_end; ++v_it, ++s_it){
	scalarField2->fdata()[*s_it] = op(scalarField1->fdata()[*v_it]);
      }
    } else {
      typename ScalarField1::mesh_type::Node::iterator v_it; s1mh->begin(v_it);
      typename ScalarField1::mesh_type::Node::iterator v_end; s1mh->end(v_end);
      typename ScalarField2::mesh_type::Node::iterator s_it; s2mh->begin(s_it);
      
      for( ; v_it != v_end; ++v_it, ++s_it){
	scalarField2->fdata()[*s_it] = op(scalarField1->fdata()[*v_it]);
      }
    }  
  } else {
    int max_workers = Max(Thread::numProcessors()/3, 4);
    Semaphore* thread_sema = scinew Semaphore( "scalar Average semaphore",
					       max_workers); 
    typedef typename ScalarField1::value_type Data;
    vector<ShareAssignArray3<Data> >& sdata = scalarField1->fdata();
    vector<ShareAssignArray3<Data> >::iterator vit = sdata.begin();
    vector<ShareAssignArray3<Data> >::iterator vit_end = sdata.end();
    IntVector offset( (*vit).getLowIndex() );
    for(;vit != vit_end; ++vit) {
      thread_sema->down();
      Thread *thrd = 
	scinew Thread(
		      scinew OperatorThread< Data, ScalarField2, ScalarOp >
		      ( *vit, scalarField2, offset, op, thread_sema ),
		      "scalar operator worker");
      thrd->detach();
    }
    thread_sema->down(max_workers);
    if(thread_sema) delete thread_sema;
  }
}
 
template<class ScalarField1, class ScalarField2>       
void ScalarFieldAverage::computeAverages(ScalarField1* scalarField1,
					  ScalarField2* scalarField2)
{
  // so far only node and cell centered data
  ASSERT( scalarField1->data_at() == Field::CELL ||
	  scalarField1->data_at() == Field::NODE );


  typename ScalarField1::mesh_handle_type s1mh =
    scalarField1->get_typed_mesh();
  typename ScalarField2::mesh_handle_type s2mh =
    scalarField2->get_typed_mesh();
  double ave = 0;
  int counter = 0;
  if( scalarField1->get_type_name(0) != "LevelField"){
    if( scalarField1->data_at() == Field::CELL){
      typename ScalarField1::mesh_type::Cell::iterator v_it; s1mh->begin(v_it);
      typename ScalarField1::mesh_type::Cell::iterator v_end; s1mh->end(v_end);
      typename ScalarField2::mesh_type::Cell::iterator s_it; s2mh->begin(s_it);
      for( ; v_it != v_end; ++v_it, ++s_it){
	scalarField2->fdata()[*s_it] =
	  (scalarField2->fdata()[*s_it] * scalarField1->fdata()[*v_it])/2.0;
	ave = scalarField2->fdata()[*s_it];
	++counter;
      }
    } else {
      typename ScalarField1::mesh_type::Node::iterator v_it; s1mh->begin(v_it);
      typename ScalarField1::mesh_type::Node::iterator v_end; s1mh->end(v_end);
      typename ScalarField2::mesh_type::Node::iterator s_it; s2mh->begin(s_it);
      
      for( ; v_it != v_end; ++v_it, ++s_it){
	scalarField2->fdata()[*s_it] =
	  (scalarField2->fdata()[*s_it] * scalarField1->fdata()[*v_it])/2.0;
	ave = scalarField2->fdata()[*s_it];
	++counter;
      }
    } 
  } else {
    int max_workers = Max(Thread::numProcessors()/3, 4);
    Semaphore* thread_sema = scinew Semaphore( "scalar Average semaphore",
					       max_workers); 
    Mutex mutex("average thread mutex");
    typedef typename ScalarField1::value_type Data;
    vector<ShareAssignArray3<Data> >& sdata = scalarField1->fdata();
    vector<ShareAssignArray3<Data> >::iterator vit = sdata.begin();
    vector<ShareAssignArray3<Data> >::iterator vit_end = sdata.end();
    IntVector offset( (*vit).getLowIndex() );
    for(;vit != vit_end; ++vit) {
      ++counter;
      thread_sema->down();
      Thread *thrd = 
	scinew Thread(
		      scinew AverageThread< Data, ScalarField2>
		      ( *vit, scalarField2, offset, ave, thread_sema, &mutex ),
		      "scalar Average worker");
      thrd->detach();
    }
    thread_sema->down(max_workers);
    if(thread_sema) delete thread_sema;
  }
}
  
}
#endif // __OPERATORS_SCALARFIELDAVERAGE_H__

