#ifndef __OPERATORS_SCALARFIELDAVERAGE_H__
#define __OPERATORS_SCALARFIELDAVERAGE_H__

#include "OperatorThread.h"
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/LatVolField.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using std::string;
using namespace SCIRun;

  
class ScalarFieldAverage: public Module {
public:
  ScalarFieldAverage(GuiContext* ctx);
  virtual ~ScalarFieldAverage() {}
    
  virtual void execute(void);
private:
    
  GuiDouble t0_;
  GuiDouble t1_;
  GuiInt tsteps_;
    
  FieldIPort *in;
  FieldOPort *sfout;
    
  LatVolField<double> *aveField;
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
  fmh->set_ni(smh->get_ni());
  fmh->set_nj(smh->get_nj());
  fmh->set_nk(smh->get_nk());
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
  //double ave = 0;
  int counter = 0;
  if( scalarField1->data_at() == Field::CELL){
    typename ScalarField1::mesh_type::Cell::iterator v_it; s1mh->begin(v_it);
    typename ScalarField1::mesh_type::Cell::iterator v_end; s1mh->end(v_end);
    typename ScalarField2::mesh_type::Cell::iterator s_it; s2mh->begin(s_it);
    for( ; v_it != v_end; ++v_it, ++s_it){
      scalarField2->fdata()[*s_it] =
	(scalarField2->fdata()[*s_it] + scalarField1->fdata()[*v_it])/2.0;
      //ave = scalarField2->fdata()[*s_it];
      ++counter;
    }
  } else {
    typename ScalarField1::mesh_type::Node::iterator v_it; s1mh->begin(v_it);
    typename ScalarField1::mesh_type::Node::iterator v_end; s1mh->end(v_end);
    typename ScalarField2::mesh_type::Node::iterator s_it; s2mh->begin(s_it);
      
    for( ; v_it != v_end; ++v_it, ++s_it){
      scalarField2->fdata()[*s_it] =
	(scalarField2->fdata()[*s_it] + scalarField1->fdata()[*v_it])/2.0;
      //ave = scalarField2->fdata()[*s_it];
      ++counter;
    }
  } 
}
  
}
#endif // __OPERATORS_SCALARFIELDAVERAGE_H__

