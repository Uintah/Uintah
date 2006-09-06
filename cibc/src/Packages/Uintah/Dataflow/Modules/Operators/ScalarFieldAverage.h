#ifndef __OPERATORS_SCALARFIELDAVERAGE_H__
#define __OPERATORS_SCALARFIELDAVERAGE_H__


#include "ScalarOperatorFunctors.h"
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using std::string;
using namespace SCIRun;

class ScalarFieldAverageAlgo: public DynamicAlgoBase
{
public:

  virtual void fillField(FieldHandle sfh, FieldHandle afh) = 0;
  virtual void averageField(FieldHandle sfh, FieldHandle afh) = 0;

  static CompileInfoHandle get_compile_info(const SCIRun::TypeDescription *td1,
                                            const SCIRun::TypeDescription *td2);
};

template <class ScalarField1, class ScalarField2>
class ScalarFieldAverageAlgoT: public ScalarFieldAverageAlgo
{
public:
  virtual void fillField(FieldHandle fh, FieldHandle afh);
  virtual void averageField(FieldHandle fh, FieldHandle afh);

private:
  void initField(ScalarField1* scalarField1,
                 ScalarField2* scalarField2);
  
  template<class ScalarOp>
  void computeScalars(ScalarField1* scalarField1,
                      ScalarField2* scalarField2,
                      ScalarOp op);
  void computeAverages(ScalarField1* scalarField1,
                       ScalarField2* scalarField2);
  
};

    

template<class ScalarField1, class ScalarField2>
void
ScalarFieldAverageAlgoT<ScalarField1,
                        ScalarField2>::fillField(FieldHandle sfh, 
                                                 FieldHandle afh)
{
  ScalarField1 *inField = (ScalarField1 *)sfh.get_rep();
  ScalarField2 *aveField = (ScalarField1 *)afh.get_rep();
  initField( inField, aveField);
  computeScalars( inField, aveField, NoOp() ); 
  computeAverages( inField, aveField); 
}

template<class ScalarField1, class ScalarField2>
void
ScalarFieldAverageAlgoT<ScalarField1,
                        ScalarField2>::averageField(FieldHandle sfh, 
                                                    FieldHandle afh)
{
  ScalarField1 *inField = (ScalarField1 *)sfh.get_rep();
  ScalarField2 *aveField = (ScalarField1 *)afh.get_rep();
  computeAverages( inField, aveField); 
}

template<class ScalarField1, class ScalarField2>
void 
ScalarFieldAverageAlgoT<ScalarField1,
                        ScalarField2>::initField(ScalarField1* scalarField1,
                                                 ScalarField2* scalarField2)
{
  ASSERT( scalarField1->basis_order() == 0 ||
	  scalarField1->basis_order() == 1 );

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


template<class ScalarField1, class ScalarField2>
    template<class ScalarOp>
void 
ScalarFieldAverageAlgoT<ScalarField1,ScalarField2>::computeScalars(
                                             ScalarField1* scalarField1,
                                             ScalarField2* scalarField2,
                                             ScalarOp op)
{
  // so far only node and cell centered data
  ASSERT( scalarField1->basis_order() == 0 ||
	  scalarField1->basis_order() == 1 );


  typename ScalarField1::mesh_handle_type s1mh =
    scalarField1->get_typed_mesh();
  typename ScalarField2::mesh_handle_type s2mh =
    scalarField2->get_typed_mesh();
 
  if( scalarField1->basis_order() == 0){
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
void 
ScalarFieldAverageAlgoT<ScalarField1,ScalarField2>::computeAverages(
                                          ScalarField1* scalarField1,
					  ScalarField2* scalarField2)
{
  // so far only node and cell centered data
  ASSERT( scalarField1->basis_order() == 0 ||
	  scalarField1->basis_order() == 1 );


  typename ScalarField1::mesh_handle_type s1mh =
    scalarField1->get_typed_mesh();
  typename ScalarField2::mesh_handle_type s2mh =
    scalarField2->get_typed_mesh();
  //double ave = 0;
  int counter = 0;
  if( scalarField1->basis_order() == 0){
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

