/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

//    File   : ScalarMinMax.h
//    Author : Kurt Zimmerman
//    Date   : March 2004

#if !defined(SCALARMINMAX_H)
#define SCALARMINMAX_H

#include <Core/Datatypes/Datatype.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Network/Module.h>

namespace Uintah {
using namespace::SCIRun;

using SCIRun::LatVolMesh;

class ScalarMinMaxAlgoCount : public DynamicAlgoBase
{
public:
  virtual void execute(FieldHandle src, double& min_val, IntVector& min_idx,
		       int& n_mins, double& max_val, IntVector& max_idx,
		       int& n_maxs) =  0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *msrc);
};


template <class FIELD>
class ScalarMinMaxAlgoCountT : public ScalarMinMaxAlgoCount
{
public:
  typedef typename FIELD::value_type       value_type;

  //! virtual interface. 
  virtual void execute(FieldHandle src, double& min_val, IntVector& min_idx,
		       int& n_mins, double& max_val, IntVector& max_idx,
		       int& n_maxs);
};

template <class FIELD>
void 
ScalarMinMaxAlgoCountT<FIELD>::execute(FieldHandle field,
				       double& min_val, IntVector& min_idx,
				       int& n_mins, double& max_val,
				       IntVector& max_idx, int& n_maxs)
{
  if( LatVolField< value_type >* fld = 
      dynamic_cast<LatVolField< value_type >* >(field.get_rep())){
   
    typename FIELD::mesh_type *m = fld->get_typed_mesh().get_rep();

    IntVector offset(0,0,0);
    fld->get_property( "offset", offset);
      

    if( field->basis_order() == 0 ){
      typename FIELD::mesh_type::Cell::iterator iter; m->begin( iter );
      typename FIELD::mesh_type::Cell::iterator iter_end; m->end( iter_end );

      for( ; iter != iter_end; ++iter){
	double value = fld->fdata()[*iter];
	if( min_val >= value){
	  if( min_val==value ){
	    n_mins++;
	  } else {
	    min_val = value;
	    n_mins = 1;
	    min_idx.x(iter.i_ + offset.x());
	    min_idx.y(iter.j_ + offset.y());
	    min_idx.z(iter.k_ + offset.z());
	  }
	}
	if( max_val <= value) {
	  if( max_val == value ){
	    n_maxs++;
	  } else {
	    max_val = value;
	    n_maxs = 1;
	    max_idx.x(iter.i_ + offset.x());
	    max_idx.y(iter.j_ + offset.y());
	    max_idx.z(iter.k_ + offset.z());
	  }
	}
      }
    } else {
      typename FIELD::mesh_type::Node::iterator iter;  m->begin( iter );
      typename FIELD::mesh_type::Node::iterator iter_end; m->end( iter_end );

      for( ; iter != iter_end; ++iter){
	double value  = fld->fdata()[*iter];
	if( min_val >= value){
	  if( min_val==value ){
	    n_mins++;
	  } else {
	    min_val = value;
	    n_mins = 1;
	    min_idx.x(iter.i_ + offset.x());
	    min_idx.y(iter.j_ + offset.y());
	    min_idx.z(iter.k_ + offset.z());
	  }
	}
	if( max_val <= value) {
	  if( max_val == value ){
	    n_maxs++;
	  } else {
	    max_val = value;
	    n_maxs = 1;
	    max_idx.x(iter.i_ + offset.x());
	    max_idx.y(iter.j_ + offset.y());
	    max_idx.z(iter.k_ + offset.z());
	  }
	}
      }
    }
  } else {
  cerr<<"Not a Lattice type---should not be here\n";
  }
}

class ScalarMinMax : public Module {

public:
  ScalarMinMax(GuiContext* ctx);
  virtual ~ScalarMinMax();
  virtual void execute();
private:
  GuiString gui_min_data_;
  GuiString gui_max_data_;
  GuiString gui_min_index_;
  GuiString gui_max_index_;
  GuiString gui_min_values_;
  GuiString gui_max_values_;

  int              generation_;

  void clear_vals();
  void update_input_attributes(FieldHandle);

  ProgressReporter my_reporter_;
  template<class Reporter> bool get_info( Reporter *, FieldHandle);
};

template<class Reporter>
bool 
ScalarMinMax::get_info( Reporter * reporter, FieldHandle f)
{
  const TypeDescription *td = f->get_type_description();
  CompileInfoHandle ci = ScalarMinMaxAlgoCount::get_compile_info(td);
  LockingHandle<ScalarMinMaxAlgoCount> algo;
  if (!DynamicCompilation::compile(ci, algo, reporter)){
    reporter->error("ScalarMinMax cannot work on this Field");
    return false;
  }

  IntVector min_idx(MAXINT, MAXINT, MAXINT);
  IntVector max_idx(-MAXINT, -MAXINT, -MAXINT);
  double min_val = MAXDOUBLE;
  double max_val = -MAXDOUBLE;
  int n_min_vals = 0;
  int n_max_vals = 0;

  algo->execute(f, min_val, min_idx, n_min_vals,
		max_val, max_idx, n_max_vals);

  gui_min_data_.set( to_string( min_val ));
  gui_max_data_.set( to_string( max_val ));
  gui_min_values_.set( to_string( n_min_vals ));
  gui_max_values_.set( to_string( n_max_vals ));

  ostringstream min_os, max_os;
  min_os<<min_idx;
  max_os<<max_idx;
  gui_min_index_.set( min_os.str() );
  gui_max_index_.set( max_os.str() );
  
  return true;
}  

} // end namespace Uintah

#endif
