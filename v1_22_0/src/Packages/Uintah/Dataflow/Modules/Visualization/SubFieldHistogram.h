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

#ifndef SUBFIELDHISTOGRAM_H
#define SUBFIELDHISTOGRAM_H
/*
 * SubFieldHistogram.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Geom/Material.h>
#include <iostream>

namespace Uintah {
using std::cerr;
using std::endl;
using namespace SCIRun;

class SubFieldHistogram : public Module {

public:
  SubFieldHistogram(GuiContext* ctx);

  virtual ~SubFieldHistogram();

  virtual void execute();
  //  void tcl_command( TCLArgs&, void* );
  
  template <class F1, class F2>
    bool fill_histogram( F1* f1, F2* f2);

  virtual void widget_moved(bool last);

  MaterialHandle white;
private:
  
  FieldHandle field;
  FieldHandle sub_field;

  ColorMapIPort* incolormap;
  FieldIPort* infield;
  FieldIPort* in_subfield;
  GeometryOPort* ogeom;
   
  int cmap_id;  // id associated with color map...
  
  GuiInt is_fixed_;
  GuiDouble min_, max_;

  int count_[256];
  int min_i, max_i;

  double setval( double val );
};


double 
SubFieldHistogram::setval(double val)
{
     return (val - min_.get())*255/(max_.get() - min_.get());
/*   double v = (val - min_.get())*255/(max_.get() - min_.get()); */
/*   if ( v < 0 ) return 0; */
/*   else if (v > 255) return 255; */
/*   else return (int)v; */
}

template <class F1, class F2>
bool SubFieldHistogram::fill_histogram(F1* f1, F2* f2)
{
  ASSERT( f1->data_at() == F1::CELL ||
	  f1->data_at() == F1::NODE );
  

  typename F1::mesh_handle_type s1mh =
    f1->get_typed_mesh();
  typename F2::mesh_handle_type s2mh =
    f2->get_typed_mesh();

  min_i = max_i = 0;
  if( f1->data_at() == F1::CELL){
    typename F1::mesh_type::Cell::iterator v_it; s1mh->begin(v_it);
    typename F1::mesh_type::Cell::iterator v_end; s1mh->end(v_end);
    typename F2::mesh_type::Cell::iterator s_it; s2mh->begin(s_it);
    for( ; v_it != v_end; ++v_it, ++s_it){
      double value = f2->fdata()[*s_it] * f1->fdata()[*v_it];
      double val = setval( value );
      if( int(val) >= 0 || int(val) <= 255 ){
	count_[ int(val) ]++;
	min_i = ((count_[min_i] < count_[int(val)] ) ? min_i : int(val));
	max_i = ((count_[max_i] > count_[int(val)] ) ? max_i : int(val));
	if(!is_fixed_.get()){
	  min_.set((val < min_.get()) ? value:min_.get());
	  max_.set((val > max_.get()) ? value:max_.get());
	}
      }
    }
    return true;
  } else {
    typename F1::mesh_type::Node::iterator v_it; s1mh->begin(v_it);
    typename F1::mesh_type::Node::iterator v_end; s1mh->end(v_end);
    typename F2::mesh_type::Node::iterator s_it; s2mh->begin(s_it);
      
    for( ; v_it != v_end; ++v_it, ++s_it)
    {
      double value = f2->fdata()[*s_it] * f1->fdata()[*v_it];
      double val = setval( value );
      if( int(val) >= 0 || int(val) <= 255 ){
	count_[ int(val) ]++;
	min_i = ((count_[min_i] < count_[int(val)] ) ? min_i : int(val));
	max_i = ((count_[max_i] > count_[int(val)] ) ? max_i : int(val));
	if(!is_fixed_.get()){
	  min_.set((val < min_.get()) ? value:min_.get());
	  max_.set((val > max_.get()) ? value:max_.get());
	}
      }
    }
    cerr<<"min max = "<<min_.get()<<" "<<max_.get()<<endl;
    return true;
  }
}   
} // End namespace Uintah

#endif
