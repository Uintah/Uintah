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

//    File   : FusionSlicePlot.h
//    Author : Michael Callahan
//    Date   : April 2002

#if !defined(FusionSlicePlot_h)
#define FusionSlicePlot_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/QuadSurfField.h>
#include <Core/Datatypes/FieldInterface.h>

namespace Fusion {

using namespace SCIRun;

class FusionSlicePlotAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src,
			      double scale) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *ftd,
				       const TypeDescription *ttd);
};


#ifdef __sgi
template< class FIELD, class TYPE >
#else
template< template<class> class FIELD, class TYPE >
#endif
class FusionSlicePlotAlgoT : public FusionSlicePlotAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src, double scale);
};


#ifdef __sgi
template< class FIELD, class TYPE >
#else
template< template<class> class FIELD, class TYPE >
#endif
FieldHandle
FusionSlicePlotAlgoT<FIELD, TYPE>::execute(FieldHandle field_h,
					   double scale)
{
  FIELD<TYPE> *ifield = (FIELD<TYPE> *) field_h.get_rep();

  ScalarFieldInterface *sfi = ifield->query_scalar_interface();

  if( sfi ) {

    typename FIELD<TYPE>::mesh_handle_type imesh = ifield->get_typed_mesh();
    typename FIELD<double>::mesh_handle_type omesh = imesh;
    omesh.detach();
    
    FIELD<double> *ofield = scinew FIELD<double>(omesh, ifield->data_at());


    
    typename FIELD<TYPE>::fdata_type::iterator in  = ifield->fdata().begin();
    typename FIELD<TYPE>::fdata_type::iterator end = ifield->fdata().end();
    typename FIELD<double>::fdata_type::iterator out = ofield->fdata().begin();

    typename FIELD<TYPE>::mesh_type::Node::index_type node = 0;
    unsigned int counter = 0;

    Point pt;
    Vector vec;

    pair<double, double> minmax;
    
    if ( !ifield->get_property("minmax", minmax)) {
      sfi->compute_min_max(minmax.first, minmax.second);

      // cache this potentially expensive to compute value.
      ifield->set_property("minmax", minmax, true);
    }
 
    scale /= minmax.second;

    while (in != end) {

      *out = *in;  // Copy the data.

      // Get the normal from the orginal surface since it should be planar.
      imesh->get_normal(vec, node);      

      // Get the orginal point on the mesh.
      omesh->get_point(pt, node);
      
      // Normalize then scale the offset value before adding to the point.
      pt += ( (*in - minmax.first) * scale ) * vec;

      omesh->set_point(pt, node);

      ++in; ++out;

      node = ++counter;

    }

    return FieldHandle( ofield );
  }
  else {
    cerr << "FusionSlicePlot - Only availible for Scalar data" << endl;

    return NULL;
  }
}

} // end namespace SCIRun

#endif // FusionSlicePlot_h
