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
  virtual FieldHandle execute(FieldHandle src, double scale) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd);
};


template< class FIELD >
class FusionSlicePlotAlgoT : public FusionSlicePlotAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src, double scale);
};


template< class FIELD >
FieldHandle
FusionSlicePlotAlgoT<FIELD>::execute(FieldHandle field_h, double scale)
{
  FIELD *ifield = (FIELD *) field_h.get_rep();

  ScalarFieldInterfaceHandle sfi = ifield->query_scalar_interface();

  if( sfi.get_rep() ) {

    FIELD *ifield = (FIELD *) field_h.get_rep();
    FIELD *ofield = ifield->clone();

    ofield->mesh_detach();

    typename FIELD::mesh_type *omesh = ofield->get_typed_mesh().get_rep();

    typename FIELD::fdata_type::iterator in  = ofield->fdata().begin();
    typename FIELD::fdata_type::iterator end = ofield->fdata().end();

    typename FIELD::mesh_type::Node::iterator inodeItr;

    omesh->begin( inodeItr );

    omesh->synchronize( Mesh::NORMALS_E );

    Point pt;
    Vector vec;

    pair<double, double> minmax;
    sfi->compute_min_max(minmax.first, minmax.second);
 
    scale /= (minmax.second-minmax.first);

    while (in != end) {

      // Get the orginal point on the mesh.
      omesh->get_point(pt, *inodeItr);
      
      // Get the normal from the orginal surface since it should be planar.
      omesh->get_normal(vec, *inodeItr);      

      // Normalize then scale the offset value before adding to the point.
      pt += ( (*in - minmax.first) * scale ) * vec;

      omesh->set_point(pt, *inodeItr);

      ++in;

      ++inodeItr;
    }

    return FieldHandle( ofield );
  } else
    return NULL;

}

} // end namespace SCIRun

#endif // FusionSlicePlot_h
