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

//    File   : Centroids.h
//    Author : Michael Callahan
//    Date   : June 2002

#if !defined(Centroids_h)
#define Centroids_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/PointCloudField.h>

namespace SCIRun {

class CentroidsAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc);
};


template <class FIELD>
class CentroidsAlgoT : public CentroidsAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src);
};


template <class FIELD>
FieldHandle
CentroidsAlgoT<FIELD>::execute(FieldHandle field_h)
{
  FIELD *ifield = dynamic_cast<FIELD *>(field_h.get_rep());
  typename FIELD::mesh_handle_type mesh = ifield->get_typed_mesh();

  PointCloudMeshHandle pcm(scinew PointCloudMesh);

  typename FIELD::mesh_type::Elem::iterator bi, ei;
  mesh->begin(bi);
  mesh->end(ei);

  Point p;
  while (bi != ei)
  {
    mesh->get_center(p, *bi);
    pcm->add_node(p);
    ++bi;
  }

  return scinew PointCloudField<double>(pcm, Field::NODE);
}


} // end namespace SCIRun

#endif // Centroids_h
