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

//    File   : MapDataToMeshCoord.h
//    Author : David Weinstein
//    Date   : June 2002

#if !defined(MapDataToMeshCoord_h)
#define MapDataToMeshCoord_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Math/function.h>
#include <Core/Datatypes/Field.h>

namespace SCIRun {

class MapDataToMeshCoordAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src, int coord) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc);
};


template <class FIELD>
class MapDataToMeshCoordAlgoT : public MapDataToMeshCoordAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src, int coord);
};


template <class FIELD>
FieldHandle
MapDataToMeshCoordAlgoT<FIELD>::execute(FieldHandle field_h, int coord)
{
  FIELD *ifield = dynamic_cast<FIELD *>(field_h.get_rep());
  FIELD *ofield = ifield->clone();

  ofield->mesh_detach();

  typename FIELD::mesh_type *omesh = ofield->get_typed_mesh().get_rep();


  typename FIELD::mesh_type::Node::iterator bn, en;
  omesh->begin(bn);
  omesh->end(en);
  typename FIELD::fdata_type::iterator di = ofield->fdata().begin();

  if (coord == 3) {
    omesh->synchronize(Mesh::NORMALS_E);
  }

  while (bn != en)
  {
    typename FIELD::value_type val = *di;
    double tmp = (double)val;
    Point p;
    omesh->get_point(p, *bn);
    if (coord == 0) p.x(tmp);
    else if (coord == 1) p.y(tmp);
    else if (coord == 2) p.z(tmp);
    else {
      Vector n;
      omesh->get_normal(n, *bn);
      p += n*tmp;
    }
    omesh->set_point(p, *bn);
    ++bn;
    ++di;
  }

  return ofield;
}


} // end namespace SCIRun

#endif // MapDataToMeshCoord_h
