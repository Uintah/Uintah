/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


//    File   : MapDataToMeshCoord.h
//    Author : David Weinstein
//    Date   : June 2002

#if !defined(MapDataToMeshCoord_h)
#define MapDataToMeshCoord_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
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
