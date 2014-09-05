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


//    File   : TransformMesh.h
//    Author : Michael Callahan
//    Date   : January 2005

#if !defined(TransformMesh_h)
#define TransformMesh_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>

namespace SCIRun {

class TransformMeshAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src) = 0;

  virtual string identify() = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    string function,
					    int hashoffset);
};


template <class FIELD>
class TransformMeshAlgoT : public TransformMeshAlgo
{
public:
  virtual void function(Point &result, const Point &p) = 0;

  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src);
};


template <class FIELD>
FieldHandle
TransformMeshAlgoT<FIELD>::execute(FieldHandle field_h)
{
  FIELD *ifield = dynamic_cast<FIELD *>(field_h.get_rep());
  FIELD *ofield = ifield->clone();
  ofield->mesh_detach();
  typename FIELD::mesh_handle_type mesh = ofield->get_typed_mesh();

  typename FIELD::mesh_type::Node::iterator ibi, iei;
  mesh->begin(ibi);
  mesh->end(iei);

  Point p;
  Point result;
  while (ibi != iei)
  {
    mesh->get_center(p, *ibi);
    result = p;
    function(result, p);
    mesh->set_point(result, *ibi);

    ++ibi;
  }

  return ofield;
}


} // end namespace SCIRun

#endif // TransformMesh_h
