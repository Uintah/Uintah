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


//    File   : AttractNormals.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(AttractNormals_h)
#define AttractNormals_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Containers/Handle.h>
#include <Core/Geometry/Vector.h>

namespace SCIRun {


class Attractor : public Datatype
{
public:
  virtual void execute(Vector &v, const Point &p);

  virtual void io(Piostream &s);
};

typedef Handle<Attractor> AttractorHandle;


class AttractNormalsAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src, AttractorHandle a) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *floc,
					    const TypeDescription *msrc,
					    bool scale_p);
};


template <class MSRC, class FLOC, class FDST>
class AttractNormalsAlgoT : public AttractNormalsAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src, AttractorHandle a);
};



template <class MSRC, class FLOC, class FDST>
FieldHandle
AttractNormalsAlgoT<MSRC, FLOC, FDST>::execute(FieldHandle field_h,
					       AttractorHandle attr)
{
  MSRC *mesh = static_cast<MSRC *>(field_h->mesh().get_rep());
  int order = field_h->basis_order();
  if (order == -1) { order = 1; }
  FDST *fdst = scinew FDST(mesh, 1);

  typename FLOC::iterator bi, ei;
  mesh->begin(bi);
  mesh->end(ei);

  while (bi != ei)
  {
    Point c;

    mesh->get_center(c, *bi);

    Vector vec;
    attr->execute(vec, c);

    fdst->set_value(vec, *bi);

    ++bi;
  }
  
  return fdst;
}


template <class FSRC, class FLOC, class FDST>
class AttractNormalsScaleAlgoT : public AttractNormalsAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src, AttractorHandle a);
};



template <class FSRC, class FLOC, class FDST>
FieldHandle
AttractNormalsScaleAlgoT<FSRC, FLOC, FDST>::execute(FieldHandle field_h,
						    AttractorHandle attr)
{
  FSRC *fsrc = static_cast<FSRC *>(field_h.get_rep());
  typename FSRC::mesh_handle_type mesh = fsrc->get_typed_mesh();
  FDST *fdst = scinew FDST(mesh, field_h->basis_order());

  typename FLOC::iterator bi, ei;
  mesh->begin(bi);
  mesh->end(ei);

  while (bi != ei)
  {
    Point c;

    mesh->get_center(c, *bi);

    Vector vec;
    attr->execute(vec, c);

    typename FSRC::value_type val;
    fsrc->value(val, *bi);
    if (val == 0) { val = 1.0e-3; }
    vec = vec * val;
    
    fdst->set_value(vec, *bi);

    ++bi;
  }
  
  return fdst;
}


} // end namespace SCIRun

#endif // AttractNormals_h
