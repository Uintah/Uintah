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
  Field::data_location at = field_h->data_at();
  if (at == Field::NONE) { at = Field::NODE; }
  FDST *fdst = scinew FDST(mesh, at);

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
  FDST *fdst = scinew FDST(mesh, field_h->data_at());

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
