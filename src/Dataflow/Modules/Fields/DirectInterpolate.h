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

//    File   : DirectInterpolate.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(DirectInterpolate_h)
#define DirectInterpolate_h

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/DynamicLoader.h>

namespace SCIRun {

//! DirectInterpBaseBase supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! DirectInterpBaseBase from the DynamicAlgoBase they will have a pointer to.
class DirectInterpScalarAlgoBase : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle f, ScalarFieldInterface *sfi) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *field,
				       const TypeDescription *element);
};


template <class Fld, class Loc>
class DirectInterpScalarAlgo : public DirectInterpScalarAlgoBase
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle f, ScalarFieldInterface *sfi);
};


template <class Fld, class Loc>
FieldHandle
DirectInterpScalarAlgo<Fld, Loc>::execute(FieldHandle fldhandle,
					  ScalarFieldInterface *sfi)
{
  Fld *fld2 = dynamic_cast<Fld *>(fldhandle.get_rep());
  if (!fld2->is_scalar()) { return 0; }
  Fld *fld = fld2->clone();
  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();

  typename Loc::iterator itr, itr_end;
  mesh->begin(itr);
  mesh->end(itr_end);
  while (itr != itr_end)
  {
    Point p;
    mesh->get_center(p, *itr);

    double val;
    if (!sfi->interpolate(val, p))
    {
      sfi->find_closest(val, p);
    }
    fld->set_value((typename Fld::value_type)val, *itr);
      

    ++itr;
  }

  FieldHandle ofh(fld);
  return ofh;
}



class DirectInterpVectorAlgoBase : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle f, VectorFieldInterface *vfi) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *field,
				       const TypeDescription *element);
};


template <class Fld, class Loc>
class DirectInterpVectorAlgo : public DirectInterpVectorAlgoBase
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle f, VectorFieldInterface *vfi);
};


template <class Fld, class Loc>
FieldHandle
DirectInterpVectorAlgo<Fld, Loc>::execute(FieldHandle fldhandle,
					  VectorFieldInterface *vfi)
{
  Fld *fld2 = dynamic_cast<Fld *>(fldhandle.get_rep());
  if (!fld2->is_scalar()) { return 0; }
  Fld *fld = fld2->clone();
  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();

  typename Loc::iterator itr, itr_end;
  mesh->begin(itr);
  mesh->end(itr_end);
  while (itr != itr_end)
  {
    Point p;
    mesh->get_center(p, *itr);

    Vector val;
    if (!vfi->interpolate(val, p))
    {
      vfi->find_closest(val, p);
    }
    fld->set_value(val, *itr);

    ++itr;
  }

  FieldHandle ofh(fld);
  return ofh;
}


class DirectInterpTensorAlgoBase : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle f, TensorFieldInterface *tfi) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *field,
				       const TypeDescription *element);
};


template <class Fld, class Loc>
class DirectInterpTensorAlgo : public DirectInterpTensorAlgoBase
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle f, TensorFieldInterface *tfi);
};


template <class Fld, class Loc>
FieldHandle
DirectInterpTensorAlgo<Fld, Loc>::execute(FieldHandle fldhandle,
					  TensorFieldInterface *tfi)
{
  Fld *fld2 = dynamic_cast<Fld *>(fldhandle.get_rep());
  if (!fld2->is_scalar()) { return 0; }
  Fld *fld = fld2->clone();
  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();

  typename Loc::iterator itr, itr_end;
  mesh->begin(itr);
  mesh->end(itr_end);
  while (itr != itr_end)
  {
    Point p;
    mesh->get_center(p, *itr);

    Tensor val;
    if (!tfi->interpolate(val, p))
    {
      tfi->find_closest(val, p);
    }
    fld->set_value(val, *itr);

    ++itr;
  }

  FieldHandle ofh(fld);
  return ofh;
}


} // end namespace SCIRun

#endif // DirectInterpolate_h
