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

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Geometry/Tensor.h>

namespace SCIRun {

//! DirectInterpBaseBase supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! DirectInterpBaseBase from the DynamicAlgoBase they will have a pointer to.
class DirectInterpScalarAlgoBase : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle f, ScalarFieldInterface *sfi,
			      bool interp, bool closest, double dist) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *field,
				       const TypeDescription *element);
};


template <class Fld, class Loc>
class DirectInterpScalarAlgo : public DirectInterpScalarAlgoBase
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle f, ScalarFieldInterface *sfi,
			      bool interp, bool closest, double dist);
};


template <class Fld, class Loc>
FieldHandle
DirectInterpScalarAlgo<Fld, Loc>::execute(FieldHandle fldhandle,
					  ScalarFieldInterface *sfi,
					  bool interp, bool closest,
					  double dist)
{
  Fld *fld2 = dynamic_cast<Fld *>(fldhandle.get_rep());
  if (!fld2->query_scalar_interface()) { return 0; }
  Fld *fld = fld2->clone();
  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();

  typename Loc::iterator itr, itr_end;
  mesh->begin(itr);
  mesh->end(itr_end);

  double val = 0;
  Point p;

  while (itr != itr_end)
  {
    mesh->get_center(p, *itr);

    if (interp && sfi->interpolate(val, p))
    {
      fld->set_value((typename Fld::value_type)val, *itr);
    }
    else if (closest)
    {
      if (sfi->find_closest(val, p) < dist)
      {
	fld->set_value((typename Fld::value_type)val, *itr);
      }
    }

    ++itr;
  }

  FieldHandle ofh(fld);
  return ofh;
}



class DirectInterpVectorAlgoBase : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle f, VectorFieldInterface *vfi,
			      bool interp, bool closest, double dist) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *field,
				       const TypeDescription *element);
};


template <class Fld, class Loc>
class DirectInterpVectorAlgo : public DirectInterpVectorAlgoBase
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle f, VectorFieldInterface *vfi,
			      bool interp, bool closest, double dist);
};


template <class Fld, class Loc>
FieldHandle
DirectInterpVectorAlgo<Fld, Loc>::execute(FieldHandle fldhandle,
					  VectorFieldInterface *vfi,
					  bool interp, bool closest,
					  double dist)
{
  Fld *fld2 = dynamic_cast<Fld *>(fldhandle.get_rep());
  if (!fld2->query_vector_interface()) { return 0; }
  Fld *fld = fld2->clone();
  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();

  typename Loc::iterator itr, itr_end;
  mesh->begin(itr);
  mesh->end(itr_end);

  Point p;
  Vector val;

  while (itr != itr_end)
  {
    mesh->get_center(p, *itr);

    if (interp && vfi->interpolate(val, p))
    {
      fld->set_value((typename Fld::value_type)val, *itr);
    }
    else if (closest)
    {
      if (vfi->find_closest(val, p) < dist)
      {
	fld->set_value((typename Fld::value_type)val, *itr);
      }
    }

    ++itr;
  }

  FieldHandle ofh(fld);
  return ofh;
}


class DirectInterpTensorAlgoBase : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle f, TensorFieldInterface *tfi,
			      bool interp, bool closest, double dist) = 0;
  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *field,
				       const TypeDescription *element);
};


template <class Fld, class Loc>
class DirectInterpTensorAlgo : public DirectInterpTensorAlgoBase
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle f, TensorFieldInterface *tfi,
			      bool interp, bool closest, double dist);
};


template <class Fld, class Loc>
FieldHandle
DirectInterpTensorAlgo<Fld, Loc>::execute(FieldHandle fldhandle,
					  TensorFieldInterface *tfi,
					  bool interp, bool closest,
					  double dist)
{
  Fld *fld2 = dynamic_cast<Fld *>(fldhandle.get_rep());
  if (!fld2->query_tensor_interface()) { return 0; }
  Fld *fld = fld2->clone();
  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();

  typename Loc::iterator itr, itr_end;
  mesh->begin(itr);
  mesh->end(itr_end);

  Point p;
  Tensor val;

  while (itr != itr_end)
  {
    mesh->get_center(p, *itr);

    if (interp && tfi->interpolate(val, p))
    {
      fld->set_value((typename Fld::value_type)val, *itr);
    }
    else if (closest)
    {
      if (tfi->find_closest(val, p) < dist)
      {
	fld->set_value((typename Fld::value_type)val, *itr);
      }
    }

    ++itr;
  }

  FieldHandle ofh(fld);
  return ofh;
}


} // end namespace SCIRun

#endif // DirectInterpolate_h





