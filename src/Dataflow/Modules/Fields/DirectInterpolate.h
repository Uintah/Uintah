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
  virtual FieldHandle execute(MeshHandle m, Field::data_location at,
			      ScalarFieldInterface *sfi,
			      bool interp, bool closest, double dist) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *fdst,
					    const TypeDescription *element);
};


template <class Fld, class Loc>
class DirectInterpScalarAlgo : public DirectInterpScalarAlgoBase
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(MeshHandle m, Field::data_location at,
			      ScalarFieldInterface *sfi,
			      bool interp, bool closest, double dist);
};


template <class Fld, class Loc>
FieldHandle
DirectInterpScalarAlgo<Fld, Loc>::execute(MeshHandle meshhandle,
					  Field::data_location at,
					  ScalarFieldInterface *sfi,
					  bool interp, bool closest,
					  double dist)
{
  typename Fld::mesh_handle_type mesh =
    dynamic_cast<typename Fld::mesh_type *>(meshhandle.get_rep());
  Fld *fld = scinew Fld(mesh, at);

  if (at == Field::NODE) mesh->synchronize(Mesh::NODES_E);
  else if (at == Field::CELL) mesh->synchronize(Mesh::CELLS_E);
  else if (at == Field::FACE) mesh->synchronize(Mesh::FACES_E);
  else if (at == Field::EDGE) mesh->synchronize(Mesh::EDGES_E);

  typename Loc::iterator itr, itr_end;
  mesh->begin(itr);
  mesh->end(itr_end);

  double val = 0.0;
  Point p;
  while (itr != itr_end)
  {
    mesh->get_center(p, *itr);

    if (interp && sfi->interpolate(val, p))
    {
      fld->set_value((typename Fld::value_type)val, *itr);
    } 
    else if (closest && sfi->find_closest(val, p) < dist)
    {
      fld->set_value((typename Fld::value_type)val, *itr);
    }
    else
    {
      fld->set_value((typename Fld::value_type)0, *itr);
    }
    ++itr;
  }
  
  FieldHandle ofh(fld);
  return ofh;
}



class DirectInterpVectorAlgoBase : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(MeshHandle m, Field::data_location at,
			      VectorFieldInterface *vfi,
			      bool interp, bool closest, double dist) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *fdst,
					    const TypeDescription *element);
};


template <class Fld, class Loc>
class DirectInterpVectorAlgo : public DirectInterpVectorAlgoBase
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(MeshHandle m, Field::data_location at,
			      VectorFieldInterface *vfi,
			      bool interp, bool closest, double dist);
};


template <class Fld, class Loc>
FieldHandle
DirectInterpVectorAlgo<Fld, Loc>::execute(MeshHandle meshhandle,
					  Field::data_location at,
					  VectorFieldInterface *vfi,
					  bool interp, bool closest,
					  double dist)
{
  typename Fld::mesh_handle_type mesh =
    dynamic_cast<typename Fld::mesh_type *>(meshhandle.get_rep());
  Fld *fld = scinew Fld(mesh, at);

  if (at == Field::NODE) mesh->synchronize(Mesh::NODES_E);
  else if (at == Field::CELL) mesh->synchronize(Mesh::CELLS_E);
  else if (at == Field::FACE) mesh->synchronize(Mesh::FACES_E);
  else if (at == Field::EDGE) mesh->synchronize(Mesh::EDGES_E);

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
    else if (closest && vfi->find_closest(val, p) < dist)
    {
      fld->set_value((typename Fld::value_type)val, *itr);
    }
    else
    {
      fld->set_value((typename Fld::value_type)0, *itr);
    }

    ++itr;
  }

  FieldHandle ofh(fld);
  return ofh;
}


class DirectInterpTensorAlgoBase : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(MeshHandle m, Field::data_location at,
			      TensorFieldInterface *tfi,
			      bool interp, bool closest, double dist) = 0;
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *fdst,
					    const TypeDescription *element);
};


template <class Fld, class Loc>
class DirectInterpTensorAlgo : public DirectInterpTensorAlgoBase
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(MeshHandle m, Field::data_location at,
			      TensorFieldInterface *tfi,
			      bool interp, bool closest, double dist);
};


template <class Fld, class Loc>
FieldHandle
DirectInterpTensorAlgo<Fld, Loc>::execute(MeshHandle meshhandle,
					  Field::data_location at,
					  TensorFieldInterface *tfi,
					  bool interp, bool closest,
					  double dist)
{
  typename Fld::mesh_handle_type mesh =
    dynamic_cast<typename Fld::mesh_type *>(meshhandle.get_rep());
  Fld *fld = scinew Fld(mesh, at);

  if (at == Field::NODE) mesh->synchronize(Mesh::NODES_E);
  else if (at == Field::CELL) mesh->synchronize(Mesh::CELLS_E);
  else if (at == Field::FACE) mesh->synchronize(Mesh::FACES_E);
  else if (at == Field::EDGE) mesh->synchronize(Mesh::EDGES_E);

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
    else if (closest && tfi->find_closest(val, p) < dist)
    {
      fld->set_value((typename Fld::value_type)val, *itr);
    }
    else
    {
      fld->set_value((typename Fld::value_type)0, *itr);
    }

    ++itr;
  }

  FieldHandle ofh(fld);
  return ofh;
}


} // end namespace SCIRun

#endif // DirectInterpolate_h





