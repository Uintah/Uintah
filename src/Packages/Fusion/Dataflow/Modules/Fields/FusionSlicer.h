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

//    File   : FusionSlicer.h
//    Author : Michael Callahan
//    Date   : April 2002

#if !defined(FusionSlicer_h)
#define FusionSlicer_h

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/DynamicLoader.h>
#include <Core/Datatypes/QuadSurfField.h>

namespace Fusion {

using namespace SCIRun;

class FusionSlicerAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src,
			      unsigned int index,
			      int axis) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *fsrc);
};


template <class IFIELD, class OFIELD>
class FusionSlicerAlgoT : public FusionSlicerAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src,
			      unsigned int index,
			      int axis);
};


template <class IFIELD, class OFIELD>
FieldHandle
FusionSlicerAlgoT<IFIELD, OFIELD>::execute(FieldHandle field_h,
					   unsigned int index,
					   int axis)
{
  IFIELD *ifield = dynamic_cast<IFIELD *>(field_h.get_rep());
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  typename OFIELD::mesh_type *omesh = scinew typename OFIELD::mesh_type;

  const unsigned int onx = imesh->get_nx();
  const unsigned int ony = imesh->get_ny();
  const unsigned int onz = imesh->get_nz();

  unsigned int nx, ny;

  if (axis == 0)
  {
    nx = ony;
    ny = onz;
  }
  else if (axis == 1)
  {
    nx = onx;
    ny = onz;
  }
  else
  {
    nx = onx;
    ny = ony;
  }

  for (unsigned int i = 0; i < nx; i++)
  {
    for (unsigned int j = 0; j < ny; j++)
    {
      unsigned int x, y, z;
      if (axis == 0)
      {
	x = index;
	y = i;
	z = j;
      }
      else if (axis == 1)
      {
	x = i;
	y = index;
	z = j;
      }
      else
      {
	x = i;
	y = j;
	z = index;
      }
      typename IFIELD::mesh_type::Node::index_type node(x, y, z);
      Point p;
      imesh->get_center(p, node);

      omesh->add_point(p);
    }
  }

  for (unsigned int i = 0; i < nx-1; i++)
  {
    for (unsigned int j = 0; j < ny-1; j++)
    {
      typename OFIELD::mesh_type::Node::index_type a = (i+0) * ny + (j+0);
      typename OFIELD::mesh_type::Node::index_type b = (i+1) * ny + (j+0);
      typename OFIELD::mesh_type::Node::index_type c = (i+1) * ny + (j+1);
      typename OFIELD::mesh_type::Node::index_type d = (i+0) * ny + (j+1);

      omesh->add_quad(a, b, c, d);
    }
  }

  OFIELD *ofield = scinew OFIELD(omesh, Field::NODE);

  unsigned int counter = 0;
  for (unsigned int i = 0; i < nx; i++)
  {
    for (unsigned int j = 0; j < ny; j++)
    {
      unsigned int x, y, z;
      if (axis == 0)
      {
	x = index;
	y = i;
	z = j;
      }
      else if (axis == 1)
      {
	x = i;
	y = index;
	z = j;
      }
      else
      {
	x = i;
	y = j;
	z = index;
      }
      typename IFIELD::mesh_type::Node::index_type node(x, y, z);
      typename IFIELD::value_type v;

      ifield->value(v, node);
      
      typename OFIELD::mesh_type::Node::index_type onode(counter);
      ofield->set_value(v, onode);
      counter++;
    }
  }

  return ofield;
}

} // end namespace SCIRun

#endif // FusionSlicer_h
