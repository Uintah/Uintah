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

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/StructQuadSurfField.h>

namespace Fusion {

using namespace SCIRun;

class FusionSlicerAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src,
			      unsigned int index,
			      int axis) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *ftd,
				       const TypeDescription *ttd);
};


#ifdef __sgi
template< class FIELD, class TYPE >
#else
template< template<class> class FIELD, class TYPE >
#endif
class FusionSlicerAlgoT : public FusionSlicerAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src,
			      unsigned int index,
			      int axis);
};


#ifdef __sgi
template< class FIELD, class TYPE >
#else
template< template<class> class FIELD, class TYPE >
#endif
FieldHandle
FusionSlicerAlgoT<FIELD, TYPE>::execute(FieldHandle field_h,
					unsigned int index,
					int axis)
{
  FIELD<TYPE> *ifield = (FIELD<TYPE> *) field_h.get_rep();

  typename FIELD<TYPE>::mesh_handle_type imesh = ifield->get_typed_mesh();

  const unsigned int onx = imesh->get_nx();
  const unsigned int ony = imesh->get_ny();
  const unsigned int onz = imesh->get_nz();

  unsigned int nx, ny;

  if (axis == 0) {
    nx = ony;
    ny = onz;
  }
  else if (axis == 1) {
    nx = onx;
    ny = onz;
  }
  else {
    nx = onx;
    ny = ony;
  }

  typename StructQuadSurfField<TYPE>::mesh_type *omesh =
    scinew typename StructQuadSurfField<TYPE>::mesh_type(nx,ny);

  typename FIELD<TYPE>::mesh_type::Node::index_type node;
  typename StructQuadSurfField<TYPE>::mesh_type::Node::index_type onode;
  Point p;

  for (unsigned int j=0; j < ny; j++) {
    onode.j_ = j;
    for (unsigned int i=0; i < nx; i++) {
      onode.i_ = i;

      if (axis == 0) {
	node.i_ = index;
	node.j_ = i;
	node.k_ = j;
      }
      else if (axis == 1) {
	node.i_ = i;
	node.j_ = index;
	node.k_ = j;
      }
      else {
	node.i_ = i;
	node.j_ = j;
	node.k_ = index;
      }

      imesh->get_center(p, node);
      omesh->set_point(onode, p);
    }
  }

  StructQuadSurfField<TYPE> *ofield = scinew StructQuadSurfField<TYPE>(omesh, Field::NODE);

  typename FIELD<TYPE>::value_type v;

  for (unsigned int j = 0; j < ny; j++) {
    onode.j_ = j;
    for (unsigned int i = 0; i < nx; i++) {
      onode.i_ = i;

      if (axis == 0) {
	node.i_ = index;
	node.j_ = i;
	node.k_ = j;
      }
      else if (axis == 1) {
	node.i_ = i;
	node.j_ = index;
	node.k_ = j;
      }
      else {
	node.i_ = i;
	node.j_ = j;
	node.k_ = index;
      }

      ifield->value(v, node);
      ofield->set_value(v, onode);
    }
  }

  return ofield;
}

} // end namespace SCIRun

#endif // FusionSlicer_h




