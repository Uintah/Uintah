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

//    File   : MaskLattice.h
//    Author : Michael Callahan
//    Date   : May 2003

#if !defined(MaskLattice_h)
#define MaskLattice_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/MaskedLatVolField.h>

namespace SCIRun {

class MaskLatticeAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src) = 0;

  virtual string identify() = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *lsrc,
					    string clipfunction,
					    int hashoffset);
};


template <class FDST, class LDST, class FSRC, class LSRC>
class MaskLatticeAlgoT : public MaskLatticeAlgo
{
public:
  virtual bool vinside_p(double x, double y, double z,
			 typename FSRC::value_type v)
  {
    return x > 0.5 && y > 0.5;
  }

  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src);
};


template <class FDST, class LDST, class FSRC, class LSRC>
FieldHandle
MaskLatticeAlgoT<FDST, LDST, FSRC, LSRC>::execute(FieldHandle field_h)
{
  FSRC *ifield = dynamic_cast<FSRC *>(field_h.get_rep());
  LatVolMeshHandle inmesh = ifield->get_typed_mesh();
  MaskedLatVolMesh *mesh = scinew MaskedLatVolMesh(inmesh->get_ni(), 
						   inmesh->get_nj(), 
						   inmesh->get_nk(),
						   Point(0.0, 0.0, 0.0),
						   Point(1.0, 1.0, 1.0));

  Transform trans = inmesh->get_transform();
  mesh->set_transform(trans);
  
  FDST *ofield = scinew FDST(mesh, ifield->data_at());

  typename LDST::iterator iter, iend;
  mesh->begin(iter);
  mesh->end(iend);
  while (iter != iend)
  {
    Point p;
    mesh->get_center(p, *iter);

    typename FDST::value_type val;

    ifield->value(val, *(typename LSRC::index_type *)&(*iter));
    ofield->set_value(val, *iter);

    if (!vinside_p(p.x(), p.y(), p.z(), val))
    {
      MaskedLatVolMesh::Cell::index_type 
	idx((*iter).mesh_, (*iter).i_, (*iter).j_, (*iter).k_);
      mesh->mask_cell(idx);
    }
    ++iter;
  }
  return ofield;
}

} // end namespace SCIRun

#endif // MaskLattice_h
