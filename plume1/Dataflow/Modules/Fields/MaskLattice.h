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
  
  FDST *ofield = scinew FDST(mesh, ifield->basis_order());

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
