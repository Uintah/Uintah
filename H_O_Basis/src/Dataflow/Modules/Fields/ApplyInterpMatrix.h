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


//    File   : ApplyInterpMatrix.h
//    Author : Michael Callahan
//    Date   : June 2002

#if !defined(ApplyInterpMatrix_h)
#define ApplyInterpMatrix_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Datatypes/Matrix.h>

namespace SCIRun {

class ApplyInterpMatrixAlgo : public DynamicAlgoBase
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src, MeshHandle dst,
			      MatrixHandle interp,
			      int basis_order) = 0;

  virtual void execute_aux(FieldHandle src, FieldHandle dst,
			   MatrixHandle interp) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *lsrc,
					    const TypeDescription *fdst,
					    const TypeDescription *ldst,
					    const string &accum,
					    bool fout_use_accum);
};


template <class FSRC, class LSRC, class FDST, class LDST, class ACCUM>
class ApplyInterpMatrixAlgoT : public ApplyInterpMatrixAlgo
{
public:

  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src, MeshHandle dst,
			      MatrixHandle interp, int basis_order);

  virtual void execute_aux(FieldHandle src, FieldHandle dst,
			   MatrixHandle interp);
};


template <class FSRC, class LSRC, class FDST, class LDST, class ACCUM>
FieldHandle
ApplyInterpMatrixAlgoT<FSRC, LSRC, 
		       FDST, LDST, ACCUM>::execute(FieldHandle src_h,
						   MeshHandle dst_h,
						   MatrixHandle interp,
						   int basis_order)
{
  typename FDST::mesh_type *dstmesh =
    dynamic_cast<typename FDST::mesh_type *>(dst_h.get_rep());

  FieldHandle ofield = scinew FDST(dstmesh, basis_order);
  execute_aux(src_h, ofield, interp);

  return ofield;
}


template <class FSRC, class LSRC, class FDST, class LDST, class ACCUM>
void
ApplyInterpMatrixAlgoT<FSRC, LSRC, 
		       FDST, LDST, ACCUM>::execute_aux(FieldHandle src_h,
						       FieldHandle dst_h,
						       MatrixHandle interp)
{
  FSRC *fsrc = dynamic_cast<FSRC *>(src_h.get_rep());
  FDST *fdst = dynamic_cast<FDST *>(dst_h.get_rep());
  typename FSRC::mesh_handle_type msrc = fsrc->get_typed_mesh();

  //ASSERT(interp.is_sparse());
  ASSERT((unsigned int)(interp->nrows()) == fdst->fdata().size())
  ASSERT((unsigned int)(interp->ncols()) == fsrc->fdata().size())

  typename LDST::iterator dbi, dei;
  Array1<int> idx;
  Array1<double> val;
  int i;
  unsigned int counter = 0; 

  fdst->get_typed_mesh()->begin(dbi);
  fdst->get_typed_mesh()->end(dei);

  while (dbi != dei)
  {
    idx.remove_all();
    val.remove_all();
    interp->getRowNonzeros(counter, idx, val);
    
    ACCUM accum(0);
    for (i = 0; i < idx.size(); i++)
    {
      typename FSRC::value_type v;
      typename LSRC::index_type index;
      msrc->to_index(index, (unsigned int)idx[i]);
      fsrc->value(v, index);
      accum += v * val[i];
    }
    fdst->set_value((typename FDST::value_type)accum, *dbi);
    
    ++counter;
    ++dbi;
  }
}


} // end namespace SCIRun

#endif // ApplyInterpMatrix_h
