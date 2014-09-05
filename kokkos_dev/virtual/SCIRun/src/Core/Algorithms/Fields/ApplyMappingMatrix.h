/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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


//    File   : ApplyMappingMatrix.h
//    Author : Michael Callahan
//    Date   : June 2002

#if !defined(ApplyMappingMatrix_h)
#define ApplyMappingMatrix_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/Field.h>

#include <Core/Algorithms/Fields/share.h>

namespace SCIRun {

class SCISHARE ApplyMappingMatrixAlgo : public DynamicAlgoBase
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter,
                              FieldHandle src, MeshHandle dst,
			      MatrixHandle mapping) = 0;

  virtual void execute_aux(ProgressReporter *reporter,
                           FieldHandle src, FieldHandle dst,
			   MatrixHandle mapping) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *lsrc,
					    const TypeDescription *fdst,
					    const string &fdststr,
					    const TypeDescription *ldst,
					    const TypeDescription *dsrc,
					    const string &accum);
};


template <class FSRC, class LSRC, class FDST, class LDST, class ACCUM>
class ApplyMappingMatrixAlgoT : public ApplyMappingMatrixAlgo
{
public:

  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter,
                              FieldHandle src, MeshHandle dst,
			      MatrixHandle mapping);

  virtual void execute_aux(ProgressReporter *reporter,
                           FieldHandle src, FieldHandle dst,
			   MatrixHandle mapping);
};


template <class FSRC, class LSRC, class FDST, class LDST, class ACCUM>
FieldHandle
ApplyMappingMatrixAlgoT<FSRC, LSRC, 
		       FDST, LDST, ACCUM>::execute(ProgressReporter *reporter,
                                                   FieldHandle src_h,
						   MeshHandle dst_h,
						   MatrixHandle mapping)
{
  typename FDST::mesh_type *dstmesh =
    dynamic_cast<typename FDST::mesh_type *>(dst_h.get_rep());

  FieldHandle ofield = scinew FDST(dstmesh);
  execute_aux(reporter, src_h, ofield, mapping);

  return ofield;
}


template <class FSRC, class LSRC, class FDST, class LDST, class ACCUM>
void
ApplyMappingMatrixAlgoT<FSRC, LSRC, 
		       FDST, LDST, ACCUM>::execute_aux(ProgressReporter *reporter,
                                                       FieldHandle src_h,
						       FieldHandle dst_h,
						       MatrixHandle mapping)
{
  FSRC *fsrc = dynamic_cast<FSRC *>(src_h.get_rep());
  FDST *fdst = dynamic_cast<FDST *>(dst_h.get_rep());
  typename FSRC::mesh_handle_type msrc = fsrc->get_typed_mesh();

  ASSERT((unsigned int)(mapping->nrows()) == fdst->fdata().size())
  ASSERT((unsigned int)(mapping->ncols()) == fsrc->fdata().size())

  typename LDST::iterator dbi, dei;
  int *idx;
  double *val;
  int idxsize;
  int idxstride;

  int i;
  unsigned int counter = 0; 

  fdst->get_typed_mesh()->begin(dbi);
  fdst->get_typed_mesh()->end(dei);

  typename LDST::size_type prsizetmp;
  fdst->get_typed_mesh()->size(prsizetmp);
  const unsigned int prsize = (unsigned int)prsizetmp;

  while (dbi != dei)
  {
    reporter->update_progress(counter, prsize);

    mapping->getRowNonzerosNoCopy(counter, idxsize, idxstride, idx, val);
    
    ACCUM accum(0);
    for (i = 0; i < idxsize; i++)
    {
      typename FSRC::value_type v;
      typename LSRC::index_type index;
      msrc->to_index(index, idx?(unsigned int)idx[i*idxstride]:i);
      fsrc->value(v, index);
      accum += v * val[i*idxstride];
    }
    fdst->set_value((typename FDST::value_type)accum, *dbi);
    ++counter;
    ++dbi;
  }
}


} // end namespace SCIRun

#endif // ApplyMappingMatrix_h
