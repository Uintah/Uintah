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

//    File   : InterpolantToTransferMatrix.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(InterpolantToTransferMatrix_h)
#define InterpolantToTransferMatrix_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Util/ModuleReporter.h>

namespace SCIRun {

class Interp2TransferAlgo : public DynamicAlgoBase
{
public:
  virtual MatrixHandle execute(ModuleReporter *m, FieldHandle itp) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fitp,
					    const TypeDescription *litp);
};


template <class FITP, class LITP>
class Interp2TransferAlgoT : public Interp2TransferAlgo
{
public:
  //! virtual interface. 
  virtual MatrixHandle execute(ModuleReporter *m, FieldHandle itp);
};


template <class FITP, class LITP>
MatrixHandle
Interp2TransferAlgoT<FITP, LITP>::execute(ModuleReporter *m,
					  FieldHandle fitp_h)
{
  FITP *fitp = (FITP *)(fitp_h.get_rep());

  unsigned int i;
  unsigned int range;
  if (!fitp->get_property("interp-source-range", range))
  {
    m->error("No column size given in the interpolant field.");
    m->error("Unable to determine output matrix size.");
    return 0;
  }

  typename LITP::size_type itrsize;
  fitp->get_typed_mesh()->size(itrsize);

  DenseMatrix *matrix = scinew DenseMatrix((unsigned int)itrsize, range);

  typename LITP::iterator iter, eiter;
  fitp->get_typed_mesh()->begin(iter);
  fitp->get_typed_mesh()->end(eiter);
  unsigned int counter = 0;
  while (iter != eiter)
  {
    typename FITP::value_type v;
    fitp->value(v, *iter);
    for (i = 0; i < v.size(); i++)
    {
      const unsigned int tmp = (unsigned int)v[i].first;
      matrix->put(counter, tmp, v[i].second);
    }
    ++counter;
    ++iter;
  }

  return matrix;
}


} // end namespace SCIRun

#endif // InterpolantToTransferMatrix_h
