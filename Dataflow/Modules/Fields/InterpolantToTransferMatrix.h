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
#include <Core/Util/ProgressReporter.h>
#include <algorithm>

namespace SCIRun {

class Interp2TransferAlgo : public DynamicAlgoBase
{
public:
  virtual MatrixHandle execute(ProgressReporter *m, FieldHandle itp) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fitp,
					    const TypeDescription *litp);

protected:

  static bool pair_less(const pair<int, float> &a, const pair<int, float> &b)
  {
    return a.first < b.first;
  }
};


template <class FITP, class LITP>
class Interp2TransferAlgoT : public Interp2TransferAlgo
{
public:
  //! virtual interface. 
  virtual MatrixHandle execute(ProgressReporter *m, FieldHandle itp);
};


template <class FITP, class LITP>
MatrixHandle
Interp2TransferAlgoT<FITP, LITP>::execute(ProgressReporter *m,
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

  typename LITP::iterator iter, eiter;
  fitp->get_typed_mesh()->begin(iter);
  fitp->get_typed_mesh()->end(eiter);
  unsigned int counter = 0;
  int *rowdata = scinew int[((unsigned int)itrsize)+1];
  vector<int> coldatav;
  vector<double> datav;
  rowdata[0] = 0;
  while (iter != eiter)
  {
    typename FITP::value_type vtmp;
    fitp->value(vtmp, *iter);
    vector<pair<int, double> > v(vtmp.size());
    for (i=0; i<vtmp.size(); i++)
    {
      v[i].first = (int)(vtmp[i].first);
      v[i].second = vtmp[i].second;
    }

    rowdata[counter+1] = rowdata[counter] + v.size();
    std::sort(v.begin(), v.end(), pair_less);
    for (i = 0; i < v.size(); i++)
    {
      coldatav.push_back(v[i].first);
      datav.push_back(v[i].second);
    }
    ++counter;
    ++iter;
  }

  int *coldata = scinew int[coldatav.size()];
  for (i=0; i < coldatav.size(); i++)
  {
    coldata[i] = coldatav[i];
  }

  double *data = scinew double[datav.size()];
  for (i=0; i < datav.size(); i++)
  {
    data[i] = datav[i];
  }

  SparseRowMatrix *matrix =
    scinew SparseRowMatrix((unsigned int)itrsize, range, rowdata, coldata,
			   datav.size(), data);

  return matrix;
}


} // end namespace SCIRun

#endif // InterpolantToTransferMatrix_h
