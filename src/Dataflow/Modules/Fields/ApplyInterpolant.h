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


//    File   : ApplyInterpolant.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(ApplyInterpolant_h)
#define ApplyInterpolant_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

namespace SCIRun {

class ApplyInterpAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src, FieldHandle itp) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *fitp,
					    const TypeDescription *litp);
};


template <class FSRC, class FITP, class LITP, class FOUT>
class ApplyInterpAlgoT : public ApplyInterpAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src, FieldHandle itp);
};


template <class FSRC, class FITP, class LITP, class FOUT>
FieldHandle
ApplyInterpAlgoT<FSRC, FITP, LITP, FOUT>::execute(FieldHandle fsrc_h,
						  FieldHandle fitp_h)
{
  FSRC *fsrc = (FSRC *)(fsrc_h.get_rep()); // dynamic cast fails on sgi
  FITP *fitp = (FITP *)(fitp_h.get_rep());
  FOUT *fout = scinew FOUT(fitp->get_typed_mesh(), fitp->basis_order());
  FieldHandle fhout(fout);

  typename LITP::iterator iter, eiter;
  fout->get_typed_mesh()->begin(iter);
  fout->get_typed_mesh()->end(eiter);
  while (iter != eiter)
  {
    typename FITP::value_type v;
    fitp->value(v, *iter);
    if (!v.empty())
    {
      typename FSRC::value_type val =
	(typename FSRC::value_type)(fsrc->value(v[0].first) * v[0].second);
      unsigned int j;
      for (j = 1; j < v.size(); j++)
      {
	val += (typename FSRC::value_type)
	  (fsrc->value(v[j].first) * v[j].second);
      }
      fout->set_value(val, *iter);
    } else {
      typename FSRC::value_type val(0);
      fout->set_value(val, *iter);
    }
    ++iter;
  }
  return fout;
}


} // end namespace SCIRun

#endif // ApplyInterpolant_h
