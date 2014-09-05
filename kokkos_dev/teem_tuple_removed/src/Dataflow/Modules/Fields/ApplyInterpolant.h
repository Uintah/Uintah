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
  FOUT *fout = scinew FOUT(fitp->get_typed_mesh(), fitp->data_at());
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
