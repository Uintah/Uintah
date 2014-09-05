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

//    File   : TransformVectorData.h
//    Author : Michael Callahan
//    Date   : June 2002

#if !defined(TransformVectorData_h)
#define TransformVectorData_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Math/function.h>

namespace SCIRun {

class TransformVectorDataAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src,
			      Function *fx, Function *fy, Function *fz,
			      bool pre_normalize_p, bool post_normalize_p) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *lsrc);
};


template <class FIELD, class LOC>
class TransformVectorDataAlgoT : public TransformVectorDataAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src,
			      Function *fx, Function *fy, Function *fz,
			      bool pre_normalize_p, bool post_normalize_p);
};


template <class FIELD, class LOC>
FieldHandle
TransformVectorDataAlgoT<FIELD, LOC>::execute(FieldHandle field_h,
					      Function *fx,
					      Function *fy,
					      Function *fz,
					      bool pre_normalize_p,
					      bool post_normalize_p)
{
  FIELD *ifield = dynamic_cast<FIELD *>(field_h.get_rep());
  FIELD *ofield = ifield->clone();

  typename FIELD::fdata_type::iterator ibi, iei, obi;
  ibi = ifield->fdata().begin();
  iei = ifield->fdata().end();
  obi = ofield->fdata().begin();

  while (ibi != iei)
  {
    Vector val(*ibi);

    if (pre_normalize_p) { val.safe_normalize(); }
    Vector tmp(fx->eval(&val[0]), fy->eval(&val[0]), fz->eval(&val[0]));
    if (post_normalize_p) { tmp.safe_normalize(); }

    *obi = tmp;

    ++ibi;
    ++obi;
  }

  return ofield;
}

} // end namespace SCIRun

#endif // TransformVectorData_h
