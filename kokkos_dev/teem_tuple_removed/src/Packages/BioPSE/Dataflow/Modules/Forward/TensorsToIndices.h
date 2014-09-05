#ifndef SCIRun_TensorsToIndices_H
#define SCIRun_TensorsToIndices_H
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

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Geometry/Tensor.h>

namespace SCIRun {

// just get the first point from the field
class TensorsToIndicesAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle srcH) = 0;

  //! support the dynamically compiled algorithm concept
 static CompileInfoHandle get_compile_info(const TypeDescription *field_src_td,
				           const string &field_dst_name);
};


template <class FSRC, class FDST>
class TensorsToIndicesAlgoT : public TensorsToIndicesAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle srcH);
};

template <class FSRC, class FDST>
FieldHandle
TensorsToIndicesAlgoT<FSRC, FDST>::execute(FieldHandle srcH)
{
  FSRC *src = dynamic_cast<FSRC *>(srcH.get_rep());
  FDST *dst = scinew FDST(src->get_typed_mesh(), src->data_at());
  vector<pair<string, Tensor> > conds;
  typename FSRC::fdata_type::iterator in = src->fdata().begin();
  typename FDST::fdata_type::iterator out = dst->fdata().begin();
  typename FSRC::fdata_type::iterator end = src->fdata().end();
  int count=0;
  while (in != end) {
    conds.push_back(pair<string, Tensor>("x", *in));
    *out = count;
    ++in; ++out; ++count;
  }
  dst->set_property("conductivity_table", conds, false);
  return dst;
}
} // End namespace BioPSE

#endif
