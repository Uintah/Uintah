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

//    File   : TransformScalarData.h
//    Author : Michael Callahan
//    Date   : June 2002

#if !defined(TransformScalarData_h)
#define TransformScalarData_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Math/function.h>

namespace SCIRun {

class TransformScalarDataAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src, Function *f) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *fsrc,
				       const TypeDescription *lsrc);
};


template <class FIELD, class LOC>
class TransformScalarDataAlgoT : public TransformScalarDataAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src, Function *f);
};


template <class FIELD, class LOC>
FieldHandle
TransformScalarDataAlgoT<FIELD, LOC>::execute(FieldHandle field_h,
					     Function *f)
{
  FIELD *ifield = dynamic_cast<FIELD *>(field_h.get_rep());
  FIELD *ofield = ifield->clone();
  typename FIELD::mesh_handle_type mesh = ifield->get_typed_mesh();

  typename LOC::iterator itr, eitr;
  mesh->begin(itr);
  mesh->end(eitr);
  while (itr != eitr)
  {
    typename FIELD::value_type val;
    ifield->value(val, *itr);
    double tmp = (double)val;
    val = (typename FIELD::value_type)(f->eval(&tmp));
    ofield->set_value(val, *itr);
    ++itr;
  }

  return ofield;
}


} // end namespace SCIRun

#endif // TransformScalarData_h
