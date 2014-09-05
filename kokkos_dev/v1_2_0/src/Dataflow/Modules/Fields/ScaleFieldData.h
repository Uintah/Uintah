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

//    File   : ScaleFieldData.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(ScaleFieldData_h)
#define ScaleFieldData_h

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/DynamicLoader.h>
#include <Core/Datatypes/Matrix.h>

namespace SCIRun {

class ScaleFieldDataAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src, MatrixHandle mat) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *fsrc,
				       const TypeDescription *lsrc);
};


template <class FIELD, class LOC>
class ScaleFieldDataAlgoT : public ScaleFieldDataAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src, MatrixHandle mat);
};


template <class FIELD, class LOC>
FieldHandle
ScaleFieldDataAlgoT<FIELD, LOC>::execute(FieldHandle field_h,
					 MatrixHandle matrix)
{
  FIELD *ifield = dynamic_cast<FIELD *>(field_h.get_rep());
  FIELD *ofield = ifield->clone();
  typename FIELD::mesh_handle_type mesh = ifield->get_typed_mesh();

  int index = 0;
  int rows = matrix->nrows();
  typename LOC::iterator itr = mesh->tbegin((typename LOC::iterator *) 0);
  typename LOC::iterator eitr = mesh->tend((typename LOC::iterator *) 0);
  while (itr != eitr)
  {
    typename FIELD::value_type val;
    ifield->value(val, *itr);
    val *= matrix->get(index % rows, 0);
    ofield->set_value(val, *itr);
   
    ++index;
    ++itr;
  }

  return ofield;
}


} // end namespace SCIRun

#endif // ScaleFieldData_h
