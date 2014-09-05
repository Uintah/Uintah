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

//    File   : TransformData.h
//    Author : Michael Callahan
//    Date   : June 2002

#if !defined(TransformData_h)
#define TransformData_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Math/function.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>

namespace SCIRun {

class TransformDataAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src) = 0;

  virtual string identify() = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    string ofieldtypenam,
					    const TypeDescription *lsrc,
					    string function,
					    int hashoffset);
};


template <class IFIELD, class OFIELD, class LOC>
class TransformDataAlgoT : public TransformDataAlgo
{
public:
  virtual void function(typename OFIELD::value_type &result,
			double x, double y, double z,
			const typename IFIELD::value_type &v) = 0;

  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src);
};


template <class IFIELD, class OFIELD, class LOC>
FieldHandle
TransformDataAlgoT<IFIELD, OFIELD, LOC>::execute(FieldHandle field_h)
{
  IFIELD *ifield = dynamic_cast<IFIELD *>(field_h.get_rep());
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();
  OFIELD *ofield = scinew OFIELD(imesh, ifield->data_at());

  typename LOC::iterator ibi, iei;
  imesh->begin(ibi);
  imesh->end(iei);

  while (ibi != iei)
  {
    typename IFIELD::value_type val;
    ifield->value(val, *ibi);

    Point p;
    imesh->get_center(p, *ibi);

    typename OFIELD::value_type result;
    function(result, p.x(), p.y(), p.z(), val);

    ofield->set_value(result, *ibi);

    ++ibi;
  }

  return ofield;
}


} // end namespace SCIRun

#endif // TransformData_h
