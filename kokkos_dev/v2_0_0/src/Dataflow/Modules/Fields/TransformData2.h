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

//    File   : TransformData2.h
//    Author : Michael Callahan
//    Date   : June 2002

#if !defined(TransformData2_h)
#define TransformData2_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Math/function.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>

namespace SCIRun {

class TransformData2Algo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src0, FieldHandle src1) = 0;

  virtual string identify() = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc0,
					    const TypeDescription *fsrc1,
					    string ofieldtypenam,
					    const TypeDescription *lsrc,
					    string function,
					    int hashoffset);
};


template <class IFIELD0, class IFIELD1, class OFIELD, class LOC>
class TransformData2AlgoT : public TransformData2Algo
{
public:
  virtual void function(typename OFIELD::value_type &result,
			double x, double y, double z,
			const typename IFIELD0::value_type &v0,
			const typename IFIELD1::value_type &v1) = 0;

  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src0, FieldHandle src1);
};


template <class IFIELD0, class IFIELD1, class OFIELD, class LOC>
FieldHandle
TransformData2AlgoT<IFIELD0, IFIELD1, OFIELD, LOC>::execute(FieldHandle f0_h,
							   FieldHandle f1_h)
{
  IFIELD0 *ifield0 = dynamic_cast<IFIELD0 *>(f0_h.get_rep());
  IFIELD1 *ifield1 = dynamic_cast<IFIELD1 *>(f1_h.get_rep());
  typename IFIELD0::mesh_handle_type imesh = ifield0->get_typed_mesh();
  OFIELD *ofield = scinew OFIELD(imesh, ifield0->data_at());

  typename LOC::iterator ibi, iei;
  imesh->begin(ibi);
  imesh->end(iei);

  while (ibi != iei)
  {
    typename IFIELD0::value_type val0;
    ifield0->value(val0, *ibi);

    typename IFIELD1::value_type val1;
    ifield1->value(val1, *ibi);

    Point p;
    imesh->get_center(p, *ibi);

    typename OFIELD::value_type result;
    function(result, p.x(), p.y(), p.z(), val0, val1);

    ofield->set_value(result, *ibi);

    ++ibi;
  }

  return ofield;
}


} // end namespace SCIRun

#endif // TransformData2_h
