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

//    File   : TransformData3.h
//    Author : Michael Callahan
//    Date   : June 2002

#if !defined(TransformData3_h)
#define TransformData3_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Math/function.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>

namespace SCIRun {

class TransformData3Algo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src0,
			      FieldHandle src1,
			      FieldHandle src2) = 0;

  virtual string identify() = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc0,
					    const TypeDescription *fsrc1,
					    const TypeDescription *fsrc2,
					    string ofieldtypenam,
					    const TypeDescription *lsrc,
					    string function,
					    int hashoffset);
};


template <class IFIELD0, class IFIELD1, class IFIELD2, class OFIELD, class LOC>
class TransformData3AlgoT : public TransformData3Algo
{
public:
  virtual void function(typename OFIELD::value_type &result,
			double x, double y, double z,
			const typename IFIELD0::value_type &v0,
			const typename IFIELD1::value_type &v1,
			const typename IFIELD2::value_type &v2) = 0;

  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src0,
			      FieldHandle src1,
			      FieldHandle src2);
};


template <class IFIELD0, class IFIELD1, class IFIELD2, class OFIELD, class LOC>
FieldHandle
TransformData3AlgoT<IFIELD0, IFIELD1, IFIELD2, OFIELD, LOC>::
execute(FieldHandle f0_h, FieldHandle f1_h, FieldHandle f2_h)
{
  IFIELD0 *ifield0 = dynamic_cast<IFIELD0 *>(f0_h.get_rep());
  IFIELD1 *ifield1 = dynamic_cast<IFIELD1 *>(f1_h.get_rep());
  IFIELD1 *ifield2 = dynamic_cast<IFIELD2 *>(f2_h.get_rep());
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

    typename IFIELD2::value_type val2;
    ifield2->value(val2, *ibi);

    Point p;
    imesh->get_center(p, *ibi);

    typename OFIELD::value_type result;
    function(result, p.x(), p.y(), p.z(), val0, val1, val2);

    ofield->set_value(result, *ibi);

    ++ibi;
  }

  return ofield;
}


} // end namespace SCIRun

#endif // TransformData3_h
