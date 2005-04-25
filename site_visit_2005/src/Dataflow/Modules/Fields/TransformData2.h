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


//    File   : TransformData2.h
//    Author : Michael Callahan
//    Date   : June 2002

#if !defined(TransformData2_h)
#define TransformData2_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
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
  OFIELD *ofield = scinew OFIELD(imesh, ifield0->basis_order());

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
