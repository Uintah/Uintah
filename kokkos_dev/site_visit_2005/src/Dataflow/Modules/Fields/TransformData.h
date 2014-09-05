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


//    File   : TransformData.h
//    Author : Michael Callahan
//    Date   : June 2002

#if !defined(TransformData_h)
#define TransformData_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
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
  OFIELD *ofield = scinew OFIELD(imesh, ifield->basis_order());

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
