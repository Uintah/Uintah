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


/*
 *  VectorMagnitude.h:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computering
 *   University of Utah
 *   May 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#if !defined(VectorMagnitude_h)
#define VectorMagnitude_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

namespace SCIRun {

class VectorMagnitudeAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle& src) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *iftd,
					    const TypeDescription *oftd);
};


template< class IFIELD, class OFIELD >
class VectorMagnitudeAlgoT : public VectorMagnitudeAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle& src);
};


template< class IFIELD, class OFIELD >
FieldHandle
VectorMagnitudeAlgoT<IFIELD, OFIELD>::execute(FieldHandle& field_h)
{
  IFIELD *ifield = (IFIELD *) field_h.get_rep();

  OFIELD *ofield = 
    scinew OFIELD(ifield->get_typed_mesh(), ifield->basis_order());

  typename IFIELD::fdata_type::iterator in  = ifield->fdata().begin();
  typename IFIELD::fdata_type::iterator end = ifield->fdata().end();
  typename OFIELD::fdata_type::iterator out = ofield->fdata().begin();

  while (in != end)
  {
    *out = in->length();;
    ++in; ++out;
  }

  return FieldHandle( ofield );
}


} // end namespace SCIRun

#endif // VectorMagnitude_h















