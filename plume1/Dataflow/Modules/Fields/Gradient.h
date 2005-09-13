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
 *  Gradient.h:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computering
 *   University of Utah
 *   May 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#if !defined(Gradient_h)
#define Gradient_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

#include <Core/Datatypes/Mesh.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>


namespace SCIRun {

class GradientAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle& src) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd,
					    const TypeDescription *ttd,
					    const TypeDescription *otd);
};


template< class IFIELD, class OFIELD >
class GradientAlgoT : public GradientAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle& src);
};


template< class IFIELD, class OFIELD >
FieldHandle
GradientAlgoT<IFIELD, OFIELD>::execute(FieldHandle& field_h)
{
  IFIELD *ifield = (IFIELD *) field_h.get_rep();

  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();
    
  OFIELD *ofield = scinew OFIELD(imesh, 0);

  typename IFIELD::mesh_type::Cell::iterator in, end;
  typename OFIELD::mesh_type::Cell::iterator out;

  imesh->begin( in );
  imesh->end( end );

  ofield->get_typed_mesh()->begin( out );

  typename OFIELD::value_type gradient;

  while (in != end) {
    gradient = ifield->cell_gradient(*in);
    ofield->set_value(gradient, *out);
    ++in; ++out;
  }

  ofield->freeze();

  return FieldHandle( ofield );
}

} // end namespace SCIRun

#endif // Gradient_h
