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
    
  OFIELD *ofield = scinew OFIELD(imesh, Field::CELL);

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
