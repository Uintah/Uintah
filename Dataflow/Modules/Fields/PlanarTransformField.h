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
 *  PlanarTransformField.h:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computering
 *   University of Utah
 *   August 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

#if !defined(PlanarTransformField_h)
#define PlanarTransformField_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Geometry/BBox.h>

namespace SCIRun {

class PlanarTransformFieldAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle& src,
			      int axis, int tx, int ty) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *oftd);
};


template< class FIELD >
class PlanarTransformFieldAlgoT : public PlanarTransformFieldAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle& src,
			      int axis, int tx, int ty);
};


template< class FIELD >
FieldHandle
PlanarTransformFieldAlgoT<FIELD>::execute(FieldHandle& ifield_h,
					  int axis, int tx, int ty)
{
  FIELD *ifield = (FIELD *) ifield_h.get_rep();
  FIELD *ofield(ifield->clone());
  ofield->mesh_detach();

  typename FIELD::mesh_type::Node::iterator inodeItr;
  ofield->get_typed_mesh()->begin(inodeItr);

  Transform trans;

  // Translate the mesh to the center of the view.
  const BBox bbox = ofield->mesh()->get_bounding_box();

  if (bbox.valid()) {
    Point center = -bbox.center();
    trans.post_translate( Vector( center ) );
  }

  // Rotate the mesh in to the correct plane.
  Point p0;
  ofield->get_typed_mesh()->get_point(p0, *inodeItr);

  if( axis == 0 )       // X
    trans.pre_rotate( atan2(p0.x(), p0.y()), Vector(1,0,0));
  else if( axis == 1 )  // Y
    trans.pre_rotate( atan2(p0.x(), p0.y()), Vector(0,1,0));
  else if( axis == 2 )  // Z
    trans.pre_rotate( atan2(p0.x(), p0.y()), Vector(0,0,1));

  ofield->mesh()->transform(trans);

  // Optionally translate the mesh away from the center of the view.
  if (tx || ty) {
    const BBox bbox = ofield->mesh()->get_bounding_box();
      
    if (bbox.valid()) {
      Vector size = bbox.diagonal();
      
      Transform trans;
      trans.post_translate( Vector( 0, tx * size.y(), ty * size.z() ) );

      ofield->mesh()->transform(trans);
    }
  }

  return FieldHandle( ofield );
}


} // end namespace SCIRun

#endif // PlanarTransformField_h
