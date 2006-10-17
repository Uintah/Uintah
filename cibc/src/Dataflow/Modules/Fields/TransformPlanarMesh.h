/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  TransformPlanarMesh.h:
 *
 *  Written by:
 *   Allen Sanderson
 *   Scientific Computing and Imaging Institute
 *   University of Utah
 *   April 2006
 *
 *  Copyright (C) 2006 SCI Inst
 */

#if !defined(TransformPlanarMesh_h)
#define TransformPlanarMesh_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

#include <Core/Geometry/BBox.h>

namespace SCIRun {

class TransformPlanarMeshAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle& src,
			      int axis, int invert, int tx, int ty) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *oftd);
};


template< class FIELD >
class TransformPlanarMeshAlgoT : public TransformPlanarMeshAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle& src,
			      int axis, int invert, int tx, int ty);
};


template< class FIELD >
FieldHandle
TransformPlanarMeshAlgoT<FIELD>::execute(FieldHandle& ifield_h,
					  int axis, int invert, int tx, int ty)
{
  FIELD *ifield = (FIELD *) ifield_h.get_rep();
  FIELD *ofield(ifield->clone());
  ofield->mesh_detach();

  Transform trans;
  Vector axisVec;

  double sign = (invert ? -1 : 1);

  Point p0;
  Point p1;
    
  typename FIELD::mesh_type::Node::iterator inodeItr;
  typename FIELD::mesh_type::Node::iterator inodeEnd;
  ofield->get_typed_mesh()->begin(inodeItr);
  ofield->get_typed_mesh()->end(inodeEnd);

  //! Get the first point along the line.
  if (inodeItr != inodeEnd) {
    ofield->get_typed_mesh()->get_point(p0, *inodeItr);
    ++inodeItr;
  } else {
    cerr << "Can not get the first point - No mesh" << endl;
    return 0;
  }

  // Get the last point along the line.
  unsigned int cc = 0;
  if (inodeItr != inodeEnd) {
    while (inodeItr != inodeEnd) {
      ofield->get_typed_mesh()->get_point(p1, *inodeItr);
      ++inodeItr;
      ++cc;
    }
  } else {
    cerr << "Can not get the last point - No mesh" << endl;
    return 0;
  }

  string if_name =
    ifield->get_type_description(Field::MESH_TD_E)->get_name();

  //! For a line assume that it is colinear.
  if (if_name.find("CurveMesh") != string::npos ) {
    axisVec = Vector(p1 - p0);

  } else {
    //! For a surface assume that it is planar.

    // Translate the mesh to the center of the view.
    const BBox bbox = ofield->mesh()->get_bounding_box();
    
    if (bbox.valid()) {
      Point center = -bbox.center();
      trans.post_translate( Vector( center ) );
    }

    Point p2;
    
    cc /= 2;

    ofield->get_typed_mesh()->begin(inodeItr);
    ofield->get_typed_mesh()->end(inodeEnd);
    ++inodeItr;

    //! Get the middle point on the surface.
    if (inodeItr != inodeEnd) {
      while (inodeItr != inodeEnd && cc ) {
	ofield->get_typed_mesh()->get_point(p2, *inodeItr);
	++inodeItr;
	--cc;
      }
    } else {
      cerr << "Can not get the last  point - No mesh" << endl;
      return 0;
    }

    Vector vec0 = Vector(p2 - p0);
    vec0.safe_normalize();

    Vector vec1 = Vector(p2 - p1);
    vec1.safe_normalize();

    axisVec = Cross( vec0, vec1 );

    axisVec.safe_normalize();

    cerr << axisVec << endl;
  }

  axisVec.safe_normalize();

  //! Rotate only if not in the -Z or +Z axis.
  if( axis != 2 || fabs( fabs( axisVec.z() ) - 1.0 ) > 1.0e-4 ) {
    double theta = atan2( axisVec.y(), axisVec.x() );
    double phi   = acos( axisVec.z() / axisVec.length() );

    //! Rotate the line into the xz plane.
    trans.pre_rotate( -theta, Vector(0,0,1) );
    //! Rotate the line into the z axis.
    trans.pre_rotate( -phi,   Vector(0,1,0) );
  }

  if( axis == 0 ) {
    //! Rotate the line into the x axis.
    trans.pre_rotate( sign * M_PI/2.0, Vector(0,1,0));
  } else if( axis == 1 ) {
    //! Rotate the line into the y axis.
    trans.pre_rotate( sign * -M_PI/2.0, Vector(1,0,0));
  } else if( invert ) {
    //! Rotate the line into the z axis.
    trans.pre_rotate( M_PI, Vector(1,0,0));
  }

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

#endif // TransformPlanarMesh_h
