/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __SPHERE_GEOMETRY_OBJECT_H__
#define __SPHERE_GEOMETRY_OBJECT_H__

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Geometry/Point.h>

#ifndef M_PI
# define M_PI           3.14159265358979323846  /* pi */
#endif

namespace Uintah {

/**************************************
        
CLASS
   SphereGeometryPiece
        
   Creates a sphere from the xml input file description.
        
GENERAL INFORMATION
        
   SphereGeometryPiece.h
        
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
        
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
        
 
        
KEYWORDS
   SphereGeometryPiece  BoundingBox inside
        
DESCRIPTION
   Creates a sphere from the xml input file description.
   Requires two inputs: origin and a radius.  
   There are methods for checking if a point is inside the sphere
   and also for determining the bounding box for the sphere.
   The input form looks like this:
       <sphere>
         <origin>[0.,0.,0.]</origin>
         <radius>2.0</radius>
       </sphere>
        
        
WARNING
        
****************************************/


class SphereGeometryPiece : public GeometryPiece {
         
public:
  //////////
  //  Constructor that takes a ProblemSpecP argument.   It reads the xml 
  // input specification and builds a sphere.
  SphereGeometryPiece(ProblemSpecP &);
         
  //////////
  //  Constructor that takes a origin and radius
  SphereGeometryPiece(const Point& origin, double radius);

  //////////
  // Destructor
  virtual ~SphereGeometryPiece();

  static const string TYPE_NAME;
  virtual std::string getType() const { return TYPE_NAME; }

  /// Make a clone
  virtual GeometryPieceP clone() const;

  //////////
  // Determines whether a point is inside the sphere. 
  virtual bool inside(const Point &p) const;
         
  //////////
  // Returns the bounding box surrounding the box.
  virtual Box getBoundingBox() const;
         
  //////////
  // Returns the voulme of the sphere
  inline double volume() const
  {
    return ((4.0/3.0)*M_PI*pow(d_radius,3.0));
  }

  //////////
  // Returns the surface area of the sphere
  inline double surfaceArea() const
  {
    return (4.0*M_PI*d_radius*d_radius);
  }

  //////////
  // Calculate the unit normal vector to center from point
  inline Vector radialDirection(const Point& pt) const
  {
    Vector normal = pt-d_origin;  
    return (normal/normal.length());
  }
  //////////
  // Get the center and radius
  //
  inline Point  origin() const {return d_origin;}
  inline double radius() const {return d_radius;}

private:

  virtual void outputHelper( ProblemSpecP & ps ) const;
         
  Point d_origin;
  double d_radius;
};

} // End namespace Uintah

#endif // __SPHERE_GEOMETRY_PIECE_H__
