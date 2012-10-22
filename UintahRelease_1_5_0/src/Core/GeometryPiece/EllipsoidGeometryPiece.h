/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __ELLIPSOID_GEOMETRY_OBJECT_H__
#define __ELLIPSOID_GEOMETRY_OBJECT_H__

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Geometry/Point.h>

#ifndef M_PI
# define M_PI           3.14159265358979323846  /* pi */
#endif

namespace Uintah {

/**************************************
        
CLASS
   EllipsoidGeometryPiece
        
   Creates a sphere from the xml input file description.
        
GENERAL INFORMATION
        
   EllipsoidGeometryPiece.h
        
   Joseph R. Peterson
   Department of Chemistry
   University of Utah
        
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
        
 
        
KEYWORDS
   EllipsoidGeometryPiece  BoundingBox inside
        
DESCRIPTION
   Creates an ellispoid from the xml input file description.
   Requires four inputs.  <origin> is always required in input,
   however there are two ways to specify the three axes.  One is a 
   set of three orthogonal Vectors (the model will throw an error if 
   vectors are not orthagonal withing 1e-12 precision).  This model
   takes preference over the second type, if both sets of inputs
   are present in the XML input file.  The second type of input 
   assumes that the ellipsoid is aligned in xyz space and gives the 
   radii of each vector.
 
   There are methods for checking if a point that has been transformed
   from ellipsoid to sphere relative to origin.
   and also for determining the bounding box for the sphere.
   The input form looks like this:
       <sphere>
         <origin>[0.,0.,0.]</origin>
         <!-- Vectors must be orthagonal -->
         <!--  High Priority -->
         <v1>    [1.,0.,0.]</v1>
         <v2>    [0.,1.,0.]</v2>
         <v3>    [0.,0.,1.]</v3>
 
         <!-- Or aligned with xyz -->
         <!--  Low Priority  -->
         <rx>    1.0       </rx>
         <ry>    1.0       </ry>
         <rz>    1.0       </rz>
       </sphere>
        
        
WARNING
        
****************************************/
  

class EllipsoidGeometryPiece : public GeometryPiece {
         
public:
  //////////
  //  Constructor that takes a ProblemSpecP argument.   It reads the xml 
  // input specification and builds a sphere.
  EllipsoidGeometryPiece(ProblemSpecP &);
         
  //////////
  //  Constructor that takes a origin and radii along orthagonal axes
  EllipsoidGeometryPiece(const Point& origin, double radx, double rady, double radz);

  //////////
  //  Constructor that takes a origin and vectors to define axes
  EllipsoidGeometryPiece(const Point& origin, Vector v1, Vector v2, Vector v3);
  
  //////////
  // Destructor
  virtual ~EllipsoidGeometryPiece();

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
    return ( 4.0*M_PI*d_radiusX*d_radiusY*d_radiusZ );
  }

  //////////
  // Returns the surface area of the sphere
  inline double surfaceArea() const
  {
    //  Knud Thomsen's formula (1.061% max error in surface area)
    double ap = pow(d_radiusX, 1.6075);
    double bp = pow(d_radiusY, 1.6075);
    double cp = pow(d_radiusZ, 1.6075);
    
    return ( 4.0*M_PI*pow(((ap*bp + ap*cp + bp*cp)/3.0), 1.6075) );
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
  inline double rX() const {return d_radiusX;}
  inline double rY() const {return d_radiusY;}
  inline double rZ() const {return d_radiusZ;}

private:

  virtual void outputHelper( ProblemSpecP & ps ) const;
  virtual void initializeEllipsoidData();
         
  Point d_origin;
  // Radii of each axis
  double d_radiusX;
  double d_radiusY;
  double d_radiusZ;
  
  double thetax, thetay, thetaz;

  // Vectors of each axis
  Vector d_v1;
  Vector d_v2;
  Vector d_v3;
  
  bool xyzAligned;
};

} // End namespace Uintah

#endif // __ELLIPSOID_GEOMETRY_PIECE_H__
