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

#ifndef __TORUS_GEOMETRY_OBJECT_H__
#define __TORUS_GEOMETRY_OBJECT_H__

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#ifndef M_PI
# define M_PI           3.14159265358979323846  /* pi */
#endif

namespace Uintah {

/**************************************

CLASS
   TorusGeometryPiece

   Creates a generalized cylinder from the xml input file description.

GENERAL INFORMATION

   TorusGeometryPiece.h

   Jim Guilkey
   Perforating Research
   Schlumberger Technology Corporation

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

KEYWORDS
   TorusGeometryPiece BoundingBox inside

DESCRIPTION
   Creates a z-axis aligned torus from the xml input file description.
   Requires three inputs: center point, and major and minor radii 
   There are methods for checking if a point is inside the torus
   and also for determining the bounding box for the torus.
   The input form looks like this:
       <torus>
         <center>[0.,0.,0.]</center>
         <major_radius>2.0</major_radius>
         <minor_radius>1.0</minor_radius>
         <axis>x</axis>
       </torus>

WARNING

****************************************/

  class TorusGeometryPiece : public GeometryPiece {
    
  public:
    //////////
    // Constructor that takes a ProblemSpecP argument.   It reads the xml 
    // input specification and builds a generalized cylinder.
    //
    TorusGeometryPiece(ProblemSpecP &);
    
    //////////
    // Constructor that takes top, bottom and radius
    //
    TorusGeometryPiece(const Point& center, 
                       const double minor,
                       const double major,
                       const string axis,
                       const double theta);
    
    //////////
    // Destructor
    //
    virtual ~TorusGeometryPiece();
    
    static const string TYPE_NAME;
    virtual std::string getType() const { return TYPE_NAME; }

    /// Make a clone
    virtual GeometryPieceP clone() const;
    
    //////////
    // Determines whether a point is inside the cylinder.
    //
    virtual bool inside(const Point &p) const;
    
    //////////
    // Returns the bounding box surrounding the cylinder.
    virtual Box getBoundingBox() const;
    
    //////////
    // Calculate the surface area
    //
    virtual inline double surfaceArea() const
      {
        return (4.0*M_PI*M_PI*d_major_radius*d_minor_radius);
      }

    //////////
    // Calculate the volume
    //
    virtual inline double volume() const
      {
        return ((2.0*M_PI*M_PI*d_major_radius*d_minor_radius*d_minor_radius));
      }
    
    //////////
    // Calculate the unit normal vector to axis from point
    //
    Vector radialDirection(const Point& pt) const;
    
    //////////
    // Get the top, bottom, radius, height
    //
    inline Point center() const {return d_center;}
    inline double major_radius() const {return d_major_radius;}
    inline double minor_radius() const {return d_minor_radius;}
    inline string axis() const {return d_axis;}

  protected:
    
    virtual void outputHelper( ProblemSpecP & ps ) const;
    
    //////////
    // Constructor needed for subclasses
    //
    TorusGeometryPiece();
    Point d_center;
    double d_major_radius;
    double d_minor_radius;
    string d_axis;
    double d_theta;
  };
} // End namespace Uintah
      
#endif // __TORUS_GEOMTRY_Piece_H__
