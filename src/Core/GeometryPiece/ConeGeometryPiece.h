/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

#ifndef __CONE_GEOMETRY_OBJECT_H__
#define __CONE_GEOMETRY_OBJECT_H__

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/CylinderGeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace Uintah {

/**************************************
        
CLASS
   ConeGeometryPiece
        
   Creates a oriented right circular cone/frustrum of a cone
   from the xml input file description.
        
GENERAL INFORMATION
        
   ConeGeometryPiece.h
        
   Biswajit Banerjee
   Department of Mechanical Engineering
   University of Utah
        
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
        
KEYWORDS
   ConeGeometryPiece BoundingBox inside
        
DESCRIPTION
   Creates a oriented right circular cone or frustrum of a cone
   from the xml input file description.
   Requires four inputs: bottom point, top point 
                         bootom radius, top radius
   There are methods for checking if a point is inside the cylinder
   and also for determining the bounding box for the cylinder.
   The input form looks like this:
       <cone>
         <bottom>[0.,0.,0.]</bottom>
         <top>[0.,0.,0.]</top>
         <bot_radius>2.0</bot_radius>
         <top_radius>1.0</top_radius>
       </cone>
   If any of the radii are ommitted the corresponding radius is 
   assumed to be zero, i.e. this point is the vertex of the right 
   circular cone.  Stores bottom radius in d_radius of 
   CylinderGeometryPiece.
        
WARNING
        
****************************************/

  class ConeGeometryPiece : public CylinderGeometryPiece {
    
  public:
    //////////
    // Constructor that takes a ProblemSpecP argument.   
    // It reads the xml input specification and builds 
    // a generalized cone.
    ConeGeometryPiece(ProblemSpecP &);
    
    //////////
    // Constructor that takes top, bottom and radius
    //
    ConeGeometryPiece(const Point& top, 
                      const Point& bottom,
                      double topRad,
                      double botRad);
    
    
    static const string TYPE_NAME;
    virtual std::string getType() const { return TYPE_NAME; }

    virtual GeometryPieceP clone() const;

    //////////
    // Destructor
    virtual ~ConeGeometryPiece();

    //////////
    // Determines whether a point is inside the cone.
    virtual bool inside(const Point &p) const;
    
    //////////
    // Returns the bounding box surrounding the cone.
    virtual Box getBoundingBox() const;
    
    //////////
    // Calculate the lateral surface area of the cone
    virtual double surfaceArea() const;
    
    //////////
    // Calculate the volume
    virtual double volume() const;
    
    //////////
    // Get the top and bottom radius
    //
    inline double topRadius() const {return d_topRad;}
    inline double botRadius() const {return d_radius;}
    
  private:
    virtual void outputHelper( ProblemSpecP & ps ) const;

    double d_topRad;
  };

} // End namespace Uintah

#endif // __CONE_GEOMTRY_Piece_H__
