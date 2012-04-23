/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


#ifndef __CYLINDER_GEOMETRY_OBJECT_H__
#define __CYLINDER_GEOMETRY_OBJECT_H__

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#ifndef M_PI
# define M_PI           3.14159265358979323846  /* pi */
#endif

namespace Uintah {

/**************************************
	
CLASS
   CylinderGeometryPiece
	
   Creates a generalized cylinder from the xml input file description.
	
GENERAL INFORMATION
	
   CylinderGeometryPiece.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   CylinderGeometryPiece BoundingBox inside
	
DESCRIPTION
   Creates a generalized cylinder from the xml input file description.
   Requires three inputs: bottom point, top point and a radius.  
   There are methods for checking if a point is inside the cylinder
   and also for determining the bounding box for the cylinder.
   The input form looks like this:
       <cylinder>
         <bottom>[0.,0.,0.]</bottom>
	 <top>[0.,0.,0.]</top>
	 <radius>2.0</radius>
       </cylinder>
	
WARNING
	
****************************************/

  class CylinderGeometryPiece : public GeometryPiece {
    
  public:
    //////////
    // Constructor that takes a ProblemSpecP argument.   It reads the xml 
    // input specification and builds a generalized cylinder.
    //
    CylinderGeometryPiece(ProblemSpecP &);
    
    //////////
    // Constructor that takes top, bottom and radius
    //
    CylinderGeometryPiece(const Point& top, 
                          const Point& bottom,
                          double radius);
    
    //////////
    // Destructor
    //
    virtual ~CylinderGeometryPiece();
    
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
        return ((2.0*M_PI*d_radius)*height());
      }

    //////////
    // Calculate the surface area
    //
    virtual inline double surfaceAreaEndCaps() const
      {
        return ( 2.*M_PI*d_radius*d_radius);
      }
    
    //////////
    // Calculate the volume
    //
    virtual inline double volume() const
      {
        return ((M_PI*d_radius*d_radius)*height());
      }
    
    //////////
    // Calculate the unit normal vector to axis from point
    //
    Vector radialDirection(const Point& pt) const;
    
    //////////
    // Get the top, bottom, radius, height
    //
    inline Point top() const {return d_top;}
    inline Point bottom() const {return d_bottom;}
    inline double radius() const {return d_radius;}
    inline double height() const { return (d_top-d_bottom).length();}
    inline bool cylinder_end() const {return d_cylinder_end;}
    inline bool axisymmetric_end() const {return d_axisymmetric_end;}
    inline bool axisymmetric_side() const {return d_axisymmetric_side;}
    
  protected:
    
    virtual void outputHelper( ProblemSpecP & ps ) const;
    
    //////////
    // Constructor needed for subclasses
    //
    CylinderGeometryPiece();
    Point d_bottom;
    Point d_top;
    double d_radius;
    bool d_cylinder_end;
    bool d_axisymmetric_end;
    bool d_axisymmetric_side;
  };
} // End namespace Uintah
      
#endif // __CYLINDER_GEOMTRY_Piece_H__
