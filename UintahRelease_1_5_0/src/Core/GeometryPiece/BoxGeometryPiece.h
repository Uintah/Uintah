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


#ifndef __BOX_GEOMETRY_OBJECT_H__
#define __BOX_GEOMETRY_OBJECT_H__

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Box.h>

namespace Uintah {

/**************************************
	
CLASS
   BoxGeometryPiece
	
   Creates a box from the xml input file description.
	
GENERAL INFORMATION
	
   BoxGeometryPiece.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   BoxGeometryPiece BoundingBox inside
	
DESCRIPTION
   Creates a box from the xml input file description.
   Requires two inputs: lower left point and upper right point.  
   There are methods for checking if a point is inside the box
   and also for determining the bounding box for the box (which
   just returns the box itself).
   The input form looks like this:
       <box>
         <min>[0.,0.,0.]</min>
	 <max>[1.,1.,1.]</max>
       </box>
	
	
WARNING
	
****************************************/


      class BoxGeometryPiece : public GeometryPiece {
	 
      public:
	 //////////
	 // Constructor that takes a ProblemSpecP argument.   It reads the xml 
	 // input specification and builds a generalized box.
	 BoxGeometryPiece(ProblemSpecP&);

	 //////////
	 // Construct a box from a min/max point
	 BoxGeometryPiece(const Point& p1, const Point& p2);
	 
	 //////////
	 // Destructor
	 virtual ~BoxGeometryPiece();

         static const string TYPE_NAME;
         virtual std::string getType() const { return TYPE_NAME; }

	 /// Make a clone
	 virtual GeometryPieceP clone() const;

	 //////////
	 // Determines whether a point is inside the box.
	 virtual bool inside(const Point &p) const;
	 
	 //////////
	 //  Returns the bounding box surrounding the box (ie, the box itself).
	 virtual Box getBoundingBox() const;
	 
	 //////////
	 //  Returns the volume of the box
	 double volume() const;

	 //////////
	 //  Returns the length pf the smallest side
	 double smallestSide() const;

	 //////////
	 //  Returns the thickness direction (direction
	 //  of smallest side)
	 unsigned int thicknessDirection() const;

      private:
         virtual void outputHelper( ProblemSpecP & ps ) const;

	 Box d_box;
	 
      };
} // End namespace Uintah

#endif // __BOX_GEOMTRY_Piece_H__
