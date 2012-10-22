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

#ifndef __NULL_GEOMETRY_OBJECT_H__
#define __NULL_GEOMETRY_OBJECT_H__

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Box.h>
#include <Core/Geometry/Point.h>
#include <vector>

namespace Uintah {

/**************************************
	
CLASS
   NullGeometryPiece
	
   Reads in a set of points and optionally a volume for each point from an
   input text file.
	
GENERAL INFORMATION
	
   NullGeometryPiece.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   NullGeometryPiece BoundingBox inside
	
DESCRIPTION
   Reads in a set of points from an input file.  Optionally, if the
   <var> tag is present, the volume will be read in for each point set.
   Requires one input: file name <name>points.txt</name>
   Optional input : <var>p.volume </var>
   There are methods for checking if a point is inside the box
   and also for determining the bounding box for the box (which
   just returns the box itself).
   The input form looks like this:
       <name>file_name.txt</name>
         <var>p.volume</var>
	
	
WARNING
	
****************************************/


      class NullGeometryPiece : public GeometryPiece {
	 
      public:
	 //////////
	 // Constructor that takes a ProblemSpecP argument.   It reads the xml 
	 // input specification and builds a generalized box.
	 NullGeometryPiece(ProblemSpecP&);

	 //////////
	 // Construct a box from a min/max point
	 NullGeometryPiece(const string& file_name);
	 
	 //////////
	 // Destructor
	 virtual ~NullGeometryPiece();

         static const string TYPE_NAME;
         virtual std::string getType() const { return TYPE_NAME; }

	 /// Make a clone
         virtual GeometryPieceP clone() const;

	 //////////
	 // Determines whether a point is inside the box.
	 virtual bool inside(const Point &p) const;
	 
	 //////////
	 //  Returns the bounding box surrounding the cylinder.
	 virtual Box getBoundingBox() const;

      private:
         virtual void outputHelper( ProblemSpecP & ps) const;

	 Box d_box;
      };

} // End namespace Uintah

#endif // __NULL_GEOMTRY_Piece_H__
