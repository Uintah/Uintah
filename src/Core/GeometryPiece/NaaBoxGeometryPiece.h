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

#ifndef __NAABOX_GEOMETRY_OBJECT_H__
#define __NAABOX_GEOMETRY_OBJECT_H__

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Box.h>
#include <Core/Math/Matrix3.h>

#include <Core/Geometry/Vector.h>

namespace Uintah {

/**************************************
	
CLASS
   NaaBoxGeometryPiece
	
   Creates a NON-ACCESS-ALIGNED box from the xml input file description.
	
GENERAL INFORMATION
	
   NaaBoxGeometryPiece.h
	
   J. Davison de St. Germain
   SCI Institute
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
KEYWORDS
   Non-Access-Aligned NaaBoxGeometryPiece BoundingBox inside Parallelepiped
	
DESCRIPTION

	
****************************************/

class NaaBoxGeometryPiece : public GeometryPiece {
	 
public:
  //////////
  // Construct a box from four points.  //
  //                                    //
  //       *------------------*         //
  //      / \                / \        //
  //    P4...\..............*   \       //
  //      \   \             .    \      //
  //       \   P2-----------------*     //
  //        \  /             .   /      //
  //         \/               . /       //
  //         P1---------------P3        //
  //                                    //
  // (The order of p2, p3, and p4 don't really matter.)
  NaaBoxGeometryPiece( const Point& p1, const Point& p2, 
                       const Point& p3, const Point& p4 );

  //////////
  // Constructor that takes a ProblemSpecP argument.   It reads the xml 
  // input specification and builds a generalized box.  UPS file should
  // use:
  //      <parallelepiped label = "cube">
  //          <p1>           [1.0, 1.0, 1.0]   </p1>
  //          <p2>           [1.0, 1.5, 1.0]   </p2>
  //          <p3>           [1.5, 1.0, 1.0]   </p3>
  //          <p4>           [1.0, 1.0, 1.5]   </p4>
  //      </parallelepiped>
  NaaBoxGeometryPiece(ProblemSpecP&);

  //////////
  // Destructor
  virtual ~NaaBoxGeometryPiece();

  static const string TYPE_NAME;
  virtual std::string getType() const { return TYPE_NAME; }

  /// Make a clone
  virtual GeometryPieceP clone() const;

  //////////
  // Determines whether a point is inside the box.
  virtual bool inside( const Point & pt ) const;
	 
  //////////
  //  Returns the bounding box surrounding the NaaBox
  virtual Box getBoundingBox() const;
	 
private:

  virtual void outputHelper( ProblemSpecP & ps ) const;

  // Called by the different constructors to create the NaaBox
  void init( const Point& p1, const Point& p2, 
             const Point& p3, const Point& p4 );

  Point   p1_, p2_, p3_, p4_;
  Matrix3 toUnitCube_;

  Box boundingBox_;
	 
}; // end class NaaBoxGeometryPiece

} // End namespace Uintah

#endif // __NAABOX_GEOMTRY_Piece_H__
