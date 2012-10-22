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

#include <Core/GeometryPiece/NaaBoxGeometryPiece.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Box.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>

#include <iostream>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

static DebugStream dbg( "GeometryPiece", false );

const string NaaBoxGeometryPiece::TYPE_NAME = "parallelepiped";

NaaBoxGeometryPiece::NaaBoxGeometryPiece(ProblemSpecP& ps)
{
  string gp_label = "Unamed";

  if( !ps->getAttribute( "label", gp_label ) ) {
    // "label" and "name" are both used... so check for "label" first, and if it isn't found, then check for "name".
    ps->getAttribute( "name", gp_label );
  }

  name_ = gp_label + " " + TYPE_NAME + " from PS";

  Point p1, p2, p3, p4;
  ps->require("p1", p1);
  ps->require("p2", p2); 
  ps->require("p3", p3);
  ps->require("p4", p4); 
  
  init( p1, p2, p3, p4 );
}

NaaBoxGeometryPiece::NaaBoxGeometryPiece( const Point& p1,
                                          const Point& p2,
                                          const Point& p3,
                                          const Point& p4 )
{
  name_ = "Unnamed " + TYPE_NAME + " from points";
  init( p1, p2, p3, p4 );
}
  
void
NaaBoxGeometryPiece::init( const Point& p1,
                           const Point& p2,
                           const Point& p3,
                           const Point& p4 )
{
  p1_ = p1;
  p2_ = p2;
  p3_ = p3;
  p4_ = p4;

  Vector  p2minusP1, p3minusP1, p4minusP1;
  p2minusP1 = p2 - p1;
  p3minusP1 = p3 - p1;
  p4minusP1 = p4 - p1;

  // p5 is the opposite corner to p1 and is used for the bounding box.
  Point p5 = p1 + (p2minusP1 + p3minusP1 + p4minusP1);

  // Find the bounding box with the following gross code
  double lowX = min(min(min(p1.x(),p2.x()),min(p2.x(),p3.x())),p4.x());
  double lowY = min(min(min(p1.y(),p2.y()),min(p2.y(),p3.y())),p4.y());
  double lowZ = min(min(min(p1.z(),p2.z()),min(p2.z(),p3.z())),p4.z());
  double highX = max(max(max(p1.x(),p2.x()),max(p2.x(),p3.x())),p4.x());
  double highY = max(max(max(p1.y(),p2.y()),max(p2.y(),p3.y())),p4.y());
  double highZ = max(max(max(p1.z(),p2.z()),max(p2.z(),p3.z())),p4.z());

  Point blow = Point(lowX,lowY,lowZ);
  Point bhigh = Point(highX,highY,highZ);

  boundingBox_ = Box( blow, bhigh );

  if( boundingBox_.degenerate() ) {
    // 1st point must be '<' second point, so flip them.
    boundingBox_.fixBoundingBox();
    if( boundingBox_.degenerate() ) {
      // If there are still problems, throw an exception...

      std::ostringstream error;
      error << "NaaBoxGeometryPiece.cc: boundingBox_ for '" + name_ + "' is degenerate..." << boundingBox_ << "\n";
      error << "See src/Core/GeometryPiece/NaaBoxGeometryPiece.h or the Users Guide for details\n";

      throw ProblemSetupException( error.str(), __FILE__, __LINE__ );
    }
  }

  dbg << "Creating NaaBoxx with BBox of: " << boundingBox_ << "\n";

  // Map the arbitrary box to a unit cube... 
  Matrix3 mat( p2minusP1.x(), p3minusP1.x(),  p4minusP1.x(), 
               p2minusP1.y(), p3minusP1.y(),  p4minusP1.y(), 
               p2minusP1.z(), p3minusP1.z(),  p4minusP1.z() );

  toUnitCube_ = mat.Inverse();
}

NaaBoxGeometryPiece::~NaaBoxGeometryPiece()
{
}

void
NaaBoxGeometryPiece::outputHelper( ProblemSpecP & ps ) const {
  ps->appendElement( "p1", p1_ );
  ps->appendElement( "p2", p2_ );
  ps->appendElement( "p3", p3_ );
  ps->appendElement( "p4", p4_ );
}

GeometryPieceP
NaaBoxGeometryPiece::clone() const
{
  return scinew NaaBoxGeometryPiece(*this);
}

//********************************************
//                                          //
//             *-------------*              //
//            / .           / \             //
//           /   .         /   \            //
//          P4-------------*    \           //
//           \    .         \    \          //
//            \   P2.........\....*         //
//             \ .            \  /          //
//             P1--------------P3           //
//
//  Returns true if the point is inside (or on) the parallelepiped.
//  (The order of p2, p3, and p4 don't really matter.)
//
//  The arbitrary box has been transformed into a unit cube... we take
//  the Point to check and transform it the same way, then just check
//  to see if the Pt is in the unit cube.
//
bool
NaaBoxGeometryPiece::inside( const Point& pt ) const
{
  Vector result = toUnitCube_ * (pt - p1_);

  if( ( result.minComponent() > 0 ) && ( result.maxComponent() <= 1.0 ) )
    return true;
  else
    return false;
}

Box
NaaBoxGeometryPiece::getBoundingBox() const
{
  return boundingBox_;
}

