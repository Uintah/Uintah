#include <Packages/Uintah/Core/GeometryPiece/NaaBoxGeometryPiece.h>

#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>

#include <iostream>

using namespace std;

using namespace Uintah;
using namespace SCIRun;

static DebugStream dbg( "GeometryPiece", false );

const string NaaBoxGeometryPiece::TYPE_NAME = "parallelepiped";

NaaBoxGeometryPiece::NaaBoxGeometryPiece(ProblemSpecP& ps)
{
  name_ = "Unnamed " + TYPE_NAME + " from PS";

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

  boundingBox_ = Box( p1, p5 );

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

