#include <Packages/Uintah/Core/GeometryPiece/NaaBoxGeometryPiece.h>

#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>

using namespace std;

using namespace Uintah;
using namespace SCIRun;

NaaBoxGeometryPiece::NaaBoxGeometryPiece(ProblemSpecP& ps)
{
  setName("NaaBox");

  Point p1, p2, p3, p4;
  ps->require("p1", p1);
  ps->require("p2", p2); 
  ps->require("p3", p3);
  ps->require("p4", p4); 
  
  //  SCI_THROW(ProblemSetupException("Input File Error: box max <= min coordinates", __FILE__, __LINE__));

  init( p1, p2, p3, p4 );
}

NaaBoxGeometryPiece::NaaBoxGeometryPiece( const Point& p1,
                                          const Point& p2,
                                          const Point& p3,
                                          const Point& p4 )
{
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

  // Calculate the bounding box
  Point max = Max( Max( Max( p1, p2 ), p3 ), p4 );
  Point min = Min( Min( Min( p1, p2 ), p3 ), p4 );

  boundingBox_ = Box( min, max );

  p2minusP1_ = p2 - p1;
  p3minusP1_ = p3 - p1;
  p4minusP1_ = p4 - p1;

  p2minusP1mag_ = p2minusP1_.length();
  p3minusP1mag_ = p3minusP1_.length();
  p4minusP1mag_ = p4minusP1_.length();

  if( p2minusP1mag_ < 0.000001 || p3minusP1mag_ < 0.000001 || p4minusP1mag_ < 0.000001 ) {
    SCI_THROW(ProblemSetupException("degenerate box", __FILE__, __LINE__));
  }
}

NaaBoxGeometryPiece::~NaaBoxGeometryPiece()
{
}

void
NaaBoxGeometryPiece::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP box_ps = ps->appendChild("parallelepiped");

  box_ps->appendElement( "p1", p1_ );
  box_ps->appendElement( "p2", p2_ );
  box_ps->appendElement( "p3", p3_ );
  box_ps->appendElement( "p4", p4_ );
}

NaaBoxGeometryPiece*
NaaBoxGeometryPiece::clone()
{
  return scinew NaaBoxGeometryPiece(*this);
}

//********************************************************************
// This code is a modified version of the code found here:
//
// http://www.csit.fsu.edu/~burkardt/m_src/geometry/parallelepiped_contains_point_3d.m
//
//  The author, John Burkardt, has graciously permitted us to use it.
//
//  Discussion:
//
//    A parallelepiped is a "slanted box", that is, opposite
//    sides are parallel planes.
//
//         *------------------*
//        / \                / \
//       /   \              /   \
//      /     \            /     \
//    P4------------------*       \
//      \        .         \       \
//       \        .         \       \
//        \        .         \       \
//         \       P2.........\-------\
//          \     /            \     /
//           \   /              \   /
//            \ /                \ /
//             P1----------------P3
//
//  Author:
//
//    Based on the code by John Burkardt
//
//  Returns true if the point is inside (or on) the parallelepiped.
//
bool
NaaBoxGeometryPiece::inside( const Point& pt ) const
{
  //cout << "check inside for pt: " << pt << " ... ";

  double dot;

  Vector ptMinusP1 = pt - p1_;

  dot = Dot( p2minusP1_, ptMinusP1 );

  if ( dot < 0.0 ) {
    //cout << "no\n";
    return false;
  }
  else if( Dot( p2minusP1_, p2minusP1_ ) < dot ) {
    //cout << "no\n";
    return false;
  }

  dot = Dot( p3minusP1_, ptMinusP1 );
  if ( dot < 0.0 ) {
    //cout << "no\n";
    return false;
  }
  else if ( Dot( p3minusP1_, p3minusP1_ ) < dot ) {
    //cout << "no\n";
    return false;
  }

  dot = Dot( p4minusP1_, ptMinusP1 );
  if ( dot < 0.0 ) {
    //cout << "no\n";
    return false;
  }
  else if ( Dot( p4minusP1_, p4minusP1_ ) < dot ) {
    //cout << "no\n";
    return false;
  }
  //cout << "yes\n";
  return true;
}

Box
NaaBoxGeometryPiece::getBoundingBox() const
{
  return boundingBox_;
}

