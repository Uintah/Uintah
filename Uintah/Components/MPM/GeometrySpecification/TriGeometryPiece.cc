#include "TriGeometryPiece.h"
#include "GeometryPieceFactory.h"
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Box.h>

using namespace Uintah::Components;

TriGeometryPiece::TriGeometryPiece(ProblemSpecP &ps)
{

  std::string file;

  ps->require("file",file);
  
 
}

TriGeometryPiece::~TriGeometryPiece()
{
}

bool TriGeometryPiece::inside(const Point &p) const
{

}

Box TriGeometryPiece::getBoundingBox() const
{

}

//
// $Log$
// Revision 1.2  2000/04/24 21:04:33  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.5  2000/04/20 22:58:14  sparker
// Resolved undefined symbols
// Trying to make stuff work
//
// Revision 1.4  2000/04/20 22:37:14  jas
// Fixed up the GeometryObjectFactory.  Added findBlock() and findNextBlock()
// to ProblemSpec stuff.  This will iterate through all of the nodes (hopefully).
//
// Revision 1.3  2000/04/20 18:56:23  sparker
// Updates to MPM
//
// Revision 1.2  2000/04/20 15:09:26  jas
// Added factory methods for GeometryObjects.
//
// Revision 1.1  2000/04/19 21:31:09  jas
// Revamping of the way objects are defined.  The different geometry object
// subtypes only do a few simple things such as testing whether a point
// falls inside the object and also gets the bounding box for the object.
// The constructive solid geometry objects:union,difference, and intersection
// have the same simple operations.
//
