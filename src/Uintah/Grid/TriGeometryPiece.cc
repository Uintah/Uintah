#include "TriGeometryPiece.h"
#include "GeometryPieceFactory.h"
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Box.h>
#include <iostream>

using namespace Uintah::MPM;
using namespace Uintah;
using namespace std;

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
    cerr << "TriGeometry piece not finished!\n";
    return false;
}

Box TriGeometryPiece::getBoundingBox() const
{
    cerr << "TriGeometry piece not finished!\n";
    return Box();
}

//
// $Log$
// Revision 1.3  2000/09/26 00:44:54  witzel
// needed "#include <iostream>" and "using namespace std;"
//
// Revision 1.2  2000/09/25 20:37:43  sparker
// Quiet g++ compiler warnings
// Work around g++ compiler bug instantiating vector<NCVariable<Vector> >
// Added computeVariableExtents to (eventually) simplify data warehouses
//
// Revision 1.1  2000/06/09 18:38:22  jas
// Moved geometry piece stuff to Grid/ from MPM/GeometryPiece/.
//
// Revision 1.3  2000/04/26 06:48:26  sparker
// Streamlined namespaces
//
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
