#include "BoundCond.h"

BoundCond::BoundCond()
{
}
BoundCond::~BoundCond()
{
}

void BoundCond::setBC(int obj, int piece, int surf, double force[4])
{
  boundaryObject = obj;
  boundaryPiece	= piece;
  boundarySurf = surf;
  boundaryForce[1]= force[1];
  boundaryForce[2]= force[2];
  boundaryForce[3]= force[3];
	
}

int BoundCond::getBCObj(){ return boundaryObject; }
int BoundCond::getBCPiece(){ return boundaryPiece; }
int BoundCond::getBCSurf(){ return boundarySurf; }

double BoundCond::getBCForce(int i) { return boundaryForce[i]; } 

// $Log$
// Revision 1.1  2000/02/24 06:11:52  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:46  sparker
// Stuff may actually work someday...
//
// Revision 1.1  1999/06/14 06:23:40  cgl
// - src/mpm/Makefile modified to work for IRIX64 or Linux
// - src/grid/Grid.cc added length to character array, since it
// 	was only 4 long, but was being sprintf'd with a 4 character
// 	number, leaving no room for the terminating 0.
// - added smpm directory. to house the samrai version of mpm.
//
// Revision 1.2  1999/01/26 21:53:32  campbell
// Added logging capabilities
//
