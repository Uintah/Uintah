#include "GeometryPiece.h"

GeomPiece::GeomPiece() {}

GeomPiece::~GeomPiece()
{
}

void GeomPiece::setPieceType(int pt) { pieceType = pt; }

void GeomPiece::setPosNeg(int pn) { piecePosNeg = pn; }

void GeomPiece::setMaterialNum(int mt) { pieceMatNum = mt; }

void GeomPiece::setVelFieldNum(int vf_num) { pieceVelFieldNum = vf_num; }

int GeomPiece::getPosNeg() { return piecePosNeg; }

int GeomPiece::getPieceType() { return pieceType; }

int GeomPiece::getMaterialNum() { return pieceMatNum; }

int GeomPiece::getVFNum() { return pieceVelFieldNum; }

double GeomPiece::getGeomBounds(int j) { return geomBounds[j]; }

void GeomPiece::setGeomBounds(double bnds[7])
{
  int i;
  
  for(i=1;i<=6;i++){
    geomBounds[i] = bnds[i];
  }
}
void GeomPiece::setInitialConditions(double icv[4])
{
  int i;
  
  for(i=1;i<=3;i++){
    initVel[i] = icv[i];
  }
}

double GeomPiece::getInitVel(int i) { return initVel[i]; }

// $Log$
// Revision 1.1  2000/03/14 22:36:05  jas
// Readded geometry specification source files.
//
// Revision 1.1  2000/02/24 06:11:56  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:51  sparker
// Stuff may actually work someday...
//
// Revision 1.1  1999/06/14 06:23:41  cgl
// - src/mpm/Makefile modified to work for IRIX64 or Linux
// - src/grid/Grid.cc added length to character array, since it
// 	was only 4 long, but was being sprintf'd with a 4 character
// 	number, leaving no room for the terminating 0.
// - added smpm directory. to house the samrai version of mpm.
//
// Revision 1.4  1999/02/10 20:53:10  guilkey
// Updated to release 2-0
//
// Revision 1.3  1999/01/26 21:53:33  campbell
// Added logging capabilities
//
