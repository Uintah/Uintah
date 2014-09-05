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

