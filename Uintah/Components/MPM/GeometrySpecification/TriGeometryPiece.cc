#include "TriGeometryPiece.h"

TriGeometryPiece::TriGeometryPiece() {}

TriGeometryPiece::~TriGeometryPiece()
{
}


int TriGeometryPiece::checkShapesPositive(Point check_point, int &np, 
					     int piece_num,
					     Vector part_spacing, int ppold)
{
  return 0;
}

int TriGeometryPiece::checkShapesNegative(Point check_point, int &np, 
					     int piece_num,
					     Vector part_spacing, int ppold)
{
  return 0;
}

void TriGeometryPiece::computeNorm(Vector &norm, Point part_pos, 
					      int sf[7], int inPiece, int &np)
{


}

// $Log$
// Revision 1.1  2000/04/14 02:05:47  jas
// Subclassed out the GeometryPiece into 4 types: Box,Cylinder,Sphere, and
// Tri.  This made the GeometryObject class simpler since many of the
// methods are now relegated to the GeometryPiece subclasses.
//
