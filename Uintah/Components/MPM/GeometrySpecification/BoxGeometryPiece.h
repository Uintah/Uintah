#ifndef __BOX_GEOMETRY_PIECE_H__
#define __BOX_GEOMETRY_PIECE_H__

#include "GeometryPiece.h"


class BoxGeometryPiece : public GeometryPiece {

 public:

  BoxGeometryPiece();
  BoxGeometryPiece(Point lower, Point upper);
  virtual ~BoxGeometryPiece();


  virtual int checkShapesPositive(Point check_point, int &np, int piece_num,
			  Vector part_spacing,int ppold);
  virtual int checkShapesNegative(Point check_point, int &np, int piece_num,
			  Vector part_spacing,int ppold);

  virtual void computeNorm(Vector &norm,Point part_pos, int surf[7], 
			   int ptype, int &np);

 private:
  Point d_lower;
  Point d_upper;

};

#endif // __BOX_GEOMTRY_PIECE_H__

// $Log$
// Revision 1.1  2000/04/14 02:05:45  jas
// Subclassed out the GeometryPiece into 4 types: Box,Cylinder,Sphere, and
// Tri.  This made the GeometryObject class simpler since many of the
// methods are now relegated to the GeometryPiece subclasses.
//
