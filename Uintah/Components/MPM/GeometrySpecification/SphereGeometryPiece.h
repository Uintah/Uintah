#ifndef __SPHERE_GEOMETRY_PIECE_H__
#define __SPHERE_GEOMETRY_PIECE_H__

#include "GeometryPiece.h"
#include <math.h>

class SphereGeometryPiece : public GeometryPiece {

 public:

  SphereGeometryPiece();
  SphereGeometryPiece(const double r, const Point o);
  virtual ~SphereGeometryPiece();
 
  virtual int checkShapesPositive(Point check_point, int &np, int piece_num,
				  Vector part_spacing,int ppold);
  virtual int checkShapesNegative(Point check_point, int &np, int piece_num,
				  Vector part_spacing,int ppold);

  virtual void computeNorm(Vector &norm, Point part_pos, int sf[7], 
			   int inPiece, int &np);



 private:
 
  Point d_origin;
  double d_radius;
};

#endif // __SPHERE_GEOMETRY_PIECE_H__

// $Log$
// Revision 1.1  2000/04/14 02:05:46  jas
// Subclassed out the GeometryPiece into 4 types: Box,Cylinder,Sphere, and
// Tri.  This made the GeometryObject class simpler since many of the
// methods are now relegated to the GeometryPiece subclasses.
//
