#ifndef __CYLINDER_GEOMETRY_PIECE_H__
#define __CYLINDER_GEOMETRY_PIECE_H__

#include "GeometryPiece.h"

class CylinderGeometryPiece : public GeometryPiece {

 public:

  enum AXIS {X = 1, Y = 2, Z = 3};

  CylinderGeometryPiece();
  CylinderGeometryPiece(AXIS axis, Point origin, double len, double rad);
  virtual ~CylinderGeometryPiece();

 
  virtual int checkShapesPositive(Point check_point, int &np,int piece_num,
				  Vector part_spacing,
				  int ppold);
  virtual int checkShapesNegative(Point check_point, int &np,int piece_num,
				  Vector part_spacing,
				  int ppold);

  virtual void computeNorm(Vector &norm, Point part_pos, int sf[7], 
			   int inPiece, int &np);


 private:
  AXIS  d_axis;
  Point d_origin;
  double d_length;
  double d_radius;
 
  

};

#endif // __CYLINDER_GEOMTRY_PIECE_H__

// $Log$
// Revision 1.1  2000/04/14 02:05:45  jas
// Subclassed out the GeometryPiece into 4 types: Box,Cylinder,Sphere, and
// Tri.  This made the GeometryObject class simpler since many of the
// methods are now relegated to the GeometryPiece subclasses.
//
