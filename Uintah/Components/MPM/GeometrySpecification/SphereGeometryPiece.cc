#include "SphereGeometryPiece.h"
#include <SCICore/Geometry/Vector.h>


SphereGeometryPiece::SphereGeometryPiece()
{
}

SphereGeometryPiece::SphereGeometryPiece(const double r, const  Point o):
  d_origin(o),d_radius(r)
{
}

SphereGeometryPiece::~SphereGeometryPiece()
{
}

int SphereGeometryPiece::checkShapesPositive(Point check_point, 
					     int &np, int piece_num,
					     Vector part_spacing,
					     int ppold)
{
 
  int pp = 0;
  Point point_diff(check_point - d_origin);
  Vector diff(point_diff.x(),point_diff.y(),point_diff.z());
  double len = diff.length();
  double len_sq = len*len;
  double radius_sq = d_radius*d_radius;

   // Positive space piece
  

  if (len_sq <= radius_sq) {
    pp = 3;
    d_in_piece=piece_num;
  }
  
  
  return pp;
}

int SphereGeometryPiece::checkShapesNegative(Point check_point, 
					     int &np, int piece_num,
					     Vector part_spacing,
					     int ppold)
{

  int pp = 0;
  Point point_diff(check_point - d_origin);
  Vector diff(point_diff.x(),point_diff.y(),point_diff.z());
  double len = diff.length();
  double len_sq = len*len;
  double radius_sq = d_radius*d_radius;

 
  // Negative space piece

  if (len_sq < radius_sq) {
    pp = -30;
    np = piece_num;
  }
  
  
  return pp;
}

void SphereGeometryPiece::computeNorm(Vector &norm, Point part_pos, 
					      int sf[7], int inPiece, int &np)
{

  Vector dir(0.0,0.0,0.0);
  norm = Vector(0.0,0.0,0.0);
  int small = 1;

  for(int i=1;i<=6;i++) {
    if(sf[i] < small) { 
      small = sf[i]; 
    }
  }
  
  // Not a surface point
  if(small == 1){ 
    return; 
  }		
   
  for(int i=1;i<=6;i++){
    if(sf[i]==(0)){	
      // sphere's surface
      Point tmp = Point((part_pos - d_origin)*2.);
      dir = Vector(tmp.x(),tmp.y(),tmp.z());
      dir.normalize();
      norm+=dir;
    }
  }
  
  // The point is on the surface of a "negative" object.
  // Get the geometry information for the negative object so we can determine
  // a surface normal.

  if(small < -10) {	
    for(int i=1;i<=6;i++){
      if(sf[i]==(-30)){		
	// sphere's surface
	Point tmp = Point((part_pos - d_origin)*(-2.));
	dir = Vector(tmp.x(),tmp.y(),tmp.z());
	dir.normalize();
	norm+=dir;
      }
    }
  }
  
  norm.normalize();

}


// $Log$
// Revision 1.2  2000/04/14 03:29:14  jas
// Fixed routines to use SCICore's point and vector stuff.
//
// Revision 1.1  2000/04/14 02:05:46  jas
// Subclassed out the GeometryPiece into 4 types: Box,Cylinder,Sphere, and
// Tri.  This made the GeometryObject class simpler since many of the
// methods are now relegated to the GeometryPiece subclasses.
//

