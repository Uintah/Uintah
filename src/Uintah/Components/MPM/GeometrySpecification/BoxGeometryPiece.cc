#include "BoxGeometryPiece.h"

BoxGeometryPiece::BoxGeometryPiece() {}

BoxGeometryPiece::BoxGeometryPiece(Point lo,Point up) :
  d_lower(lo),d_upper(up)
{
}

BoxGeometryPiece::~BoxGeometryPiece()
{
}

int BoxGeometryPiece::checkShapesPositive(Point check_point, 
					  int &np,int piece_num,
					  Vector particle_spacing,
					  int ppold)
{
  int  pp;
  double xlow,xhigh,ylow,yhigh,zlow,zhigh;

  /*hexahedron*/
  xlow = d_lower.x();
  xhigh= d_upper.x();
  ylow = d_lower.y();
  yhigh= d_upper.y();
  zlow = d_lower.z();
  zhigh= d_upper.z();
  
  if ((check_point.x()-xlow>=0.0) && (xhigh-check_point.x()>=0.0) &&
      (check_point.y()-ylow>=0.0) && (yhigh-check_point.y()>=0.0) &&
      (check_point.z()-zlow>=0.0) && (zhigh-check_point.z()>=0.0) ){
    pp = 1;
    //	  pp = 3; 
    d_in_piece=piece_num;
  } 
  else if(xlow-check_point.x() <= particle_spacing.x() && 
	  xlow-check_point.x() > 0.0){
    pp= -1;
  }
  else if(xhigh-check_point.x() >= -particle_spacing.x() &&
	  xhigh-check_point.x() < 0.0){
    pp= -2;
  }
  else if(ylow-check_point.y() <= particle_spacing.y() &&
	  ylow-check_point.y() > 0.0){
    pp= -3;
  }
  else if(yhigh-check_point.y() >= -particle_spacing.y() &&
	  yhigh-check_point.y() < 0.0){
    pp= -4;
  }
  else if(zlow-check_point.z() <= particle_spacing.z() &&
	  zlow-check_point.z() > 0.0){
    pp= -5;
  }
  else if(zhigh-check_point.z() >= -particle_spacing.z() &&
	  zhigh-check_point.z() < 0.0){
    pp= -6;
  }

  return pp;
}

int BoxGeometryPiece::checkShapesNegative(Point check_point,
					  int &np,int piece_num,
					  Vector particle_spacing,
					  int ppold)
{

  /*hexahedron*/
 
  double xlow = d_lower.x();
  double xhigh= d_upper.x();
  double ylow = d_lower.y();
  double yhigh= d_upper.y();
  double zlow = d_lower.z();
  double zhigh= d_upper.z();
  
  int pp;

  if((check_point.x()-xlow>0.0) && (xhigh-check_point.x()>0.0) &&
     (check_point.y()-ylow>0.0) && (yhigh-check_point.y()>0.0) &&
     (check_point.z()-zlow>0.0) && (zhigh-check_point.z()>0.0) ){
    pp = -10;
    np = piece_num;
  }
  
  if(xlow-check_point.x() >= -particle_spacing.x() &&
     xlow-check_point.x() < 0.0   &&
     check_point.y()-ylow>0.0 && yhigh-check_point.y()>0.0 &&
     check_point.z()-zlow>0.0 && zhigh-check_point.z()>0.0 ){
    pp= -11;
    np = piece_num;
  }
  if(xhigh-check_point.x() <= particle_spacing.x() &&
     xhigh-check_point.x() > 0.0   &&
     check_point.y()-ylow>0.0 && yhigh-check_point.y()>0.0 &&
     check_point.z()-zlow>0.0 && zhigh-check_point.z()>0.0 ){
    pp= -21;
    np = piece_num;
  }
  if(ylow-check_point.y() >= -particle_spacing.y() &&
     ylow-check_point.y() < 0.0    &&
     check_point.x()-xlow>0.0 && xhigh-check_point.x()>0.0 &&
     check_point.z()-zlow>0.0 && zhigh-check_point.z()>0.0 ){
    pp= -31;
    np = piece_num;
  }
  if(yhigh-check_point.y() <= particle_spacing.y() &&
     yhigh-check_point.y() > 0.0  &&
     check_point.x()-xlow>0.0 && xhigh-check_point.x()>0.0 &&
     check_point.z()-zlow>0.0 && zhigh-check_point.z()>0.0 ){
    pp= -41;
    np = piece_num;
  }
  if(zlow-check_point.z() >= -particle_spacing.z() &&
     zlow-check_point.z() < 0.0    &&
     check_point.x()-xlow>0.0 && xhigh-check_point.x()>0.0 &&
     check_point.y()-ylow>0.0 && yhigh-check_point.y()>0.0){
    pp= -51;
    np = piece_num;
  }
  if(zhigh-check_point.z() <= particle_spacing.z() &&
     zhigh-check_point.z() > 0.0   &&
     check_point.x()-xlow>0.0 && xhigh-check_point.x()>0.0 &&
     check_point.y()-ylow>0.0 && yhigh-check_point.y()>0.0){
    pp= -61;
    np = piece_num;
  }
  
  

  return pp;
}

void BoxGeometryPiece::computeNorm(Vector &norm,Point part_pos, 
					   int sf[7], int ptype, int &np)
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
    if(sf[i]==(-1)){		// low x
      dir = Vector(-1.0,0.0,0.0);
      norm+=dir;
    }
    else if(sf[i]==(-2)){		// high x
      dir = Vector(1.0,0.0,0.0);
      norm+=dir;
    }
    else if(sf[i]==(-3)){		// low y
      dir = Vector(0.0,-1.0,0.0);
      norm+=dir;
    }
    else if(sf[i]==(-4)){		// high y
      dir = Vector(0.0,1.0,0.0);
      norm+=dir;
    }
    else if(sf[i]==(-5)){		// low z
      dir = Vector(0.0,0.0,-1.0);
      norm+=dir;
    }
    else if(sf[i]==(-6)){		// high z
      dir = Vector(0.0,0.0,1.0);
      norm+=dir;
    }
  }

  if (small < -10) {
    for(int i=1;i<=6;i++){
      if(sf[i]==(-11)){			// low x
	dir = Vector(1.0,0.0,0.0);
	norm+=dir;
      }
      else if(sf[i]==(-21)){		// high x
	dir = Vector(-1.0,0.0,0.0);
	norm+=dir;
      }
      else if(sf[i]==(-31)){		// low y
	dir = Vector(0.0,1.0,0.0);
	norm+=dir;
      }
      else if(sf[i]==(-41)){		// high y
	dir = Vector(0.0,-1.0,0.0);
	norm+=dir;
      }
      else if(sf[i]==(-51)){		// low z
	dir = Vector(0.0,0.0,1.0);
	norm+=dir;
      }
      else if(sf[i]==(-61)){		// high z
	dir = Vector(0.0,0.0,-1.0);
	norm+=dir;
      }
    }
  }

  // Normalize the surface normal vector
  norm.normalize();

}

// $Log$
// Revision 1.2  2000/04/14 03:29:13  jas
// Fixed routines to use SCICore's point and vector stuff.
//
// Revision 1.1  2000/04/14 02:05:45  jas
// Subclassed out the GeometryPiece into 4 types: Box,Cylinder,Sphere, and
// Tri.  This made the GeometryObject class simpler since many of the
// methods are now relegated to the GeometryPiece subclasses.
//
