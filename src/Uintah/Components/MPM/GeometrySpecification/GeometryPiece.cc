#include "GeometryPiece.h"

GeometryPiece::GeometryPiece()
{
}

GeometryPiece::~GeometryPiece()
{
}


void GeometryPiece::setPosNeg(int pn) 
{ 
  d_piece_pos_neg = pn; 
}

void GeometryPiece::setMaterialNum(int mt)
{
  d_piece_mat_num = mt; 
}

void GeometryPiece::setVelFieldNum(int vf_num) 
{ 
  d_piece_vel_field_num = vf_num; 
}

int GeometryPiece::getPosNeg() 
{ 
  return d_piece_pos_neg; 
}


int GeometryPiece::getMaterialNum() 
{ 
  return d_piece_mat_num; 
}

int GeometryPiece::getVFNum() 
{ 
  return d_piece_vel_field_num; 
}


void GeometryPiece::setInitialConditions(Vector icv)
{
  d_init_cond_vel = icv;
}

Vector GeometryPiece::getInitVel() 
{  
  return d_init_cond_vel;; 

}

int GeometryPiece::getInPiece() 
{
  return d_in_piece;
}
// $Log$
// Revision 1.3  2000/04/14 03:45:40  jas
// Added getInPiece method.
//
// Revision 1.2  2000/04/14 02:05:45  jas
// Subclassed out the GeometryPiece into 4 types: Box,Cylinder,Sphere, and
// Tri.  This made the GeometryObject class simpler since many of the
// methods are now relegated to the GeometryPiece subclasses.
//
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
