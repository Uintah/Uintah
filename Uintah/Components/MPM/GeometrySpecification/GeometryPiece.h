#ifndef __GEOMETRY_PIECE_H__
#define __GEOMETRY_PIECE_H__

#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/IntVector.h>
#include <SCICore/Geometry/Point.h>

using SCICore::Geometry::Vector;
using SCICore::Geometry::IntVector;
using SCICore::Geometry::Point;

class GeometryPiece {

 public:

  GeometryPiece();
  virtual ~GeometryPiece();
  
  void    setPosNeg(int pn);
  void    setMaterialNum(int mt);
  void    setVelFieldNum(int vf_num);
  int     getPosNeg();
  int     getMaterialNum();
  int	  getVFNum();
  void	  setInitialConditions(Vector icv);
  Vector  getInitVel();
  int     getInPiece();

  virtual int checkShapesPositive(Point check_point, int &np,
                                  int piece_num, Vector part_spacing,
                                  int ppold) = 0;
  virtual int checkShapesNegative(Point check_point, int &np,
                                  int piece_num, Vector part_spacing,
                                  int ppold) = 0;
 
  virtual void computeNorm(Vector &norm,Point part_pos, int surf[7],
                           int ptype, int &np) = 0;        

 protected:
  
  int             d_piece_pos_neg;
  int             d_piece_mat_num;
  int		  d_piece_vel_field_num;
  Vector          d_init_cond_vel;
  int             d_in_piece;
  IntVector  d_num_particles_cell;  

};

#endif // __GEOEMTRY_PIECE_H__

// $Log$
// Revision 1.2  2000/04/14 02:05:46  jas
// Subclassed out the GeometryPiece into 4 types: Box,Cylinder,Sphere, and
// Tri.  This made the GeometryObject class simpler since many of the
// methods are now relegated to the GeometryPiece subclasses.
//
// Revision 1.1  2000/03/14 22:10:49  jas
// Initial creation of the geometry specification directory with the legacy
// problem setup.
//
// Revision 1.1  2000/02/24 06:11:57  sparker
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
// Revision 1.3  1999/01/26 21:53:34  campbell
// Added logging capabilities
//
