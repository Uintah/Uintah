#ifndef __BOUND_COND_H__
#define __BOUND_COND_H__

class BoundCond {
private:

  int boundaryObject,boundaryPiece,boundarySurf;
  double boundaryForce[4];

public:

  BoundCond();
  ~BoundCond();

  void setBC(int obj,int piece,int surf,double force[4]);
  int	getBCObj();
  int	getBCPiece();
  int	getBCSurf();
  double getBCForce(int i);

};

#endif // __BOUND_COND_H__

// $Log$
// Revision 1.1  2000/02/24 06:11:52  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:46  sparker
// Stuff may actually work someday...
//
// Revision 1.1  1999/06/14 06:23:40  cgl
// - src/mpm/Makefile modified to work for IRIX64 or Linux
// - src/grid/Grid.cc added length to character array, since it
// 	was only 4 long, but was being sprintf'd with a 4 character
// 	number, leaving no room for the terminating 0.
// - added smpm directory. to house the samrai version of mpm.
//
// Revision 1.2  1999/01/26 21:53:33  campbell
// Added logging capabilities
// 
