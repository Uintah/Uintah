#ifndef __GEOMETRY_OBJECT_H__
#define __GEOMETRY_OBJECT_H__

#include "GeometryPiece.h"
#include "BoundCond.h"
#include "DataWarehouseP.h"
#include <iosfwd>
#include <vector>
namespace SCICore {
    namespace Geometry {
	class Vector;
    }
}
class Material;
class ParticleSet;
class Region;

class GeomObject {
private:

  int                     numPieces,objectNumber,inPiece;
  double                  dXYZ[4];
  int                     numParPCell[4];
  GeomPiece               gP[50];
  double                  probBounds[7];
  double                  dxpp,dypp,dzpp; // distance between particles
       
public:

  GeomObject();
  ~GeomObject();
  
  void setObjInfo(int n, double bds[7], int np,double dx[4],int nppc[4]);
  double * getObjInfoBounds();
  void setObjInfoBounds(double bds[7]);
  int getObjInfoNumPieces();
  double * getObjInfoDx();
  int * getObjInfoNumParticlesPerCell();
  void addPieces(std::ifstream &filename);
  int  getPieceInfo();
  void readFromFile(std::ifstream &filename);
  void FillWParticles(std::vector<Material *> materials,
		      std::vector<BoundCond> BC,
		      const Region* region,
		      DataWarehouseP&);
  int CheckShapes(double x[3], int &np);
  void Surface(double x[3], int surf[7], int &np);
  void Norm(SCICore::Geometry::Vector &norm,double x[3], int surf[7], int ptype, int &np);

};

#endif // __GEOMETRY_OBJECT_H__
// $Log$
// Revision 1.1  2000/02/24 06:11:56  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:51  sparker
// Stuff may actually work someday...
//
// Revision 1.4  1999/09/04 23:00:32  jas
// Changed from Particle * to Particle to fix a memory leak.
//
// Revision 1.3  1999/08/18 19:23:09  jas
// Now can return information about the problem specification and object
// bounds.  This is necessary for the creation of particles on the processor
// that they reside.
//
// Revision 1.2  1999/06/18 05:44:52  cgl
// - Major work on the make environment for smpm.  See doc/smpm.make
// - fixed getSize(), (un)packStream() for all constitutive models
//   and Particle so that size reported and packed amount are the same.
// - Added infomation to Particle.packStream().
// - fixed internal force summation equation to keep objects from exploding.
// - speed up interpolateParticlesToPatchData()
// - Changed lists of Particles to lists of Particle*s.
// - Added a command line option for smpm `-c npatch'.  Valid values are 1 2 4
//
// Revision 1.1  1999/06/14 06:23:41  cgl
// - src/mpm/Makefile modified to work for IRIX64 or Linux
// - src/grid/Grid.cc added length to character array, since it
// 	was only 4 long, but was being sprintf'd with a 4 character
// 	number, leaving no room for the terminating 0.
// - added smpm directory. to house the samrai version of mpm.
//
// Revision 1.4  1999/05/26 20:30:51  guilkey
// Added the capability for determining which particles are on the surface
// and what the surface normal of those particles is.  This information
// is output to part.pos.
//
// Revision 1.3  1999/01/26 21:53:33  campbell
// Added logging capabilities
//
