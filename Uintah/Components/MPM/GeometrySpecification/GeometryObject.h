#ifndef __GEOMETRY_OBJECT_H__
#define __GEOMETRY_OBJECT_H__

#include <Uintah/Components/MPM/GeometrySpecification/GeometryPiece.h>
#include <Uintah/Components/MPM/BoundCond.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <iosfwd>
#include <vector>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/IntVector.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Box.h>


using Uintah::Interface::ProblemSpecP;
using Uintah::Interface::ProblemSpec;
using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;
using SCICore::Geometry::IntVector;
using Uintah::Grid::Box;

namespace SCICore {
    namespace Geometry {
	class Vector;
    }
}
namespace Uintah {
namespace Components {
class Material;
class ParticleSet;
class Region;

class GeometryObject {
       
 public:

  GeometryObject();
  virtual ~GeometryObject();
  
  void setObjInfo(int n, Point low, Point hi, Vector dx,IntVector ppc);
  Point getObjInfoBoundsLower();
  Point getObjInfoBoundsUpper();
  void setObjInfoBounds(Point lo, Point up);
  int getObjInfoNumPieces();
  Vector getObjInfoDx();
  IntVector getObjInfoNumParticlesPerCell();
  void addPieces(ProblemSpecP prob_spec);
  int  getPieceInfo();
  void readFromFile(std::ifstream &filename);
  void fillWithParticles(std::vector<Material *> &materials,
		      std::vector<BoundCond> &BC,
		      const Region* region,
		      Uintah::Interface::DataWarehouseP&);
  int checkShapes(Point part_pos, int &np);
  void surface(Point part_pos, int surf[7], int &np);
  void norm(Vector &norm,Point part_pos, int surf[7], int ptype, int &np);

  virtual void add(GeometryObject* go);
  virtual Box getBoundingBox() const;
  virtual bool inside(const Point &p) const;

 private:

  int d_num_pieces;
  int d_object_number;
  int d_in_piece;
  Vector d_xyz;
  IntVector d_num_par_per_cell;
  std::vector<GeometryPiece *> d_geom_pieces;
  Point d_upper_prob_coord;
  Point d_lower_prob_coord;
  Vector d_particle_spacing; // distance between particles

};
} // end namespace Components
} // end namespace Uintah

#endif // __GEOMETRY_OBJECT_H__
// $Log$
// Revision 1.7  2000/04/19 21:31:08  jas
// Revamping of the way objects are defined.  The different geometry object
// subtypes only do a few simple things such as testing whether a point
// falls inside the object and also gets the bounding box for the object.
// The constructive solid geometry objects:union,difference, and intersection
// have the same simple operations.
//
// Revision 1.6  2000/04/19 05:26:07  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.5  2000/04/14 02:05:45  jas
// Subclassed out the GeometryPiece into 4 types: Box,Cylinder,Sphere, and
// Tri.  This made the GeometryObject class simpler since many of the
// methods are now relegated to the GeometryPiece subclasses.
//
// Revision 1.4  2000/03/20 17:17:14  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.3  2000/03/15 22:13:08  jas
// Added log and changed header file locations.
//
// Revision 1.2  2000/03/15 21:58:24  jas
// Added logging and put guards in.
//
// Revision 1.1  2000/03/14 22:10:49  jas
// Initial creation of the geometry specification directory with the legacy
// problem setup.
//
// Revision 1.2  2000/02/27 07:48:41  sparker
// Homebrew code all compiles now
// First step toward PSE integration
// Added a "Standalone Uintah Simulation" (sus) executable
// MPM does NOT run yet
//
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
