#ifndef __PROBLEM_H__
#define __PROBLEM_H__

#include <Uintah/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <Uintah/Components/MPM/BoundCond.h>
#include <Uintah/Components/MPM/GeometrySpecification/Material.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <string>
#include <vector>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/ProblemSpecP.h>

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using Uintah::Interface::ProblemSpec;
using Uintah::Interface::ProblemSpecP;

namespace Uintah {
    namespace Grid {
	class Region;
    }
}
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/GridP.h>

class Problem {
  
 public:
  Problem();
  ~Problem();
  
    
  void preProcessor(Uintah::Interface::ProblemSpecP prob_spec,
		    Uintah::Grid::GridP grid);
  void createParticles(const Uintah::Grid::Region* region, 
		       Uintah::Interface::DataWarehouseP&);
   
  int  getNumMaterial() const;
  int getNumObjects() const;
  std::vector<GeometryObject>* getObjects();
 

 private:
  int d_num_material; //
  std::vector<Material *> d_materials;   //
  int  d_num_objects;  //
  std::vector<GeometryObject> d_objects;
  int d_num_bcs;      // number of boundary conditions;
  std::vector<BoundCond>  d_bcs;          // boundary conditions

};

#endif // __PROBLEM_H__

// $Log$
// Revision 1.4  2000/04/14 02:05:46  jas
// Subclassed out the GeometryPiece into 4 types: Box,Cylinder,Sphere, and
// Tri.  This made the GeometryObject class simpler since many of the
// methods are now relegated to the GeometryPiece subclasses.
//
// Revision 1.3  2000/03/20 17:17:15  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.2  2000/03/15 22:13:08  jas
// Added log and changed header file locations.
//
// Revision 1.1  2000/03/14 22:10:49  jas
// Initial creation of the geometry specification directory with the legacy
// problem setup.
//
// Revision 1.2  2000/02/27 07:48:42  sparker
// Homebrew code all compiles now
// First step toward PSE integration
// Added a "Standalone Uintah Simulation" (sus) executable
// MPM does NOT run yet
//
// Revision 1.1  2000/02/24 06:12:00  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:56  sparker
// Stuff may actually work someday...
//
// Revision 1.6  1999/10/28 23:24:48  jas
// Eliminated the -n option for manually creating multiple boxes.  SAMRAI does
// this automatically now.
//
// Revision 1.5  1999/09/04 23:00:32  jas
// Changed from Particle * to Particle to fix a memory leak.
//
// Revision 1.4  1999/08/18 19:23:09  jas
// Now can return information about the problem specification and object
// bounds.  This is necessary for the creation of particles on the processor
// that they reside.
//
// Revision 1.3  1999/07/28 16:54:55  cgl
// - Multiple Velocity fields in smpm
// - stop creating extra particles
//
// Revision 1.2  1999/06/18 05:44:53  cgl
// - Major work on the make environment for smpm.  See doc/smpm.make
// - fixed getSize(), (un)packStream() for all constitutive models
//   and Particle so that size reported and packed amount are the same.
// - Added infomation to Particle.packStream().
// - fixed internal force summation equation to keep objects from exploding.
// - speed up interpolateParticlesToPatchData()
// - Changed lists of Particles to lists of Particle*s.
// - Added a command line option for smpm `-c npatch'.  Valid values are 1 2 4
//
// Revision 1.1  1999/06/14 06:23:42  cgl
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
