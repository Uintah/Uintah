#ifndef __PROBLEM_H__
#define __PROBLEM_H__

#include <Uintah/Components/MPM/GeometrySpecification/GeometryPiece.h>
#include <Uintah/Components/MPM/BoundCond.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <string>
#include <vector>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/GridP.h>

namespace Uintah {
   class Region;
   namespace MPM {
      class GeometryObject;
      using SCICore::Geometry::Vector;
      using SCICore::Geometry::Point;
   
      class Problem {
      
      public:
	 Problem();
	 ~Problem();
      
      
	 void preProcessor(const ProblemSpecP& prob_spec, GridP& grid,
			   SimulationStateP& sharedState);
	 void createParticles(const Region* region, 
			      DataWarehouseP&);
	 
	 int getNumObjects() const;
	 std::vector<GeometryObject>* getObjects();
	 
	 
      private:
	 int d_num_bcs;      // number of boundary conditions;
	 std::vector<BoundCond>  d_bcs;          // boundary conditions
	 
      };
   } // end namespace MPM
} // end namespace Uintah

#endif // __PROBLEM_H__

// $Log$
// Revision 1.9  2000/04/26 06:48:25  sparker
// Streamlined namespaces
//
// Revision 1.8  2000/04/24 21:04:32  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.7  2000/04/20 18:56:22  sparker
// Updates to MPM
//
// Revision 1.6  2000/04/19 05:26:08  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.5  2000/04/14 03:29:13  jas
// Fixed routines to use SCICore's point and vector stuff.
//
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
