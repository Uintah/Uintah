#ifndef __PROBLEM_H__
#define __PROBLEM_H__

#include "GeometryObject.h"
#include "BoundCond.h"
#include "Material.h"
#include <string>
#include <vector>
class Region;
#include <Uintah/Interface/DataWarehouseP.h>

class Problem {
private:
  int                NumMaterial; //
    std::vector<Material *> materials;   //
  double             bnds[7];     // 1-6 are x-z boundaries xlo, xhi, ylo, yhi
  double             dx[4];       // 1-3 are spacing
  int                nppcel[4];   // 1-3 number of particles per cell
  int                NumObjects;  //
    std::vector<int>        npieces;     // number of pieces for each object
    std::vector<GeomObject> Objects;
  int                numbcs;      // number of boundary conditions;
    std::vector<BoundCond>  BC;          // boundary conditions

  
public:
  Problem();
  ~Problem();

    void getBnds(double bnds[7]) {
	for(int i=1;i<7;i++)
	    bnds[i] = this->bnds[i];
    }
    void getDx(double dx[4]) {
	for(int i=1;i<4;i++)
	    dx[i] = this->dx[i];
    }

  void preProcessor(std::string filename);
  void createParticles(const Region* region, DataWarehouseP&);
  void writeGridFiles(std::string gridposname, std::string gridcellname);
  void writeSAMRAIGridFile(std::string gridname);
  int  getNumMaterial() const;
  int getNumObjects() const;
    std::vector<GeomObject> * getObjects();
};

#endif // __PROBLEM_H__

// $Log$
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
