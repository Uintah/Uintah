#include "Problem.h"
#include <Uintah/Components/MPM/BoundCond.h>
#include <iostream>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <SCICore/Geometry/Point.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Components/MPM/GeometrySpecification/GeometryPieceFactory.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/SimulationState.h>

using SCICore::Geometry::Point;

using namespace std;
using namespace Uintah::MPM;

class GeometryObject;

Problem::Problem()
  : d_num_bcs(0)
{
}

Problem::~Problem()
{
}

void Problem::preProcessor(const ProblemSpecP& prob_spec, GridP&,
			   SimulationStateP& sharedState)
{
  cerr << "In the preprocessor . . ." << endl;
    
  // Search for the MaterialProperties block and then get the MPM section
  
  ProblemSpecP mat_ps =  prob_spec->findBlock("MaterialProperties");
 
  ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");  

  for (ProblemSpecP ps = mpm_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {
    // Extract out the type of constitutive model and the 
    // associated parameters
     MPMMaterial *mat = new MPMMaterial(ps);
     sharedState->registerMaterial(mat);
  }                                                                
      

  // Extract out the BCS
    
#if 0
  
  infile >>stuff;
  if(stuff != string("Number_of_Force_Boundary_Conditions"))
    {
      cerr << "Bad input file: " << filename << endl;
      cerr << "Input file needs: Number_of_Force_Boundary_Conditions" << endl;
      cerr << "You have probably misspelled Conditions i.e. Condtions" << endl;
      exit(1);
    }
  infile >> numbcs;
  
  for(n = 0; n < numbcs; n++)
    {
      infile >> stuff;
      infile >> obj >> piece >> surf >> force[1] >> force[2] >> force[3];
      
      BCTemp.setBC(obj, piece, surf, force);
      BC.push_back(BCTemp);
    }

#endif 
}

#if 0
void Problem::createParticles(const Region* region, DataWarehouseP& dw)
{
  for (int i = 0; i < d_objects.size(); i++)
    {

#ifdef WONT_COMPILE_YET
      d_objects[i].fillWithParticles(d_materials, d_bcs, region, dw);
#endif
    }
}
#endif

// $Log$
// Revision 1.14  2000/04/28 07:35:31  sparker
// Started implementation of DataWarehouse
// MPM particle initialization now works
//
// Revision 1.13  2000/04/26 06:48:25  sparker
// Streamlined namespaces
//
// Revision 1.12  2000/04/24 21:04:32  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.11  2000/04/20 23:57:01  jas
// Fixed the findBlock() and findNextBlock() to iterate through all the
// nodes.  Now we can go thru the MPM setup without an error.  There is
// still the problem of where the <res> tag should go.
//
// Revision 1.10  2000/04/20 22:37:14  jas
// Fixed up the GeometryObjectFactory.  Added findBlock() and findNextBlock()
// to ProblemSpec stuff.  This will iterate through all of the nodes (hopefully).
//
// Revision 1.9  2000/04/20 18:56:22  sparker
// Updates to MPM
//
// Revision 1.8  2000/04/20 15:09:26  jas
// Added factory methods for GeometryObjects.
//
// Revision 1.7  2000/04/19 05:26:08  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.6  2000/04/14 15:51:37  jas
// Changed a cout to cerr.
//
// Revision 1.5  2000/04/14 03:29:13  jas
// Fixed routines to use SCICore's point and vector stuff.
//
// Revision 1.4  2000/04/14 02:05:46  jas
// Subclassed out the GeometryPiece into 4 types: Box,Cylinder,Sphere, and
// Tri.  This made the GeometryObject class simpler since many of the
// methods are now relegated to the GeometryPiece subclasses.
//
// Revision 1.3  2000/03/22 23:41:23  sparker
// Working towards getting arches to compile/run
//
// Revision 1.2  2000/03/20 17:17:15  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.1  2000/03/14 22:36:06  jas
// Readded geometry specification source files.
//
// Revision 1.1  2000/02/24 06:11:59  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:56  sparker
// Stuff may actually work someday...
//
// Revision 1.8  1999/10/28 23:24:48  jas
// Eliminated the -n option for manually creating multiple boxes.  SAMRAI does
// this automatically now.
//
// Revision 1.7  1999/09/04 23:00:32  jas
// Changed from Particle * to Particle to fix a memory leak.
//
// Revision 1.6  1999/08/18 19:23:09  jas
// Now can return information about the problem specification and object
// bounds.  This is necessary for the creation of particles on the processor
// that they reside.
//
// Revision 1.5  1999/08/06 21:14:45  jas
// Fixed largest patch size to use as big as patch as possible
//
// Revision 1.4  1999/07/28 16:54:55  cgl
// - Multiple Velocity fields in smpm
// - stop creating extra particles
//
// Revision 1.3  1999/07/22 20:17:20  jas
// grid.amr contains all the input needed by SAMRAI to run a problem.  We
// no longer need the test.amr file.
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
// Revision 1.6  1999/02/10 20:53:10  guilkey
// Updated to release 2-0
//
// Revision 1.5  1999/01/26 21:53:34  campbell
// Added logging capabilities
//
