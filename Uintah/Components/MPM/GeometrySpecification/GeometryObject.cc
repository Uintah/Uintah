#include "GeometryObject.h"
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <iostream>

using namespace Uintah::MPM;

GeometryObject::GeometryObject(MPMMaterial* mpm_matl,
                               GeometryPiece* piece,
			       ProblemSpecP& ps)
   : d_piece(piece)
{
   ps->require("res", d_resolution);
   ps->require("velocity", d_initialVel);
   ps->require("temperature", d_initialTemperature);
   
   if(mpm_matl->getFractureModel()) {
     ps->require("tensile_strength_min", d_tensileStrengthMin);
     ps->require("tensile_strength_max", d_tensileStrengthMax);
     ps->require("tensile_strength_variation", d_tensileStrengthVariation);
   }
}

GeometryObject::~GeometryObject()
{
}

IntVector GeometryObject::getNumParticlesPerCell()
{
  return d_resolution;
}

// $Log$
// Revision 1.19  2000/09/22 07:11:50  tan
// MPM code works with fracture in three point bending.
//
// Revision 1.18  2000/07/25 19:10:28  guilkey
// Changed code relating to particle combustion as well as the
// heat conduction.
//
// Revision 1.16  2000/06/23 02:51:48  tan
// temperature is required when MPMPhysicalModules::heatConductionModel
// considered.
//
// Revision 1.15  2000/05/31 17:17:22  guilkey
// Added getInitialTemperature to GeometryObject.
//
// Revision 1.14  2000/05/30 20:19:14  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.13  2000/05/03 23:52:47  guilkey
// Fixed some small errors in the MPM code to make it work
// and give what appear to be correct answers.
//
// Revision 1.12  2000/05/01 16:18:14  sparker
// Completed more of datawarehouse
// Initial more of MPM data
// Changed constitutive model for bar
//
// Revision 1.11  2000/04/27 23:18:46  sparker
// Added problem initialization for MPM
//
// Revision 1.10  2000/04/26 06:48:24  sparker
// Streamlined namespaces
//
// Revision 1.9  2000/04/25 18:43:30  jas
// Changed variable name of d_num_par_per_cell to d_resolution.
//
// Revision 1.8  2000/04/24 21:04:30  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.7  2000/04/20 18:56:21  sparker
// Updates to MPM
//
// Revision 1.6  2000/04/20 15:09:25  jas
// Added factory methods for GeometryObjects.
//
// Revision 1.5  2000/04/19 21:31:08  jas
// Revamping of the way objects are defined.  The different geometry object
// subtypes only do a few simple things such as testing whether a point
// falls inside the object and also gets the bounding box for the object.
// The constructive solid geometry objects:union,difference, and intersection
// have the same simple operations.
//
// Revision 1.4  2000/04/19 05:26:07  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.3  2000/04/14 02:05:45  jas
// Subclassed out the GeometryPiece into 4 types: Box,Cylinder,Sphere, and
// Tri.  This made the GeometryObject class simpler since many of the
// methods are now relegated to the GeometryPiece subclasses.
//
// Revision 1.2  2000/03/20 17:17:14  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.1  2000/03/14 22:36:05  jas
// Readded geometry specification source files.
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
// Revision 1.4  2000/01/27 06:40:30  sparker
// Working, semi-optimized serial version
//
// Revision 1.3  2000/01/26 01:08:06  sparker
// Things work now
//
// Revision 1.2  2000/01/25 18:28:44  sparker
// Now links and kinda runs
//
// Revision 1.1  2000/01/24 22:48:51  sparker
// Stuff may actually work someday...
//
// Revision 1.8  1999/09/04 23:00:31  jas
// Changed from Particle * to Particle to fix a memory leak.
//
// Revision 1.7  1999/09/03 23:02:17  guilkey
// Fixed some old errors with GeometryObject.cc in CheckShapes, specifically
// with "negative" cylindrical pieces.
//
// Revision 1.5  1999/08/31 15:52:25  jas
// Added functions to set the geometry bound for a geometry object.  This will
// allow a processor to query how big a space it has and then have the
// geometry object fill itself with only the particles that actually fill
// the space controlled by the processor.
//
// Revision 1.4  1999/08/25 17:52:48  guilkey
// Added functionality to allow for an "bond strength" momentum exchange
// routine to implemented.  Most of these functions are currently commented
// out in the files that I'm checking in.  These include applyExternalForce,
// exchangeMomentumBond(2),  and computeTractions.  Also, the printParticle
// function has been updated so that the grid values are now also printed out,
// again in tecplot format.  These files should also work with the SCIRun
// viz tools, although this hasn't been tested.
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
// Revision 1.1  1999/06/14 06:23:40  cgl
// - src/mpm/Makefile modified to work for IRIX64 or Linux
// - src/grid/Grid.cc added length to character array, since it
// 	was only 4 long, but was being sprintf'd with a 4 character
// 	number, leaving no room for the terminating 0.
// - added smpm directory. to house the samrai version of mpm.
//
// Revision 1.10  1999/05/31 19:36:14  cgl
// Work in stand-alone version of MPM:
//
// - Added materials_dat.cc in src/constitutive_model to generate the
//   materials.dat file for preMPM.
// - Eliminated references to ConstitutiveModel in Grid.cc and GeometryObject.cc
//   Now only Particle and Material know about ConstitutiveModel.
// - Added reads/writes of Particle start and restart information as member
//   functions of Particle
// - "part.pos" now has identicle format to the restart files.
//   mpm.cc modified to take advantage of this.
//
// Revision 1.9  1999/05/30 02:10:49  cgl
// The stand-alone version of ConstitutiveModel and derived classes
// are now more isolated from the rest of the code.  A new class
// ConstitutiveModelFactory has been added to handle all of the
// switching on model type.  Between the ConstitutiveModelFactory
// class functions and a couple of new virtual functions in the
// ConstitutiveModel class, new models can be added without any
// source modifications to any classes outside of the constitutive_model
// directory.  See csafe/Uintah/src/CD/src/constitutive_model/HOWTOADDANEWMODEL
// for updated details on how to add a new model.
//
// --cgl
//
// Revision 1.8  1999/05/26 20:30:50  guilkey
// Added the capability for determining which particles are on the surface
// and what the surface normal of those particles is.  This information
// is output to part.pos.
//
// Revision 1.7  1999/05/24 21:06:00  guilkey
// Added a new constitutive model, and tried to make it easier for
// others to add new models in the future.
//
// Revision 1.6  1999/02/25 22:32:32  guilkey
// Fixed some functions associated with the HyperElastic constitutive model.
//
// Revision 1.5  1999/02/10 20:53:10  guilkey
// Updated to release 2-0
//
// Revision 1.4  1999/01/26 21:53:33  campbell
// Added logging capabilities
//
