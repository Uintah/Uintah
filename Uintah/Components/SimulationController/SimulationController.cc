/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/SimulationController/SimulationController.h>
#include <SCICore/Geometry/IntVector.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Containers/Array3.h>
#include <SCICore/Thread/Time.h>
#include <Uintah/Exceptions/ProblemSetupException.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/SimulationTime.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SoleVariable.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Interface/CFDInterface.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Interface/MPMInterface.h>
#include <Uintah/Interface/MDInterface.h>
#include <Uintah/Interface/Output.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/ProblemSpecInterface.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Parallel/ProcessorContext.h>
#include <Uintah/Grid/VarTypes.h>
#include <iostream>
#include <values.h>
using std::cerr;
using std::cout;

using SCICore::Geometry::IntVector;
using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;
using SCICore::Math::Abs;
using SCICore::Geometry::Min;
using SCICore::Thread::Time;
using namespace Uintah;
using SCICore::Containers::Array3;

SimulationController::SimulationController( int MpiRank, int MpiProcesses ) :
  UintahParallelComponent( MpiRank, MpiProcesses )
{
   d_restarting = false;
   d_generation = 0;
   d_dwMpiHandler = scinew DWMpiHandler();
}

SimulationController::~SimulationController()
{
}

void SimulationController::run()
{
   UintahParallelPort* pp = getPort("problem spec");
   ProblemSpecInterface* psi = dynamic_cast<ProblemSpecInterface*>(pp);
   
   // Get the problem specification
   ProblemSpecP ups = psi->readInputFile();
   if(!ups)
      throw ProblemSetupException("Cannot read problem specification");
   
   releasePort("problem spec");
   
   if(ups->getNodeName() != "Uintah_specification")
      throw ProblemSetupException("Input file is not a Uintah specification");
   
   Output* output = dynamic_cast<Output*>(getPort("output"));
   output->problemSetup(ups);
   
   // Setup the initial grid
   GridP grid=scinew Grid();
   
   problemSetup(ups, grid);
   
   if(grid->numLevels() == 0){
      cerr << "No problem specified.  Exiting SimulationController.\n";
      return;
   }
   
   // Check the grid
   grid->performConsistencyCheck();
   // Print out meta data
   grid->printStatistics();

   SimulationStateP sharedState = scinew SimulationState(ups);
   
   // Initialize the CFD and/or MPM components
   CFDInterface* cfd = dynamic_cast<CFDInterface*>(getPort("cfd"));
   if(cfd)
      cfd->problemSetup(ups, grid, sharedState);
   
   MPMInterface* mpm = dynamic_cast<MPMInterface*>(getPort("mpm"));
   if(mpm)
      mpm->problemSetup(ups, grid, sharedState);

   // Initialize the MD components --tan
   MDInterface* md = dynamic_cast<MDInterface*>(getPort("md"));
   if(md)
      md->problemSetup(ups, grid, sharedState);
   
   Scheduler* sched = dynamic_cast<Scheduler*>(getPort("scheduler"));
   SchedulerP scheduler(sched);

   DataWarehouseP old_ds = scheduler->createDataWarehouse( d_generation );
   d_generation++;

   d_dwMpiHandler->registerDW( old_ds );
   // Should I check to see if MpiProcesses is > 1 before making this
   // thread?  If not, the thread will just die when it realizes
   // that there are only one MpiProcesses.  Should I have if tests
   // around all the d_dwMpiHandler calls to test for this, or just
   // let it do the single assignment operation that will actually
   // have no effect?
   if( d_MpiProcesses > 1 ) {
     d_MpiThread = scinew Thread( d_dwMpiHandler, "DWMpiHandler" );
   }

   old_ds->setGrid(grid);
   
   scheduler->initialize();
   if(d_restarting){
      cerr << "Restart not finised!\n";
   } else {
      // Initialize the CFD and/or MPM data
      for(int i=0;i<grid->numLevels();i++){
	 LevelP level = grid->getLevel(i);
	 scheduleInitialize(level, scheduler, old_ds, cfd, mpm, md);
      }
   }
   
   // For AMR, this will need to change
   if(grid->numLevels() != 1)
      throw ProblemSetupException("AMR problem specified; cannot do it yet");
   LevelP level = grid->getLevel(0);
   
   // Parse time struct
   SimulationTime timeinfo(ups);
   
   double start_time = Time::currentSeconds();
   double t = timeinfo.initTime;
   
   scheduleComputeStableTimestep(level, scheduler, old_ds,
				 cfd, mpm, md);
   
   ProcessorContext* pc = ProcessorContext::getRootContext();

   scheduler->execute(pc, old_ds);
   
   while(t < timeinfo.maxTime) {
      double wallTime = Time::currentSeconds() - start_time;

      delt_vartype delt_var;
      old_ds->get(delt_var, sharedState->get_delt_label());

      double delt = delt_var;
      delt *= timeinfo.delt_factor;

      if(delt < timeinfo.delt_min){
	 cerr << "WARNING: raising delt from " << delt
	      << " to minimum: " << timeinfo.delt_min << '\n';
	 delt = timeinfo.delt_min;
      }
      if(delt > timeinfo.delt_max){
	 cerr << "WARNING: lowering delt from " << delt 
	      << " to maxmimum: " << timeinfo.delt_max << '\n';
	 delt = timeinfo.delt_max;
      }
      old_ds->override(delt_vartype(delt), sharedState->get_delt_label());
      cout << "Time=" << t << ", delt=" << delt 
	   << ", elapsed time = " << wallTime << '\n';

      scheduler->initialize();

      DataWarehouseP new_ds = scheduler->createDataWarehouse( d_generation );
      d_generation++;

      d_dwMpiHandler->registerDW( new_ds );
      new_ds->carryForward(old_ds);
      scheduleTimeAdvance(t, delt, level, scheduler, old_ds, new_ds,
			  cfd, mpm, md);
      if(output)
	 output->finalizeTimestep(t, delt, level, scheduler, old_ds, new_ds);
      t += delt;
      
      // Begin next time step...
      scheduleComputeStableTimestep(level, scheduler, new_ds, cfd, mpm, md);
      scheduler->execute(pc, new_ds);
      
      old_ds = new_ds;
   }

   if( d_MpiRank == 0 && d_MpiProcesses > 1 ) {
     d_dwMpiHandler->shutdown( d_MpiProcesses );
   }
}

void SimulationController::problemSetup(const ProblemSpecP& params,
					GridP& grid)
{
   ProblemSpecP grid_ps = params->findBlock("Grid");
   if(!grid_ps)
      return;
   
   for(ProblemSpecP level_ps = grid_ps->findBlock("Level");
       level_ps != 0; level_ps = level_ps->findNextBlock("Level")){
      // Make two passes through the patches.  The first time, we
      // want to find the spacing and the lower left corner of the
      // problem domain.  Spacing can be specified with a dx,dy,dz
      // on the level, or with a resolution on the patch.  If a
      // resolution is used on a problem with more than one patch,
      // the resulting grid spacing must be consistent.
      Point anchor(MAXDOUBLE, MAXDOUBLE, MAXDOUBLE);
      Vector spacing;
      bool have_levelspacing=false;
      if(level_ps->get("spacing", spacing))
	 have_levelspacing=true;
      bool have_patchspacing=false;
	 
      for(ProblemSpecP box_ps = level_ps->findBlock("Box");
	  box_ps != 0; box_ps = box_ps->findNextBlock("Box")){
	 Point lower;
	 box_ps->require("lower", lower);
	 Point upper;
	 box_ps->require("upper", upper);
	 anchor=SCICore::Geometry::Min(lower, anchor);

	 IntVector resolution;
	 if(box_ps->get("resolution", resolution)){
	    if(have_levelspacing){
	       throw ProblemSetupException("Cannot specify level spacing and patch resolution");
	    } else {
	       Vector newspacing = (upper-lower)/resolution;
	       if(have_patchspacing){
		  Vector diff = spacing-newspacing;
		  if(diff.length() > 1.e-6)
		     throw ProblemSetupException("Using patch resolution, and the patch spacings are inconsistent");
	       } else {
		  spacing = newspacing;
	       }
	       have_patchspacing=true;
	    }
	 }
      }
	 
      if(!have_levelspacing && !have_patchspacing)
	 throw ProblemSetupException("Box resolution is not specified");
	 
      LevelP level = scinew Level(grid.get_rep(), anchor, spacing);
      
      for(ProblemSpecP box_ps = level_ps->findBlock("Box");
	  box_ps != 0; box_ps = box_ps->findNextBlock("Box")){
	 Point lower;
	 box_ps->require("lower", lower);
	 Point upper;
	 box_ps->require("upper", upper);

	 IntVector lowCell = level->getCellIndex(lower);
	 IntVector highCell = level->getCellIndex(upper);
	 Point lower2 = level->getNodePosition(lowCell);
	 Point upper2 = level->getNodePosition(highCell);
	 double diff_lower = (lower2-lower).length();
	 if(diff_lower > 1.e-6)
	    throw ProblemSetupException("Box lower corner does not coincide with grid");
	 double diff_upper = (upper2-upper).length();
	 if(diff_upper > 1.e-6){
	    cerr << "upper=" << upper << '\n';
	    cerr << "lowCell =" << lowCell << '\n';
	    cerr << "highCell =" << highCell << '\n';
	    cerr << "upper2=" << upper2 << '\n';
	    cerr << "diff=" << diff_upper << '\n';
	    throw ProblemSetupException("Box upper corner does not coincide with grid");
	 }
	 IntVector extraCells;
	 if(box_ps->get("extraCells", extraCells)){
	    lowCell = lowCell-extraCells;
	    highCell = highCell+extraCells;
	 }

	 IntVector resolution(highCell-lowCell);
	 if(resolution.x() < 1 || resolution.y() < 1 || resolution.z() < 1)
	    throw ProblemSetupException("Degeneration patch");

	 IntVector patches;
	 if(box_ps->get("patches", patches)){
	    for(int i=0;i<patches.x();i++){
	       for(int j=0;j<patches.y();j++){
		  for(int k=0;k<patches.z();k++){
		     IntVector startcell = resolution*IntVector(i,j,k)/patches;
		     IntVector endcell = resolution*IntVector(i+1,j+1,k+1)/patches;
		     level->addPatch(startcell+lowCell, endcell+lowCell);
		  }
	       }
	    }
	 } else {
	    level->addPatch(lowCell, highCell);
	 }
      }
      level->finalizeLevel();
      grid->addLevel(level);
   }
}

void SimulationController::scheduleInitialize(LevelP& level,
					      SchedulerP& sched,
					      DataWarehouseP& new_ds,
					      CFDInterface* cfd,
					      MPMInterface* mpm,
					      MDInterface* md)
{
  if(cfd) {
    cfd->scheduleInitialize(level, sched, new_ds);
  }
  if(mpm) {
    mpm->scheduleInitialize(level, sched, new_ds);
  }
  if(md) {
    md->scheduleInitialize(level, sched, new_ds);
  }
}

void SimulationController::scheduleComputeStableTimestep(LevelP& level,
							 SchedulerP& sched,
							 DataWarehouseP& new_ds,
							 CFDInterface* cfd,
							 MPMInterface* mpm,
							 MDInterface* md)
{
   if(cfd)
      cfd->scheduleComputeStableTimestep(level, sched, new_ds);
   if(mpm)
      mpm->scheduleComputeStableTimestep(level, sched, new_ds);
   if(md)
      md->scheduleComputeStableTimestep(level, sched, new_ds);
}

void SimulationController::scheduleTimeAdvance(double t, double delt,
					       LevelP& level,
					       SchedulerP& sched,
					       DataWarehouseP& old_ds,
					       DataWarehouseP& new_ds,
					       CFDInterface* cfd,
					       MPMInterface* mpm,
					       MDInterface* md)
{
   // Temporary - when cfd/mpm are coupled this will need help
   if(cfd)
      cfd->scheduleTimeAdvance(t, delt, level, sched, old_ds, new_ds);
   if(mpm)
      mpm->scheduleTimeAdvance(t, delt, level, sched, old_ds, new_ds);
      
   // Added molecular dynamics module, currently it will not be coupled with 
   // cfd/mpm.  --tan
   if(md)
      md->scheduleTimeAdvance(t, delt, level, sched, old_ds, new_ds);
   
#if 0
   
   /* If we aren't doing any chemistry, skip this step */
#if 0
   if(chem)
      chem->calculateChemistryEffects();
#endif
   
   /* If we aren't doing MPM, skip this step */
   if(mpm){
#if 0
      mpm->zeroMPMGridData();
      mpm->interpolateParticlesToGrid(/* consume */ p_mass, p_velocity,
				      p_extForce, p_temperature,
				      /* produce */ g_mass, g_velocity, g_exForce,
				      g_volFraction, g_temperature);
#endif
   }
   if(mpm && !cfd){  // In other words, doing MPM only
#if 0
      mpm->exchangeMomentum2();
      mpm->computeVelocityGradients(/* arguments left as an exercise */);
      mpm->computeStressTensor();
      mpm->computeInternalForces(/* arguments left as an exercise */);
#endif
   }
   
   /* If we aren't doing CFD, sking this step */
   if(cfd && !mpm){
#if 0
      cfd->pressureAndVelocitySolve(/* consume */ g_density, g_massFraction,
				    g_temperature,
				    maybe other stuff,
				    
				    /* produce */ g_velocity, g_pressure);
#endif
   }
   
   if(mpm && cfd){
#if 0
      coupling->pressureVelocityStressSolve();
      /* produce */ cell centered pressure,
		       face centered velocity,
		       particle stresses
		       mpm->computeInternalForces();
#endif
   }
   
   if(mpm){
#if 0
      mpm->solveEquationsOfMotion(/* consume */ g_deltaPress, p_stress,
				  some boundary conditions,
				  /* produce */ p_acceleration);
      mpm->integrateAceleration(/* consume */ p_acceleration);
#endif
   }
   if(cfd){
#if 0
      /* This may be different, or a no-op for arches. - Rajesh? */
      cfd->addLagragianEffects(...);
#endif
   }
   /* Is this CFD or MPM, or what? */
   /* It's "or what" hence I prefer using the coupling module so
      neither of the others have to know about it.               */
   if(mpm && cfd){       // Do coupling's version of Exchange
#if 0
      coupling->calculateMomentumAndEnergyExchange( ... );
#endif
   }
   else if(mpm && !cfd){ // Do mpm's version of Exchange
#if 0
      mpm->exchangeMomentum();
#endif
   }
   else if(cfd && !mpm){ // Do cfd's version of Exchange
#if 0
      cfd->momentumExchange();
#endif
   }
   
   if(cfd){
#if 0
      cfd->advect(...);
      cfd->advanceInTime(...);
#endif
   }
   if(mpm){
#if 0
      mpm->interpolateGridToParticles(...);
      mpm->updateParticleVelocityAndPosition(...);
#endif
   }
#endif
}

//
// $Log$
// Revision 1.34  2000/06/16 05:03:09  sparker
// Moved timestep multiplier to simulation controller
// Fixed timestep min/max clamping so that it really works now
// Implemented "override" for reduction variables that will
//   allow the value of a reduction variable to be overridden
//
// Revision 1.33  2000/06/15 23:14:09  sparker
// Cleaned up scheduler code
// Renamed BrainDamagedScheduler to SingleProcessorScheduler
// Created MPIScheduler to (eventually) do the MPI work
//
// Revision 1.32  2000/06/15 21:57:13  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.31  2000/06/09 23:36:33  bard
// Removed cosmetic implementation of fudge factor (time step multiplier).
// It needs to be put in the constitutive routines to have the correct effect.
//
// Revision 1.30  2000/06/09 17:20:12  tan
// Added MD(molecular dynamics) module to SimulationController.
//
// Revision 1.29  2000/06/08 21:00:40  jas
// Added timestep multiplier (fudge factor).
//
// Revision 1.28  2000/06/05 19:51:56  guilkey
// Formatting
//
// Revision 1.27  2000/05/31 23:44:55  rawat
// modified arches and properties
//
// Revision 1.26  2000/05/30 20:19:25  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.25  2000/05/30 19:43:28  guilkey
// Unfixed the maximum timestep size fix, which wasn't much of a
// fix at all...
//
// Revision 1.24  2000/05/30 17:09:54  dav
// MPI stuff
//
// Revision 1.23  2000/05/26 18:58:15  guilkey
// Uncommented code to allow a maximum timestep to be set and effectively
// used.  The minimum time step still doesn't work, but for an explicit
// code, who cares?
//
// Revision 1.22  2000/05/20 08:09:18  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.21  2000/05/18 18:49:16  jas
// SimulationState constructor now uses the input file.
//
// Revision 1.20  2000/05/15 19:39:45  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.19  2000/05/11 21:25:51  sparker
// Fixed main time loop
//
// Revision 1.18  2000/05/11 20:10:20  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.17  2000/05/10 20:02:55  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made patches have a single uniform index space - still needs work
//
// Revision 1.16  2000/05/07 06:02:10  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.15  2000/05/05 06:42:44  dav
// Added some _hopefully_ good code mods as I work to get the MPI stuff to work.
//
// Revision 1.14  2000/05/04 18:38:32  jas
// Now the max_dt is used in the simulation if it is smaller than the
// stable dt computed in the constitutive model.
//
// Revision 1.13  2000/05/02 17:54:30  sparker
// Implemented more of SerialMPM
//
// Revision 1.12  2000/05/02 06:07:18  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.11  2000/04/26 06:48:36  sparker
// Streamlined namespaces
//
// Revision 1.10  2000/04/20 18:56:28  sparker
// Updates to MPM
//
// Revision 1.9  2000/04/19 20:59:25  dav
// adding MPI support
//
// Revision 1.8  2000/04/19 05:26:12  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.7  2000/04/13 20:05:56  sparker
// Compile more of arches
// Made SimulationController work somewhat
//
// Revision 1.6  2000/04/13 06:50:59  sparker
// More implementation to get this to work
//
// Revision 1.5  2000/04/12 23:00:09  sparker
// Start of reading grids
//
// Revision 1.4  2000/04/11 07:10:42  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.3  2000/03/17 20:58:31  dav
// namespace updates
//
//
