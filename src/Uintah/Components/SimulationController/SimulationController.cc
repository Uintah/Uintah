/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/SimulationController/SimulationController.h>
#include <Uintah/Exceptions/ProblemSetupException.h>
#include <Uintah/Interface/ProblemSpecInterface.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <SCICore/Thread/Time.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/IntVector.h>
#include <SCICore/Math/MiscMath.h>
#include <iostream>
using std::cerr;

using SCICore::Thread::Time;

using Uintah::Exceptions::ProblemSetupException;
using Uintah::Interface::ProblemSpecInterface;
using Uintah::Components::SimulationController;
using Uintah::Parallel::UintahParallelPort;
using Uintah::Grid::Grid;
using Uintah::Grid::Level;
using SCICore::Geometry::IntVector;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::Math::Abs;

SimulationController::SimulationController()
{
}

SimulationController::~SimulationController()
{
}

void SimulationController::run()
{
    UintahParallelPort* pp = getPort("problem spec");
    ProblemSpecInterface* psi = dynamic_cast<ProblemSpecInterface*>(pp);

    // Get the problem specification
    ProblemSpecP params = psi->readInputFile();
    if(!params)
	throw ProblemSetupException("Cannot read problem specification");

    releasePort("problem spec");

    ProblemSpecP ups = params->findBlock("Uintah_specification");
    if(!ups)
	throw ProblemSetupException("Input file is not a Uintah specification");

    // Setup the initial grid
    GridP grid=new Uintah::Grid::Grid();

    problemSetup(ups, grid);

    if(grid->numLevels() == 0){
	cerr << "No problem specified.  Exiting SimulationController.\n";
	return;
    }

    grid->performConsistencyCheck();
    grid->printStatistics();

#if 0

    DataWarehouseP old_ds = scheduler->createDataWarehouse();

    //old_ds->put(grid, "grid");

    if(cfd)
	cfd->problemSetup(params, grid, old_ds);
    if(mpm)
	mpm->problemSetup(params, grid, old_ds);

    // For AMR, this will need to change
    if(grid->numLevels() != 1)
	throw ProblemSetupException("AMR problem specified; cannot do it yet");
    LevelP level = grid->getLevel(0);

    double start_time = Time::currentSeconds();
    double t = params->getStartTime();

    scheduler->initialize();
    computeStableTimestep(level, scheduler, old_ds);
    ProcessorContext* pc = ProcessorContext::getRootContext();
    scheduler->execute(pc);
    SoleVariable<double> delt;
    old_ds->get(delt, "delt");
    do {
	double wallTime = Time::currentSeconds() - start_time;
	cout << "Time=" << t << ", delt=" << delt << ", elapsed time = " << wallTime << '\n';
	scheduler->initialize();
	DataWarehouseP new_ds = scheduler->createDataWarehouse();
	timeAdvance(t, delt, level, scheduler, old_ds, new_ds);
	output->finalizeTimestep(t, delt, level, scheduler, new_ds);
	t += delt;
	
	// Begin next time step...
	computeStableTimestep(level, scheduler, new_ds);
	scheduler->addTarget("delt");
	scheduler->execute(pc);

	new_ds->get(delt, "delt");
	old_ds = new_ds;
    } while(t < params->getMaximumTime());

    cerr << "nlevels: " << grid->numLevels() << '\n';
#endif
}

void SimulationController::problemSetup(const ProblemSpecP& params,
					GridP& grid)
{
    ProblemSpecP grid_ps = params->findBlock("Grid");
    if(!grid_ps)
	return;

    for(ProblemSpecP level_ps = grid_ps->findBlock("Level");
	level_ps != 0; level_ps = level_ps->findNextBlock("Level")){
	LevelP level = new Level();
	
	for(ProblemSpecP box_ps = level_ps->findBlock("Box");
	    box_ps != 0; box_ps = box_ps->findNextBlock("Box")){
	    Point lower;
	    box_ps->require("lower", lower);
	    Point upper;
	    box_ps->require("upper", upper);

	    IntVector resolution;
	    bool have_res=false;
	    if(box_ps->get("resolution", resolution)){
		have_res=true;
	    }
	    Vector spacing;
	    if(box_ps->get("spacing", spacing)){
		if(have_res)
		    throw ProblemSetupException("Cannot specify spacing AND resolution for Box");

		have_res=true;
		Vector diag = upper-lower;
		Vector res = diag/spacing;
		resolution.x((int)(res.x()+0.5));
		resolution.y((int)(res.y()+0.5));
		resolution.z((int)(res.z()+0.5));
		if(Abs(resolution.x() - res.x()) > 1.e-6
		   || Abs(resolution.y() - res.y()) > 1.e-6
		   || Abs(resolution.z() - res.z()) > 1.e-6)
		    throw ProblemSetupException("Grid spacing does not allow an integer number of cells");
	    }


	    if(!have_res)
		throw ProblemSetupException("Box resolution is not specified");

	    IntVector patches;
	    if(box_ps->get("patches", patches)){
		Vector diag(upper-lower);
		Vector scale(1./patches.x(), 1./patches.y(), 1./patches.z());
		for(int i=0;i<patches.x();i++){
		    for(int j=0;j<patches.y();j++){
			for(int k=0;k<patches.z();k++){
			    IntVector startcell = resolution*IntVector(i,j,k)/patches;
			    IntVector endcell = resolution*IntVector(i+1,j+1,k+1)/patches;
			    IntVector ncells = endcell-startcell;
			    level->addRegion(lower+diag*Vector(i,j,k)*scale,
					     lower+diag*Vector(i+1,j+1,k+1)*scale,
					     resolution);
			}
		    }
		}
	    } else {
		level->addRegion(lower, upper, resolution);
	    }
	}
	grid->addLevel(level);
    }
}

void SimulationController::computeStableTimestep(LevelP& level,
						 SchedulerP& sched,
						 DataWarehouseP& new_ds)
{
#if 0
    if(cfd && mpm){
	cfd->computeStableTimestep(level, sched, new_ds);
	mpm->computeStableTimestep(level, sched, new_ds);
	//throw ProblemSetupException("MPM+CFD doesn't work");
	/*
	double dt_cfd = cfd->computeStableTimestep(params, grid, Scheduler);
	double dt_mpm = mpm->computeStableTimestep(params, grid, Scheduler);
	return dt_cfd<dt_mpm? dt_cfd:dt_mpm;
	*/
    } else if(cfd){
	cfd->computeStableTimestep(level, sched, new_ds);
    } else if(mpm){
	mpm->computeStableTimestep(level, sched, new_ds);
    } else {
	throw ProblemSetupException("Neither MPM or CFD specified");
    }
#endif
}

void SimulationController::timeAdvance(double t, double delt,
				       LevelP& level,
				       SchedulerP& sched,
				       const DataWarehouseP& old_ds,
				       DataWarehouseP& new_ds)
{
#if 0

    /* If we aren't doing any chemistry, skip this step */
#if 0
    if(chem)
       chem->calculateChemistryEffects();
#endif

    // Temporary
    if(cfd)
	cfd->timeStep(t, delt, level, sched, old_ds, new_ds);
    if(mpm)
	mpm->timeStep(t, delt, level, sched, old_ds, new_ds);

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
