
#include "SimulationController.h"
#include "BrainDamagedScheduler.h"
#include "CFDInterface.h"
#include "DataWarehouse.h"
#include "ICE.h"
#include "ChemistryInterface.h"
#include "MPMInterface.h"
#include "Grid.h"
#include "Level.h"
#include "Output.h"
#include "ProblemSetupException.h"
#include "ProblemSpec.h"
#include "ProcessorContext.h"
#include "SoleVariable.h"
#include "SerialMPM.h"
#include "ThreadedMPM.h"
#include <SCICore/Thread/Time.h>
using SCICore::Thread::Time;
#include <iostream>
using std::cerr;
using std::cout;

SimulationController::SimulationController(int argc, char* argv[])
{
    mpm = new SerialMPM(); // DEFAULT
    for(int i=1;i<argc;i++){
	std::string arg(argv[i]);

	if(arg == "-ice")
	    cfd = 0; //new ICE();
	else if(arg == "-nompm")
	    mpm = 0;
	else if(arg == "-threaded")
	    mpm = new ThreadedMPM();
	else if(arg == "-nthreads") {
	    i++;
	    int numThreads = atoi(argv[i]);
	    ProcessorContext::getRootContext()->setNumThreads(numThreads);
	}
	else
	    cerr << "Unknown argument: " << arg << '\n';
    }
    scheduler = new BrainDamagedScheduler();
}

SimulationController::~SimulationController()
{
}

void SimulationController::run()
{
    // Get the problem specification.  Hard-coded for now - create a 
    // component later
    ProblemSpecP params = new ProblemSpec;

    // Componentize later
    OutputP output = new Output;
    
    // Setup the initial grid
    GridP grid=new Grid();

    problemSetup(params, grid);

    if(grid->numLevels() == 0){
	cerr << "No problem specified.  Exiting SimulationController.\n";
	return;
    }

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
}

void SimulationController::problemSetup(const ProblemSpecP& params,
					GridP& grid)
{
    LevelP mainLevel = new Level();
    grid->addLevel(mainLevel);
}

void SimulationController::computeStableTimestep(LevelP& level,
						 SchedulerP& sched,
						 DataWarehouseP& new_ds)
{
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
}

void SimulationController::timeAdvance(double t, double delt,
				       LevelP& level,
				       SchedulerP& sched,
				       const DataWarehouseP& old_ds,
				       DataWarehouseP& new_ds)
{

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
}
