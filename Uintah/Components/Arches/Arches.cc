/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/Arches.h>
#include <Uintah/Components/Arches/PicardNonlinearSolver.h>
#include <Uintah/Components/Arches/PhysicalConstants.h>
#include <Uintah/Components/Arches/SmagorinskyModel.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/Properties.h>
#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/FCVariable.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SoleVariable.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Exceptions/ParameterNotFound.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <iostream>
using std::cerr;
using std::endl;

using namespace Uintah::ArchesSpace;

Arches::Arches( int MpiRank, int MpiProcesses ) :
  UintahParallelComponent( MpiRank, MpiProcesses )
{
  d_densityLabel = scinew VarLabel("density", 
				CCVariable<double>::getTypeDescription() );
  d_pressureLabel = scinew VarLabel("pressure", 
				 CCVariable<double>::getTypeDescription() );
  d_scalarLabel = scinew VarLabel("scalars", 
			       CCVariable<Vector>::getTypeDescription() );
  d_velocityLabel = scinew VarLabel("velocity", 
				 CCVariable<Vector>::getTypeDescription() );
  d_viscosityLabel = scinew VarLabel("viscosity", 
				  CCVariable<double>::getTypeDescription() );
}

Arches::~Arches()
{

}

void 
Arches::problemSetup(const ProblemSpecP& params, GridP&,
		     SimulationStateP& sharedState)
{
  d_sharedState = sharedState;
  ProblemSpecP db = params->findBlock("CFD")->findBlock("ARCHES");
  // not sure, do we need to reduce and put in datawarehouse
  db->require("grow_dt", d_deltaT);

  // physical constants
  d_physicalConsts = scinew PhysicalConstants();

  // ** BB 5/19/2000 ** For now read the Physical constants from the 
  // CFD-ARCHES block
  // for gravity, read it from shared state 
  //d_physicalConsts->problemSetup(params);
  d_physicalConsts->problemSetup(db);

  // read properties, boundary and turbulence model
  d_props = scinew Properties();
  d_props->problemSetup(db);
  string turbModel;
  db->require("turbulence_model", turbModel);
  if (turbModel == "smagorinsky") 
    d_turbModel = scinew SmagorinskyModel(d_physicalConsts);
  else 
    throw InvalidValue("Turbulence Model not supported" + turbModel);
  d_turbModel->problemSetup(db);
  d_boundaryCondition = scinew BoundaryCondition(d_turbModel, d_props);
  // send params, boundary type defined at the level of Grid
  d_boundaryCondition->problemSetup(db);
  string nlSolver;
  db->require("nonlinear_solver", nlSolver);
  if(nlSolver == "picard")
    d_nlSolver = scinew PicardNonlinearSolver(d_props, d_boundaryCondition,
					   d_turbModel, d_physicalConsts);
  else
    throw InvalidValue("Nonlinear solver not supported: "+nlSolver);

  //d_nlSolver->problemSetup(db, dw); /* 2 params ? */
  d_nlSolver->problemSetup(db);
}

void 
Arches::problemInit(const LevelP& ,
		    SchedulerP& , 
		    DataWarehouseP& ,
		    bool )
{
  cerr << "** NOTE ** Problem init has been called for ARCHES";
#if 0
  // initializes variables
  if (!restrt) {
    sched_paramInit(level, sched, dw);
    // initialize velocity, scalars and properties at the boundary
    d_boundaryCondition->sched_setProfile(level, sched, dw);
  }
  d_properties->sched_computeProperties(level, sched, dw, dw);
  d_turbModel->sched_computeTurbSubmodel(level, sched, dw, dw);
  d_boundaryCondition->sched_pressureBC(level, sched, dw, dw);
#endif
}

void 
Arches::scheduleInitialize(const LevelP& level,
			   SchedulerP& sched,
			   DataWarehouseP& dw)
{
  // BB : 5/19/2000 : Start scheduling the initializations one by one
  // Parameter initialization
  //  cerr << "Schedule parameter initialization\n" ;
  //  sched_paramInit(level, sched, dw);
  // cerr << "SerialArches::scheduleInitialize not completely done\n";
  
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = new Task("Arches::paramInit",
			   patch, dw, dw, this,
			   &Arches::paramInit);
      cerr << "New task created successfully\n";
      tsk->computes(dw, d_velocityLabel, 0, patch);
      tsk->computes(dw, d_pressureLabel, 0, patch);
      tsk->computes(dw, d_scalarLabel, 0, patch);
      tsk->computes(dw, d_densityLabel, 0, patch);
      tsk->computes(dw, d_viscosityLabel, 0, patch);
      sched->addTask(tsk);
      cerr << "New task added successfully to scheduler\n";
    }
  }
}

void 
Arches::scheduleComputeStableTimestep(const LevelP&,
				      SchedulerP&,
				      DataWarehouseP& dw)
{
  dw->put(delt_vartype(d_deltaT),  d_sharedState->get_delt_label()); 
}

void 
Arches::scheduleTimeAdvance(double time, double dt,
			    const LevelP& level, SchedulerP& sched,
			    const DataWarehouseP& old_dw, DataWarehouseP& new_dw)
{
  cerr << "Arches::scheduleTimeAdvance\n";

#ifdef WONT_COMPILE_YET
  int error_code = d_nlSolver->nonlinearSolve(time, dt, level, 
					      sched, old_dw, new_dw);
  if (!error_code) {
#if 0
    old_dw = new_dw;
#endif
  }
  else {
    cerr << "Nonlinear Solver didn't converge" << endl;
  }
#endif
  cerr << "Done: Arches::scheduleTimeAdvance\n";
}

void 
Arches::sched_paramInit(const LevelP& level,
			SchedulerP& sched, 
			DataWarehouseP& dw)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = new Task("Arches::paramInit",
			   patch, dw, dw, this,
			   &Arches::paramInit);
      cerr << "New task created successfully\n";
      tsk->computes(dw, d_velocityLabel, 0, patch);
      tsk->computes(dw, d_pressureLabel, 0, patch);
      tsk->computes(dw, d_scalarLabel, 0, patch);
      tsk->computes(dw, d_densityLabel, 0, patch);
      tsk->computes(dw, d_viscosityLabel, 0, patch);
      sched->addTask(tsk);
      cerr << "New task added successfully to scheduler\n";
    }
  }
}

void
Arches::paramInit(const ProcessorContext* ,
		  const Patch* patch,
		  DataWarehouseP& old_dw,
		  DataWarehouseP& new_dw)
{
  // ....but will only compute for computational domain

  cerr << "Arches::paramInit\n";

  CCVariable<Vector> velocity;
  CCVariable<double> pressure;
  CCVariable<Vector> scalar;
  CCVariable<double> density;
  CCVariable<double> viscosity;

  cerr << "Actual initialization - before allocation : old_dw = " 
       << old_dw <<"\n";
  old_dw->allocate(velocity, d_velocityLabel, 0, patch);
  old_dw->allocate(pressure, d_pressureLabel, 0, patch);
  old_dw->allocate(scalar, d_scalarLabel, 0, patch);
  old_dw->allocate(density, d_densityLabel, 0, patch);
  old_dw->allocate(viscosity, d_viscosityLabel, 0, patch);
  cerr << "Actual initialization - after allocation\n";

  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

#ifdef WONT_COMPILE_YET
  FORT_INIT(velocity, pressure, scalar, density, viscosity,
	    lowIndex, highIndex);
#endif
  cerr << "Actual initialization - before put : old_dw = " 
       << old_dw <<"\n";
  old_dw->put(velocity, d_velocityLabel, 0, patch);
  old_dw->put(pressure, d_pressureLabel, 0, patch);
  old_dw->put(scalar, d_scalarLabel, 0, patch);
  old_dw->put(density, d_densityLabel, 0, patch);
  old_dw->put(viscosity, d_viscosityLabel, 0, patch);
  cerr << "Actual initialization - after put \n";
}
  

//
// $Log$
// Revision 1.32  2000/06/01 19:29:46  rawat
// Modified BoundaryCondition to read multiple flowinlets
//
// Revision 1.31  2000/05/31 23:44:52  rawat
// modified arches and properties
//
// Revision 1.29  2000/05/30 20:18:45  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.28  2000/05/30 17:06:11  dav
// added Cocoon doc template.  fixed non compilation problem.
//
// Revision 1.27  2000/05/30 15:44:57  rawat
// modified computeStableTimestep
//
// Revision 1.26  2000/05/20 22:54:14  bbanerje
// Again, adding the first set of changes to get the scheduler to add tasks.
//
// Revision 1.25  2000/05/09 22:56:22  sparker
// Changed name of namespace
//
// Revision 1.24  2000/04/28 07:35:23  sparker
// Started implementation of DataWarehouse
// MPM particle initialization now works
//
// Revision 1.23  2000/04/26 06:48:00  sparker
// Streamlined namespaces
//
// Revision 1.22  2000/04/24 21:04:19  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.21  2000/04/20 18:56:10  sparker
// Updates to MPM
//
// Revision 1.20  2000/04/19 20:59:11  dav
// adding MPI support
//
// Revision 1.19  2000/04/19 05:25:56  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.18  2000/04/13 06:50:50  sparker
// More implementation to get this to work
//
// Revision 1.17  2000/04/12 22:58:29  sparker
// Resolved conflicts
// Making it compile
//
// Revision 1.16  2000/04/12 22:46:42  dav
// changes to make it compile
//
// Revision 1.15  2000/04/11 19:55:51  rawat
// modified nonlinear solver for initialization
//
// Revision 1.14  2000/04/11 07:10:35  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.13  2000/04/10 23:11:23  rawat
// Added sub-grid scale turbulence models
//
// Revision 1.12  2000/04/09 00:53:50  rawat
// added PhysicalConstants and Properties classes
//
// Revision 1.11  2000/04/07 18:30:12  rawat
// Added problem initialization function in Arches.cc
//
// Revision 1.10  2000/03/31 17:35:05  moulding
// A call to d_nlSolver->problemSetup() was attempted with 2 parameters, causing
// a compile error (too many parameters).  I changed it to one parameter.
// The change is at line 40.
//
// Revision 1.9  2000/03/29 21:18:16  rawat
// modified boundarycondition.cc for inlet bcs
//
// Revision 1.8  2000/03/23 20:05:13  jas
// Changed the location of ProblemSpec from Grid to Interface in the include
// file path.
//
// Revision 1.7  2000/03/21 23:14:47  rawat
// Implemented more of problem setup routines
//
// Revision 1.6  2000/03/21 21:27:03  dav
// namespace fixs
//
//

