//----- Arches.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/Arches.h>
#include <Uintah/Components/Arches/ArchesFort.h>
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

//****************************************************************************
// Actual constructor for Arches
//****************************************************************************
Arches::Arches( int MpiRank, int MpiProcesses ) :
  UintahParallelComponent( MpiRank, MpiProcesses )
{
  d_densityLabel = scinew VarLabel("density", 
				   CCVariable<double>::getTypeDescription() );
  d_pressureLabel = scinew VarLabel("pressure", 
				    CCVariable<double>::getTypeDescription() );
  d_uVelocityLabel = scinew VarLabel("uVelocity", 
				    CCVariable<double>::getTypeDescription() );
  d_vVelocityLabel = scinew VarLabel("vVelocity", 
				    CCVariable<double>::getTypeDescription() );
  d_wVelocityLabel = scinew VarLabel("wVelocity", 
				    CCVariable<double>::getTypeDescription() );
  d_scalarLabel = scinew VarLabel("scalar", 
				  CCVariable<double>::getTypeDescription() );
  d_viscosityLabel = scinew VarLabel("viscosity", 
				     CCVariable<double>::getTypeDescription() );
}

//****************************************************************************
// Destructor
//****************************************************************************
Arches::~Arches()
{

}

//****************************************************************************
// problem set up
//****************************************************************************
void 
Arches::problemSetup(const ProblemSpecP& params, 
		     GridP&,
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

  // read properties
  d_props = scinew Properties();
  d_props->problemSetup(db);
  d_nofScalars = d_props->getNumMixVars();

  // read turbulence model
  string turbModel;
  db->require("turbulence_model", turbModel);
  if (turbModel == "smagorinsky") 
    d_turbModel = scinew SmagorinskyModel(d_physicalConsts);
  else 
    throw InvalidValue("Turbulence Model not supported" + turbModel);
  d_turbModel->problemSetup(db);

  // read boundary
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

/*
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
*/

//****************************************************************************
// Schedule initialization
//****************************************************************************
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
      int matlIndex = 0;
      tsk->computes(dw, d_uVelocityLabel, matlIndex, patch);
      tsk->computes(dw, d_vVelocityLabel, matlIndex, patch);
      tsk->computes(dw, d_wVelocityLabel, matlIndex, patch);
      tsk->computes(dw, d_pressureLabel, matlIndex, patch);
      for (int ii = 0; ii < d_nofScalars; ii++) 
	tsk->computes(dw, d_scalarLabel, ii, patch);
      tsk->computes(dw, d_densityLabel, matlIndex, patch);
      tsk->computes(dw, d_viscosityLabel, matlIndex, patch);
      sched->addTask(tsk);
      cerr << "New task added successfully to scheduler\n";
    }
  }
}

//****************************************************************************
// schedule computation of stable time step
//****************************************************************************
void 
Arches::scheduleComputeStableTimestep(const LevelP&,
				      SchedulerP&,
				      DataWarehouseP& dw)
{
  dw->put(delt_vartype(d_deltaT),  d_sharedState->get_delt_label()); 
}

//****************************************************************************
// Schedule time advance
//****************************************************************************
void 
Arches::scheduleTimeAdvance(double time, double dt,
			    const LevelP& level, 
			    SchedulerP& sched,
			    DataWarehouseP& old_dw, 
			    DataWarehouseP& new_dw)
{
  cerr << "Arches::scheduleTimeAdvance\n";

  int error_code = d_nlSolver->nonlinearSolve(level, sched, old_dw, new_dw,
					      time, dt);
  if (!error_code) {
    old_dw = new_dw;
  }
  else {
    cerr << "Nonlinear Solver didn't converge" << endl;
  }
  cerr << "Done: Arches::scheduleTimeAdvance\n";
}

/*
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
*/

//****************************************************************************
// Actual initialization
//****************************************************************************
void
Arches::paramInit(const ProcessorContext* ,
		  const Patch* patch,
		  DataWarehouseP& old_dw,
		  DataWarehouseP& )
{
  // ....but will only compute for computational domain

  cerr << "Arches::paramInit\n";

  CCVariable<double> uVelocity;
  CCVariable<double> vVelocity;
  CCVariable<double> wVelocity;
  CCVariable<double> pressure;
  vector<CCVariable<double> > scalar(d_nofScalars);
  CCVariable<double> density;
  CCVariable<double> viscosity;

  cerr << "Actual initialization - before allocation : old_dw = " 
       << old_dw <<"\n";
  int matlIndex = 0;
  old_dw->allocate(uVelocity, d_uVelocityLabel, matlIndex, patch);
  old_dw->allocate(vVelocity, d_vVelocityLabel, matlIndex, patch);
  old_dw->allocate(wVelocity, d_wVelocityLabel, matlIndex, patch);
  old_dw->allocate(pressure, d_pressureLabel, matlIndex, patch);
  for (int ii = 0; ii < d_nofScalars; ii++) {
    old_dw->allocate(scalar[ii], d_scalarLabel, ii, patch);
  }
  old_dw->allocate(density, d_densityLabel, matlIndex, patch);
  old_dw->allocate(viscosity, d_viscosityLabel, matlIndex, patch);
  cerr << "Actual initialization - after allocation\n";

  // ** WARNING **  this needs to be changed soon (6/9/2000)
  // IntVector domainLow = patch->getCellLowIndex();
  // IntVector domainHigh = patch->getCellHighIndex();
  // IntVector indexLow = patch->getCellLowIndex();
  // IntVector indexHigh = patch->getCellHighIndex();
  int domainLow[3], domainHigh[3];
  int indexLow[3], indexHigh[3];
  domainLow[0] = (patch->getCellLowIndex()).x()+1;
  domainLow[1] = (patch->getCellLowIndex()).y()+1;
  domainLow[2] = (patch->getCellLowIndex()).z()+1;
  domainHigh[0] = (patch->getCellHighIndex()).x();
  domainHigh[1] = (patch->getCellHighIndex()).y();
  domainHigh[2] = (patch->getCellHighIndex()).z();
  for (int ii = 0; ii < 3; ii++) {
    indexLow[ii] = domainLow[ii]+1;
    indexHigh[ii] = domainHigh[ii]-1;
  }
 
  double uVal = 0.0, vVal = 0.0, wVal = 0.0;
  double pVal = 0.0, denVal = 0.0;
  double visVal = d_physicalConsts->getMolecularViscosity();
  FORT_INIT(domainLow, domainHigh, indexLow, indexHigh,
	    uVelocity.getPointer(), &uVal, vVelocity.getPointer(), &vVal, 
	    wVelocity.getPointer(), &wVal,
	    pressure.getPointer(), &pVal, density.getPointer(), &denVal, 
	    viscosity.getPointer(), &visVal);
  for (int ii = 0; ii < d_nofScalars; ii++) {
    double scalVal = 0.0;
    FORT_INIT_SCALAR(domainLow, domainHigh,
		     indexLow, indexHigh, 
		     scalar[ii].getPointer(), &scalVal);
  }

  cerr << "Actual initialization - before put : old_dw = " 
       << old_dw <<"\n";
  old_dw->put(uVelocity, d_uVelocityLabel, matlIndex, patch);
  old_dw->put(vVelocity, d_vVelocityLabel, matlIndex, patch);
  old_dw->put(wVelocity, d_wVelocityLabel, matlIndex, patch);
  old_dw->put(pressure, d_pressureLabel, matlIndex, patch);
  for (int ii = 0; ii < d_nofScalars; ii++) {
    old_dw->put(scalar[ii], d_scalarLabel, ii, patch);
  }
  old_dw->put(density, d_densityLabel, matlIndex, patch);
  old_dw->put(viscosity, d_viscosityLabel, matlIndex, patch);
  cerr << "Actual initialization - after put \n";
}
  
//****************************************************************************
// Private default constructor for Arches
//****************************************************************************
//Arches::Arches():UintahParallelComponent()
//{
//}

//****************************************************************************
// Private copy constructor for Arches
//****************************************************************************
//Arches::Arches(const Arches&):UintahParallelComponent()
//{
//}

//****************************************************************************
// private operator=
//****************************************************************************
//Arches&
//Arches::operator=(const Arches&)
//{
//}

//
// $Log$
// Revision 1.36  2000/06/13 06:02:30  bbanerje
// Added some more StencilMatrices and vector<CCVariable> types.
//
// Revision 1.35  2000/06/12 21:29:59  bbanerje
// Added first Fortran routines, added Stencil Matrix where needed,
// removed unnecessary CCVariables (e.g., sources etc.)
//
// Revision 1.34  2000/06/07 06:13:53  bbanerje
// Changed CCVariable<Vector> to CCVariable<double> for most cases.
// Some of these variables may not be 3D Vectors .. they may be Stencils
// or more than 3D arrays. Need help here.
//
// Revision 1.33  2000/06/04 23:57:46  bbanerje
// Updated Arches to do ScheduleTimeAdvance.
//
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

