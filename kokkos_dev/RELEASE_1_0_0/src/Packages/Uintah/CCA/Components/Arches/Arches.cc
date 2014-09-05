//----- Arches.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesFort.h>
#include <Packages/Uintah/CCA/Components/Arches/PicardNonlinearSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/SmagorinskyModel.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/Properties.h>
#include <Packages/Uintah/Core/Grid/Array3Index.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/NotFinished.h>

#include <iostream>
using std::cerr;
using std::endl;

using namespace Uintah;
using namespace SCIRun;

// ****************************************************************************
// Actual constructor for Arches
// ****************************************************************************
Arches::Arches(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
  d_lab = scinew ArchesLabel();
  d_MAlab = 0; // will be set by setMPMArchesLabel
}

// ****************************************************************************
// Destructor
// ****************************************************************************
Arches::~Arches()
{

}

// ****************************************************************************
// problem set up
// ****************************************************************************
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
#ifdef multimaterialform
  bool multimaterial;
  db->require("MultiMaterial",multimaterial);
  if (multimaterial) {
    d_mmInterface = new MultiMaterialInterface();
    d_mmInterface(db, d_sharedState);
  }
  else
    d_mmInterface = 0;
  d_props = scinew Properties(d_lab);
  d_props->problemSetup(db, d_mmInterface);
  d_nofScalars = d_props->getNumMixVars();

  // read turbulence model
  string turbModel;
  db->require("turbulence_model", turbModel);
  if (turbModel == "smagorinsky") 
    d_turbModel = scinew SmagorinskyModel(d_lab, d_physicalConsts);
  else 
    throw InvalidValue("Turbulence Model not supported" + turbModel);
  d_turbModel->problemSetup(db);

  // read boundary
  d_boundaryCondition = scinew BoundaryCondition(d_lab, d_turbModel, d_props);
  // send params, boundary type defined at the level of Grid
  d_boundaryCondition->problemSetup(db);
  d_props->setBC(d_boundaryCondition);

  string nlSolver;
  db->require("nonlinear_solver", nlSolver);
  if(nlSolver == "picard") {
    d_nlSolver = scinew PicardNonlinearSolver(d_lab, d_props, d_boundaryCondition,
					      d_turbModel, d_physicalConsts,
					      d_myworld);
  }
  else
    throw InvalidValue("Nonlinear solver not supported: "+nlSolver);

  d_nlSolver->problemSetup(db, d_mmInterface);
#endif
  // read properties
  // d_MAlab = multimaterial arches common labels
  d_props = scinew Properties(d_lab, d_MAlab);
  d_props->problemSetup(db);
  d_nofScalars = d_props->getNumMixVars();

  // read turbulence model
  string turbModel;
  db->require("turbulence_model", turbModel);
  if (turbModel == "smagorinsky") 
    d_turbModel = scinew SmagorinskyModel(d_lab, d_MAlab, d_physicalConsts);
  else 
    throw InvalidValue("Turbulence Model not supported" + turbModel);
  d_turbModel->problemSetup(db);

  // read boundary
  d_boundaryCondition = scinew BoundaryCondition(d_lab, d_MAlab, d_turbModel, d_props);
  // send params, boundary type defined at the level of Grid
  d_boundaryCondition->problemSetup(db);
  d_props->setBC(d_boundaryCondition);

  string nlSolver;
  db->require("nonlinear_solver", nlSolver);
  if(nlSolver == "picard") {
    d_nlSolver = scinew PicardNonlinearSolver(d_lab, d_MAlab, d_props, 
					      d_boundaryCondition,
					      d_turbModel, d_physicalConsts,
					      d_myworld);
  }
  else
    throw InvalidValue("Nonlinear solver not supported: "+nlSolver);

  d_nlSolver->problemSetup(db);
}

// ****************************************************************************
// Schedule initialization
// ****************************************************************************
void 
Arches::scheduleInitialize(const LevelP& level,
			   SchedulerP& sched)
{
#if 0
  // schedule the initialization of parameters
  // require : None
  // compute : [u,v,w]VelocityIN, pressureIN, scalarIN, densityIN,
  //           viscosityIN
  sched_paramInit(level, sched, dw, dw);

  // schedule init of cell type
  // require : NONE
  // compute : cellType
  d_boundaryCondition->sched_cellTypeInit(level, sched, dw, dw);

  // computing flow inlet areas
  d_boundaryCondition->sched_calculateArea(level, sched, dw, dw);

  // Set the profile (output Varlabel have SP appended to them)
  // require : densityIN,[u,v,w]VelocityIN
  // compute : densitySP, [u,v,w]VelocitySP, scalarSP
  d_boundaryCondition->sched_setProfile(level, sched, dw, dw);

  // Compute props (output Varlabel have CP appended to them)
  // require : densitySP
  // require scalarSP
  // compute : densityCP
  d_props->sched_computeProps(level, sched, dw, dw);

  // Compute Turb subscale model (output Varlabel have CTS appended to them)
  // require : densityCP, viscosityIN, [u,v,w]VelocitySP
  // compute : viscosityCTS
  d_turbModel->sched_computeTurbSubmodel(level, sched, dw, dw);

  // Computes velocities at apecified pressure b.c's
  // require : densityCP, pressureIN, [u,v,w]VelocitySP
  // compute : pressureSPBC, [u,v,w]VelocitySPBC
  if (d_boundaryCondition->getPressureBC()) 
    d_boundaryCondition->sched_computePressureBC(level, sched, dw, dw);
#else
  NOT_FINISHED("new task stuff");
#endif
}

// ****************************************************************************
// schedule the initialization of parameters
// ****************************************************************************
void 
Arches::sched_paramInit(SchedulerP& sched, const PatchSet* patches,
			const MaterialSet* matls)
{
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;

    // primitive variable initialization
    Task* tsk = scinew Task("Arches::paramInit",
			 patch, old_dw, new_dw, this,
			 &Arches::paramInit);
    int matlIndex = 0;
    tsk->computes(new_dw, d_lab->d_uVelocityINLabel, matlIndex, patch);
    tsk->computes(new_dw, d_lab->d_vVelocityINLabel, matlIndex, patch);
    tsk->computes(new_dw, d_lab->d_wVelocityINLabel, matlIndex, patch);
    tsk->computes(new_dw, d_lab->d_pressureINLabel, matlIndex, patch);
    for (int ii = 0; ii < d_nofScalars; ii++) 
      tsk->computes(new_dw, d_lab->d_scalarINLabel, ii, patch);
    tsk->computes(new_dw, d_lab->d_densityINLabel, matlIndex, patch);
    tsk->computes(new_dw, d_lab->d_viscosityINLabel, matlIndex, patch);
    sched->addTask(tsk);
  }
#else
  NOT_FINISHED("new task stuff");
#endif
}

// ****************************************************************************
// schedule computation of stable time step
// ****************************************************************************
void 
Arches::scheduleComputeStableTimestep(const LevelP&,
				      SchedulerP&)
{
#if 0
  dw->put(delt_vartype(d_deltaT),  d_sharedState->get_delt_label()); 
#else
  NOT_FINISHED("new task stuff");
#endif
}

// ****************************************************************************
// Schedule time advance
// ****************************************************************************
void 
Arches::scheduleTimeAdvance(double time, double dt,
			    const LevelP& level, 
			    SchedulerP& sched)
{
#if 0
#ifdef ARCHES_MAIN_DEBUG
  cerr << "Begin: Arches::scheduleTimeAdvance\n";
#endif
  // Schedule the non-linear solve
  // require : densityCP, viscosityCTS, [u,v,w]VelocitySP, 
  //           pressureIN. scalarIN
  // compute : densityRCP, viscosityRCTS, [u,v,w]VelocityMS,
  //           pressurePS, scalarSS 
  d_nlSolver->nonlinearSolve(level, sched, old_dw, new_dw,
					      time, dt);
  //  if (!error_code) {
  //    old_dw = new_dw;
  //  }
  //  else {
  //    cerr << "Nonlinear Solver didn't converge" << endl;
  //  }
#ifdef ARCHES_MAIN_DEBUG
  cerr << "Done: Arches::scheduleTimeAdvance\n";
#endif
#else
  NOT_FINISHED("new task stuff");
#endif
}


// ****************************************************************************
// Actual initialization
// ****************************************************************************
void
Arches::paramInit(const ProcessorGroup* ,
		  const Patch* patch,
		  DataWarehouseP& old_dw,
		  DataWarehouseP& )
{
  // ....but will only compute for computational domain
  SFCXVariable<double> uVelocity;
  SFCYVariable<double> vVelocity;
  SFCZVariable<double> wVelocity;
  CCVariable<double> pressure;
  vector<CCVariable<double> > scalar(d_nofScalars);
  CCVariable<double> density;
  CCVariable<double> viscosity;

  int matlIndex = 0;
  old_dw->allocate(uVelocity, d_lab->d_uVelocityINLabel, matlIndex, patch);
  old_dw->allocate(vVelocity, d_lab->d_vVelocityINLabel, matlIndex, patch);
  old_dw->allocate(wVelocity, d_lab->d_wVelocityINLabel, matlIndex, patch);
  old_dw->allocate(pressure, d_lab->d_pressureINLabel, matlIndex, patch);
  for (int ii = 0; ii < d_nofScalars; ii++) {
    old_dw->allocate(scalar[ii], d_lab->d_scalarINLabel, ii, patch);
  }
  old_dw->allocate(density, d_lab->d_densityINLabel, matlIndex, patch);
  old_dw->allocate(viscosity, d_lab->d_viscosityINLabel, matlIndex, patch);

  // ** WARNING **  this needs to be changed soon (6/9/2000)
  IntVector domLoU = uVelocity.getFortLowIndex();
  IntVector domHiU = uVelocity.getFortHighIndex();
  IntVector idxLoU = domLoU;
  IntVector idxHiU = domHiU;
  IntVector domLoV = vVelocity.getFortLowIndex();
  IntVector domHiV = vVelocity.getFortHighIndex();
  IntVector idxLoV = domLoV;
  IntVector idxHiV = domHiV;
  IntVector domLoW = wVelocity.getFortLowIndex();
  IntVector domHiW = wVelocity.getFortHighIndex();
  IntVector idxLoW = domLoW;
  IntVector idxHiW = domHiW;
  IntVector domLo = pressure.getFortLowIndex();
  IntVector domHi = pressure.getFortHighIndex();
  IntVector idxLo = domLo;
  IntVector idxHi = domHi;

  //can read these values from input file 
  double uVal = 0.0, vVal = 0.0, wVal = 0.0;
  double pVal = 0.0, denVal = 0.0;
  double visVal = d_physicalConsts->getMolecularViscosity();
  FORT_INIT(domLoU.get_pointer(), domHiU.get_pointer(), 
	    idxLoU.get_pointer(), idxHiU.get_pointer(),
	    uVelocity.getPointer(), &uVal, 
	    domLoV.get_pointer(), domHiV.get_pointer(), 
	    idxLoV.get_pointer(), idxHiV.get_pointer(),
	    vVelocity.getPointer(), &vVal, 
	    domLoW.get_pointer(), domHiW.get_pointer(), 
	    idxLoW.get_pointer(), idxHiW.get_pointer(),
	    wVelocity.getPointer(), &wVal,
	    domLo.get_pointer(), domHi.get_pointer(), 
	    idxLo.get_pointer(), idxHi.get_pointer(),
	    pressure.getPointer(), &pVal, 
	    density.getPointer(), &denVal, 
	    viscosity.getPointer(), &visVal);
  for (int ii = 0; ii < d_nofScalars; ii++) {
    double scalVal = 0.0;
    FORT_INIT_SCALAR(domLo.get_pointer(), domHi.get_pointer(),
		     idxLo.get_pointer(), idxHi.get_pointer(), 
		     scalar[ii].getPointer(), &scalVal);
  }

  old_dw->put(uVelocity, d_lab->d_uVelocityINLabel, matlIndex, patch);
  old_dw->put(vVelocity, d_lab->d_vVelocityINLabel, matlIndex, patch);
  old_dw->put(wVelocity, d_lab->d_wVelocityINLabel, matlIndex, patch);
  old_dw->put(pressure, d_lab->d_pressureINLabel, matlIndex, patch);
  for (int ii = 0; ii < d_nofScalars; ii++) {
    old_dw->put(scalar[ii], d_lab->d_scalarINLabel, ii, patch);
  }
  old_dw->put(density, d_lab->d_densityINLabel, matlIndex, patch);
  old_dw->put(viscosity, d_lab->d_viscosityINLabel, matlIndex, patch);

}
