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
#include <Uintah/Grid/SFCXVariable.h>
#include <Uintah/Grid/SFCYVariable.h>
#include <Uintah/Grid/SFCZVariable.h>
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
Arches::Arches(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
  d_densityINLabel = scinew VarLabel("densityIN", 
				   CCVariable<double>::getTypeDescription() );
  d_pressureINLabel = scinew VarLabel("pressureIN", 
				   CCVariable<double>::getTypeDescription() );
  d_uVelocityINLabel = scinew VarLabel("uVelocityIN", 
				   SFCXVariable<double>::getTypeDescription() );
  d_vVelocityINLabel = scinew VarLabel("vVelocityIN", 
				   SFCYVariable<double>::getTypeDescription() );
  d_wVelocityINLabel = scinew VarLabel("wVelocityIN", 
				   SFCZVariable<double>::getTypeDescription() );
  d_scalarINLabel = scinew VarLabel("scalarIN", 
				   CCVariable<double>::getTypeDescription() );
  d_viscosityINLabel = scinew VarLabel("viscosityIN", 
				   CCVariable<double>::getTypeDescription() );
  d_cellTypeLabel = scinew VarLabel("cellType", 
				   CCVariable<int>::getTypeDescription() );
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

  d_nlSolver->problemSetup(db);
}

//****************************************************************************
// Schedule initialization
//****************************************************************************
void 
Arches::scheduleInitialize(const LevelP& level,
			   SchedulerP& sched,
			   DataWarehouseP& dw)
{
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
  if (d_boundaryCondition->getPressureBC()) 
    d_boundaryCondition->sched_computePressureBC(level, sched, dw, dw);
}

//****************************************************************************
// schedule the initialization of parameters
//****************************************************************************
void 
Arches::sched_paramInit(const LevelP& level,
			SchedulerP& sched,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;

    // primitive variable initialization
    Task* tsk = new Task("Arches::paramInit",
			 patch, old_dw, new_dw, this,
			 &Arches::paramInit);
    int matlIndex = 0;
    tsk->computes(new_dw, d_uVelocityINLabel, matlIndex, patch);
    tsk->computes(new_dw, d_vVelocityINLabel, matlIndex, patch);
    tsk->computes(new_dw, d_wVelocityINLabel, matlIndex, patch);
    tsk->computes(new_dw, d_pressureINLabel, matlIndex, patch);
    for (int ii = 0; ii < d_nofScalars; ii++) 
      tsk->computes(new_dw, d_scalarINLabel, ii, patch);
    tsk->computes(new_dw, d_densityINLabel, matlIndex, patch);
    tsk->computes(new_dw, d_viscosityINLabel, matlIndex, patch);
    sched->addTask(tsk);
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

  cerr << "Begin: Arches::scheduleTimeAdvance\n";
  // Schedule the non-linear solve
  // require : densityCP, viscosityCTS, [u,v,w]VelocitySP, 
  //           pressureIN. scalarIN
  // compute : densityRCP, viscosityRCTS, [u,v,w]VelocityMS,
  //           pressurePS, scalarSS 
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


//****************************************************************************
// Actual initialization
//****************************************************************************
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
  old_dw->allocate(uVelocity, d_uVelocityINLabel, matlIndex, patch);
  old_dw->allocate(vVelocity, d_vVelocityINLabel, matlIndex, patch);
  old_dw->allocate(wVelocity, d_wVelocityINLabel, matlIndex, patch);
  old_dw->allocate(pressure, d_pressureINLabel, matlIndex, patch);
  for (int ii = 0; ii < d_nofScalars; ii++) {
    old_dw->allocate(scalar[ii], d_scalarINLabel, ii, patch);
  }
  old_dw->allocate(density, d_densityINLabel, matlIndex, patch);
  old_dw->allocate(viscosity, d_viscosityINLabel, matlIndex, patch);

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

  old_dw->put(uVelocity, d_uVelocityINLabel, matlIndex, patch);
  old_dw->put(vVelocity, d_vVelocityINLabel, matlIndex, patch);
  old_dw->put(wVelocity, d_wVelocityINLabel, matlIndex, patch);
  old_dw->put(pressure, d_pressureINLabel, matlIndex, patch);
  for (int ii = 0; ii < d_nofScalars; ii++) {
    old_dw->put(scalar[ii], d_scalarINLabel, ii, patch);
  }
  old_dw->put(density, d_densityINLabel, matlIndex, patch);
  old_dw->put(viscosity, d_viscosityINLabel, matlIndex, patch);

  // Testing if correct values have been put
  /*
  cout << " In C++ (Arches.cc) " << endl;
  for (int kk = indexLow.z(); kk <= indexHigh.z(); kk++) {
    for (int jj = indexLow.y(); jj <= indexHigh.y(); jj++) {
      for (int ii = indexLow.x(); ii <= indexHigh.x(); ii++) {
	cout << "(" << ii << "," << jj << "," << kk << ") : "
	     << " UU = " << uVelocity[IntVector(ii,jj,kk)]
	     << " DEN = " << density[IntVector(ii,jj,kk)]
	     << " VIS = " << viscosity[IntVector(ii,jj,kk)] << endl;
      }
    }
  }
  */
}
  
//
// $Log$
// Revision 1.51  2000/07/11 15:46:26  rawat
// added setInitialGuess in PicardNonlinearSolver and also added uVelSrc
//
// Revision 1.50  2000/07/07 23:07:44  rawat
// added inlet bc's
//
// Revision 1.49  2000/06/30 06:29:41  bbanerje
// Got Inlet Area to be calculated correctly .. but now two CellInformation
// variables are being created (Rawat ... check that).
//
// Revision 1.48  2000/06/30 04:19:16  rawat
// added turbulence model and compute properties
//
// Revision 1.47  2000/06/29 06:22:47  bbanerje
// Updated FCVariable to SFCX, SFCY, SFCZVariables and made corresponding
// changes to profv.  Code is broken until the changes are reflected
// thru all the files.
//
// Revision 1.46  2000/06/28 08:14:52  bbanerje
// Changed the init routines a bit.
//
// Revision 1.45  2000/06/22 23:06:32  bbanerje
// Changed velocity related variables to FCVariable type.
// ** NOTE ** We may need 3 types of FCVariables (one for each direction)
//
// Revision 1.44  2000/06/20 20:42:35  rawat
// added some more boundary stuff and modified interface to IntVector. Before
// compiling the code you need to update /SCICore/Geometry/IntVector.h
//
// Revision 1.43  2000/06/19 18:00:28  rawat
// added function to compute velocity and density profiles and inlet bc.
// Fixed bugs in CellInformation.cc
//
// Revision 1.42  2000/06/18 01:20:14  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
// Revision 1.41  2000/06/17 07:06:22  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.40  2000/06/16 21:50:47  bbanerje
// Changed the Varlabels so that sequence in understood in init stage.
// First cycle detected in task graph.
//
// Revision 1.39  2000/06/16 07:06:16  bbanerje
// Added init of props, pressure bcs and turbulence model in Arches.cc
// Changed duplicate task names (setProfile) in BoundaryCondition.cc
// Commented out nolinear_dw creation in PicardNonlinearSolver.cc
//
// Revision 1.38  2000/06/15 22:13:21  rawat
// modified boundary stuff
//
// Revision 1.37  2000/06/14 20:40:47  rawat
// modified boundarycondition for physical boundaries and
// added CellInformation class
//
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
// MPM particle initalization now works
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

