//----- BoundaryCondition.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/MPMArches/MPMArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesConstVariables.h>
#include <Packages/Uintah/CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Packages/Uintah/Core/Grid/Variables/Stencil.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/Properties.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/Reductions.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/GeometryPiece/UnionGeometryPiece.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Containers/StaticArray.h>
#include <iostream>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

#include <Packages/Uintah/CCA/Components/Arches/fortran/celltypeInit_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/areain_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/profscalar_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/calpbc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/inlbcs_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/inlpresbcinout_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcscalar_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcuvel_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcvvel_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcwvel_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcpress_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/profv_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcenthalpy_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/enthalpyradwallbc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/addpressuregrad_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/intrusion_computevel_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/mmbcenthalpy_energyex_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/mmbcvelocity_momex_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/mmbcvelocity_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/mmcelltypeinit_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/mmenthalpywallbc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/mmscalarwallbc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/mmwallbc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/mmwallbc_trans_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/mm_computevel_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/mm_explicit_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/mm_explicit_oldvalue_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/mm_explicit_vel_fort.h>

//****************************************************************************
// Constructor for BoundaryCondition
//****************************************************************************
BoundaryCondition::BoundaryCondition(const ArchesLabel* label,
				     const MPMArchesLabel* MAlb,
				     PhysicalConstants* phys_const,
				     Properties* props,
				     bool calcReactScalar,
				     bool calcEnthalpy):
                                     d_lab(label), d_MAlab(MAlb),
				     d_physicalConsts(phys_const), 
				     d_props(props),
				     d_reactingScalarSolve(calcReactScalar),
				     d_enthalpySolve(calcEnthalpy)
{
  d_nofScalars = d_props->getNumMixVars();
  MM_CUTOFF_VOID_FRAC = 0.5;
  d_wallBdry = 0;
  d_pressureBC = 0;
  d_outletBC = 0;
}

//****************************************************************************
// Destructor
//****************************************************************************
BoundaryCondition::~BoundaryCondition()
{
  delete d_wallBdry;
  delete d_pressureBC;
  delete d_outletBC;
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void 
BoundaryCondition::problemSetup(const ProblemSpecP& params)
{

  ProblemSpecP db = params->findBlock("BoundaryConditions");
  d_flowfieldCellTypeVal = -1;
  d_numInlets = 0;
  int total_cellTypes = 0;
  int numMixingScalars = d_props->getNumMixVars();
  
// flow ramping is on or off, it is the same for all inlets  
  db->getWithDefault("ramping_inlet_flowrate", d_ramping_inlet_flowrate, true);
  db->getWithDefault("carbon_balance", d_carbon_balance, false);
  if (ProblemSpecP inlet_db = db->findBlock("FlowInlet")) {
    d_inletBoundary = true;
    for (ProblemSpecP inlet_db = db->findBlock("FlowInlet");
         inlet_db != 0; inlet_db = inlet_db->findNextBlock("FlowInlet")) {
      d_flowInlets.push_back(FlowInlet(numMixingScalars, total_cellTypes));
      d_flowInlets[d_numInlets].problemSetup(inlet_db);
      // compute density and other dependent properties
      d_flowInlets[d_numInlets].streamMixturefraction.d_initEnthalpy=true;
      d_flowInlets[d_numInlets].streamMixturefraction.d_scalarDisp=0.0;
      d_props->computeInletProperties(
                        d_flowInlets[d_numInlets].streamMixturefraction,
		        d_flowInlets[d_numInlets].calcStream);
      ++total_cellTypes;
      ++d_numInlets;
    }
  }
  else {
    cout << "Flow inlet boundary not specified" << endl;
    d_inletBoundary = false;
  }
 
  if (ProblemSpecP wall_db = db->findBlock("WallBC")) {
    d_wallBoundary = true;
    d_wallBdry = scinew WallBdry(total_cellTypes);
    d_wallBdry->problemSetup(wall_db);
    ++total_cellTypes;
  }
  else {
    cout << "Wall boundary not specified" << endl;
    d_wallBoundary = false;
  }
  
  if (ProblemSpecP press_db = db->findBlock("PressureBC")) {
    d_pressureBoundary = true;
    d_pressureBC = scinew PressureInlet(numMixingScalars, total_cellTypes);
    d_pressureBC->problemSetup(press_db);
    // compute density and other dependent properties
    d_pressureBC->streamMixturefraction.d_initEnthalpy=true;
    d_pressureBC->streamMixturefraction.d_scalarDisp=0.0;
    d_props->computeInletProperties(
                		 d_pressureBC->streamMixturefraction, 
		        	 d_pressureBC->calcStream);
    ++total_cellTypes;
  }
  else {
    cout << "Pressure boundary not specified" << endl;
    d_pressureBoundary = false;
  }
  
  if (ProblemSpecP outlet_db = db->findBlock("outletBC")) {
    d_outletBoundary = true;
    d_outletBC = scinew FlowOutlet(numMixingScalars, total_cellTypes);
    d_outletBC->problemSetup(outlet_db);
    // compute density and other dependent properties
    d_outletBC->streamMixturefraction.d_initEnthalpy=true;
    d_outletBC->streamMixturefraction.d_scalarDisp=0.0;
    d_props->computeInletProperties(
			       d_outletBC->streamMixturefraction, 
			       d_outletBC->calcStream);
    ++total_cellTypes;
  }
  else {
    cout << "Outlet boundary not specified" << endl;
    d_outletBoundary = false;
  }

  if (ProblemSpecP intrusion_db = db->findBlock("intrusions")) {
    d_intrusionBoundary = true;
    d_intrusionBC = scinew IntrusionBdry(total_cellTypes);
    d_intrusionBC->problemSetup(intrusion_db);
    ++total_cellTypes;
  }
  else {
    cout << "Intrusion boundary not specified" << endl;
    d_intrusionBoundary = false;
  }

  d_mmWallID = -10; // invalid cell type
  // if multimaterial then add an id for multimaterial wall
  if (d_MAlab) 
    d_mmWallID = total_cellTypes;

}

//****************************************************************************
// schedule the initialization of cell types
//****************************************************************************
void 
BoundaryCondition::sched_cellTypeInit(SchedulerP& sched, const PatchSet* patches,
				      const MaterialSet* matls)
{
  // cell type initialization
  Task* tsk = scinew Task("BoundaryCondition::cellTypeInit",
			  this,
			  &BoundaryCondition::cellTypeInit);
  tsk->computes(d_lab->d_cellTypeLabel);
  sched->addTask(tsk, patches, matls);

}

//****************************************************************************
// Actual initialization of celltype
//****************************************************************************
void 
BoundaryCondition::cellTypeInit(const ProcessorGroup*,
				const PatchSubset* patches,
				const MaterialSubset*,
				DataWarehouse*,
				DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    CCVariable<int> cellType;
    new_dw->allocateAndPut(cellType, d_lab->d_cellTypeLabel, matlIndex, patch);

    IntVector domLo = cellType.getFortLowIndex();
    IntVector domHi = cellType.getFortHighIndex();
    IntVector idxLo = domLo;
    IntVector idxHi = domHi;
 
    // initialize CCVariable to -1 which corresponds to flowfield
    // fort_celltypeinit(idxLo, idxHi, cellType, d_flowfieldCellTypeVal);
    cellType.initialize(-1);
    
    // Find the geometry of the patch
    Box patchBox = patch->getBox();

    // wall boundary type
    int celltypeval;
    {
      if (d_wallBoundary) {
        int nofGeomPieces = (int)d_wallBdry->d_geomPiece.size();
        for (int ii = 0; ii < nofGeomPieces; ii++) {
	  GeometryPiece*  piece = d_wallBdry->d_geomPiece[ii];
	  Box geomBox = piece->getBoundingBox();
	  Box b = geomBox.intersect(patchBox);
	  // check for another geometry
	  if (!(b.degenerate())) {
	    CellIterator iter = patch->getCellCenterIterator(b);
	    IntVector idxLo = iter.begin();
	    IntVector idxHi = iter.end() - IntVector(1,1,1);
	    celltypeval = d_wallBdry->d_cellTypeID;
	    fort_celltypeinit(idxLo, idxHi, cellType, celltypeval);
	  }
        }
      }
    }
    // initialization for pressure boundary
    {
      if (d_pressureBoundary) {
	int nofGeomPieces = (int)d_pressureBC->d_geomPiece.size();
	for (int ii = 0; ii < nofGeomPieces; ii++) {
	  GeometryPiece*  piece = d_pressureBC->d_geomPiece[ii];
	  Box geomBox = piece->getBoundingBox();
	  Box b = geomBox.intersect(patchBox);
	  // check for another geometry
	  if (!(b.degenerate())) {
	    CellIterator iter = patch->getCellCenterIterator(b);
	    IntVector idxLo = iter.begin();
	    IntVector idxHi = iter.end() - IntVector(1,1,1);
	    celltypeval = d_pressureBC->d_cellTypeID;
	    fort_celltypeinit(idxLo, idxHi, cellType, celltypeval);
	  }
	}
      }
    }
    // initialization for outlet boundary
    {
      if (d_outletBoundary) {
	int nofGeomPieces = (int)d_outletBC->d_geomPiece.size();
	for (int ii = 0; ii < nofGeomPieces; ii++) {
	  GeometryPiece*  piece = d_outletBC->d_geomPiece[ii];
	  Box geomBox = piece->getBoundingBox();
	  Box b = geomBox.intersect(patchBox);
	  // check for another geometry
	  if (!(b.degenerate())) {
	    CellIterator iter = patch->getCellCenterIterator(b);
	    IntVector idxLo = iter.begin();
	    IntVector idxHi = iter.end() - IntVector(1,1,1);
	    celltypeval = d_outletBC->d_cellTypeID;
	    fort_celltypeinit(idxLo, idxHi, cellType, celltypeval);
	  }
	}
      }
    }
    // set boundary type for inlet flow field
    {
      if (d_inletBoundary) {
        for (int ii = 0; ii < d_numInlets; ii++) {
          int nofGeomPieces = (int)d_flowInlets[ii].d_geomPiece.size();
          for (int jj = 0; jj < nofGeomPieces; jj++) {
	    GeometryPiece*  piece = d_flowInlets[ii].d_geomPiece[jj];
	    Box geomBox = piece->getBoundingBox();
	    Box b = geomBox.intersect(patchBox);
	    // check for another geometry
	    if (b.degenerate())
	    continue; // continue the loop for other inlets
	    // iterates thru box b, converts from geometry space to index space
	    // make sure this works
#if 0
	    CellIterator iter = patch->getCellCenterIterator(b);
	    IntVector idxLo = iter.begin();
	    IntVector idxHi = iter.end() - IntVector(1,1,1);
	    celltypeval = d_flowInlets[ii].d_cellTypeID;
	    fort_celltypeinit(idxLo, idxHi, cellType, celltypeval);
#endif
	    for (CellIterator iter = patch->getCellCenterIterator(b);
	         !iter.done(); iter++) {
	      Point p = patch->cellPosition(*iter);
	      if (piece->inside(p)) 
	        cellType[*iter] = d_flowInlets[ii].d_cellTypeID;
	    }
          }
        }
      }
    }
    
    {
      if (d_intrusionBoundary) {
	int nofGeomPieces = (int)d_intrusionBC->d_geomPiece.size();
	for (int ii = 0; ii < nofGeomPieces; ii++) {
	  GeometryPiece*  piece = d_intrusionBC->d_geomPiece[ii];
	  Box geomBox = piece->getBoundingBox();
	  Box b = geomBox.intersect(patchBox);
	  if (!(b.degenerate())) {
	    CellIterator iter = patch->getCellCenterIterator(b);
	    IntVector idxLo = iter.begin();
	    IntVector idxHi = iter.end() - IntVector(1,1,1);
	    celltypeval = d_intrusionBC->d_cellTypeID;
	    fort_celltypeinit(idxLo, idxHi, cellType, celltypeval);
	  }
	}
      }
    }
  }
}  

// for multimaterial
//****************************************************************************
// schedule the initialization of mm wall cell types
//****************************************************************************
void 
BoundaryCondition::sched_mmWallCellTypeInit(SchedulerP& sched, const PatchSet* patches,
					    const MaterialSet* matls, bool fixCellType)
{
  // cell type initialization
  Task* tsk = scinew Task("BoundaryCondition::mmWallCellTypeInit",
			  this,
			  &BoundaryCondition::mmWallCellTypeInit,
			  fixCellType);
  
  int numGhostcells = 0;

  // New DW warehouse variables to calculate cell types if we are
  // recalculating cell types and resetting void fractions

  double time = d_lab->d_sharedState->getElapsedTime();
  bool recalculateCellType = false;
  int dwnumber = d_lab->d_sharedState->getCurrentTopLevelTimeStep();
  //  cout << "Current DW number = " << dwnumber << endl;
  //  if (time < 1.0e-10 || !fixCellType) recalculateCellType = true;
  if (dwnumber < 2 || !fixCellType) recalculateCellType = true;
  //  cout << "recalculateCellType =" << recalculateCellType << endl;

  tsk->requires(Task::OldDW, d_lab->d_mmgasVolFracLabel,
		Ghost::None, numGhostcells);
  tsk->requires(Task::OldDW, d_lab->d_mmcellTypeLabel, 
		Ghost::None, numGhostcells);
  tsk->requires(Task::OldDW, d_MAlab->mmCellType_MPMLabel, 
		Ghost::None, numGhostcells);
  if (d_cutCells)
    tsk->requires(Task::OldDW, d_MAlab->mmCellType_CutCellLabel,
		  Ghost::None, numGhostcells);

  if (recalculateCellType) {

    tsk->requires(Task::NewDW, d_MAlab->void_frac_CCLabel, 
		  Ghost::None, numGhostcells);
    tsk->requires(Task::OldDW, d_lab->d_cellTypeLabel, 
		  Ghost::None, numGhostcells);

    tsk->computes(d_lab->d_mmgasVolFracLabel);
    tsk->computes(d_lab->d_mmcellTypeLabel);
    tsk->computes(d_MAlab->mmCellType_MPMLabel);
    if (d_cutCells)
      tsk->computes(d_MAlab->mmCellType_CutCellLabel);

    tsk->modifies(d_MAlab->void_frac_MPM_CCLabel);
    if (d_cutCells)
      tsk->modifies(d_MAlab->void_frac_CutCell_CCLabel);
  }
  else {

    tsk->computes(d_lab->d_mmgasVolFracLabel);
    tsk->computes(d_lab->d_mmcellTypeLabel);
    tsk->computes(d_MAlab->mmCellType_MPMLabel);
    if (d_cutCells)
      tsk->computes(d_MAlab->mmCellType_CutCellLabel);

  }

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Actual initialization of celltype
//****************************************************************************
void 
BoundaryCondition::mmWallCellTypeInit(const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset*,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw,
				      bool fixCellType)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    int numGhostcells = 0;

    double time = d_lab->d_sharedState->getElapsedTime();
    bool recalculateCellType = false;
    int dwnumber = d_lab->d_sharedState->getCurrentTopLevelTimeStep();
    //    cout << "Current DW number = " << dwnumber << endl;
    //    if (time < 1.0e-10 || !fixCellType) recalculateCellType = true;
    if (dwnumber < 2 || !fixCellType) recalculateCellType = true;
    //    cout << "recalculateCellType = " << recalculateCellType << endl;

    // New DW void fraction to decide cell types and reset void fractions

    constCCVariable<double> voidFrac;
    constCCVariable<int> cellType;

    CCVariable<double> mmGasVolFrac;
    CCVariable<int> mmCellType;
    CCVariable<int> mmCellTypeMPM;
    CCVariable<int> mmCellTypeCutCell;
    CCVariable<double> voidFracMPM;
    CCVariable<double> voidFracCutCell;

    constCCVariable<double> oldGasVolFrac;
    constCCVariable<int> mmCellTypeOld;
    constCCVariable<int> mmCellTypeMPMOld;
    constCCVariable<int> mmCellTypeCutCellOld;
    constCCVariable<double> voidFracMPMOld;
    constCCVariable<double> voidFracCutCellOld;

    old_dw->get(oldGasVolFrac, d_lab->d_mmgasVolFracLabel, matlIndex, patch,
		Ghost::None, numGhostcells);
    old_dw->get(mmCellTypeOld, d_lab->d_mmcellTypeLabel, matlIndex, patch,
		Ghost::None, numGhostcells);
    old_dw->get(mmCellTypeMPMOld, d_MAlab->mmCellType_MPMLabel, matlIndex, patch,
		Ghost::None, numGhostcells);
    if (d_cutCells)
      old_dw->get(mmCellTypeCutCellOld, d_MAlab->mmCellType_CutCellLabel, matlIndex, patch,
		  Ghost::None, numGhostcells);

    if (recalculateCellType) {

      new_dw->get(voidFrac, d_MAlab->void_frac_CCLabel, matlIndex, patch,
		  Ghost::None, numGhostcells);
      old_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch,
		  Ghost::None, numGhostcells);

      new_dw->allocateAndPut(mmGasVolFrac, d_lab->d_mmgasVolFracLabel, 
			     matlIndex, patch);
      new_dw->allocateAndPut(mmCellType, d_lab->d_mmcellTypeLabel, 
			     matlIndex, patch);
      new_dw->allocateAndPut(mmCellTypeMPM, d_MAlab->mmCellType_MPMLabel, 
			     matlIndex, patch);
      if (d_cutCells)
	new_dw->allocateAndPut(mmCellTypeCutCell, d_MAlab->mmCellType_CutCellLabel, 
			       matlIndex, patch);

      new_dw->getModifiable(voidFracMPM, d_MAlab->void_frac_MPM_CCLabel, 
			    matlIndex, patch);
      if (d_cutCells)
	new_dw->getModifiable(voidFracCutCell, d_MAlab->void_frac_CutCell_CCLabel, 
			      matlIndex, patch);

    }
    else {

      new_dw->allocateAndPut(mmGasVolFrac, d_lab->d_mmgasVolFracLabel, 
			     matlIndex, patch);
      new_dw->allocateAndPut(mmCellType, d_lab->d_mmcellTypeLabel, 
			     matlIndex, patch);
      new_dw->allocateAndPut(mmCellTypeMPM, d_MAlab->mmCellType_MPMLabel, 
			     matlIndex, patch);
      if (d_cutCells)
	new_dw->allocateAndPut(mmCellTypeCutCell, d_MAlab->mmCellType_CutCellLabel, 
			       matlIndex, patch);
    }

    IntVector domLo = mmCellType.getFortLowIndex();
    IntVector domHi = mmCellType.getFortHighIndex();
    IntVector idxLo = domLo;
    IntVector idxHi = domHi;

    if (recalculateCellType) {

      mmGasVolFrac.copyData(voidFrac);
      mmCellType.copyData(cellType);
      mmCellTypeMPM.copyData(cellType);
      if (d_cutCells)
	mmCellTypeCutCell.copyData(cellType);

      // resets old mmwall type back to flow field and sets cells with void fraction
      // of less than .5 to mmWall

      fort_mmcelltypeinit(idxLo, idxHi, mmGasVolFrac, mmCellType, d_mmWallID,
      			  d_flowfieldCellTypeVal, MM_CUTOFF_VOID_FRAC);  

      fort_mmcelltypeinit(idxLo, idxHi, voidFracMPM, mmCellTypeMPM, d_mmWallID,
      			  d_flowfieldCellTypeVal, MM_CUTOFF_VOID_FRAC);  
      if (d_cutCells)
	fort_mmcelltypeinit(idxLo, idxHi, voidFracCutCell, mmCellTypeCutCell, d_mmWallID,
			    d_flowfieldCellTypeVal, MM_CUTOFF_VOID_FRAC);  
    }
    else {

      mmGasVolFrac.copyData(oldGasVolFrac);
      mmCellType.copyData(mmCellTypeOld);
      mmCellTypeMPM.copyData(mmCellTypeMPMOld);
      if (d_cutCells)
	mmCellTypeCutCell.copyData(mmCellTypeCutCellOld);
    }
  }  
}

// for multimaterial
//****************************************************************************
// schedule the initialization of mm wall cell types for the very first
// time step
//****************************************************************************
void 
BoundaryCondition::sched_mmWallCellTypeInit_first(SchedulerP& sched, 
						  const PatchSet* patches,
						  const MaterialSet* matls)
{
  // cell type initialization
  Task* tsk = scinew Task("BoundaryCondition::mmWallCellTypeInit_first",
			  this,
			  &BoundaryCondition::mmWallCellTypeInit_first);
  
  int numGhostcells = 0;
  tsk->requires(Task::NewDW, d_MAlab->void_frac_CCLabel, 
		Ghost::None, numGhostcells);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		Ghost::None, numGhostcells);
  tsk->computes(d_lab->d_mmcellTypeLabel);
  tsk->modifies(d_lab->d_mmgasVolFracLabel);
  
  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Actual initialization of celltype
//****************************************************************************
void 
BoundaryCondition::mmWallCellTypeInit_first(const ProcessorGroup*,
					    const PatchSubset* patches,
					    const MaterialSubset*,
					    DataWarehouse* ,
					    DataWarehouse* new_dw)	
{

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    int numGhostcells = 0;
    constCCVariable<int> cellType;
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch,
		Ghost::None, numGhostcells);
    constCCVariable<double> voidFrac;
    new_dw->get(voidFrac, d_MAlab->void_frac_CCLabel, matlIndex, patch,
		Ghost::None, numGhostcells);
    CCVariable<int> mmcellType;
    new_dw->allocateAndPut(mmcellType, d_lab->d_mmcellTypeLabel, matlIndex, patch);
    mmcellType.copyData(cellType);
    CCVariable<double> mmvoidFrac;
    new_dw->getModifiable(mmvoidFrac, d_lab->d_mmgasVolFracLabel, matlIndex, patch);
    mmvoidFrac.copyData(voidFrac);
	
    IntVector domLo = mmcellType.getFortLowIndex();
    IntVector domHi = mmcellType.getFortHighIndex();
    IntVector idxLo = domLo;
    IntVector idxHi = domHi;

    // resets old mmwall type back to flow field and sets cells with void fraction
    // of less than .01 to mmWall

    fort_mmcelltypeinit(idxLo, idxHi, mmvoidFrac, mmcellType, d_mmWallID,
			d_flowfieldCellTypeVal, MM_CUTOFF_VOID_FRAC);

    // allocateAndPut instead:
    /* new_dw->put(mmcellType, d_lab->d_mmcellTypeLabel, matlIndex, patch); */;
    // save in arches label
    // allocateAndPut instead:
    /* new_dw->put(mmvoidFrac, d_lab->d_mmgasVolFracLabel, matlIndex, patch); */;
  }  
}


    
//****************************************************************************
// Actual initialization of celltype
//****************************************************************************
void 
BoundaryCondition::computeInletFlowArea(const ProcessorGroup*,
					const PatchSubset* patches,
					const MaterialSubset*,
					DataWarehouse*,
					DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Create the cellType variable
    constCCVariable<int> cellType;
    
    // Get the cell type data from the old_dw
    // **WARNING** numGhostcells, Ghost::None may change in the future
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch,
		Ghost::None, Arches::ZEROGHOSTCELLS);
    
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellInfo = cellInfoP.get().get_rep();
    
    // Get the low and high index for the variable and the patch
    IntVector domLo = cellType.getFortLowIndex();
    IntVector domHi = cellType.getFortHighIndex();
    
    // Get the geometry of the patch
    Box patchBox = patch->getBox();
    
    // Go thru the number of inlets
    for (int ii = 0; ii < d_numInlets; ii++) {
      
      // Loop thru the number of geometry pieces in each inlet
      int nofGeomPieces = (int)d_flowInlets[ii].d_geomPiece.size();
      for (int jj = 0; jj < nofGeomPieces; jj++) {
	
	// Intersect the geometry piece with the patch box
	GeometryPiece*  piece = d_flowInlets[ii].d_geomPiece[jj];
	Box geomBox = piece->getBoundingBox();
	Box b = geomBox.intersect(patchBox);
	// check for another geometry
	if (b.degenerate()){
	  new_dw->put(sum_vartype(0),d_flowInlets[ii].d_area_label);
	  continue; // continue the loop for other inlets
	}
	
	// iterates thru box b, converts from geometry space to index space
	// make sure this works
	CellIterator iter = patch->getCellCenterIterator(b);
	IntVector idxLo = iter.begin();
	IntVector idxHi = iter.end() - IntVector(1,1,1);
	
	// Calculate the inlet area
	double inlet_area;
	int cellid = d_flowInlets[ii].d_cellTypeID;
	
	bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
	bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
	bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
	bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
	bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
	bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

	fort_areain(domLo, domHi, idxLo, idxHi, cellInfo->sew, cellInfo->sns,
		    cellInfo->stb, inlet_area, cellType, cellid,
		    d_flowfieldCellTypeVal,
		    xminus, xplus, yminus, yplus, zminus, zplus);
	
	// Write the inlet area to the old_dw
	new_dw->put(sum_vartype(inlet_area),d_flowInlets[ii].d_area_label);
      }
    }
  }
}
    
//****************************************************************************
// Schedule the computation of the presures bcs
//****************************************************************************
void 
BoundaryCondition::sched_computePressureBC(SchedulerP& sched, const PatchSet* patches,
					   const MaterialSet* matls)
{
  //copies old db to new_db and then uses non-linear
  //solver to compute new values
  Task* tsk = scinew Task("BoundaryCondition::computePressureBC",
			  this,
			  &BoundaryCondition::computePressureBC);

  // This task requires the pressure

#ifdef ExactMPMArchesInitialize
  if (d_MAlab)
    tsk->requires(Task::NewDW, d_lab->d_mmcellTypeLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  else
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
#else
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
#endif

  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
      // This task computes new uVelocity, vVelocity and wVelocity
  tsk->modifies(d_lab->d_pressurePSLabel);
  tsk->modifies(d_lab->d_uVelocitySPBCLabel);
  tsk->modifies(d_lab->d_vVelocitySPBCLabel);
  tsk->modifies(d_lab->d_wVelocitySPBCLabel);

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Actually calculate the pressure BCs
//****************************************************************************
void 
BoundaryCondition::computePressureBC(const ProcessorGroup* ,
				  const PatchSubset* patches,
				  const MaterialSubset*,
				  DataWarehouse*,
				  DataWarehouse* new_dw) 
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    constCCVariable<int> cellType;
    constCCVariable<double> density;
    constCCVariable<double> viscosity;
    CCVariable<double> pressure;
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;

    // get cellType, pressure and velocity

#ifdef ExactMPMArchesInitialize
    if (d_MAlab)
      new_dw->get(cellType, d_lab->d_mmcellTypeLabel, matlIndex, patch, Ghost::None,
		  Arches::ZEROGHOSTCELLS);
    else
      new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		  Arches::ZEROGHOSTCELLS);
#else
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
#endif

    new_dw->get(density, d_lab->d_densityCPLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->get(viscosity, d_lab->d_viscosityCTSLabel, matlIndex, patch, 
		Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->getModifiable(uVelocity, d_lab->d_uVelocitySPBCLabel, matlIndex, patch);
    new_dw->getModifiable(vVelocity, d_lab->d_vVelocitySPBCLabel, matlIndex, patch);
    new_dw->getModifiable(wVelocity, d_lab->d_wVelocitySPBCLabel, matlIndex, patch);
    new_dw->getModifiable(pressure, d_lab->d_pressurePSLabel, matlIndex, patch);
    PerPatch<CellInformationP> cellInfoP;

    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 

      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    else {

      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    }

    CellInformation* cellinfo = cellInfoP.get().get_rep();

    // Get the low and high index for the patch and the variables
    //  IntVector domLoScalar = density.getFortLowIndex();
    //  IntVector domHiScalar = density.getFortHighIndex();
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
    
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    
    fort_calpbc(uVelocity, vVelocity, wVelocity, idxLo, idxHi,
		pressure, density, viscosity, cellType, 
		d_pressureBC->d_cellTypeID,
		d_pressureBC->refPressure,
		cellinfo->dxepu, cellinfo->dynpv, cellinfo->dztpw,
		xminus, xplus, yminus, yplus, zminus, zplus);
    
  } 
}





//****************************************************************************
// Schedule computes inlet areas
// computes inlet area for inlet bc
//****************************************************************************
void 
BoundaryCondition::sched_calculateArea(SchedulerP& sched, const PatchSet* patches,
				       const MaterialSet* matls)
{
  Task* tsk = new Task("BoundaryCondition::calculateArea",
		       this,
		       &BoundaryCondition::computeInletFlowArea);

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  // ***warning checkpointing
  //      tsk->computes(old_dw, d_lab->d_cellInfoLabel, matlIndex, patch);
  for (int ii = 0; ii < d_numInlets; ii++) 
    tsk->computes(d_flowInlets[ii].d_area_label);

  sched->addTask(tsk, patches, matls);
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = new Task("BoundaryCondition::calculateArea",
			   patch, old_dw, new_dw, this,
			   &BoundaryCondition::computeInletFlowArea);
      int matlIndex = 0;
      tsk->requires(old_dw, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		    Arches::ZEROGHOSTCELLS);
      // ***warning checkpointing
      //      tsk->computes(old_dw, d_lab->d_cellInfoLabel, matlIndex, patch);
      for (int ii = 0; ii < d_numInlets; ii++) {
	// make it simple by adding matlindex for reduction vars
	tsk->computes(old_dw, d_flowInlets[ii].d_area_label);
      }
      sched->addTask(tsk);
    }
  }
#endif
}

//****************************************************************************
// Schedule set profile
// assigns flat velocity profiles for primary and secondary inlets
// Also sets flat profiles for density
//****************************************************************************
void 
BoundaryCondition::sched_setProfile(SchedulerP& sched, const PatchSet* patches,
				    const MaterialSet* matls)
{
  Task* tsk = scinew Task("BoundaryCondition::setProfile",
			  this,
			  &BoundaryCondition::setFlatProfile);

  // This task requires cellTypeVariable and areaLabel for inlet boundary
  // Also densityIN, [u,v,w] velocityIN, scalarIN
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  for (int ii = 0; ii < d_numInlets; ii++) {
    tsk->requires(Task::NewDW, d_flowInlets[ii].d_area_label);
  }
  if (d_enthalpySolve) {
    tsk->modifies(d_lab->d_enthalpySPLabel);
  }
  if (d_reactingScalarSolve) {
    tsk->modifies(d_lab->d_reactscalarSPLabel);
  }
    
  // This task computes new density, uVelocity, vVelocity and wVelocity, scalars
  tsk->modifies(d_lab->d_densityCPLabel);
  tsk->modifies(d_lab->d_uVelocitySPBCLabel);
  tsk->modifies(d_lab->d_vVelocitySPBCLabel);
  tsk->modifies(d_lab->d_wVelocitySPBCLabel);
  tsk->modifies(d_lab->d_uVelRhoHatLabel);
  tsk->modifies(d_lab->d_vVelRhoHatLabel);
  tsk->modifies(d_lab->d_wVelRhoHatLabel);

  for (int ii = 0; ii < d_props->getNumMixVars(); ii++) 
    tsk->modifies(d_lab->d_scalarSPLabel);

  for (int ii = 0; ii < d_numInlets; ii++) 
    tsk->computes(d_flowInlets[ii].d_flowRate_label);

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Actually set flat profile at flow inlet boundary
//****************************************************************************
void 
BoundaryCondition::setFlatProfile(const ProcessorGroup* /*pc*/,
				  const PatchSubset* patches,
				  const MaterialSubset*,
				  DataWarehouse*,
				  DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    constCCVariable<int> cellType;
    CCVariable<double> density;
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    SFCXVariable<double> uVelRhoHat;
    SFCYVariable<double> vVelRhoHat;
    SFCZVariable<double> wVelRhoHat;
    StaticArray<CCVariable<double> > scalar(d_nofScalars);
    CCVariable<double> reactscalar;
    CCVariable<double> enthalpy;

    new_dw->getModifiable(uVelocity, d_lab->d_uVelocitySPBCLabel, matlIndex, patch);
    new_dw->getModifiable(vVelocity, d_lab->d_vVelocitySPBCLabel, matlIndex, patch);
    new_dw->getModifiable(wVelocity, d_lab->d_wVelocitySPBCLabel, matlIndex, patch);
    new_dw->getModifiable(uVelRhoHat, d_lab->d_uVelRhoHatLabel, matlIndex, patch);
    new_dw->getModifiable(vVelRhoHat, d_lab->d_vVelRhoHatLabel, matlIndex, patch);
    new_dw->getModifiable(wVelRhoHat, d_lab->d_wVelRhoHatLabel, matlIndex, patch);
    
    // get cellType, density and velocity
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->getModifiable(density, d_lab->d_densityCPLabel, matlIndex, patch);
    for (int ii = 0; ii < d_nofScalars; ii++) {
      new_dw->getModifiable(scalar[ii], d_lab->d_scalarSPLabel, matlIndex, patch);
    }
      // reactscalar will be zero at the boundaries, so no further calculation
      // is required.
    if (d_reactingScalarSolve)
      new_dw->getModifiable(reactscalar, d_lab->d_reactscalarSPLabel, matlIndex, patch);
    
    if (d_enthalpySolve) 
      new_dw->getModifiable(enthalpy, d_lab->d_enthalpySPLabel, matlIndex, patch);

    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

    // loop thru the flow inlets to set all the components of velocity and density
    if (d_inletBoundary) {
      double time = 0.0; 
      for (int indx = 0; indx < d_numInlets; indx++) {
        sum_vartype area_var;
        new_dw->get(area_var, d_flowInlets[indx].d_area_label);
        double area = area_var;
	double actual_flow_rate;
      
        // Get a copy of the current flow inlet
        // check if given patch intersects with the inlet boundary of type index
        FlowInlet fi = d_flowInlets[indx];
        //cerr << " inlet area" << area << " flowrate" << fi.flowRate << endl;
        //cerr << "density=" << fi.calcStream.d_density << endl;
        fort_profv(uVelocity, vVelocity, wVelocity, idxLo, idxHi,
		   cellType, area, fi.d_cellTypeID, fi.flowRate, fi.inletVel,
		   fi.calcStream.d_density,
		   xminus, xplus, yminus, yplus, zminus, zplus, time,
		   d_ramping_inlet_flowrate, actual_flow_rate);

	d_flowInlets[indx].flowRate = actual_flow_rate;
	new_dw->put(delt_vartype(actual_flow_rate),
		    d_flowInlets[indx].d_flowRate_label);

        fort_profscalar(idxLo, idxHi, density, cellType,
		        fi.calcStream.d_density, fi.d_cellTypeID,
		        xminus, xplus, yminus, yplus, zminus, zplus);
        if (d_enthalpySolve)
        fort_profscalar(idxLo, idxHi, enthalpy, cellType,
		        fi.calcStream.d_enthalpy, fi.d_cellTypeID,
		        xminus, xplus, yminus, yplus, zminus, zplus);

      }
    }
    if (d_pressureBoundary) {
      // set density
      fort_profscalar(idxLo, idxHi, density, cellType,
		      d_pressureBC->calcStream.d_density,
		      d_pressureBC->d_cellTypeID,
		      xminus, xplus, yminus, yplus, zminus, zplus);
      if (d_enthalpySolve)
	fort_profscalar(idxLo, idxHi, enthalpy, cellType,
			d_pressureBC->calcStream.d_enthalpy,
			d_pressureBC->d_cellTypeID,
			xminus, xplus, yminus, yplus, zminus, zplus);
    }
    if (d_outletBoundary) {
      // set density
      fort_profscalar(idxLo, idxHi, density, cellType,
		      d_outletBC->calcStream.d_density,
		      d_outletBC->d_cellTypeID,
		      xminus, xplus, yminus, yplus, zminus, zplus);
      if (d_enthalpySolve)
	fort_profscalar(idxLo, idxHi, enthalpy, cellType,
			d_outletBC->calcStream.d_enthalpy,
			d_outletBC->d_cellTypeID,
			xminus, xplus, yminus, yplus, zminus, zplus);
    }

    for (int indx = 0; indx < d_nofScalars; indx++) {
      if (d_inletBoundary) {
        for (int ii = 0; ii < d_numInlets; ii++) {
	  double scalarValue = 
		 d_flowInlets[ii].streamMixturefraction.d_mixVars[indx];
	  fort_profscalar(idxLo, idxHi, scalar[indx], cellType,
			  scalarValue, d_flowInlets[ii].d_cellTypeID,
			  xminus, xplus, yminus, yplus, zminus, zplus);
	}
      }
      if (d_pressureBoundary) {
	double scalarValue = 
	       d_pressureBC->streamMixturefraction.d_mixVars[indx];
	fort_profscalar(idxLo, idxHi, scalar[indx], cellType, scalarValue,
			d_pressureBC->d_cellTypeID,
			xminus, xplus, yminus, yplus, zminus, zplus);
      }
      if (d_outletBoundary) {
	double scalarValue = 
	       d_outletBC->streamMixturefraction.d_mixVars[indx];
	fort_profscalar(idxLo, idxHi, scalar[indx], cellType, scalarValue,
			d_outletBC->d_cellTypeID,
			xminus, xplus, yminus, yplus, zminus, zplus);
      }
    }
    uVelRhoHat.copyData(uVelocity); 
    vVelRhoHat.copyData(vVelocity); 
    wVelRhoHat.copyData(wVelocity); 
  }
}

//****************************************************************************
// Actually calculate the velocity BC
//****************************************************************************
void 
BoundaryCondition::velocityBC(const ProcessorGroup*,
			      const Patch* patch,
			      int index,
			      CellInformation* cellinfo,
			      ArchesVariables* vars,
			      ArchesConstVariables* constvars) 
{
  // Call the fortran routines
  switch(index) {
  case 1:
    uVelocityBC(patch, cellinfo, vars, constvars);
    break;
  case 2:
    vVelocityBC(patch, cellinfo, vars, constvars);
    break;
  case 3:
    wVelocityBC(patch, cellinfo, vars, constvars);
    break;
  default:
    cerr << "Invalid Index value" << endl;
    break;
  }
}

//****************************************************************************
// call fortran routine to calculate the U Velocity BC
//****************************************************************************
void 
BoundaryCondition::uVelocityBC(const Patch* patch,
			       CellInformation* cellinfo,
			       ArchesVariables* vars,
		               ArchesConstVariables* constvars)
{
  int wall_celltypeval = wallCellType();
  int pressure_celltypeval = pressureCellType();
  int outlet_celltypeval = outletCellType();
  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  // computes momentum source term due to wall
  // uses total viscosity for wall source, not just molecular viscosity
  //double molViscosity = d_physicalConsts->getMolecularViscosity();
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  fort_bcuvel(constvars->uVelocity, vars->uVelocityCoeff[Arches::AP],
	      vars->uVelocityCoeff[Arches::AE],
	      vars->uVelocityCoeff[Arches::AW], 
	      vars->uVelocityCoeff[Arches::AN], 
	      vars->uVelocityCoeff[Arches::AS], 
	      vars->uVelocityCoeff[Arches::AT], 
	      vars->uVelocityCoeff[Arches::AB], 
	      vars->uVelNonlinearSrc, vars->uVelLinearSrc,
	      vars->uVelocityConvectCoeff[Arches::AE],
	      vars->uVelocityConvectCoeff[Arches::AW],
	      vars->uVelocityConvectCoeff[Arches::AN],
	      vars->uVelocityConvectCoeff[Arches::AS],
	      vars->uVelocityConvectCoeff[Arches::AT],
	      vars->uVelocityConvectCoeff[Arches::AB],
	      constvars->vVelocity, constvars->wVelocity,
	      idxLo, idxHi, constvars->cellType,
	      wall_celltypeval, outlet_celltypeval, pressure_celltypeval,
	      constvars->viscosity,
	      cellinfo->sewu, cellinfo->sns, cellinfo->stb,
	      cellinfo->yy, cellinfo->yv, cellinfo->zz, cellinfo->zw,
	      xminus, xplus, yminus, yplus, zminus, zplus);

}

//****************************************************************************
// call fortran routine to calculate the V Velocity BC
//****************************************************************************
void 
BoundaryCondition::vVelocityBC(const Patch* patch,
			       CellInformation* cellinfo,
			       ArchesVariables* vars,
		               ArchesConstVariables* constvars)
{
  int wall_celltypeval = wallCellType();
  int pressure_celltypeval = pressureCellType();
  int outlet_celltypeval = outletCellType();
  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  // computes momentum source term due to wall
  // uses total viscosity for wall source, not just molecular viscosity
  //double molViscosity = d_physicalConsts->getMolecularViscosity();
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  fort_bcvvel(constvars->vVelocity, vars->vVelocityCoeff[Arches::AP],
	      vars->vVelocityCoeff[Arches::AE],
	      vars->vVelocityCoeff[Arches::AW], 
	      vars->vVelocityCoeff[Arches::AN], 
	      vars->vVelocityCoeff[Arches::AS], 
	      vars->vVelocityCoeff[Arches::AT], 
	      vars->vVelocityCoeff[Arches::AB], 
	      vars->vVelNonlinearSrc, vars->vVelLinearSrc,
	      vars->vVelocityConvectCoeff[Arches::AE],
	      vars->vVelocityConvectCoeff[Arches::AW],
	      vars->vVelocityConvectCoeff[Arches::AN],
	      vars->vVelocityConvectCoeff[Arches::AS],
	      vars->vVelocityConvectCoeff[Arches::AT],
	      vars->vVelocityConvectCoeff[Arches::AB],
	      constvars->uVelocity, constvars->wVelocity,
	      idxLo, idxHi, constvars->cellType,
	      wall_celltypeval, outlet_celltypeval, pressure_celltypeval,
	      constvars->viscosity,
	      cellinfo->sew, cellinfo->snsv, cellinfo->stb,
	      cellinfo->xx, cellinfo->xu, cellinfo->zz, cellinfo->zw,
	      xminus, xplus, yminus, yplus, zminus, zplus);

}

//****************************************************************************
// call fortran routine to calculate the W Velocity BC
//****************************************************************************
void 
BoundaryCondition::wVelocityBC(const Patch* patch,
			       CellInformation* cellinfo,
			       ArchesVariables* vars,
		               ArchesConstVariables* constvars)
{
  int wall_celltypeval = wallCellType();
  int pressure_celltypeval = pressureCellType();
  int outlet_celltypeval = outletCellType();
  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  // computes momentum source term due to wall
  // uses total viscosity for wall source, not just molecular viscosity
  //double molViscosity = d_physicalConsts->getMolecularViscosity();
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  fort_bcwvel(constvars->wVelocity, vars->wVelocityCoeff[Arches::AP],
	      vars->wVelocityCoeff[Arches::AE],
	      vars->wVelocityCoeff[Arches::AW], 
	      vars->wVelocityCoeff[Arches::AN], 
	      vars->wVelocityCoeff[Arches::AS], 
	      vars->wVelocityCoeff[Arches::AT], 
	      vars->wVelocityCoeff[Arches::AB], 
	      vars->wVelNonlinearSrc, vars->wVelLinearSrc,
	      vars->wVelocityConvectCoeff[Arches::AE],
	      vars->wVelocityConvectCoeff[Arches::AW],
	      vars->wVelocityConvectCoeff[Arches::AN],
	      vars->wVelocityConvectCoeff[Arches::AS],
	      vars->wVelocityConvectCoeff[Arches::AT],
	      vars->wVelocityConvectCoeff[Arches::AB],
	      constvars->uVelocity, constvars->vVelocity,
	      idxLo, idxHi, constvars->cellType,
	      wall_celltypeval, outlet_celltypeval, pressure_celltypeval,
	      constvars->viscosity,
	      cellinfo->sew, cellinfo->sns, cellinfo->stbw,
	      cellinfo->xx, cellinfo->xu, cellinfo->yy, cellinfo->yv,
	      xminus, xplus, yminus, yplus, zminus, zplus);

}

//****************************************************************************
// Actually compute the pressure bcs
//****************************************************************************
void 
BoundaryCondition::pressureBC(const ProcessorGroup*,
			      const Patch* patch,
			      DataWarehouse* /*old_dw*/,
			      DataWarehouse* /*new_dw*/,
			      CellInformation* /*cellinfo*/,
			      ArchesVariables* vars,
			      ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector domLo = constvars->cellType.getFortLowIndex();
  IntVector domHi = constvars->cellType.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  IntVector domLong = vars->pressLinearSrc.getFortLowIndex();
  IntVector domHing = vars->pressLinearSrc.getFortHighIndex();
  for(int i=0;i<7;i++){
     ASSERTEQ(domLong,
	      vars->pressCoeff[i].getWindow()->getLowIndex());
     ASSERTEQ(domHing+IntVector(1,1,1),
	      vars->pressCoeff[i].getWindow()->getHighIndex());
  }
  ASSERTEQ(domLong, vars->pressNonlinearSrc.getWindow()->getLowIndex());
  ASSERTEQ(domHing+IntVector(1,1,1), vars->pressNonlinearSrc.getWindow()->getHighIndex());

  // Get the wall boundary and flow field codes
  int wall_celltypeval = wallCellType();
  int pressure_celltypeval = pressureCellType();
  int outlet_celltypeval = outletCellType();
  // ** WARNING ** Symmetry is hardcoded to -3
  // int symmetry_celltypeval = -3;

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  int neumann_bc = -1;
  int dirichlet_bc = 1;
  //fortran call
  fort_bcpress(domLo, domHi, idxLo, idxHi, constvars->pressure,
	       vars->pressCoeff[Arches::AP],
	       vars->pressCoeff[Arches::AE], vars->pressCoeff[Arches::AW],
	       vars->pressCoeff[Arches::AN], vars->pressCoeff[Arches::AS],
	       vars->pressCoeff[Arches::AT], vars->pressCoeff[Arches::AB],
	       vars->pressNonlinearSrc, vars->pressLinearSrc,
	       constvars->cellType, wall_celltypeval, wall_celltypeval,
	       neumann_bc,
	       xminus, xplus, yminus, yplus, zminus, zplus);

  fort_bcpress(domLo, domHi, idxLo, idxHi, constvars->pressure,
	       vars->pressCoeff[Arches::AP],
	       vars->pressCoeff[Arches::AE], vars->pressCoeff[Arches::AW],
	       vars->pressCoeff[Arches::AN], vars->pressCoeff[Arches::AS],
	       vars->pressCoeff[Arches::AT], vars->pressCoeff[Arches::AB],
	       vars->pressNonlinearSrc, vars->pressLinearSrc,
	       constvars->cellType, wall_celltypeval, outlet_celltypeval,
	       dirichlet_bc,
	       xminus, xplus, yminus, yplus, zminus, zplus);

  fort_bcpress(domLo, domHi, idxLo, idxHi, constvars->pressure,
	       vars->pressCoeff[Arches::AP],
	       vars->pressCoeff[Arches::AE], vars->pressCoeff[Arches::AW],
	       vars->pressCoeff[Arches::AN], vars->pressCoeff[Arches::AS],
	       vars->pressCoeff[Arches::AT], vars->pressCoeff[Arches::AB],
	       vars->pressNonlinearSrc, vars->pressLinearSrc,
	       constvars->cellType, wall_celltypeval, pressure_celltypeval,
	       dirichlet_bc,
	       xminus, xplus, yminus, yplus, zminus, zplus);
  
  for (int ii = 0; ii < d_numInlets; ii++) {
  fort_bcpress(domLo, domHi, idxLo, idxHi, constvars->pressure,
	       vars->pressCoeff[Arches::AP],
	       vars->pressCoeff[Arches::AE], vars->pressCoeff[Arches::AW],
	       vars->pressCoeff[Arches::AN], vars->pressCoeff[Arches::AS],
	       vars->pressCoeff[Arches::AT], vars->pressCoeff[Arches::AB],
	       vars->pressNonlinearSrc, vars->pressLinearSrc,
	       constvars->cellType, wall_celltypeval, 
	       d_flowInlets[ii].d_cellTypeID,
	       neumann_bc,
	       xminus, xplus, yminus, yplus, zminus, zplus);
  }
}

//****************************************************************************
// Actually compute the scalar bcs
//****************************************************************************
void 
BoundaryCondition::scalarBC(const ProcessorGroup*,
			    const Patch* patch,
			    int /*index*/,
			    CellInformation* cellinfo,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector domLo = constvars->density.getFortLowIndex();
  IntVector domHi = constvars->density.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  // Get the wall boundary and flow field codes
  int wall_celltypeval = wallCellType();
  int pressure_celltypeval = pressureCellType();
  int outlet_celltypeval = outletCellType();
  // ** WARNING ** Symmetry/sfield/outletfield/ffield hardcoded to -3,-4,-5, -6
  //               Fmixin hardcoded to 0
  int symmetry_celltypeval = -3;
  int sfield = -4;
  int ffield = -1;
  double fmixin = 0.0;
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  //fortran call
  fort_bcscalar(domLo, domHi, idxLo, idxHi, constvars->scalar,
		vars->scalarCoeff[Arches::AE], vars->scalarCoeff[Arches::AW],
		vars->scalarCoeff[Arches::AN], vars->scalarCoeff[Arches::AS],
		vars->scalarCoeff[Arches::AT], vars->scalarCoeff[Arches::AB],
		vars->scalarDiffusionCoeff[Arches::AE],
		vars->scalarDiffusionCoeff[Arches::AW],
		vars->scalarDiffusionCoeff[Arches::AN], 
		vars->scalarDiffusionCoeff[Arches::AS],
		vars->scalarDiffusionCoeff[Arches::AT],
		vars->scalarDiffusionCoeff[Arches::AB],
		vars->scalarNonlinearSrc, vars->scalarLinearSrc,
		constvars->density, fmixin, constvars->uVelocity,
		constvars->vVelocity, constvars->wVelocity,cellinfo->sew,
		cellinfo->sns, cellinfo->stb, constvars->cellType,
		wall_celltypeval, symmetry_celltypeval,
		d_flowInlets[0].d_cellTypeID, pressure_celltypeval, ffield,
		sfield, outlet_celltypeval,
		xminus, xplus, yminus, yplus, zminus, zplus);
}



//****************************************************************************
// Actually compute the scalar bcs
//****************************************************************************
void 
BoundaryCondition::enthalpyBC(const ProcessorGroup*,
			    const Patch* patch,
			    CellInformation* cellinfo,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector domLo = constvars->density.getFortLowIndex();
  IntVector domHi = constvars->density.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  // Get the wall boundary and flow field codes
  int wall_celltypeval = wallCellType();
  int pressure_celltypeval = pressureCellType();
  int outlet_celltypeval = outletCellType();
  // ** WARNING ** Symmetry/sfield/outletfield/ffield hardcoded to -3,-4,-5, -6
  //               Fmixin hardcoded to 0
  int symmetry_celltypeval = -3;
  int sfield = -4;
  int ffield = -1;
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  //fortran call
  fort_bcenthalpy(domLo, domHi, idxLo, idxHi, constvars->enthalpy,
		  vars->scalarCoeff[Arches::AE],
		  vars->scalarCoeff[Arches::AW],
		  vars->scalarCoeff[Arches::AN],
		  vars->scalarCoeff[Arches::AS],
		  vars->scalarCoeff[Arches::AT],
		  vars->scalarCoeff[Arches::AB],
		  vars->scalarNonlinearSrc, vars->scalarLinearSrc,
		  constvars->density, constvars->uVelocity,
		  constvars->vVelocity,
		  constvars->wVelocity, cellinfo->sew, cellinfo->sns,
		  cellinfo->stb, constvars->cellType, wall_celltypeval,
		  symmetry_celltypeval, d_flowInlets[0].d_cellTypeID,
		  pressure_celltypeval, ffield, sfield, outlet_celltypeval,
		  xminus, xplus, yminus, yplus, zminus, zplus);

}


void 
BoundaryCondition::enthalpyRadWallBC(const ProcessorGroup*,
				     const Patch* patch,
				     CellInformation*,
				     ArchesVariables* vars)
{

  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  // Get the wall boundary and flow field codes
  int wall_celltypeval = wallCellType();
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  //fortran call
  fort_enthalpyradwallbc(idxLo, idxHi, vars->qfluxe, vars->qfluxw,
			 vars->qfluxn, vars->qfluxs, vars->qfluxt,
			 vars->qfluxb, vars->temperature, vars->cellType,
			 wall_celltypeval, xminus, xplus, yminus, yplus,
			 zminus, zplus);
}






//****************************************************************************
// Documentation here
//****************************************************************************
void 
BoundaryCondition::addPressureGrad(const ProcessorGroup* ,
				   const Patch* patch ,
				   int index,
				   CellInformation*,
				   ArchesVariables* vars)
{
  // Get the patch and variable indices
  IntVector domLoU, domHiU;
  IntVector domLoUng, domHiUng;
  IntVector idxLoU, idxHiU;
  int ioff, joff, koff;
  switch(index) {
  case Arches::XDIR:
    domLoU = vars->pressGradUSu.getFortLowIndex();
    domHiU = vars->pressGradUSu.getFortHighIndex();
    domLoUng = vars->uVelNonlinearSrc.getFortLowIndex();
    domHiUng = vars->uVelNonlinearSrc.getFortHighIndex();
    idxLoU = patch->getSFCXFORTLowIndex();
    idxHiU = patch->getSFCXFORTHighIndex();

    ioff = 1;
    joff = 0;
    koff = 0;
    if (d_MAlab) {
      for (int colZ = idxLoU.z(); colZ < idxHiU.z(); colZ ++) {
	for (int colY = idxLoU.y(); colY < idxHiU.y(); colY ++) {
	  for (int colX = idxLoU.x(); colX < idxHiU.x(); colX ++) {
	  // Store current cell
	    IntVector currCell(colX, colY, colZ);
	    IntVector prevCell(colX-1, colY, colZ);
	    vars->pressGradUSu[currCell] *= (vars->voidFraction[currCell]+
					     vars->voidFraction[prevCell])/2;
	  }
	}
      }
    }
    fort_addpressuregrad(idxLoU, idxHiU, vars->pressGradUSu,
			 vars->uVelNonlinearSrc, vars->cellType, d_mmWallID,
			 ioff, joff, koff);
    break;
  case Arches::YDIR:
    domLoU = vars->pressGradVSu.getFortLowIndex();
    domHiU = vars->pressGradVSu.getFortHighIndex();
    domLoUng = vars->vVelNonlinearSrc.getFortLowIndex();
    domHiUng = vars->vVelNonlinearSrc.getFortHighIndex();
    idxLoU = patch->getSFCYFORTLowIndex();
    idxHiU = patch->getSFCYFORTHighIndex();

    ioff = 0;
    joff = 1;
    koff = 0;
    if (d_MAlab) {
      for (int colZ = idxLoU.z(); colZ < idxHiU.z(); colZ ++) {
	for (int colY = idxLoU.y(); colY < idxHiU.y(); colY ++) {
	  for (int colX = idxLoU.x(); colX < idxHiU.x(); colX ++) {
	  // Store current cell
	    IntVector currCell(colX, colY, colZ);
	    IntVector prevCell(colX, colY-1, colZ);
	    vars->pressGradVSu[currCell] *= (vars->voidFraction[currCell]+
					     vars->voidFraction[prevCell])/2;
	  }
	}
      }
    }
    fort_addpressuregrad(idxLoU, idxHiU, vars->pressGradVSu,
			 vars->vVelNonlinearSrc, vars->cellType, d_mmWallID,
			 ioff, joff, koff);
    break;
  case Arches::ZDIR:
    domLoU = vars->pressGradWSu.getFortLowIndex();
    domHiU = vars->pressGradWSu.getFortHighIndex();
    domLoUng = vars->wVelNonlinearSrc.getFortLowIndex();
    domHiUng = vars->wVelNonlinearSrc.getFortHighIndex();
    idxLoU = patch->getSFCZFORTLowIndex();
    idxHiU = patch->getSFCZFORTHighIndex();

    ioff = 0;
    joff = 0;
    koff = 1;
    if (d_MAlab) {
      for (int colZ = idxLoU.z(); colZ < idxHiU.z(); colZ ++) {
	for (int colY = idxLoU.y(); colY < idxHiU.y(); colY ++) {
	  for (int colX = idxLoU.x(); colX < idxHiU.x(); colX ++) {
	  // Store current cell
	    IntVector currCell(colX, colY, colZ);
	    IntVector prevCell(colX, colY, colZ-1);
	    vars->pressGradWSu[currCell] *= (vars->voidFraction[currCell]+
					     vars->voidFraction[prevCell])/2;
	  }
	}
      }
    }
    fort_addpressuregrad(idxLoU, idxHiU, vars->pressGradWSu,
			 vars->wVelNonlinearSrc, vars->cellType, d_mmWallID,
			 ioff, joff, koff);
    break;
  default:
    throw InvalidValue("Invalid index in BoundaryCondition::addPressGrad");
  }
}

void
BoundaryCondition::intrusionTemperatureBC(const ProcessorGroup*,
					  const Patch* patch,
					  constCCVariable<int>& cellType,
					  CCVariable<double>& temperature)
{
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	IntVector currCell = IntVector(colX, colY, colZ);	  
	if (cellType[currCell]==d_intrusionBC->d_cellTypeID)
	  temperature[currCell] = d_intrusionBC->d_temperature;
      }
    }
  }
}

void
BoundaryCondition::mmWallTemperatureBC(const ProcessorGroup*,
				       const Patch* patch,
				       constCCVariable<int>& cellType,
				       constCCVariable<double> solidTemp,
				       CCVariable<double>& temperature,
				       bool d_energyEx)
{
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {

	IntVector currCell = IntVector(colX, colY, colZ);

	if (cellType[currCell]==d_mmWallID) {

	  if (d_energyEx) {
	    if (d_fixTemp) 
	      temperature[currCell] = 298.0;
	    else
	      temperature[currCell] = solidTemp[currCell];
	  }
	  else
	    temperature[currCell] = 298.0;

	}
      }
    }
  }
}


// compute intrusion wall bc
void 
BoundaryCondition::intrusionVelocityBC(const ProcessorGroup*,
				const Patch* patch,
				int index, CellInformation*,
				ArchesVariables* vars,
		      	        ArchesConstVariables* constvars)
{
    // Call the fortran routines
  switch(index) {
  case 1:
    intrusionuVelocityBC(patch,
		  vars, constvars);
    break;
  case 2:
    intrusionvVelocityBC(patch,
		  vars, constvars);
    break;
  case 3:
    intrusionwVelocityBC(patch,
		  vars, constvars);
    break;
  default:
    cerr << "Invalid Index value" << endl;
    break;

  }
}

void 
BoundaryCondition::intrusionuVelocityBC(const Patch* patch,
					ArchesVariables* vars,
		      	        	ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch and the variables
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  int ioff = 1;
  int joff = 0;
  int koff = 0;

  fort_mmbcvelocity(idxLoU, idxHiU,
		    vars->uVelocityCoeff[Arches::AE],
		    vars->uVelocityCoeff[Arches::AW],
		    vars->uVelocityCoeff[Arches::AN],
		    vars->uVelocityCoeff[Arches::AS],
		    vars->uVelocityCoeff[Arches::AT],
		    vars->uVelocityCoeff[Arches::AB],
		    vars->uVelNonlinearSrc, vars->uVelLinearSrc,
		    constvars->cellType, d_intrusionBC->d_cellTypeID,
		    ioff, joff, koff);
}

void 
BoundaryCondition::intrusionvVelocityBC(const Patch* patch,
				 	ArchesVariables* vars,
		      	        	ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch and the variables
  IntVector idxLoU = patch->getSFCYFORTLowIndex();
  IntVector idxHiU = patch->getSFCYFORTHighIndex();
  
  int ioff = 0;
  int joff = 1;
  int koff = 0;

  fort_mmbcvelocity(idxLoU, idxHiU,
		    vars->vVelocityCoeff[Arches::AN],
		    vars->vVelocityCoeff[Arches::AS],
		    vars->vVelocityCoeff[Arches::AT],
		    vars->vVelocityCoeff[Arches::AB],
		    vars->vVelocityCoeff[Arches::AE],
		    vars->vVelocityCoeff[Arches::AW],
		    vars->vVelNonlinearSrc, vars->vVelLinearSrc,
		    constvars->cellType,d_intrusionBC->d_cellTypeID,
		    ioff, joff, koff);
}

void 
BoundaryCondition::intrusionwVelocityBC( const Patch* patch,
				  	ArchesVariables* vars,
		      	        	ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch and the variables
  IntVector idxLoU = patch->getSFCZFORTLowIndex();
  IntVector idxHiU = patch->getSFCZFORTHighIndex();

  int ioff = 0;
  int joff = 0;
  int koff = 1;

  fort_mmbcvelocity(idxLoU, idxHiU,
		    vars->wVelocityCoeff[Arches::AT],
		    vars->wVelocityCoeff[Arches::AB],
		    vars->wVelocityCoeff[Arches::AE],
		    vars->wVelocityCoeff[Arches::AW],
		    vars->wVelocityCoeff[Arches::AN],
		    vars->wVelocityCoeff[Arches::AS],
		    vars->wVelNonlinearSrc, vars->wVelLinearSrc,
		    constvars->cellType, d_intrusionBC->d_cellTypeID,
		    ioff, joff, koff);
}

void 
BoundaryCondition::intrusionMomExchangeBC(const ProcessorGroup*,
					  const Patch* patch,
					  int index, CellInformation* cellinfo,
					  ArchesVariables* vars,
		      		  	  ArchesConstVariables* constvars)
{
  // Call the fortran routines
  switch(index) {
  case 1:
    intrusionuVelMomExBC(patch, cellinfo, vars, constvars);
    break;
  case 2:
    intrusionvVelMomExBC(patch, cellinfo, vars, constvars);
    break;
  case 3:
    intrusionwVelMomExBC(patch, cellinfo, vars, constvars);
    break;
  default:
    cerr << "Invalid Index value" << endl;
    break;

  }
}

void 
BoundaryCondition::intrusionuVelMomExBC(const Patch* patch,
					CellInformation* cellinfo,
					ArchesVariables* vars,
		      			ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch and the variables
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  int ioff = 1;
  int joff = 0;
  int koff = 0;
  double Viscos = d_physicalConsts->getMolecularViscosity();
  fort_mmbcvelocity_momex(idxLoU, idxHiU,
			  vars->uVelocityCoeff[Arches::AN],
			  vars->uVelocityCoeff[Arches::AS],
			  vars->uVelocityCoeff[Arches::AT],
			  vars->uVelocityCoeff[Arches::AB],
			  vars->uVelLinearSrc,
			  cellinfo->sewu, cellinfo->sns, cellinfo->stb,
			  cellinfo->yy, cellinfo->yv, cellinfo->zz,
			  cellinfo->zw, Viscos, constvars->cellType, 
			  d_intrusionBC->d_cellTypeID, ioff, joff, koff);
}


void 
BoundaryCondition::intrusionvVelMomExBC(const Patch* patch,
					CellInformation* cellinfo,
					ArchesVariables* vars,
		      			ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch and the variables
  IntVector idxLoU = patch->getSFCYFORTLowIndex();
  IntVector idxHiU = patch->getSFCYFORTHighIndex();
  int ioff = 0;
  int joff = 1;
  int koff = 0;
  double Viscos = d_physicalConsts->getMolecularViscosity();
  fort_mmbcvelocity_momex(idxLoU, idxHiU,
			  vars->vVelocityCoeff[Arches::AT],
			  vars->vVelocityCoeff[Arches::AB],
			  vars->vVelocityCoeff[Arches::AE],
			  vars->vVelocityCoeff[Arches::AW],
			  vars->vVelLinearSrc,
			  cellinfo->snsv, cellinfo->stb, cellinfo->sew,
			  cellinfo->zz, cellinfo->zw, cellinfo->xx,
			  cellinfo->xu, Viscos, constvars->cellType, 
			  d_intrusionBC->d_cellTypeID, ioff, joff, koff);
}

void 
BoundaryCondition::intrusionwVelMomExBC(const Patch* patch,
					CellInformation* cellinfo,
					ArchesVariables* vars,
		      			ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch and the variables
  IntVector idxLoU = patch->getSFCZFORTLowIndex();
  IntVector idxHiU = patch->getSFCZFORTHighIndex();
  int ioff = 0;
  int joff = 0;
  int koff = 1;
  double Viscos = d_physicalConsts->getMolecularViscosity();
  fort_mmbcvelocity_momex(idxLoU, idxHiU,
			  vars->wVelocityCoeff[Arches::AE],
			  vars->wVelocityCoeff[Arches::AW],
			  vars->wVelocityCoeff[Arches::AN],
			  vars->wVelocityCoeff[Arches::AS],
			  vars->wVelLinearSrc,
			  cellinfo->stbw, cellinfo->sew, cellinfo->sns,
			  cellinfo->xx, cellinfo->xu, cellinfo->yy,
			  cellinfo->yv, Viscos, constvars->cellType, 
			  d_intrusionBC->d_cellTypeID, ioff, joff, koff);
}

void 
BoundaryCondition::intrusionEnergyExBC(const ProcessorGroup*,
				       const Patch* patch,
				       CellInformation* cellinfo,
				       ArchesVariables* vars,
				       ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  double Viscos = d_physicalConsts->getMolecularViscosity();
  fort_mmbcenthalpy_energyex(idxLo, idxHi,
			     vars->scalarNonlinearSrc,
			     constvars->temperature,
			     constvars->cp,
			     cellinfo->sew, cellinfo->sns, cellinfo->stb,
			     cellinfo->xx, cellinfo->xu,
			     cellinfo->yy, cellinfo->yv,
			     cellinfo->zz, cellinfo->zw,
			     Viscos,
			     constvars->cellType, 
			     d_intrusionBC->d_cellTypeID);
}


void 
BoundaryCondition::intrusionPressureBC(const ProcessorGroup*,
				       const Patch* patch,
				       CellInformation*,
				       ArchesVariables* vars,
			       	       ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();

  IntVector idxHi = patch->getCellFORTHighIndex();
  IntVector domLong = vars->pressLinearSrc.getFortLowIndex();
  IntVector domHing = vars->pressLinearSrc.getFortHighIndex();
  for(int i=0;i<7;i++){
     ASSERTEQ(domLong,
	      vars->pressCoeff[i].getWindow()->getLowIndex());
     ASSERTEQ(domHing+IntVector(1,1,1),
	      vars->pressCoeff[i].getWindow()->getHighIndex());
  }
  ASSERTEQ(domLong, vars->pressNonlinearSrc.getWindow()->getLowIndex());
  ASSERTEQ(domHing+IntVector(1,1,1), vars->pressNonlinearSrc.getWindow()->getHighIndex());

  //fortran call
  fort_mmwallbc(idxLo, idxHi,
		vars->pressCoeff[Arches::AE], vars->pressCoeff[Arches::AW],
		vars->pressCoeff[Arches::AN], vars->pressCoeff[Arches::AS],
		vars->pressCoeff[Arches::AT], vars->pressCoeff[Arches::AB],
		vars->pressNonlinearSrc, vars->pressLinearSrc,
		constvars->cellType, d_intrusionBC->d_cellTypeID);
}
// applies multimaterial bc's for scalars and pressure
void
BoundaryCondition::intrusionScalarBC( const ProcessorGroup*,
				      const Patch* patch,
				      CellInformation*,
				      ArchesVariables* vars,
			       	      ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  //fortran call
  fort_mmwallbc(idxLo, idxHi,
		vars->scalarCoeff[Arches::AE], vars->scalarCoeff[Arches::AW],
		vars->scalarCoeff[Arches::AN], vars->scalarCoeff[Arches::AS],
		vars->scalarCoeff[Arches::AT], vars->scalarCoeff[Arches::AB],
		vars->scalarNonlinearSrc, vars->scalarLinearSrc,
		constvars->cellType, d_intrusionBC->d_cellTypeID);
}

void
BoundaryCondition::intrusionEnthalpyBC( const ProcessorGroup*,
					const Patch* patch, double delta_t,
					CellInformation* cellinfo,
					ArchesVariables* vars,
					ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  //fortran call
  fort_mmwallbc_trans(idxLo, idxHi,
		      vars->scalarCoeff[Arches::AE],
		      vars->scalarCoeff[Arches::AW],
		      vars->scalarCoeff[Arches::AN],
		      vars->scalarCoeff[Arches::AS],
		      vars->scalarCoeff[Arches::AT],
		      vars->scalarCoeff[Arches::AB],
		      vars->scalarNonlinearSrc,
		      vars->scalarLinearSrc,
		      constvars->old_enthalpy,
		      constvars->old_density,
		      constvars->cellType, d_intrusionBC->d_cellTypeID,
		      cellinfo->sew, cellinfo->sns, cellinfo->stb, 
		      delta_t);
}


      // compute multimaterial wall bc
void 
BoundaryCondition::mmvelocityBC(const ProcessorGroup*,
				const Patch* patch,
				int index, CellInformation*,
				ArchesVariables* vars,
		       		ArchesConstVariables* constvars)
{
    // Call the fortran routines
  switch(index) {
  case 1:
    mmuVelocityBC(patch,
		  vars, constvars);
    break;
  case 2:
    mmvVelocityBC(patch,
		  vars, constvars);
    break;
  case 3:
    mmwVelocityBC(patch,
		  vars, constvars);
    break;
  default:
    cerr << "Invalid Index value" << endl;
    break;

  // add kumar's mmvelbc
  }
}

void 
BoundaryCondition::mmuVelocityBC(const Patch* patch,
				 ArchesVariables* vars,
		       		 ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch and the variables
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  int ioff = 1;
  int joff = 0;
  int koff = 0;

  fort_mmbcvelocity(idxLoU, idxHiU,
		    vars->uVelocityCoeff[Arches::AE],
		    vars->uVelocityCoeff[Arches::AW],
		    vars->uVelocityCoeff[Arches::AN],
		    vars->uVelocityCoeff[Arches::AS],
		    vars->uVelocityCoeff[Arches::AT],
		    vars->uVelocityCoeff[Arches::AB],
		    vars->uVelNonlinearSrc, vars->uVelLinearSrc,
		    constvars->cellType, d_mmWallID, ioff, joff, koff);
}

void 
BoundaryCondition::mmvVelocityBC(const Patch* patch,
				 ArchesVariables* vars,
		       		 ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch and the variables
  IntVector idxLoU = patch->getSFCYFORTLowIndex();
  IntVector idxHiU = patch->getSFCYFORTHighIndex();
  
  int ioff = 0;
  int joff = 1;
  int koff = 0;

  fort_mmbcvelocity(idxLoU, idxHiU,
		    vars->vVelocityCoeff[Arches::AN],
		    vars->vVelocityCoeff[Arches::AS],
		    vars->vVelocityCoeff[Arches::AT],
		    vars->vVelocityCoeff[Arches::AB],
		    vars->vVelocityCoeff[Arches::AE],
		    vars->vVelocityCoeff[Arches::AW],
		    vars->vVelNonlinearSrc, vars->vVelLinearSrc,
		    constvars->cellType, d_mmWallID, ioff, joff, koff);
}

void 
BoundaryCondition::mmwVelocityBC( const Patch* patch,
				 ArchesVariables* vars,
		       		 ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch and the variables
  IntVector idxLoU = patch->getSFCZFORTLowIndex();
  IntVector idxHiU = patch->getSFCZFORTHighIndex();

  int ioff = 0;
  int joff = 0;
  int koff = 1;

  fort_mmbcvelocity(idxLoU, idxHiU,
		    vars->wVelocityCoeff[Arches::AT],
		    vars->wVelocityCoeff[Arches::AB],
		    vars->wVelocityCoeff[Arches::AE],
		    vars->wVelocityCoeff[Arches::AW],
		    vars->wVelocityCoeff[Arches::AN],
		    vars->wVelocityCoeff[Arches::AS],
		    vars->wVelNonlinearSrc, vars->wVelLinearSrc,
		    constvars->cellType, d_mmWallID, ioff, joff, koff);
}

void 
BoundaryCondition::mmpressureBC(const ProcessorGroup*,
				const Patch* patch,
				CellInformation*,
				ArchesVariables* vars,
				ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  IntVector domLong = vars->pressLinearSrc.getFortLowIndex();
  IntVector domHing = vars->pressLinearSrc.getFortHighIndex();
  for(int i=0;i<7;i++){
     ASSERTEQ(domLong,
	      vars->pressCoeff[i].getWindow()->getLowIndex());
     ASSERTEQ(domHing+IntVector(1,1,1),
	      vars->pressCoeff[i].getWindow()->getHighIndex());
  }
  ASSERTEQ(domLong, vars->pressNonlinearSrc.getWindow()->getLowIndex());
  ASSERTEQ(domHing+IntVector(1,1,1), vars->pressNonlinearSrc.getWindow()->getHighIndex());

  //fortran call
  fort_mmwallbc(idxLo, idxHi,
		vars->pressCoeff[Arches::AE], vars->pressCoeff[Arches::AW],
		vars->pressCoeff[Arches::AN], vars->pressCoeff[Arches::AS],
		vars->pressCoeff[Arches::AT], vars->pressCoeff[Arches::AB],
		vars->pressNonlinearSrc, vars->pressLinearSrc,
		constvars->cellType, d_mmWallID);
}


// applies multimaterial bc's for scalars and pressure
void
BoundaryCondition::mmscalarWallBC( const ProcessorGroup*,
				   const Patch* patch,
				   CellInformation*,
				   ArchesVariables* vars,
				   ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  //fortran call
  fort_mmscalarwallbc(idxLo, idxHi,
		      vars->scalarCoeff[Arches::AE], vars->scalarCoeff[Arches::AW],
		      vars->scalarCoeff[Arches::AN], vars->scalarCoeff[Arches::AS],
		      vars->scalarCoeff[Arches::AT], vars->scalarCoeff[Arches::AB],
		      vars->scalarNonlinearSrc, vars->scalarLinearSrc,
		      constvars->cellType, d_mmWallID);
}


// applies multimaterial bc's for enthalpy
void
BoundaryCondition::mmEnthalpyWallBC( const ProcessorGroup*,
				   const Patch* patch,
				   CellInformation*,
				   ArchesVariables* vars,
				   ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  //fortran call
  fort_mmenthalpywallbc(idxLo, idxHi,
		vars->scalarCoeff[Arches::AE], vars->scalarCoeff[Arches::AW],
		vars->scalarCoeff[Arches::AN], vars->scalarCoeff[Arches::AS],
		vars->scalarCoeff[Arches::AT], vars->scalarCoeff[Arches::AB],
		vars->scalarNonlinearSrc, vars->scalarLinearSrc,
		constvars->cellType, d_mmWallID);
}


//****************************************************************************
// constructor for BoundaryCondition::WallBdry
//****************************************************************************
BoundaryCondition::WallBdry::WallBdry(int cellID):
  d_cellTypeID(cellID)
{
}

//****************************************************************************
// destructor for BoundaryCondition::WallBdry
//****************************************************************************
BoundaryCondition::WallBdry::~WallBdry()
{
  for (std::vector<GeometryPiece*>::const_iterator g = d_geomPiece.begin();
       g != d_geomPiece.end(); ++g)
    delete *g;
}


//****************************************************************************
// Problem Setup for BoundaryCondition::WallBdry
//****************************************************************************
void 
BoundaryCondition::WallBdry::problemSetup(ProblemSpecP& params)
{
  ProblemSpecP geomObjPS = params->findBlock("geom_object");
  GeometryPieceFactory::create(geomObjPS, d_geomPiece);
}

//****************************************************************************
// constructor for BoundaryCondition::WallBdry
//****************************************************************************
BoundaryCondition::IntrusionBdry::IntrusionBdry(int cellID):
  d_cellTypeID(cellID)
{
}

//****************************************************************************
// destructor for BoundaryCondition::WallBdry
//****************************************************************************
BoundaryCondition::IntrusionBdry::~IntrusionBdry()
{
  for (std::vector<GeometryPiece*>::const_iterator g = d_geomPiece.begin();
       g != d_geomPiece.end(); ++g)
    delete *g;
}

//****************************************************************************
// Problem Setup for BoundaryCondition::WallBdry
//****************************************************************************
void 
BoundaryCondition::IntrusionBdry::problemSetup(ProblemSpecP& params)
{
  if (params->findBlock("temperature"))
    params->require("temperature", d_temperature);
  else
    d_temperature = 300;
  ProblemSpecP geomObjPS = params->findBlock("geom_object");
  GeometryPieceFactory::create(geomObjPS, d_geomPiece);
}




//****************************************************************************
// constructor for BoundaryCondition::FlowInlet
//****************************************************************************
BoundaryCondition::FlowInlet::FlowInlet(int /*numMix*/, int cellID):
  d_cellTypeID(cellID)
{
  turb_lengthScale = 0.0;
  flowRate = 0.0;
  inletVel = 0.0;
  fcr = 0.0;
  // add cellId to distinguish different inlets
  d_area_label = VarLabel::create("flowarea"+cellID,
   ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_flowRate_label = VarLabel::create("flowRate"+cellID,
   ReductionVariable<double, Reductions::Min<double> >::getTypeDescription()); 
}

BoundaryCondition::FlowInlet::FlowInlet():
  d_cellTypeID(0), d_area_label(0), d_flowRate_label(0)
{
  turb_lengthScale = 0.0;
  flowRate = 0.0;
  inletVel = 0.0;
  fcr = 0.0;
}

BoundaryCondition::FlowInlet::FlowInlet(const FlowInlet& copy) :
  d_cellTypeID (copy.d_cellTypeID),
  flowRate(copy.flowRate),
  inletVel(copy.inletVel),
  fcr(copy.fcr),
  streamMixturefraction(copy.streamMixturefraction),
  turb_lengthScale(copy.turb_lengthScale),
  calcStream(copy.calcStream),
  d_area_label(copy.d_area_label),
  d_flowRate_label(copy.d_flowRate_label)
{
  for (vector<GeometryPiece*>::const_iterator it = copy.d_geomPiece.begin();
       it != copy.d_geomPiece.end(); ++it)
    d_geomPiece.push_back((*it)->clone());

  d_area_label->addReference();
  d_flowRate_label->addReference();
}

BoundaryCondition::FlowInlet& BoundaryCondition::FlowInlet::operator=(const FlowInlet& copy)
{
  // remove reference from the old label
  VarLabel::destroy(d_area_label);
  d_area_label = copy.d_area_label;
  d_area_label->addReference();
  VarLabel::destroy(d_flowRate_label);
  d_flowRate_label = copy.d_flowRate_label;
  d_flowRate_label->addReference();

  d_cellTypeID = copy.d_cellTypeID;
  flowRate = copy.flowRate;
  inletVel = copy.inletVel;
  fcr = copy.fcr;
  streamMixturefraction = copy.streamMixturefraction;
  turb_lengthScale = copy.turb_lengthScale;
  calcStream = copy.calcStream;
  d_geomPiece = copy.d_geomPiece;

  return *this;
}


BoundaryCondition::FlowInlet::~FlowInlet()
{
  VarLabel::destroy(d_area_label);
  VarLabel::destroy(d_flowRate_label);
  for (std::vector<GeometryPiece*>::const_iterator g = d_geomPiece.begin();
       g != d_geomPiece.end(); ++g)
    delete *g;
}

//****************************************************************************
// Problem Setup for BoundaryCondition::FlowInlet
//****************************************************************************
void 
BoundaryCondition::FlowInlet::problemSetup(ProblemSpecP& params)
{
  params->getWithDefault("Flow_rate", flowRate,0.0);
  params->getWithDefault("InletVelocity", inletVel,0.0);
  params->require("TurblengthScale", turb_lengthScale);
  // This parameter only needs to be set for fuel inlets for which
  // mixture fraction > 0, if there is an air inlet, and air has some CO2,
  // this air CO2 will be counted in the balance automatically
  params->getWithDefault("CarbonMassFractionInFuel", fcr, 0.0);
  // check to see if this will work
  ProblemSpecP geomObjPS = params->findBlock("geom_object");
  GeometryPieceFactory::create(geomObjPS, d_geomPiece);
  // loop thru all the inlet geometry objects
  //for (ProblemSpecP geom_obj_ps = params->findBlock("geom_object");
  //     geom_obj_ps != 0; 
  //     geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
  //  vector<GeometryPiece*> pieces;
  //  GeometryPieceFactory::create(geom_obj_ps, pieces);
  //  if(pieces.size() == 0){
  //    throw ParameterNotFound("No piece specified in geom_object");
  //  } else if(pieces.size() > 1){
  //    d_geomPiece = scinew UnionGeometryPiece(pieces);
  //  } else {
  //    d_geomPiece = pieces[0];
  //  }
  //}

  double mixfrac;
  for (ProblemSpecP mixfrac_db = params->findBlock("MixtureFraction");
       mixfrac_db != 0; 
       mixfrac_db = mixfrac_db->findNextBlock("MixtureFraction")) {
    mixfrac_db->require("Mixfrac", mixfrac);
    streamMixturefraction.d_mixVars.push_back(mixfrac);
  }
  double mixfracvar;
  for (ProblemSpecP mixfracvar_db = params->findBlock("MixtureFractionVar");
       mixfracvar_db != 0; 
       mixfracvar_db = mixfracvar_db->findNextBlock("MixtureFractionVar")) {
    mixfracvar_db->require("Mixfracvar", mixfracvar);
    streamMixturefraction.d_mixVarVariance.push_back(mixfracvar);
  }
 
}


//****************************************************************************
// constructor for BoundaryCondition::PressureInlet
//****************************************************************************
BoundaryCondition::PressureInlet::PressureInlet(int /*numMix*/, int cellID):
  d_cellTypeID(cellID)
{
  //  streamMixturefraction.setsize(numMix-1);
  turb_lengthScale = 0.0;
  refPressure = 0.0;
}

//****************************************************************************
// destructor for BoundaryCondition::PressureInlet
//****************************************************************************
BoundaryCondition::PressureInlet::~PressureInlet()
{
  for (std::vector<GeometryPiece*>::const_iterator g = d_geomPiece.begin();
       g != d_geomPiece.end(); ++g)
    delete *g;
}

//****************************************************************************
// Problem Setup for BoundaryCondition::PressureInlet
//****************************************************************************
void 
BoundaryCondition::PressureInlet::problemSetup(ProblemSpecP& params)
{
  params->require("RefPressure", refPressure);
  params->require("TurblengthScale", turb_lengthScale);
  ProblemSpecP geomObjPS = params->findBlock("geom_object");
  GeometryPieceFactory::create(geomObjPS, d_geomPiece);
  // loop thru all the pressure inlet geometry objects
  //for (ProblemSpecP geom_obj_ps = params->findBlock("geom_object");
  //     geom_obj_ps != 0; 
  //     geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
  //  vector<GeometryPiece*> pieces;
  //  GeometryPieceFactory::create(geom_obj_ps, pieces);
  //  if(pieces.size() == 0){
  //    throw ParameterNotFound("No piece specified in geom_object");
  //  } else if(pieces.size() > 1){
  //    d_geomPiece = scinew UnionGeometryPiece(pieces);
  //  } else {
  //    d_geomPiece = pieces[0];
  //  }
  //}
  double mixfrac;
  for (ProblemSpecP mixfrac_db = params->findBlock("MixtureFraction");
       mixfrac_db != 0; 
       mixfrac_db = mixfrac_db->findNextBlock("MixtureFraction")) {
    mixfrac_db->require("Mixfrac", mixfrac);
    streamMixturefraction.d_mixVars.push_back(mixfrac);
  }
  double mixfracvar;
  for (ProblemSpecP mixfracvar_db = params->findBlock("MixtureFractionVar");
       mixfracvar_db != 0; 
       mixfracvar_db = mixfracvar_db->findNextBlock("MixtureFractionVar")) {
    mixfracvar_db->require("Mixfracvar", mixfracvar);
    streamMixturefraction.d_mixVarVariance.push_back(mixfracvar);
  }
}

//****************************************************************************
// constructor for BoundaryCondition::FlowOutlet
//****************************************************************************
BoundaryCondition::FlowOutlet::FlowOutlet(int /*numMix*/, int cellID):
  d_cellTypeID(cellID)
{
  //  streamMixturefraction.setsize(numMix-1);
  turb_lengthScale = 0.0;
}

//****************************************************************************
// destructor for BoundaryCondition::FlowOutlet
//****************************************************************************
BoundaryCondition::FlowOutlet::~FlowOutlet()
{
  for (std::vector<GeometryPiece*>::const_iterator g = d_geomPiece.begin();
       g != d_geomPiece.end(); ++g)
    delete *g;
}

//****************************************************************************
// Problem Setup for BoundaryCondition::FlowOutlet
//****************************************************************************
void 
BoundaryCondition::FlowOutlet::problemSetup(ProblemSpecP& params)
{
  params->require("TurblengthScale", turb_lengthScale);
  ProblemSpecP geomObjPS = params->findBlock("geom_object");
  GeometryPieceFactory::create(geomObjPS, d_geomPiece);
  // loop thru all the inlet geometry objects
  //for (ProblemSpecP geom_obj_ps = params->findBlock("geom_object");
  //     geom_obj_ps != 0; 
  //     geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
  //  vector<GeometryPiece*> pieces;
  //  GeometryPieceFactory::create(geom_obj_ps, pieces);
  //  if(pieces.size() == 0){
  //    throw ParameterNotFound("No piece specified in geom_object");
  //  } else if(pieces.size() > 1){
  //    d_geomPiece = scinew UnionGeometryPiece(pieces);
  //  } else {
  //    d_geomPiece = pieces[0];
  //  }
  //}
  double mixfrac;
  for (ProblemSpecP mixfrac_db = params->findBlock("MixtureFraction");
       mixfrac_db != 0; 
       mixfrac_db = mixfrac_db->findNextBlock("MixtureFraction")) {
    mixfrac_db->require("Mixfrac", mixfrac);
    streamMixturefraction.d_mixVars.push_back(mixfrac);
  }
  double mixfracvar;
  for (ProblemSpecP mixfracvar_db = params->findBlock("MixtureFractionVar");
       mixfracvar_db != 0; 
       mixfracvar_db = mixfracvar_db->findNextBlock("MixtureFractionVar")) {
    mixfracvar_db->require("Mixfracvar", mixfracvar);
    streamMixturefraction.d_mixVarVariance.push_back(mixfracvar);
  }
}

void
BoundaryCondition::calculateIntrusionVel(const ProcessorGroup* ,
					 const Patch* patch,
					 int index,
					 CellInformation* ,
					 ArchesVariables* vars,
		      	       		 ArchesConstVariables* constvars)
{
  
  int ioff, joff, koff;
  IntVector idxLoU;
  IntVector idxHiU;

  switch(index) {

  case Arches::XDIR:

    idxLoU = patch->getSFCXFORTLowIndex();
    idxHiU = patch->getSFCXFORTHighIndex();
    ioff = 1; joff = 0; koff = 0;

    fort_intrusion_computevel(vars->uVelRhoHat,
			      ioff, joff, koff,
			      constvars->cellType,
			      idxLoU, idxHiU,
			      d_intrusionBC->d_cellTypeID);

    break;

  case Arches::YDIR:

    idxLoU = patch->getSFCYFORTLowIndex();
    idxHiU = patch->getSFCYFORTHighIndex();
    ioff = 0; joff = 1; koff = 0;

    fort_intrusion_computevel(vars->vVelRhoHat,
			      ioff, joff, koff,
			      constvars->cellType,
			      idxLoU, idxHiU,
			      d_intrusionBC->d_cellTypeID);

    break;

  case Arches::ZDIR:

    idxLoU = patch->getSFCZFORTLowIndex();
    idxHiU = patch->getSFCZFORTHighIndex();

    ioff = 0; joff = 0; koff = 1;
    
    fort_intrusion_computevel(vars->wVelRhoHat,
		       ioff, joff, koff,
		       constvars->cellType,
		       idxLoU, idxHiU,
		       d_intrusionBC->d_cellTypeID);

    break;

  default:
    
    throw InvalidValue("Invalid index in Source::calcVelSrc");
    
  }
  
}

void
BoundaryCondition::calculateVelocityPred_mm(const ProcessorGroup* ,
					    const Patch* patch,
					    double delta_t,
					    int index,
					    CellInformation* cellinfo,
					    ArchesVariables* vars,
					    ArchesConstVariables* constvars)
{
  
  int ioff, joff, koff;
  IntVector idxLoU;
  IntVector idxHiU;

  switch(index) {

  case Arches::XDIR:

    idxLoU = patch->getSFCXFORTLowIndex();
    idxHiU = patch->getSFCXFORTHighIndex();
    ioff = 1; joff = 0; koff = 0;

    fort_mm_computevel(
		      vars->uVelRhoHat,
		      constvars->pressure,
		      constvars->density,
		      constvars->voidFraction,
		      cellinfo->dxpw,
		      delta_t, 
		      ioff, joff, koff,
		      constvars->cellType,
		      idxLoU, idxHiU,
		      d_mmWallID);

    break;

  case Arches::YDIR:

    idxLoU = patch->getSFCYFORTLowIndex();
    idxHiU = patch->getSFCYFORTHighIndex();
    ioff = 0; joff = 1; koff = 0;

    fort_mm_computevel(
		      vars->vVelRhoHat,
		      constvars->pressure,
		      constvars->density,
		      constvars->voidFraction,
		      cellinfo->dyps,
		      delta_t, 
		      ioff, joff, koff,
		      constvars->cellType,
		      idxLoU, idxHiU,
		      d_mmWallID);

    break;

  case Arches::ZDIR:

    idxLoU = patch->getSFCZFORTLowIndex();
    idxHiU = patch->getSFCZFORTHighIndex();

    ioff = 0; joff = 0; koff = 1;

    fort_mm_computevel(
		      vars->wVelRhoHat,
		      constvars->pressure,
		      constvars->density,
		      constvars->voidFraction,
		      cellinfo->dzpb,
		      delta_t, 
		      ioff, joff, koff,
		      constvars->cellType,
		      idxLoU, idxHiU,
		      d_mmWallID);

    break;

  default:

    throw InvalidValue("Invalid index in Source::calcVelSrc");

  }

}

//****************************************************************************
// Set the boundary conditions for convection fluxes
//****************************************************************************
void 
BoundaryCondition::setFluxBC(const ProcessorGroup*,
			      const Patch* patch,
			      int index,
			      ArchesVariables* vars) 
{
  int wall_celltypeval = d_wallBdry->d_cellTypeID;

  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
  if (xplus) idxHi = idxHi + IntVector(1,0,0);
  if (yplus) idxHi = idxHi + IntVector(0,1,0);
  if (zplus) idxHi = idxHi + IntVector(0,0,1);

  switch(index) {
  case Arches::XDIR:
      if (xminus) {
	int colX = idxLo.x() - 1;
	for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
	  for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
            IntVector currCell(colX, colY, colZ);
            IntVector xplusCell(colX+1, colY, colZ);
            IntVector xplusplusCell(colX+2, colY, colZ);

	    if ((vars->cellType[currCell] == wall_celltypeval)
		&& (!(vars->cellType[xplusCell] == wall_celltypeval))) {
                     (vars->filteredRhoUjU[0])[xplusplusCell] = 0.0;
            }
	  }
	}
      }
    break;
  case Arches::YDIR:
      if (xminus) {
	int colX = idxLo.x() - 1;
	for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
	  for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
            IntVector currCell(colX, colY, colZ);
            IntVector xplusCell(colX+1, colY, colZ);
            IntVector yminusCell(colX, colY-1, colZ);

            if ((vars->cellType[currCell] == wall_celltypeval)&&
                (!((yminus)&&(colY == idxLo.y())))) {
                     (vars->filteredRhoUjV[0])[xplusCell] = 0.0;
            }
            else if ((vars->cellType[yminusCell] == wall_celltypeval)&&
                     (!(vars->cellType[currCell] == wall_celltypeval))) {
                     (vars->filteredRhoUjV[0])[xplusCell] = 0.0;
            }
	  }
	}
      }
    break;
  case Arches::ZDIR:
      if (xminus) {
	int colX = idxLo.x() - 1;
	for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
	  for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
            IntVector currCell(colX, colY, colZ);
            IntVector xplusCell(colX+1, colY, colZ);
            IntVector zminusCell(colX, colY, colZ-1);

            if ((vars->cellType[currCell] == wall_celltypeval)&&
                (!((zminus)&&(colZ == idxLo.z())))) {
                     (vars->filteredRhoUjW[0])[xplusCell] = 0.0;
            }
            else if ((vars->cellType[zminusCell] == wall_celltypeval)&&
                     (!(vars->cellType[currCell] == wall_celltypeval))) {
                     (vars->filteredRhoUjW[0])[xplusCell] = 0.0;
            }
	  }
	}
      }
    break;
  default:
    cerr << "Invalid Index value" << endl;
    break;
  }
}

void 
BoundaryCondition::calculateVelRhoHat_mm(const ProcessorGroup* ,
			    const Patch* patch,
			    int index, double delta_t,
			    CellInformation* cellinfo,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)

{
  // Get the patch bounds and the variable bounds
  IntVector idxLo;
  IntVector idxHi;
  // for explicit solver
  int ioff, joff, koff;

  switch (index) {
  case Arches::XDIR:
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();
    ioff = 1; joff = 0; koff = 0;

    fort_mm_explicit_vel(idxLo, idxHi, 
			 vars->uVelRhoHat,
			 constvars->uVelocity,
			 vars->uVelocityCoeff[Arches::AE], 
			 vars->uVelocityCoeff[Arches::AW], 
			 vars->uVelocityCoeff[Arches::AN], 
			 vars->uVelocityCoeff[Arches::AS], 
			 vars->uVelocityCoeff[Arches::AT], 
			 vars->uVelocityCoeff[Arches::AB], 
			 vars->uVelocityCoeff[Arches::AP], 
			 vars->uVelNonlinearSrc,
			 constvars->new_density,
			 cellinfo->sewu, cellinfo->sns, cellinfo->stb,
			 delta_t, ioff, joff, koff,
			 constvars->cellType,
			 d_mmWallID);
    break;
  case Arches::YDIR:
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
    ioff = 0; joff = 1; koff = 0;

    fort_mm_explicit_vel(idxLo, idxHi, 
			 vars->vVelRhoHat,
			 constvars->vVelocity,
			 vars->vVelocityCoeff[Arches::AE], 
			 vars->vVelocityCoeff[Arches::AW], 
			 vars->vVelocityCoeff[Arches::AN], 
			 vars->vVelocityCoeff[Arches::AS], 
			 vars->vVelocityCoeff[Arches::AT], 
			 vars->vVelocityCoeff[Arches::AB], 
			 vars->vVelocityCoeff[Arches::AP], 
			 vars->vVelNonlinearSrc,
			 constvars->new_density,
			 cellinfo->sew, cellinfo->snsv, cellinfo->stb,
			 delta_t, ioff, joff, koff,
			 constvars->cellType,
			 d_mmWallID);

    break;
  case Arches::ZDIR:
    idxLo = patch->getSFCZFORTLowIndex();
    idxHi = patch->getSFCZFORTHighIndex();
    ioff = 0; joff = 0; koff = 1;

    fort_mm_explicit_vel(idxLo, idxHi, 
			 vars->wVelRhoHat,
			 constvars->wVelocity,
			 vars->wVelocityCoeff[Arches::AE], 
			 vars->wVelocityCoeff[Arches::AW], 
			 vars->wVelocityCoeff[Arches::AN], 
			 vars->wVelocityCoeff[Arches::AS], 
			 vars->wVelocityCoeff[Arches::AT], 
			 vars->wVelocityCoeff[Arches::AB], 
			 vars->wVelocityCoeff[Arches::AP], 
			 vars->wVelNonlinearSrc,
			 constvars->new_density,
			 cellinfo->sew, cellinfo->sns, cellinfo->stbw,
			 delta_t, ioff, joff, koff,
			 constvars->cellType,
			 d_mmWallID);

    vars->residWVel = 1.0E-7;
    vars->truncWVel = 1.0;

    break;
  default:
    throw InvalidValue("Invalid index in LinearSolver for velocity");
  }
}

//****************************************************************************
// Scalar Solve for Multimaterial
//****************************************************************************
void 
BoundaryCondition::scalarLisolve_mm(const ProcessorGroup*,
				    const Patch* patch,
				    double delta_t,
				    ArchesVariables* vars,
				    ArchesConstVariables* constvars,
				    CellInformation* cellinfo)
{
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#if implict_defined

  IntVector domLo = vars->scalar.getFortLowIndex();
  IntVector domHi = vars->scalar.getFortHighIndex();

  IntVector domLoDen = vars->old_density.getFortLowIndex();
  IntVector domHiDen = vars->old_density.getFortHighIndex();

  Array1<double> e1;
  Array1<double> f1;
  Array1<double> e2;
  Array1<double> f2;
  Array1<double> e3;
  Array1<double> f3;

  IntVector Size = domHi - domLo + IntVector(1,1,1);

  e1.resize(Size.x());
  f1.resize(Size.x());
  e2.resize(Size.y());
  f2.resize(Size.y());
  e3.resize(Size.z());
  f3.resize(Size.z());

  sum_vartype resid;
  sum_vartype trunc;

  old_dw->get(resid, lab->d_scalarResidLabel);
  old_dw->get(trunc, lab->d_scalarTruncLabel);

  double nlResid = resid;
  double trunc_conv = trunc*1.0E-7;
  double theta = 0.5;
  int scalarIter = 0;
  double scalarResid = 0.0;
  do {
    //fortran call for lineGS solver
    fort_linegs(idxLo, idxHi,
		vars->scalar,
		vars->scalarCoeff[Arches::AE],
		vars->scalarCoeff[Arches::AW],
		vars->scalarCoeff[Arches::AN],
		vars->scalarCoeff[Arches::AS],
		vars->scalarCoeff[Arches::AT],
		vars->scalarCoeff[Arches::AB],
		vars->scalarCoeff[Arches::AP],
		vars->scalarNonlinearSrc,
		e1, f1, e2, f2, e3, f3, theta);
		
    computeScalarResidual(pc, patch, old_dw, new_dw, index, vars);
    scalarResid = vars->residScalar;
    ++scalarIter;
  } while((scalarIter < d_maxSweeps)&&((scalarResid > d_residual*nlResid)||
				      (scalarResid > trunc_conv)));
  cerr << "After scalar " << index <<" solve " << scalarIter << " " << scalarResid << endl;
  cerr << "After scalar " << index <<" solve " << nlResid << " " << trunc_conv <<  endl;
#endif

  fort_mm_explicit(idxLo, idxHi, vars->scalar, constvars->old_scalar,
		   constvars->scalarCoeff[Arches::AE], 
		   constvars->scalarCoeff[Arches::AW], 
		   constvars->scalarCoeff[Arches::AN], 
		   constvars->scalarCoeff[Arches::AS], 
		   constvars->scalarCoeff[Arches::AT], 
		   constvars->scalarCoeff[Arches::AB], 
		   constvars->scalarCoeff[Arches::AP], 
		   constvars->scalarNonlinearSrc, constvars->old_density,
		   cellinfo->sew, cellinfo->sns, cellinfo->stb, 
		   delta_t,
		   constvars->cellType, d_mmWallID);

     for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
       for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
	for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
	  IntVector currCell(ii,jj,kk);
	  if (vars->scalar[currCell] > 1.0)
	    vars->scalar[currCell] = 1.0;
	  else if (vars->scalar[currCell] < 0.0)
	    vars->scalar[currCell] = 0.0;
	}
      }
    }

    vars->residScalar = 1.0E-7;
    vars->truncScalar = 1.0;

}

//****************************************************************************
// Enthalpy Solve for Multimaterial
//****************************************************************************
void 
BoundaryCondition::enthalpyLisolve_mm(const ProcessorGroup*,
				      const Patch* patch,
				      double delta_t,
				      ArchesVariables* vars,
				      ArchesConstVariables* constvars,
				      CellInformation* cellinfo)
{
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#if implict_defined

  IntVector domLo = vars->enthalpy.getFortLowIndex();
  IntVector domHi = vars->enthalpy.getFortHighIndex();

  IntVector domLoDen = vars->old_density.getFortLowIndex();
  IntVector domHiDen = vars->old_density.getFortHighIndex();

  Array1<double> e1;
  Array1<double> f1;
  Array1<double> e2;
  Array1<double> f2;
  Array1<double> e3;
  Array1<double> f3;

  IntVector Size = domHi - domLo + IntVector(1,1,1);

  e1.resize(Size.x());
  f1.resize(Size.x());
  e2.resize(Size.y());
  f2.resize(Size.y());
  e3.resize(Size.z());
  f3.resize(Size.z());

  sum_vartype resid;
  sum_vartype trunc;

  old_dw->get(resid, lab->d_enthalpyResidLabel);
  old_dw->get(trunc, lab->d_enthalpyTruncLabel);

  double nlResid = resid;
  double trunc_conv = trunc*1.0E-7;
  double theta = 0.5;
  int scalarIter = 0;
  double scalarResid = 0.0;
  do {
    //fortran call for lineGS solver
    fort_linegs(idxLo, idxHi,
		vars->enthalpy,
		vars->scalarCoeff[Arches::AE],
		vars->scalarCoeff[Arches::AW],
		vars->scalarCoeff[Arches::AN],
		vars->scalarCoeff[Arches::AS],
		vars->scalarCoeff[Arches::AT],
		vars->scalarCoeff[Arches::AB],
		vars->scalarCoeff[Arches::AP],
		vars->scalarNonlinearSrc,
		e1, f1, e2, f2, e3, f3, theta);
		
    computeScalarResidual(pc, patch, old_dw, new_dw, index, vars);
    scalarResid = vars->residScalar;
    ++scalarIter;
  } while((scalarIter < d_maxSweeps)&&((scalarResid > d_residual*nlResid)||
				      (scalarResid > trunc_conv)));
  cerr << "After scalar " << index <<" solve " << scalarIter << " " << scalarResid << endl;
  cerr << "After scalar " << index <<" solve " << nlResid << " " << trunc_conv <<  endl;
#endif

  fort_mm_explicit_oldvalue(idxLo, idxHi, vars->enthalpy, constvars->old_enthalpy,
			    constvars->scalarCoeff[Arches::AE], 
			    constvars->scalarCoeff[Arches::AW], 
			    constvars->scalarCoeff[Arches::AN], 
			    constvars->scalarCoeff[Arches::AS], 
			    constvars->scalarCoeff[Arches::AT], 
			    constvars->scalarCoeff[Arches::AB], 
			    constvars->scalarCoeff[Arches::AP], 
			    constvars->scalarNonlinearSrc, constvars->old_density,
			    cellinfo->sew, cellinfo->sns, cellinfo->stb, 
			    delta_t,
			    constvars->cellType, d_mmWallID);

    vars->residScalar = 1.0E-7;
    vars->truncScalar = 1.0;

}

//****************************************************************************
// Set zero gradient for scalar on pressure bc
//****************************************************************************
void 
BoundaryCondition::scalarPressureBC(const ProcessorGroup*,
			    const Patch* patch,
			    int /*index*/,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  int pressure_celltypeval = pressureCellType();

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  if (xminus) {
    int colX = idxLo.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);
        if (constvars->cellType[xminusCell] == pressure_celltypeval)
          vars->scalar[xminusCell] = vars->scalar[currCell];
      }
    }
  }
  if (xplus) {
    int colX = idxHi.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        if (constvars->cellType[xplusCell] == pressure_celltypeval)
          vars->scalar[xplusCell] = vars->scalar[currCell];
      }
    }
  }
  if (yminus) {
    int colY = idxLo.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);
        if (constvars->cellType[yminusCell] == pressure_celltypeval)
          vars->scalar[yminusCell] = vars->scalar[currCell];
      }
    }
  }
  if (yplus) {
    int colY = idxHi.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        if (constvars->cellType[yplusCell] == pressure_celltypeval)
          vars->scalar[yplusCell] = vars->scalar[currCell];
      }
    }
  }
  if (zminus) {
    int colZ = idxLo.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);
        if (constvars->cellType[zminusCell] == pressure_celltypeval)
          vars->scalar[zminusCell] = vars->scalar[currCell];
      }
    }
  }
  if (zplus) {
    int colZ = idxHi.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        if (constvars->cellType[zplusCell] == pressure_celltypeval)
          vars->scalar[zplusCell] = vars->scalar[currCell];
      }
    }
  }
}



//****************************************************************************
// Set zero gradient for enthalpy on pressure bc
//****************************************************************************
void 
BoundaryCondition::enthalpyPressureBC(const ProcessorGroup*,
			    const Patch* patch,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  int pressure_celltypeval = pressureCellType();

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  if (xminus) {
    int colX = idxLo.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);
        if (constvars->cellType[xminusCell] == pressure_celltypeval)
          vars->enthalpy[xminusCell] = vars->enthalpy[currCell];
      }
    }
  }
  if (xplus) {
    int colX = idxHi.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        if (constvars->cellType[xplusCell] == pressure_celltypeval)
          vars->enthalpy[xplusCell] = vars->enthalpy[currCell];
      }
    }
  }
  if (yminus) {
    int colY = idxLo.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);
        if (constvars->cellType[yminusCell] == pressure_celltypeval)
          vars->enthalpy[yminusCell] = vars->enthalpy[currCell];
      }
    }
  }
  if (yplus) {
    int colY = idxHi.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        if (constvars->cellType[yplusCell] == pressure_celltypeval)
          vars->enthalpy[yplusCell] = vars->enthalpy[currCell];
      }
    }
  }
  if (zminus) {
    int colZ = idxLo.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);
        if (constvars->cellType[zminusCell] == pressure_celltypeval)
          vars->enthalpy[zminusCell] = vars->enthalpy[currCell];
      }
    }
  }
  if (zplus) {
    int colZ = idxHi.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        if (constvars->cellType[zplusCell] == pressure_celltypeval)
          vars->enthalpy[zplusCell] = vars->enthalpy[currCell];
      }
    }
  }
}
//****************************************************************************
// Set scalar at the outlet from 1d advection equation
//****************************************************************************
void 
BoundaryCondition::scalarOutletBC(const ProcessorGroup*,
			    const Patch* patch,
			    int /*index*/,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  int outlet_celltypeval = outletCellType();

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  if (xminus) {
    int colX = idxLo.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);
        if (constvars->cellType[xminusCell] == outlet_celltypeval)
          vars->scalar[xminusCell]= vars->scalar[currCell];
      }
    }
  }
  if (xplus) {
    int colX = idxHi.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        if (constvars->cellType[xplusCell] == outlet_celltypeval)
          vars->scalar[xplusCell]= vars->scalar[currCell];
      }
    }
  }
  if (yminus) {
    int colY = idxLo.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);
        if (constvars->cellType[yminusCell] == outlet_celltypeval)
          vars->scalar[yminusCell]= vars->scalar[currCell];
      }
    }
  }
  if (yplus) {
    int colY = idxHi.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        if (constvars->cellType[yplusCell] == outlet_celltypeval)
          vars->scalar[yplusCell]= vars->scalar[currCell];
      }
    }
  }
  if (zminus) {
    int colZ = idxLo.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);
        if (constvars->cellType[zminusCell] == outlet_celltypeval)
          vars->scalar[zminusCell]= vars->scalar[currCell];
      }
    }
  }
  if (zplus) {
    int colZ = idxHi.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        if (constvars->cellType[zplusCell] == outlet_celltypeval)
          vars->scalar[zplusCell]= vars->scalar[currCell];
      }
    }
  }
}



//****************************************************************************
// Set enthalpy at the outlet from 1d advection equation
//****************************************************************************
void 
BoundaryCondition::enthalpyOutletBC(const ProcessorGroup*,
			    const Patch* patch,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  int outlet_celltypeval = outletCellType();

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  if (xminus) {
    int colX = idxLo.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);
        if (constvars->cellType[xminusCell] == outlet_celltypeval)
          vars->enthalpy[xminusCell]= vars->enthalpy[currCell];
      }
    }
  }
  if (xplus) {
    int colX = idxHi.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        if (constvars->cellType[xplusCell] == outlet_celltypeval)
          vars->enthalpy[xplusCell]= vars->enthalpy[currCell];
      }
    }
  }
  if (yminus) {
    int colY = idxLo.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);
        if (constvars->cellType[yminusCell] == outlet_celltypeval)
          vars->enthalpy[yminusCell]= vars->enthalpy[currCell];
      }
    }
  }
  if (yplus) {
    int colY = idxHi.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        if (constvars->cellType[yplusCell] == outlet_celltypeval)
          vars->enthalpy[yplusCell]= vars->enthalpy[currCell];
      }
    }
  }
  if (zminus) {
    int colZ = idxLo.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);
        if (constvars->cellType[zminusCell] == outlet_celltypeval)
          vars->enthalpy[zminusCell]= vars->enthalpy[currCell];
      }
    }
  }
  if (zplus) {
    int colZ = idxHi.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        if (constvars->cellType[zplusCell] == outlet_celltypeval)
          vars->enthalpy[zplusCell]= vars->enthalpy[currCell];
      }
    }
  }
}
//****************************************************************************
// Set the inlet rho hat velocity BC
//****************************************************************************
void 
BoundaryCondition::velRhoHatInletBC(const ProcessorGroup* ,
			    const Patch* patch,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars,
			    double time_shift)
{
  double time = d_lab->d_sharedState->getElapsedTime();
  double current_time = time + time_shift;
  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  // stores cell type info for the patch with the ghost cell type
  for (int indx = 0; indx < d_numInlets; indx++) {
    // Get a copy of the current flow inlet
    FlowInlet fi = d_flowInlets[indx];
    
    // assign flowType the value that corresponds to flow
    //CellTypeInfo flowType = FLOW;
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    fort_inlbcs(vars->uVelRhoHat, vars->vVelRhoHat, vars->wVelRhoHat,
      	  idxLo, idxHi, constvars->new_density, constvars->cellType, 
      	  fi.d_cellTypeID, current_time,
      	  xminus, xplus, yminus, yplus, zminus, zplus,
	  d_ramping_inlet_flowrate);
    
  }
}
//****************************************************************************
// Set hat velocity on pressure bc
// Tangential bc's are not needed to be set for hat velocities
// Commented them out to avoid confusion
//****************************************************************************
void 
BoundaryCondition::velRhoHatPressureBC(const ProcessorGroup*,
			    const Patch* patch,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  int pressure_celltypeval = pressureCellType();

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  if (xminus) {
    int colX = idxLo.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        if (constvars->cellType[xminusCell] == pressure_celltypeval) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y())))
	    vars->uVelRhoHat[currCell] = 0.0;
	  else {
	  if (constvars->old_uVelocity[currCell] > 1.0e-10)
            vars->uVelRhoHat[currCell] = vars->uVelRhoHat[xplusCell];
	  else
	    vars->uVelRhoHat[currCell] = 0.0;
	  }
          vars->uVelRhoHat[xminusCell] = vars->uVelRhoHat[currCell];
         /* if (!(yminus && (colY == idxLo.y())))
            vars->vVelRhoHat[xminusCell] = vars->vVelRhoHat[currCell];
          if (!(zminus && (colZ == idxLo.z())))
            vars->wVelRhoHat[xminusCell] = vars->wVelRhoHat[currCell];*/
        }
      }
    }
  }
  if (xplus) {
    int colX = idxHi.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        IntVector xplusplusCell(colX+2, colY, colZ);
        if (constvars->cellType[xplusCell] == pressure_celltypeval) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y())))
	    vars->uVelRhoHat[xplusCell] = 0.0;
	  else {
	  if (constvars->old_uVelocity[xplusCell] < -1.0e-10)
            vars->uVelRhoHat[xplusCell] = vars->uVelRhoHat[currCell];
	  else
	    vars->uVelRhoHat[xplusCell] = 0.0;
	  }
          vars->uVelRhoHat[xplusplusCell] = vars->uVelRhoHat[xplusCell];
         /* if (!(yminus && (colY == idxLo.y())))
            vars->vVelRhoHat[xplusCell] = vars->vVelRhoHat[currCell];
          if (!(zminus && (colZ == idxLo.z())))
            vars->wVelRhoHat[xplusCell] = vars->wVelRhoHat[currCell];*/
        }
      }
    }
  }
  if (yminus) {
    int colY = idxLo.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        if (constvars->cellType[yminusCell] == pressure_celltypeval) {
        /*  if (!(xminus && (colX == idxLo.x())))
            vars->uVelRhoHat[yminusCell] = vars->uVelRhoHat[currCell];*/
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    vars->vVelRhoHat[currCell] = 0.0;
	  else {
	  if (constvars->old_vVelocity[currCell] > 1.0e-10)
            vars->vVelRhoHat[currCell] = vars->vVelRhoHat[yplusCell];
	  else
	    vars->vVelRhoHat[currCell] = 0.0;
	  }
          vars->vVelRhoHat[yminusCell] = vars->vVelRhoHat[currCell];
         /* if (!(zminus && (colZ == idxLo.z())))
            vars->wVelRhoHat[yminusCell] = vars->wVelRhoHat[currCell];*/
        }
      }
    }
  }
  if (yplus) {
    int colY = idxHi.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        IntVector yplusplusCell(colX, colY+2, colZ);
        if (constvars->cellType[yplusCell] == pressure_celltypeval) {
         /* if (!(xminus && (colX == idxLo.x())))
            vars->uVelRhoHat[yplusCell] = vars->uVelRhoHat[currCell];*/
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    vars->vVelRhoHat[yplusCell] = 0.0;
	  else {
	  if (constvars->old_vVelocity[yplusCell] < -1.0e-10)
            vars->vVelRhoHat[yplusCell] = vars->vVelRhoHat[currCell];
	  else
	    vars->vVelRhoHat[yplusCell] = 0.0;
	  }
          vars->vVelRhoHat[yplusplusCell] = vars->vVelRhoHat[yplusCell];
        /*  if (!(zminus && (colZ == idxLo.z())))
            vars->wVelRhoHat[yplusCell] = vars->wVelRhoHat[currCell];*/
        }
      }
    }
  }
  if (zminus) {
    int colZ = idxLo.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);
        IntVector zplusCell(colX, colY, colZ+1);
        if (constvars->cellType[zminusCell] == pressure_celltypeval) {
        /*  if (!(xminus && (colX == idxLo.x())))
            vars->uVelRhoHat[zminusCell] = vars->uVelRhoHat[currCell];
          if (!(yminus && (colY == idxLo.y())))
            vars->vVelRhoHat[zminusCell] = vars->vVelRhoHat[currCell];*/
          if ((yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    vars->wVelRhoHat[currCell] = 0.0;
	  else {
	  if (constvars->old_wVelocity[currCell] > 1.0e-10)
            vars->wVelRhoHat[currCell] = vars->wVelRhoHat[zplusCell];
	  else
	    vars->wVelRhoHat[currCell] = 0.0;
	  }
          vars->wVelRhoHat[zminusCell] = vars->wVelRhoHat[currCell];
        }
      }
    }
  }
  if (zplus) {
    int colZ = idxHi.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        IntVector zplusplusCell(colX, colY, colZ+2);
        if (constvars->cellType[zplusCell] == pressure_celltypeval) {
         /* if (!(xminus && (colX == idxLo.x())))
            vars->uVelRhoHat[zplusCell] = vars->uVelRhoHat[currCell];
          if (!(yminus && (colY == idxLo.y())))
            vars->vVelRhoHat[zplusCell] = vars->vVelRhoHat[currCell];*/
          if ((yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    vars->wVelRhoHat[zplusCell] = 0.0;
	  else {
	  if (constvars->old_wVelocity[zplusCell] < -1.0e-10)
            vars->wVelRhoHat[zplusCell] = vars->wVelRhoHat[currCell];
	  else
	    vars->wVelRhoHat[zplusCell] = 0.0;
	  }
          vars->wVelRhoHat[zplusplusCell] = vars->wVelRhoHat[zplusCell];
        }
      }
    }
  }
}
//****************************************************************************
// Set hat velocity at the outlet
// Tangential bc's are not needed to be set for hat velocities
// Commented them out to avoid confusion
//****************************************************************************
void 
BoundaryCondition::velRhoHatOutletBC(const ProcessorGroup*,
			    const Patch* patch,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  int outlet_celltypeval = outletCellType();

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  if (xminus) {
    int colX = idxLo.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        if (constvars->cellType[xminusCell] == outlet_celltypeval) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y())))
	    vars->uVelRhoHat[currCell] = 0.0;
	  else {
	  if (constvars->old_uVelocity[currCell] < -1.0e-10)
            vars->uVelRhoHat[currCell] = vars->uVelRhoHat[xplusCell];
	  else
	    vars->uVelRhoHat[currCell] = 0.0;
	  }
          vars->uVelRhoHat[xminusCell] = vars->uVelRhoHat[currCell];
 /*         if (!(yminus && (colY == idxLo.y())))
            vars->vVelRhoHat[xminusCell] = vars->vVelRhoHat[currCell];
          if (!(zminus && (colZ == idxLo.z())))
            vars->wVelRhoHat[xminusCell] = vars->wVelRhoHat[currCell];*/
        }
      }
    }
  }
  if (xplus) {
    int colX = idxHi.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        IntVector xplusplusCell(colX+2, colY, colZ);
        if (constvars->cellType[xplusCell] == outlet_celltypeval) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y())))
	    vars->uVelRhoHat[xplusCell] = 0.0;
	  else {
	  if (constvars->old_uVelocity[xplusCell] > 1.0e-10)
            vars->uVelRhoHat[xplusCell] = vars->uVelRhoHat[currCell];
	  else
	    vars->uVelRhoHat[xplusCell] = 0.0;
	  }
          vars->uVelRhoHat[xplusplusCell] = vars->uVelRhoHat[xplusCell];
   /*       if (!(yminus && (colY == idxLo.y())))
            vars->vVelRhoHat[xplusCell] = vars->vVelRhoHat[currCell];
          if (!(zminus && (colZ == idxLo.z())))
            vars->wVelRhoHat[xplusCell] = vars->wVelRhoHat[currCell];*/
        }
      }
    }
  }
  if (yminus) {
    int colY = idxLo.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
	IntVector yminusCell(colX, colY-1, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        if (constvars->cellType[yminusCell] == outlet_celltypeval) {
     /*     if (!(xminus && (colX == idxLo.x())))
            vars->uVelRhoHat[yminusCell] = vars->uVelRhoHat[currCell];*/
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    vars->vVelRhoHat[currCell] = 0.0;
	  else {
	  if (constvars->old_vVelocity[currCell] < -1.0e-10)
            vars->vVelRhoHat[currCell] = vars->vVelRhoHat[yplusCell];
	  else
	    vars->vVelRhoHat[currCell] = 0.0;
	  }
          vars->vVelRhoHat[yminusCell] = vars->vVelRhoHat[currCell];
     /*     if (!(zminus && (colZ == idxLo.z())))
            vars->wVelRhoHat[yminusCell] = vars->wVelRhoHat[currCell];*/
        }
      }
    }
  }
  if (yplus) {
    int colY = idxHi.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        IntVector yplusplusCell(colX, colY+2, colZ);
        if (constvars->cellType[yplusCell] == outlet_celltypeval) {
      /*    if (!(xminus && (colX == idxLo.x())))
            vars->uVelRhoHat[yplusCell] = vars->uVelRhoHat[currCell];*/
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    vars->vVelRhoHat[yplusCell] = 0.0;
	  else {
	  if (constvars->old_vVelocity[yplusCell] > 1.0e-10)
            vars->vVelRhoHat[yplusCell] = vars->vVelRhoHat[currCell];
	  else
	    vars->vVelRhoHat[yplusCell] = 0.0;
	  }
          vars->vVelRhoHat[yplusplusCell] = vars->vVelRhoHat[yplusCell];
       /*   if (!(zminus && (colZ == idxLo.z())))
            vars->wVelRhoHat[yplusCell] = vars->wVelRhoHat[currCell];*/
        }
      }
    }
  }
  if (zminus) {
    int colZ = idxLo.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
	IntVector zminusCell(colX, colY, colZ-1);
        IntVector zplusCell(colX, colY, colZ+1);
        if (constvars->cellType[zminusCell] == outlet_celltypeval) {
      /*    if (!(xminus && (colX == idxLo.x())))
            vars->uVelRhoHat[zminusCell] = vars->uVelRhoHat[currCell];
          if (!(yminus && (colY == idxLo.y())))
            vars->vVelRhoHat[zminusCell] = vars->vVelRhoHat[currCell];*/
          if ((yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    vars->wVelRhoHat[currCell] = 0.0;
	  else {
	  if (constvars->old_wVelocity[currCell] < -1.0e-10)
            vars->wVelRhoHat[currCell] = vars->wVelRhoHat[zplusCell];
	  else
	    vars->wVelRhoHat[currCell] = 0.0;
	  }
          vars->wVelRhoHat[zminusCell] = vars->wVelRhoHat[currCell];
        }
      }
    }
  }
  if (zplus) {
    int colZ = idxHi.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        IntVector zplusplusCell(colX, colY, colZ+2);
        if (constvars->cellType[zplusCell] == outlet_celltypeval) {
       /*   if (!(xminus && (colX == idxLo.x())))
            vars->uVelRhoHat[zplusCell] = vars->uVelRhoHat[currCell];
          if (!(yminus && (colY == idxLo.y())))
            vars->vVelRhoHat[zplusCell] = vars->vVelRhoHat[currCell];*/
          if ((yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    vars->wVelRhoHat[zplusCell] = 0.0;
	  else {
	  if (constvars->old_wVelocity[zplusCell] > 1.0e-10)
            vars->wVelRhoHat[zplusCell] = vars->wVelRhoHat[currCell];
	  else
	    vars->wVelRhoHat[zplusCell] = 0.0;
	  }
          vars->wVelRhoHat[zplusplusCell] = vars->wVelRhoHat[zplusCell];
        }
      }
    }
  }
}
//****************************************************************************
// Set zero gradient for velocity on pressure bc
//****************************************************************************
void 
BoundaryCondition::velocityPressureBC(const ProcessorGroup*,
			    const Patch* patch,
			    const int index,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  int outlet_celltypeval = outletCellType();
  int pressure_celltypeval = pressureCellType();

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  if (xminus) {
    int colX = idxLo.x();
    int maxY = idxHi.y();
    if (yplus) maxY++;
    int maxZ = idxHi.z();
    if (zplus) maxZ++;
    for (int colZ = idxLo.z(); colZ <= maxZ; colZ ++) {
      for (int colY = idxLo.y(); colY <= maxY; colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);
        IntVector xminusyminusCell(colX-1, colY-1, colZ);
        IntVector xminuszminusCell(colX-1, colY, colZ-1);
        if ((constvars->cellType[xminusCell] == pressure_celltypeval)||
            (constvars->cellType[xminusCell] == outlet_celltypeval)) {
          switch (index) {
           case Arches::XDIR:
           break;
           case Arches::YDIR:
             if (!(zplus && (colZ == maxZ)))
               vars->vVelRhoHat[xminusCell] = vars->vVelRhoHat[currCell];
           break;
           case Arches::ZDIR:
             if (!(yplus && (colY == maxY)))
               vars->wVelRhoHat[xminusCell] = vars->wVelRhoHat[currCell];
           break;
           default:
		throw InvalidValue("Invalid index in velocityPressureBC");
          }
        }
	else {
	  if ((constvars->cellType[xminusyminusCell] == pressure_celltypeval)
              ||(constvars->cellType[xminusyminusCell] == outlet_celltypeval)) {
            switch (index) {
             case Arches::XDIR:
             break;
             case Arches::YDIR:
               if (!(zplus && (colZ == maxZ)))
                 vars->vVelRhoHat[xminusCell] = vars->vVelRhoHat[currCell];
             break;
             case Arches::ZDIR:
             break;
             default:
		  throw InvalidValue("Invalid index in velocityPressureBC");
            }
          }
	  if ((constvars->cellType[xminuszminusCell] == pressure_celltypeval)
              ||(constvars->cellType[xminuszminusCell] == outlet_celltypeval)) {
            switch (index) {
             case Arches::XDIR:
             break;
             case Arches::YDIR:
             break;
             case Arches::ZDIR:
               if (!(yplus && (colY == maxY)))
                 vars->wVelRhoHat[xminusCell] = vars->wVelRhoHat[currCell];
             break;
             default:
		  throw InvalidValue("Invalid index in velocityPressureBC");
            }
          }
	}
      }
    }
  }
  if (xplus) {
    int colX = idxHi.x();
    int maxY = idxHi.y();
    if (yplus) maxY++;
    int maxZ = idxHi.z();
    if (zplus) maxZ++;
    for (int colZ = idxLo.z(); colZ <= maxZ; colZ ++) {
      for (int colY = idxLo.y(); colY <= maxY; colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        IntVector xplusyminusCell(colX+1, colY-1, colZ);
        IntVector xpluszminusCell(colX+1, colY, colZ-1);
        if ((constvars->cellType[xplusCell] == pressure_celltypeval)||
            (constvars->cellType[xplusCell] == outlet_celltypeval)) {
          switch (index) {
           case Arches::XDIR:
           break;
           case Arches::YDIR:
             if (!(zplus && (colZ == maxZ)))
               vars->vVelRhoHat[xplusCell] = vars->vVelRhoHat[currCell];
           break;
           case Arches::ZDIR:
             if (!(yplus && (colY == maxY)))
               vars->wVelRhoHat[xplusCell] = vars->wVelRhoHat[currCell];
           break;
           default:
		throw InvalidValue("Invalid index in velocityPressureBC");
          }
        }
	else {
	  if ((constvars->cellType[xplusyminusCell] == pressure_celltypeval)
              ||(constvars->cellType[xplusyminusCell] == outlet_celltypeval)) {
            switch (index) {
             case Arches::XDIR:
             break;
             case Arches::YDIR:
               if (!(zplus && (colZ == maxZ)))
                 vars->vVelRhoHat[xplusCell] = vars->vVelRhoHat[currCell];
             break;
             case Arches::ZDIR:
             break;
             default:
		  throw InvalidValue("Invalid index in velocityPressureBC");
            }
          }
	  if ((constvars->cellType[xpluszminusCell] == pressure_celltypeval)
              ||(constvars->cellType[xpluszminusCell] == outlet_celltypeval)) {
            switch (index) {
             case Arches::XDIR:
             break;
             case Arches::YDIR:
             break;
             case Arches::ZDIR:
               if (!(yplus && (colY == maxY)))
                 vars->wVelRhoHat[xplusCell] = vars->wVelRhoHat[currCell];
             break;
             default:
		  throw InvalidValue("Invalid index in velocityPressureBC");
            }
          }
	}
      }
    }
  }
  if (yminus) {
    int colY = idxLo.y();
    int maxX = idxHi.x();
    if (xplus) maxX++;
    int maxZ = idxHi.z();
    if (zplus) maxZ++;
    for (int colZ = idxLo.z(); colZ <= maxZ; colZ ++) {
      for (int colX = idxLo.x(); colX <= maxX; colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);
        IntVector yminusxminusCell(colX-1, colY-1, colZ);
        IntVector yminuszminusCell(colX, colY-1, colZ-1);
        if ((constvars->cellType[yminusCell] == pressure_celltypeval)||
            (constvars->cellType[yminusCell] == outlet_celltypeval)) {
          switch (index) {
           case Arches::XDIR:
             if (!(zplus && (colZ == maxZ)))
               vars->uVelRhoHat[yminusCell] = vars->uVelRhoHat[currCell];
           break;
           case Arches::YDIR:
           break;
           case Arches::ZDIR:
             if (!(xplus && (colX == maxX)))
               vars->wVelRhoHat[yminusCell] = vars->wVelRhoHat[currCell];
           break;
           default:
		throw InvalidValue("Invalid index in velocityPressureBC");
          }
        }
	else {
	  if ((constvars->cellType[yminusxminusCell] == pressure_celltypeval)
              ||(constvars->cellType[yminusxminusCell] == outlet_celltypeval)) {
            switch (index) {
             case Arches::XDIR:
               if (!(zplus && (colZ == maxZ)))
                 vars->uVelRhoHat[yminusCell] = vars->uVelRhoHat[currCell];
             break;
             case Arches::YDIR:
             break;
             case Arches::ZDIR:
             break;
             default:
		  throw InvalidValue("Invalid index in velocityPressureBC");
            }
          }
	  if ((constvars->cellType[yminuszminusCell] == pressure_celltypeval)
              ||(constvars->cellType[yminuszminusCell] == outlet_celltypeval)) {
            switch (index) {
             case Arches::XDIR:
             break;
             case Arches::YDIR:
             break;
             case Arches::ZDIR:
               if (!(xplus && (colX == maxX)))
                 vars->wVelRhoHat[yminusCell] = vars->wVelRhoHat[currCell];
             break;
             default:
		  throw InvalidValue("Invalid index in velocityPressureBC");
            }
          }
	}  
      }
    }
  }
  if (yplus) {
    int colY = idxHi.y();
    int maxX = idxHi.x();
    if (xplus) maxX++;
    int maxZ = idxHi.z();
    if (zplus) maxZ++;
    for (int colZ = idxLo.z(); colZ <= maxZ; colZ ++) {
      for (int colX = idxLo.x(); colX <= maxX; colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        IntVector yplusxminusCell(colX-1, colY+1, colZ);
        IntVector ypluszminusCell(colX, colY+1, colZ-1);
        if ((constvars->cellType[yplusCell] == pressure_celltypeval)||
            (constvars->cellType[yplusCell] == outlet_celltypeval)) {
          switch (index) {
           case Arches::XDIR:
             if (!(zplus && (colZ == maxZ)))
               vars->uVelRhoHat[yplusCell] = vars->uVelRhoHat[currCell];
           break;
           case Arches::YDIR:
           break;
           case Arches::ZDIR:
             if (!(xplus && (colX == maxX)))
               vars->wVelRhoHat[yplusCell] = vars->wVelRhoHat[currCell];
           break;
           default:
		throw InvalidValue("Invalid index in velocityPressureBC");
          }
        }
	else {
	  if ((constvars->cellType[yplusxminusCell] == pressure_celltypeval)
              ||(constvars->cellType[yplusxminusCell] == outlet_celltypeval)) {
            switch (index) {
             case Arches::XDIR:
               if (!(zplus && (colZ == maxZ)))
                 vars->uVelRhoHat[yplusCell] = vars->uVelRhoHat[currCell];
             break;
             case Arches::YDIR:
             break;
             case Arches::ZDIR:
             break;
             default:
		  throw InvalidValue("Invalid index in velocityPressureBC");
            }
          }
	  if ((constvars->cellType[ypluszminusCell] == pressure_celltypeval)
              ||(constvars->cellType[ypluszminusCell] == outlet_celltypeval)) {
            switch (index) {
             case Arches::XDIR:
             break;
             case Arches::YDIR:
             break;
             case Arches::ZDIR:
               if (!(xplus && (colX == maxX)))
                 vars->wVelRhoHat[yplusCell] = vars->wVelRhoHat[currCell];
             break;
             default:
		  throw InvalidValue("Invalid index in velocityPressureBC");
            }
          }
	}
      }
    }
  }
  if (zminus) {
    int colZ = idxLo.z();
    int maxX = idxHi.x();
    if (xplus) maxX++;
    int maxY = idxHi.y();
    if (yplus) maxY++;
    for (int colY = idxLo.y(); colY <= maxY; colY ++) {
      for (int colX = idxLo.x(); colX <= maxX; colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);
        IntVector zminusxminusCell(colX-1, colY, colZ-1);
        IntVector zminusyminusCell(colX, colY-1, colZ-1);
        if ((constvars->cellType[zminusCell] == pressure_celltypeval)||
            (constvars->cellType[zminusCell] == outlet_celltypeval)) {
          switch (index) {
           case Arches::XDIR:
             if (!(yplus && (colY == maxY)))
               vars->uVelRhoHat[zminusCell] = vars->uVelRhoHat[currCell];
           break;
           case Arches::YDIR:
             if (!(xplus && (colX == maxX)))
               vars->vVelRhoHat[zminusCell] = vars->vVelRhoHat[currCell];
           break;
           case Arches::ZDIR:
           break;
           default:
		throw InvalidValue("Invalid index in velocityPressureBC");
          }
        }
	else {
	  if ((constvars->cellType[zminusxminusCell] == pressure_celltypeval)
              ||(constvars->cellType[zminusxminusCell] == outlet_celltypeval)) {
            switch (index) {
             case Arches::XDIR:
               if (!(yplus && (colY == maxY)))
                 vars->uVelRhoHat[zminusCell] = vars->uVelRhoHat[currCell];
             break;
             case Arches::YDIR:
             break;
             case Arches::ZDIR:
             break;
             default:
		  throw InvalidValue("Invalid index in velocityPressureBC");
            }
          }
	  if ((constvars->cellType[zminusyminusCell] == pressure_celltypeval)
              ||(constvars->cellType[zminusyminusCell] == outlet_celltypeval)) {
            switch (index) {
             case Arches::XDIR:
             break;
             case Arches::YDIR:
               if (!(xplus && (colX == maxX)))
                 vars->vVelRhoHat[zminusCell] = vars->vVelRhoHat[currCell];
             break;
             case Arches::ZDIR:
             break;
             default:
		  throw InvalidValue("Invalid index in velocityPressureBC");
            }
          }
	}
      }
    }
  }
  if (zplus) {
    int colZ = idxHi.z();
    int maxX = idxHi.x();
    if (xplus) maxX++;
    int maxY = idxHi.y();
    if (yplus) maxY++;
    for (int colY = idxLo.y(); colY <= maxY; colY ++) {
      for (int colX = idxLo.x(); colX <= maxX; colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        IntVector zplusxminusCell(colX-1, colY, colZ+1);
        IntVector zplusyminusCell(colX, colY-1, colZ+1);
        if ((constvars->cellType[zplusCell] == pressure_celltypeval)||
            (constvars->cellType[zplusCell] == outlet_celltypeval)) {
          switch (index) {
           case Arches::XDIR:
             if (!(yplus && (colY == maxY)))
               vars->uVelRhoHat[zplusCell] = vars->uVelRhoHat[currCell];
           break;
           case Arches::YDIR:
             if (!(xplus && (colX == maxX)))
               vars->vVelRhoHat[zplusCell] = vars->vVelRhoHat[currCell];
           break;
           case Arches::ZDIR:
           break;
           default:
		throw InvalidValue("Invalid index in velocityPressureBC");
          }
        }
	else {
	  if ((constvars->cellType[zplusxminusCell] == pressure_celltypeval)
              ||(constvars->cellType[zplusxminusCell] == outlet_celltypeval)) {
            switch (index) {
             case Arches::XDIR:
               if (!(yplus && (colY == maxY)))
                 vars->uVelRhoHat[zplusCell] = vars->uVelRhoHat[currCell];
             break;
             case Arches::YDIR:
             break;
             case Arches::ZDIR:
             break;
             default:
		  throw InvalidValue("Invalid index in velocityPressureBC");
            }
          }
	  if ((constvars->cellType[zplusyminusCell] == pressure_celltypeval)
              ||(constvars->cellType[zplusyminusCell] == outlet_celltypeval)) {
            switch (index) {
             case Arches::XDIR:
             break;
             case Arches::YDIR:
               if (!(xplus && (colX == maxX)))
                 vars->vVelRhoHat[zplusCell] = vars->vVelRhoHat[currCell];
             break;
             case Arches::ZDIR:
             break;
             default:
		  throw InvalidValue("Invalid index in velocityPressureBC");
            }
          }
	}
      }
    }
  }
}
//****************************************************************************
// Add pressure gradient to outlet velocity
//****************************************************************************
void 
BoundaryCondition::addPresGradVelocityOutletBC(const ProcessorGroup*,
			    const Patch* patch,
			    const int index,
			    CellInformation* cellinfo,
			    const double delta_t,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  int outlet_celltypeval = outletCellType();
  int pressure_celltypeval = pressureCellType();

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  switch (index) {
  case Arches::XDIR:
  if (xminus) {
    int colX = idxLo.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);
        if ((constvars->cellType[xminusCell] == outlet_celltypeval)||
            (constvars->cellType[xminusCell] == pressure_celltypeval)) {
           double avdenlow = 0.5 * (constvars->density[currCell] +
			            constvars->density[xminusCell]);

           vars->uVelRhoHat[currCell] -= 2.0*delta_t*constvars->pressure[currCell]/
				(cellinfo->sew[colX] * avdenlow);

           vars->uVelRhoHat[xminusCell] = vars->uVelRhoHat[currCell];

        }
      }
    }
  }
  if (xplus) {
    int colX = idxHi.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        IntVector xplusplusCell(colX+2, colY, colZ);
        if ((constvars->cellType[xplusCell] == outlet_celltypeval)||
            (constvars->cellType[xplusCell] == pressure_celltypeval)) {
           double avden = 0.5 * (constvars->density[xplusCell] +
			         constvars->density[currCell]);

           vars->uVelRhoHat[xplusCell] += 2.0*delta_t*constvars->pressure[currCell]/
				(cellinfo->sew[colX] * avden);

           vars->uVelRhoHat[xplusplusCell] = vars->uVelRhoHat[xplusCell];
        }
      }
    }
  }
  break;
  case Arches::YDIR:
  if (yminus) {
    int colY = idxLo.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
	IntVector yminusCell(colX, colY-1, colZ);
        if ((constvars->cellType[yminusCell] == outlet_celltypeval)||
            (constvars->cellType[yminusCell] == pressure_celltypeval)) {
           double avdenlow = 0.5 * (constvars->density[currCell] +
			            constvars->density[yminusCell]);

           vars->vVelRhoHat[currCell] -= 2.0*delta_t*constvars->pressure[currCell]/
				(cellinfo->sns[colY] * avdenlow);

           vars->vVelRhoHat[yminusCell] = vars->vVelRhoHat[currCell];

        }
      }
    }
  }
  if (yplus) {
    int colY = idxHi.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        IntVector yplusplusCell(colX, colY+2, colZ);
        if ((constvars->cellType[yplusCell] == outlet_celltypeval)||
            (constvars->cellType[yplusCell] == pressure_celltypeval)) {
           double avden = 0.5 * (constvars->density[yplusCell] +
			         constvars->density[currCell]);

           vars->vVelRhoHat[yplusCell] += 2.0*delta_t*constvars->pressure[currCell]/
				(cellinfo->sns[colY] * avden);

           vars->vVelRhoHat[yplusplusCell] = vars->vVelRhoHat[yplusCell];

        }
      }
    }
  }
  break;
  case Arches::ZDIR:
  if (zminus) {
    int colZ = idxLo.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
	IntVector zminusCell(colX, colY, colZ-1);
        if ((constvars->cellType[zminusCell] == outlet_celltypeval)||
            (constvars->cellType[zminusCell] == pressure_celltypeval)) {
           double avdenlow = 0.5 * (constvars->density[currCell] +
			            constvars->density[zminusCell]);

           vars->wVelRhoHat[currCell] -= 2.0*delta_t*constvars->pressure[currCell]/
				(cellinfo->stb[colZ] * avdenlow);

           vars->wVelRhoHat[zminusCell] = vars->wVelRhoHat[currCell];

        }
      }
    }
  }
  if (zplus) {
    int colZ = idxHi.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        IntVector zplusplusCell(colX, colY, colZ+2);
        if ((constvars->cellType[zplusCell] == outlet_celltypeval)||
            (constvars->cellType[zplusCell] == pressure_celltypeval)) {
           double avden = 0.5 * (constvars->density[zplusCell] +
			         constvars->density[currCell]);

           vars->wVelRhoHat[zplusCell] += 2.0*delta_t*constvars->pressure[currCell]/
				(cellinfo->stb[colZ] * avden);

           vars->wVelRhoHat[zplusplusCell] = vars->wVelRhoHat[zplusCell];

        }
      }
    }
  }
  break;
  default:
   throw InvalidValue("Invalid index in addPresGradVelocityOutletBC");
  }
}
//****************************************************************************
// Schedule computation of mass balance for the outlet velocity correction
//****************************************************************************
void
BoundaryCondition::sched_getFlowINOUT(SchedulerP& sched,
				      const PatchSet* patches,
				      const MaterialSet* matls,
				      const TimeIntegratorLabel* timelabels)
{
  string taskname =  "BoundaryCondition::getFlowINOUT" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &BoundaryCondition::getFlowINOUT,
			  timelabels);
  
  tsk->requires(Task::NewDW, d_lab->d_filterdrhodtLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::AroundCells,
		Arches::ONEGHOSTCELL);

  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->computes(timelabels->flowIN);
  tsk->computes(timelabels->flowOUT);
  tsk->computes(timelabels->denAccum);
  tsk->computes(timelabels->floutbc);
  tsk->computes(timelabels->areaOUT);

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Get mass balance for the outlet velocity correction
//****************************************************************************
void 
BoundaryCondition::getFlowINOUT(const ProcessorGroup*,
				const PatchSubset* patches,
				const MaterialSubset*,
				DataWarehouse*,
				DataWarehouse* new_dw,
				const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    constCCVariable<int> cellType;
    constCCVariable<double> filterdrhodt;
    constCCVariable<double> density;
    constSFCXVariable<double> uVelocity;
    constSFCYVariable<double> vVelocity;
    constSFCZVariable<double> wVelocity;

    new_dw->get(filterdrhodt, d_lab->d_filterdrhodtLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(cellType, d_lab->d_cellTypeLabel,
		matlIndex, patch, Ghost::AroundCells,Arches::ONEGHOSTCELL);

    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(density, d_lab->d_densityCPLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(uVelocity, d_lab->d_uVelocitySPBCLabel, matlIndex,
		patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(vVelocity, d_lab->d_vVelocitySPBCLabel, matlIndex,
		patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, matlIndex,
		patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    // Get the low and high index for the patch and the variables
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();

    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    double flowIN = 0.0;
    double flowOUT = 0.0;
    double denAccum = 0.0;
    double floutbc = 0.0;
    double areaOUT = 0.0;
    double varIN  = 0.0;
    double varOUT  = 0.0;

    for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
      for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
        for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
	  IntVector currCell(ii,jj,kk);
	  denAccum += filterdrhodt[currCell];
        }
      }
    }

    if (xminus||xplus||yminus||yplus||zminus||zplus) {

      bool doing_balance = false;
      for (int indx = 0; indx < d_numInlets; indx++) {

	// Get a copy of the current flow inlet
	// assign flowType the value that corresponds to flow
	//CellTypeInfo flowType = FLOW;
	FlowInlet fi = d_flowInlets[indx];
	double fout = 0.0;
	fort_inlpresbcinout(uVelocity, vVelocity, wVelocity, idxLo, idxHi,
			   density, cellType, fi.d_cellTypeID,
			   flowIN, fout, cellinfo->sew, cellinfo->sns,
			   cellinfo->stb, xminus, xplus, yminus, yplus,
			   zminus, zplus, doing_balance,
			   density, varIN, varOUT);
	if (fout > 0.0)
		throw InvalidValue("Flow comming out of inlet");

      } 

      if (d_pressureBoundary) {
	int pressure_celltypeval = d_pressureBC->d_cellTypeID;
	fort_inlpresbcinout(uVelocity, vVelocity, wVelocity, idxLo, idxHi,
			    density, cellType, pressure_celltypeval,
			    flowIN, flowOUT, cellinfo->sew, cellinfo->sns,
			    cellinfo->stb, xminus, xplus, yminus, yplus,
			    zminus, zplus, doing_balance,
			    density, varIN, varOUT);
      }
      if (d_outletBoundary) {
	int outlet_celltypeval = d_outletBC->d_cellTypeID;
	if (xminus) {
	  int colX = idxLo.x();
  	  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
	    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
	      IntVector currCell(colX, colY, colZ);
      	      IntVector xminusCell(colX-1, colY, colZ);

	      if (cellType[xminusCell] == outlet_celltypeval) {
                 double avdenlow = 0.5 * (density[currCell] +
	      		            density[xminusCell]);
     	    	 floutbc -= avdenlow*uVelocity[currCell] *
	          	     cellinfo->sns[colY] * cellinfo->stb[colZ];
    	         areaOUT += cellinfo->sns[colY] * cellinfo->stb[colZ];
              }
	    }
	  }
 	}
	if (xplus) {
	  int colX = idxHi.x();
	  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
	    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
	      IntVector currCell(colX, colY, colZ);
	      IntVector xplusCell(colX+1, colY, colZ);

	      if (cellType[xplusCell] == outlet_celltypeval) {
	         double avden = 0.5 * (density[xplusCell] +
	      		         density[currCell]);
	         floutbc += avden*uVelocity[xplusCell] *
	       	     cellinfo->sns[colY] * cellinfo->stb[colZ];
    	         areaOUT += cellinfo->sns[colY] * cellinfo->stb[colZ];
	      }
	    }
	  }
	}
 	if (yminus) {
 	  int colY = idxLo.y();
 	  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
 	    for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
 	      IntVector currCell(colX, colY, colZ);
 	      IntVector yminusCell(colX, colY-1, colZ);

 	      if (cellType[yminusCell] == outlet_celltypeval) {
 	         double avdenlow = 0.5 * (density[currCell] +
 	      		            density[yminusCell]);
 	         flowOUT -= Min(0.0,avdenlow*vVelocity[currCell] *
	                   cellinfo->sew[colX] * cellinfo->stb[colZ]);
 	         flowIN += Max(0.0,avdenlow*vVelocity[currCell] *
	                   cellinfo->sew[colX] * cellinfo->stb[colZ]);
    	         areaOUT += cellinfo->sew[colX] * cellinfo->stb[colZ];
 	      }
 	    }
 	  }
 	}
 	if (yplus) {
 	  int colY = idxHi.y();
 	  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
 	    for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
 	      IntVector currCell(colX, colY, colZ);
 	      IntVector yplusCell(colX, colY+1, colZ);

 	      if (cellType[yplusCell] == outlet_celltypeval) {
 	         double avden = 0.5 * (density[yplusCell] +
 	      		         density[currCell]);
 	         flowOUT += Max(0.0,avden*vVelocity[yplusCell] *
 	          	     cellinfo->sew[colX] * cellinfo->stb[colZ]);
 	         flowIN -= Min(0.0,avden*vVelocity[yplusCell] *
 	          	     cellinfo->sew[colX] * cellinfo->stb[colZ]);
    	         areaOUT += cellinfo->sew[colX] * cellinfo->stb[colZ];
 	      }
 	    }
 	  }
 	}
 	if (zminus) {
 	  int colZ = idxLo.z();
 	  for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
 	    for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
 	      IntVector currCell(colX, colY, colZ);
 	      IntVector zminusCell(colX, colY, colZ-1);

 	      if (cellType[zminusCell] == outlet_celltypeval) {
 	         double avdenlow = 0.5 * (density[currCell] +
 	      		            density[zminusCell]);
 	         flowOUT -= Min(0.0,avdenlow*wVelocity[currCell] *
 	          	     cellinfo->sew[colX] * cellinfo->sns[colY]);
 	         flowIN += Max(0.0,avdenlow*wVelocity[currCell] *
 	          	     cellinfo->sew[colX] * cellinfo->sns[colY]);
    	         areaOUT += cellinfo->sew[colX] * cellinfo->sns[colY];
 	      }
 	    }
 	  }
 	}
 	if (zplus) {
 	  int colZ = idxHi.z();
 	  for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
 	    for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
 	      IntVector currCell(colX, colY, colZ);
 	      IntVector zplusCell(colX, colY, colZ+1);

 	      if (cellType[zplusCell] == outlet_celltypeval) {
 	         double avden = 0.5 * (density[zplusCell] +
 	      		         density[currCell]);
 	         flowOUT += Max(0.0,avden*wVelocity[zplusCell] *
 	          	     cellinfo->sew[colX] * cellinfo->sns[colY]);
 	         flowIN -= Min(0.0,avden*wVelocity[zplusCell] *
 	          	     cellinfo->sew[colX] * cellinfo->sns[colY]);
    	         areaOUT += cellinfo->sew[colX] * cellinfo->sns[colY];
 	      }
 	    }
 	  }
  	}
      }
    }

  new_dw->put(sum_vartype(flowIN), timelabels->flowIN);
  new_dw->put(sum_vartype(flowOUT), timelabels->flowOUT);
  new_dw->put(sum_vartype(denAccum), timelabels->denAccum);
  new_dw->put(sum_vartype(floutbc), timelabels->floutbc);
  new_dw->put(sum_vartype(areaOUT), timelabels->areaOUT);
  }
}

//****************************************************************************
// Schedule mass balance computation
// Does not perform any velocity correction
// Named for historical reasons
//****************************************************************************
void BoundaryCondition::sched_correctVelocityOutletBC(SchedulerP& sched,
					      const PatchSet* patches,
					      const MaterialSet* matls,
				         const TimeIntegratorLabel* timelabels)
{
  string taskname =  "BoundaryCondition::correctVelocityOutletBC" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &BoundaryCondition::correctVelocityOutletBC,
			  timelabels);
  
  tsk->requires(Task::NewDW, timelabels->flowIN);
  tsk->requires(Task::NewDW, timelabels->flowOUT);
  tsk->requires(Task::NewDW, timelabels->denAccum);
  tsk->requires(Task::NewDW, timelabels->floutbc);
  tsk->requires(Task::NewDW, timelabels->areaOUT);

  if (timelabels->integrator_last_step)
    tsk->computes(d_lab->d_uvwoutLabel);

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Correct outlet velocity
//****************************************************************************
void 
BoundaryCondition::correctVelocityOutletBC(const ProcessorGroup* pc,
			      const PatchSubset* ,
			      const MaterialSubset*,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw,
			      const TimeIntegratorLabel* timelabels)
{
    sum_vartype sum_totalFlowIN, sum_totalFlowOUT, sum_netflowOutbc,
                sum_totalAreaOUT, sum_denAccum;
    double totalFlowIN, totalFlowOUT, netFlowOUT_outbc, totalAreaOUT, denAccum;

    new_dw->get(sum_totalFlowIN, timelabels->flowIN);
    new_dw->get(sum_totalFlowOUT, timelabels->flowOUT);
    new_dw->get(sum_denAccum, timelabels->denAccum);
    new_dw->get(sum_netflowOutbc, timelabels->floutbc);
    new_dw->get(sum_totalAreaOUT, timelabels->areaOUT);
			  
    totalFlowIN = sum_totalFlowIN;
    totalFlowOUT = sum_totalFlowOUT;
    netFlowOUT_outbc = sum_netflowOutbc;
    totalAreaOUT = sum_totalAreaOUT;
    denAccum = sum_denAccum;
    double uvwcorr = 0.0;

    d_overallMB = fabs((totalFlowIN - denAccum - totalFlowOUT - 
			 netFlowOUT_outbc)/(totalFlowIN+1.e-20));

    int me = pc->myrank();

    if (d_outletBoundary) {
      if (totalAreaOUT > 0.0) {
	uvwcorr = (totalFlowIN - denAccum - totalFlowOUT - netFlowOUT_outbc)/
	           totalAreaOUT;
      }
      else {
	cout << "Zero area for specified outlet" << endl;
	exit(1);
      }
    }
    else
      uvwcorr = 0.0;

    if (me == 0) {
      if (d_overallMB > 0.0)
	cerr << "Overall Mass Balance " << log10(d_overallMB/1.e-7+1.e-20) << endl;
      cerr << "Total flow in " << totalFlowIN << endl;
      cerr << "Total flow out " << totalFlowOUT << endl;
      cerr << "Density accumulation " << denAccum << endl;
      cerr << "Total flow out BC " << netFlowOUT_outbc << endl;
      cerr << "Overall velocity correction " << uvwcorr << endl;
      cerr << "Total Area out " << totalAreaOUT << endl;
    }
    if (timelabels->integrator_last_step)
      new_dw->put(delt_vartype(uvwcorr), d_lab->d_uvwoutLabel);
}
//****************************************************************************
// Schedule init inlet bcs
//****************************************************************************
void 
BoundaryCondition::sched_initInletBC(SchedulerP& sched, const PatchSet* patches,
				    const MaterialSet* matls)
{
  Task* tsk = scinew Task("BoundaryCondition::initInletBC",
			  this,
			  &BoundaryCondition::initInletBC);

  // This task requires cellTypeVariable and areaLabel for inlet boundary
  // Also densityIN, [u,v,w] velocityIN, scalarIN
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
    
  tsk->modifies(d_lab->d_uVelocitySPBCLabel);
  tsk->modifies(d_lab->d_vVelocitySPBCLabel);
  tsk->modifies(d_lab->d_wVelocitySPBCLabel);
  tsk->modifies(d_lab->d_uVelRhoHatLabel);
  tsk->modifies(d_lab->d_vVelRhoHatLabel);
  tsk->modifies(d_lab->d_wVelRhoHatLabel);

  tsk->computes(d_lab->d_densityOldOldLabel);
  tsk->computes(d_lab->d_maxAbsU_label);
  tsk->computes(d_lab->d_maxAbsV_label);
  tsk->computes(d_lab->d_maxAbsW_label);
  tsk->computes(d_lab->d_maxUxplus_label);
  tsk->computes(d_lab->d_avUxplus_label);

//#ifdef divergenceconstraint
    tsk->computes(d_lab->d_divConstraintLabel);
//#endif
  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Actually initialize inlet BCs
//****************************************************************************
void 
BoundaryCondition::initInletBC(const ProcessorGroup* /*pc*/,
				  const PatchSubset* patches,
				  const MaterialSubset*,
				  DataWarehouse*,
				  DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    constCCVariable<int> cellType;
    constCCVariable<double> density;
    CCVariable<double> density_oldold;
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    SFCXVariable<double> uVelRhoHat;
    SFCYVariable<double> vVelRhoHat;
    SFCZVariable<double> wVelRhoHat;

    new_dw->getModifiable(uVelocity, d_lab->d_uVelocitySPBCLabel, matlIndex, patch);
    new_dw->getModifiable(vVelocity, d_lab->d_vVelocitySPBCLabel, matlIndex, patch);
    new_dw->getModifiable(wVelocity, d_lab->d_wVelocitySPBCLabel, matlIndex, patch);
    new_dw->getModifiable(uVelRhoHat, d_lab->d_uVelRhoHatLabel, matlIndex, patch);
    new_dw->getModifiable(vVelRhoHat, d_lab->d_vVelRhoHatLabel, matlIndex, patch);
    new_dw->getModifiable(wVelRhoHat, d_lab->d_wVelRhoHatLabel, matlIndex, patch);
    
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->get(density, d_lab->d_densityCPLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->allocateAndPut(density_oldold, d_lab->d_densityOldOldLabel, matlIndex, patch);

    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

    if (d_inletBoundary) {
      double current_time = 0.0; 
      for (int indx = 0; indx < d_numInlets; indx++) {
        // Get a copy of the current flow inlet
        FlowInlet fi = d_flowInlets[indx];
    
        fort_inlbcs(uVelocity, vVelocity, wVelocity,
      	                 idxLo, idxHi, density, cellType, 
      	                 fi.d_cellTypeID, current_time,
      	                 xminus, xplus, yminus, yplus, zminus, zplus,
	                 d_ramping_inlet_flowrate);
      }
    }  
    
    density_oldold.copyData(density); // copy old into new
    uVelRhoHat.copyData(uVelocity); 
    vVelRhoHat.copyData(vVelocity); 
    wVelRhoHat.copyData(wVelocity); 

//#ifdef divergenceconstraint    
    CCVariable<double> divergence;
    new_dw->allocateAndPut(divergence,
			     d_lab->d_divConstraintLabel, matlIndex, patch);
    divergence.initialize(0.0);
//#endif

    double maxAbsU = 0.0;
    double maxAbsV = 0.0;
    double maxAbsW = 0.0;
    IntVector indexLow;
    IntVector indexHigh;
    double maxUxplus = -10000000000.0;
    double avUxplus = 0.0;
    const Level* level = patch->getLevel();
    IntVector low, high;
    level->findCellIndexRange(low, high);
    IntVector range = high-low;
    double num_elem = range.y()*range.z();
    
      indexLow = patch->getSFCXFORTLowIndex();
      indexHigh = patch->getSFCXFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);

	      maxAbsU = Max(Abs(uVelocity[currCell]), maxAbsU);
          }
        }
      }
      new_dw->put(max_vartype(maxAbsU), d_lab->d_maxAbsU_label); 

      if ((d_outletBoundary)&&(xplus)) {
        int outlet_celltypeval = outletCellType();
        int colX = indexHigh.x();
        for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
          for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector xplusCell(colX+1, colY, colZ);

	      if (cellType[xplusCell] == outlet_celltypeval) {
	        maxUxplus = Max(uVelocity[currCell], maxUxplus);
		avUxplus += uVelocity[currCell];
	      }
          }
        }
      }
      new_dw->put(max_vartype(maxUxplus), d_lab->d_maxUxplus_label);
      avUxplus /= num_elem;
      new_dw->put(sum_vartype(avUxplus), d_lab->d_avUxplus_label);

      indexLow = patch->getSFCYFORTLowIndex();
      indexHigh = patch->getSFCYFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);

	      maxAbsV = Max(Abs(vVelocity[currCell]), maxAbsV);
          }
        }
      }
      new_dw->put(max_vartype(maxAbsV), d_lab->d_maxAbsV_label); 

      indexLow = patch->getSFCZFORTLowIndex();
      indexHigh = patch->getSFCZFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);

	      maxAbsW = Max(Abs(wVelocity[currCell]), maxAbsW);
          }
        }
      }
      new_dw->put(max_vartype(maxAbsW), d_lab->d_maxAbsW_label); 
    
  }
}
//****************************************************************************
// Schedule setting zero tangential force velocity BC
//****************************************************************************
void
BoundaryCondition::sched_setVelocityTangentialBC(SchedulerP& sched,
				      const PatchSet* patches,
				      const MaterialSet* matls,
				      const TimeIntegratorLabel* timelabels)
{
  string taskname =  "BoundaryCondition::setVelocityTangentialBC" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &BoundaryCondition::setVelocityTangentialBC,
			  timelabels);
  
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::AroundCells,
		Arches::ONEGHOSTCELL);

  tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  tsk->modifies(d_lab->d_uVelocitySPBCLabel);
  tsk->modifies(d_lab->d_vVelocitySPBCLabel);
  tsk->modifies(d_lab->d_wVelocitySPBCLabel);

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Set zero tangential force velocity BC
//****************************************************************************
void 
BoundaryCondition::setVelocityTangentialBC(const ProcessorGroup*,
				const PatchSubset* patches,
				const MaterialSubset*,
				DataWarehouse*,
				DataWarehouse* new_dw,
				const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    constCCVariable<int> cellType;
    constSFCXVariable<double> uVelocity;
    constSFCYVariable<double> vVelocity;
    constSFCZVariable<double> wVelocity;
    SFCXVariable<double> new_uVelocity;
    SFCYVariable<double> new_vVelocity;
    SFCZVariable<double> new_wVelocity;

    new_dw->get(cellType, d_lab->d_cellTypeLabel,
		matlIndex, patch, Ghost::AroundCells,Arches::ONEGHOSTCELL);

    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(uVelocity, d_lab->d_uVelRhoHatLabel, matlIndex,
		patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(vVelocity, d_lab->d_vVelRhoHatLabel, matlIndex,
		patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(wVelocity, d_lab->d_wVelRhoHatLabel, matlIndex,
		patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
   
    new_dw->getModifiable(new_uVelocity, d_lab->d_uVelocitySPBCLabel,
		          matlIndex, patch);
    new_dw->getModifiable(new_vVelocity, d_lab->d_vVelocitySPBCLabel,
		          matlIndex, patch);
    new_dw->getModifiable(new_wVelocity, d_lab->d_wVelocitySPBCLabel,
		          matlIndex, patch);

    // Get the low and high index for the patch and the variables
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();

    //int pressure_celltypeval = pressureCellType();
    int pressure_celltypeval = -10;
    int outlet_celltypeval = outletCellType();

    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

    if (xminus) {
      int colX = idxLo.x();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
          IntVector currCell(colX, colY, colZ);
          IntVector xminusCell(colX-1, colY, colZ);
          IntVector yminusCell(colX, colY-1, colZ);
          IntVector zminusCell(colX, colY, colZ-1);
          if ((cellType[xminusCell] == pressure_celltypeval)||
              (cellType[xminusCell] == outlet_celltypeval)) {
            if (!(yminus && (colY == idxLo.y())))
              new_vVelocity[xminusCell] = vVelocity[currCell] +
		(uVelocity[currCell] - uVelocity[yminusCell]) *
		cellinfo->sew[colX]/cellinfo->sns[colY];
            if (!(zminus && (colZ == idxLo.z())))
              new_wVelocity[xminusCell] = wVelocity[currCell] +
		(uVelocity[currCell] - uVelocity[zminusCell]) *
		cellinfo->sew[colX]/cellinfo->stb[colZ];
          }
        }
      }
    }
    if (xplus) {
      int colX = idxHi.x();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
          IntVector currCell(colX, colY, colZ);
          IntVector xplusCell(colX+1, colY, colZ);
          IntVector xplusyminusCell(colX+1, colY-1, colZ);
          IntVector xpluszminusCell(colX+1, colY, colZ-1);
          if ((cellType[xplusCell] == pressure_celltypeval)||
              (cellType[xplusCell] == outlet_celltypeval)) {
            if (!(yminus && (colY == idxLo.y())))
              new_vVelocity[xplusCell] = vVelocity[currCell] -
		(uVelocity[xplusCell] - uVelocity[xplusyminusCell]) *
		cellinfo->sew[colX]/cellinfo->sns[colY];
            if (!(zminus && (colZ == idxLo.z())))
              new_wVelocity[xplusCell] = wVelocity[currCell] -
		(uVelocity[xplusCell] - uVelocity[xpluszminusCell]) *
		cellinfo->sew[colX]/cellinfo->stb[colZ];
          }
        }
      }
    }
    if (yminus) {
      int colY = idxLo.y();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          IntVector xminusCell(colX-1, colY, colZ);
          IntVector yminusCell(colX, colY-1, colZ);
          IntVector zminusCell(colX, colY, colZ-1);
          if ((cellType[yminusCell] == pressure_celltypeval)||
              (cellType[yminusCell] == outlet_celltypeval)) {
            if (!(xminus && (colX == idxLo.x())))
              new_uVelocity[yminusCell] = uVelocity[currCell] +
		(vVelocity[currCell] - vVelocity[xminusCell]) *
		cellinfo->sns[colY]/cellinfo->sew[colX];
            if (!(zminus && (colZ == idxLo.z())))
              new_wVelocity[yminusCell] = wVelocity[currCell] +
		(vVelocity[currCell] - vVelocity[zminusCell]) *
		cellinfo->sns[colY]/cellinfo->stb[colZ];
          }
        }
      }
    }
    if (yplus) {
      int colY = idxHi.y();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          IntVector yplusCell(colX, colY+1, colZ);
          IntVector yplusxminusCell(colX-1, colY+1, colZ);
          IntVector ypluszminusCell(colX, colY+1, colZ-1);
          if ((cellType[yplusCell] == pressure_celltypeval)||
              (cellType[yplusCell] == outlet_celltypeval)) {
            if (!(xminus && (colX == idxLo.x())))
              new_uVelocity[yplusCell] = uVelocity[currCell] -
		(vVelocity[yplusCell] - vVelocity[yplusxminusCell]) *
		cellinfo->sns[colY]/cellinfo->sew[colX];
            if (!(zminus && (colZ == idxLo.z())))
              new_wVelocity[yplusCell] = wVelocity[currCell] -
		(vVelocity[yplusCell] - vVelocity[ypluszminusCell]) *
		cellinfo->sns[colY]/cellinfo->stb[colZ];
          }
        }
      }
    }
    if (zminus) {
      int colZ = idxLo.z();
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          IntVector xminusCell(colX-1, colY, colZ);
          IntVector yminusCell(colX, colY-1, colZ);
          IntVector zminusCell(colX, colY, colZ-1);
          if ((cellType[zminusCell] == pressure_celltypeval)||
              (cellType[zminusCell] == outlet_celltypeval)) {
            if (!(xminus && (colX == idxLo.x())))
              new_uVelocity[zminusCell] = uVelocity[currCell] +
		(wVelocity[currCell] - wVelocity[xminusCell]) *
		cellinfo->stb[colZ]/cellinfo->sew[colX];
            if (!(yminus && (colY == idxLo.y())))
              new_vVelocity[zminusCell] = vVelocity[currCell] +
		(wVelocity[currCell] - wVelocity[yminusCell]) *
		cellinfo->stb[colZ]/cellinfo->sns[colY];
          }
        }
      }
    }
    if (zplus) {
      int colZ = idxHi.z();
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          IntVector zplusCell(colX, colY, colZ+1);
          IntVector zplusxminusCell(colX-1, colY, colZ+1);
          IntVector zplusyminusCell(colX, colY-1, colZ+1);
          if ((cellType[zplusCell] == pressure_celltypeval)||
              (cellType[zplusCell] == outlet_celltypeval)) {
            if (!(xminus && (colX == idxLo.x())))
              new_uVelocity[zplusCell] = uVelocity[currCell] -
		(wVelocity[zplusCell] - wVelocity[zplusxminusCell]) *
		cellinfo->stb[colZ]/cellinfo->sew[colX];
            if (!(yminus && (colY == idxLo.y())))
              new_vVelocity[zplusCell] = vVelocity[currCell] -
		(wVelocity[zplusCell] - wVelocity[zplusyminusCell]) *
		cellinfo->stb[colZ]/cellinfo->sns[colY];
          }
        }
      }
    }
  }
}
//****************************************************************************
// Schedule computation of mixture fraction flow rate
//****************************************************************************
void
BoundaryCondition::sched_getScalarFlowRate(SchedulerP& sched,
				      const PatchSet* patches,
				      const MaterialSet* matls)
{
  string taskname =  "BoundaryCondition::getScalarFlowRate";
  Task* tsk = scinew Task(taskname, this,
			  &BoundaryCondition::getScalarFlowRate);
  
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::AroundCells,
		Arches::ONEGHOSTCELL);

  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  if (d_carbon_balance)
    tsk->requires(Task::NewDW, d_lab->d_co2INLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  if (d_enthalpySolve)
    tsk->requires(Task::NewDW, d_lab->d_enthalpySPLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->computes(d_lab->d_scalarFlowRateLabel);
  if (d_carbon_balance)
    tsk->computes(d_lab->d_CO2FlowRateLabel);
  if (d_enthalpySolve)
    tsk->computes(d_lab->d_enthalpyFlowRateLabel);

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Get mixture fraction flow rate
//****************************************************************************
void 
BoundaryCondition::getScalarFlowRate(const ProcessorGroup* pc,
				const PatchSubset* patches,
				const MaterialSubset*,
				DataWarehouse*,
				DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    ArchesConstVariables constVars;

    constCCVariable<double> scalar;
    constCCVariable<double> co2;
    constCCVariable<double> enthalpy;

    new_dw->get(constVars.cellType, d_lab->d_cellTypeLabel,
		matlIndex, patch, Ghost::AroundCells,Arches::ONEGHOSTCELL);

    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(constVars.density, d_lab->d_densityCPLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constVars.uVelocity, d_lab->d_uVelocitySPBCLabel, matlIndex,
		patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constVars.vVelocity, d_lab->d_vVelocitySPBCLabel, matlIndex,
		patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constVars.wVelocity, d_lab->d_wVelocitySPBCLabel, matlIndex,
		patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(scalar, d_lab->d_scalarSPLabel, matlIndex,
		patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    double scalarIN = 0.0;
    double scalarOUT = 0.0;
    getVariableFlowRate(pc,patch, cellinfo, &constVars, scalar,
			&scalarIN, &scalarOUT); 

    new_dw->put(sum_vartype(scalarOUT-scalarIN), d_lab->d_scalarFlowRateLabel);

    double co2IN = 0.0;
    double co2OUT = 0.0;
    if (d_carbon_balance) {
      new_dw->get(co2, d_lab->d_co2INLabel, matlIndex,
		  patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    getVariableFlowRate(pc,patch, cellinfo, &constVars, co2,
			&co2IN, &co2OUT); 
      new_dw->put(sum_vartype(co2OUT-co2IN), d_lab->d_CO2FlowRateLabel);
    }
    double enthalpyIN = 0.0;
    double enthalpyOUT = 0.0;
    if (d_enthalpySolve) {
      new_dw->get(enthalpy, d_lab->d_enthalpySPLabel, matlIndex,
		  patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    getVariableFlowRate(pc,patch, cellinfo, &constVars, enthalpy,
			&enthalpyIN, &enthalpyOUT); 
      new_dw->put(sum_vartype(enthalpyOUT-enthalpyIN), d_lab->d_enthalpyFlowRateLabel);
    }
  }
}

//****************************************************************************
// Schedule scalar efficiency computation
//****************************************************************************
void BoundaryCondition::sched_getScalarEfficiency(SchedulerP& sched,
					      const PatchSet* patches,
					      const MaterialSet* matls)
{
  string taskname =  "BoundaryCondition::getScalarEfficiency";
  Task* tsk = scinew Task(taskname, this,
			  &BoundaryCondition::getScalarEfficiency);
  
  for (int ii = 0; ii < d_numInlets; ii++) {
    tsk->requires(Task::OldDW, d_flowInlets[ii].d_flowRate_label);
    tsk->computes(d_flowInlets[ii].d_flowRate_label);
  }

  tsk->requires(Task::NewDW, d_lab->d_scalarFlowRateLabel);
  tsk->computes(d_lab->d_scalarEfficiencyLabel);

  if (d_carbon_balance) {
    tsk->requires(Task::NewDW, d_lab->d_CO2FlowRateLabel);
    tsk->computes(d_lab->d_carbonEfficiencyLabel);
  }
  if (d_enthalpySolve) {
    tsk->requires(Task::NewDW, d_lab->d_enthalpyFlowRateLabel);
    tsk->requires(Task::NewDW, d_lab->d_totalRadSrcLabel);
    tsk->computes(d_lab->d_normTotalRadSrcLabel);
    tsk->computes(d_lab->d_enthalpyEfficiencyLabel);
  }

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Correct outlet velocity
//****************************************************************************
void 
BoundaryCondition::getScalarEfficiency(const ProcessorGroup* pc,
			      const PatchSubset* ,
			      const MaterialSubset*,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw)
{
    sum_vartype sum_scalarFlowRate, sum_CO2FlowRate, sum_enthalpyFlowRate;
    sum_vartype sum_totalRadSrc;
    delt_vartype flowRate;
    double scalarFlowRate = 0.0;
    double CO2FlowRate = 0.0;
    double enthalpyFlowRate = 0.0;
    double totalFlowRate = 0.0;
    double totalCarbonFlowRate = 0.0;
    double totalEnthalpyFlowRate = 0.0;
    double scalarEfficiency = 0.0;
    double carbonEfficiency = 0.0;
    double enthalpyEfficiency = 0.0;
    double totalRadSrc = 0.0;
    double normTotalRadSrc = 0.0;

    new_dw->get(sum_scalarFlowRate, d_lab->d_scalarFlowRateLabel);
    scalarFlowRate = sum_scalarFlowRate;
    if (d_carbon_balance) {
      new_dw->get(sum_CO2FlowRate, d_lab->d_CO2FlowRateLabel);
      CO2FlowRate = sum_CO2FlowRate;
    }
    if (d_enthalpySolve) {
      new_dw->get(sum_enthalpyFlowRate, d_lab->d_enthalpyFlowRateLabel);
      enthalpyFlowRate = sum_enthalpyFlowRate;
      new_dw->get(sum_totalRadSrc, d_lab->d_totalRadSrcLabel);
      totalRadSrc = sum_totalRadSrc;
    }
    for (int indx = 0; indx < d_numInlets; indx++) {
      FlowInlet fi = d_flowInlets[indx];
      old_dw->get(flowRate, d_flowInlets[indx].d_flowRate_label);
      d_flowInlets[indx].flowRate = flowRate;
      fi.flowRate = flowRate;
      new_dw->put(flowRate, d_flowInlets[indx].d_flowRate_label);
      double scalarValue = fi.streamMixturefraction.d_mixVars[0];
      if (scalarValue > 0.0)
	  totalFlowRate += fi.flowRate;
      if ((d_carbon_balance)&&(scalarValue > 0.0))
	    totalCarbonFlowRate += fi.flowRate * fi.fcr;
      if ((d_enthalpySolve)&&(scalarValue > 0.0))
	    totalEnthalpyFlowRate += fi.flowRate * fi.calcStream.getEnthalpy();
    }
    if (totalFlowRate > 0.0)
      scalarEfficiency = scalarFlowRate / totalFlowRate;
    else 
      cout << "WARNING! No mixture fraction in the domain." << endl;
    new_dw->put(delt_vartype(scalarEfficiency), d_lab->d_scalarEfficiencyLabel);

    if (d_carbon_balance) {
      if (totalCarbonFlowRate > 0.0)
	carbonEfficiency = CO2FlowRate * 12.0/28.0 /totalCarbonFlowRate;
      else 
	throw InvalidValue("No carbon in the domain");
      new_dw->put(delt_vartype(carbonEfficiency), d_lab->d_carbonEfficiencyLabel);
    }
    if (d_enthalpySolve) {
      if (totalEnthalpyFlowRate < 0.0) {
	enthalpyEfficiency = enthalpyFlowRate/totalEnthalpyFlowRate;
	normTotalRadSrc = totalRadSrc/totalEnthalpyFlowRate;
	enthalpyEfficiency -= normTotalRadSrc;
      }
      else 
	throw InvalidValue("No enthalpy in the domain");
      new_dw->put(delt_vartype(enthalpyEfficiency), d_lab->d_enthalpyEfficiencyLabel);
      new_dw->put(delt_vartype(normTotalRadSrc), d_lab->d_normTotalRadSrcLabel);
    }
 
}
//****************************************************************************
// Get boundary flow rate for a given variable
//****************************************************************************
void 
BoundaryCondition::getVariableFlowRate(const ProcessorGroup*,
			               const Patch* patch,
			               CellInformation* cellinfo,
			               ArchesConstVariables* constvars,
				       constCCVariable<double> balance_var,
				       double* varIN, double* varOUT) 
{
    // Get the low and high index for the patch and the variables
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();

    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

    if (xminus||xplus||yminus||yplus||zminus||zplus) {

      double flowIN = 0.0;
      double flowOUT = 0.0;
      bool doing_balance = true;
      for (int indx = 0; indx < d_numInlets; indx++) {

	// Get a copy of the current flow inlet
	// assign flowType the value that corresponds to flow
	//CellTypeInfo flowType = FLOW;
	FlowInlet fi = d_flowInlets[indx];
	double varIN_inlet = 0.0;
	double varOUT_inlet = 0.0;
	fort_inlpresbcinout(constvars->uVelocity, constvars->vVelocity,
			   constvars->wVelocity, idxLo, idxHi,
			   constvars->density, constvars->cellType,
			   fi.d_cellTypeID,
			   flowIN, flowOUT, cellinfo->sew, cellinfo->sns,
			   cellinfo->stb, xminus, xplus, yminus, yplus,
			   zminus, zplus, doing_balance,
			   balance_var, varIN_inlet, varOUT_inlet);

	if (varOUT_inlet > 0.0)
		throw InvalidValue("Balance variable comming out of inlet");

	// Count balance variable comming through the air inlet
	double scalarValue = fi.streamMixturefraction.d_mixVars[0];
	if (scalarValue == 0.0)
	  *varIN += varIN_inlet;
      } 

      if (d_pressureBoundary) {
	double varIN_bc = 0.0;
	double varOUT_bc = 0.0;
	int pressure_celltypeval = d_pressureBC->d_cellTypeID;
	fort_inlpresbcinout(constvars->uVelocity, constvars->vVelocity,
			   constvars->wVelocity, idxLo, idxHi,
			   constvars->density, constvars->cellType,
			   pressure_celltypeval,
			   flowIN, flowOUT, cellinfo->sew, cellinfo->sns,
			   cellinfo->stb, xminus, xplus, yminus, yplus,
			   zminus, zplus, doing_balance,
			   balance_var, varIN_bc, varOUT_bc);
	*varIN += varIN_bc;
	*varOUT += varOUT_bc;
      }
      if (d_outletBoundary) {
	double varIN_bc = 0.0;
	double varOUT_bc = 0.0;
	int outlet_celltypeval = d_outletBC->d_cellTypeID;
	fort_inlpresbcinout(constvars->uVelocity, constvars->vVelocity,
			   constvars->wVelocity, idxLo, idxHi,
			   constvars->density, constvars->cellType,
			   outlet_celltypeval,
			   flowIN, flowOUT, cellinfo->sew, cellinfo->sns,
			   cellinfo->stb, xminus, xplus, yminus, yplus,
			   zminus, zplus, doing_balance,
			   balance_var, varIN_bc, varOUT_bc);
	*varIN += varIN_bc;
	*varOUT += varOUT_bc;
      }
    }  
}

//****************************************************************************
// schedule copy of inlet flow rates for nosolve
//****************************************************************************
void BoundaryCondition::sched_setInletFlowRates(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls)
{
  string taskname =  "BoundaryCondition::setInletFlowRates";
  Task* tsk = scinew Task(taskname, this,
			  &BoundaryCondition::setInletFlowRates);
  
  for (int ii = 0; ii < d_numInlets; ii++) {
    tsk->requires(Task::OldDW, d_flowInlets[ii].d_flowRate_label);
    tsk->computes(d_flowInlets[ii].d_flowRate_label);
  }

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// copy inlet flow rates for nosolve
//****************************************************************************
void 
BoundaryCondition::setInletFlowRates(const ProcessorGroup* pc,
				     const PatchSubset* ,
				     const MaterialSubset*,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw)
{
  delt_vartype flowRate;
  for (int indx = 0; indx < d_numInlets; indx++) {
    FlowInlet fi = d_flowInlets[indx];
    old_dw->get(flowRate, d_flowInlets[indx].d_flowRate_label);
    d_flowInlets[indx].flowRate = flowRate;
    fi.flowRate = flowRate;
    new_dw->put(flowRate, d_flowInlets[indx].d_flowRate_label);
  }
}
