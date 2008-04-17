//----- BoundaryCondition.cc ----------------------------------------------

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
#include <Packages/Uintah/CCA/Components/Arches/ExtraScalarSolver.h>
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
#include <Packages/Uintah/Core/Exceptions/VariableNotFoundInGrid.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Containers/StaticArray.h>
#include <iostream>
#include <sstream>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

#include <Packages/Uintah/CCA/Components/Arches/fortran/celltypeInit_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/areain_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/profscalar_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/inlbcs_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/inlpresbcinout_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcscalar_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcuvel_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcvvel_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcwvel_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcpress_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/profv_fort.h>
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
#include <Packages/Uintah/CCA/Components/Arches/fortran/get_ramping_factor_fort.h>

//****************************************************************************
// Constructor for BoundaryCondition
//****************************************************************************
BoundaryCondition::BoundaryCondition(const ArchesLabel* label,
                                     const MPMArchesLabel* MAlb,
                                     PhysicalConstants* phys_const,
                                     Properties* props,
                                     bool calcReactScalar,
                                     bool calcEnthalpy,
                                     bool calcVariance):
                                     d_lab(label), d_MAlab(MAlb),
                                     d_physicalConsts(phys_const), 
                                     d_props(props),
                                     d_reactingScalarSolve(calcReactScalar),
                                     d_enthalpySolve(calcEnthalpy),
                                     d_calcVariance(calcVariance)
{
  MM_CUTOFF_VOID_FRAC = 0.5;
  d_wallBdry = 0;
  d_pressureBC = 0;
  d_outletBC = 0;
  d_doAreaCalcforSourceBoundaries = 0;
}

//****************************************************************************
// Destructor
//****************************************************************************
BoundaryCondition::~BoundaryCondition()
{
  delete d_wallBdry;
  delete d_pressureBC;
  delete d_outletBC;
  for (int ii = 0; ii < d_numInlets; ii++)
    delete d_flowInlets[ii];
  if (d_calcExtraScalars)
    for (int i=0; i < static_cast<int>(d_extraScalarBCs.size()); i++)
      delete d_extraScalarBCs[i];
	  
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
  d_numSourceBoundaries = 0;
  int total_cellTypes = 0;
  
  db->getWithDefault("carbon_balance", d_carbon_balance, false);
  db->getWithDefault("sulfur_balance", d_sulfur_balance, false);
  db->getWithDefault("carbon_balance_es", d_carbon_balance_es, false);
  db->getWithDefault("sulfur_balance_es", d_sulfur_balance_es, false);

 //--- instrusions with boundary sources -----
 if (ProblemSpecP intrusionbcs_db = db->findBlock("IntrusionWithBCSource")){

	for (ProblemSpecP intrusionbcs_db = db->findBlock("IntrusionWithBCSource");
		 intrusionbcs_db != 0; intrusionbcs_db = intrusionbcs_db->findNextBlock("IntrusionWithBCSource")){
				 
			d_sourceBoundaryInfo.push_back(scinew BCSourceInfo(d_calcVariance, d_reactingScalarSolve));
			d_sourceBoundaryInfo[d_numSourceBoundaries]->problemSetup(intrusionbcs_db);
			if (d_sourceBoundaryInfo[d_numSourceBoundaries]->doAreaCalc)
				d_doAreaCalcforSourceBoundaries = true; //there has to be a better way
					
			//compute the density and other properties for this inlet stream
			d_sourceBoundaryInfo[d_numSourceBoundaries]->streamMixturefraction.d_initEnthalpy = true;
			d_sourceBoundaryInfo[d_numSourceBoundaries]->streamMixturefraction.d_scalarDisp=0.0;
			d_props->computeInletProperties(d_sourceBoundaryInfo[d_numSourceBoundaries]->streamMixturefraction,
											d_sourceBoundaryInfo[d_numSourceBoundaries]->calcStream);
				

			++d_numSourceBoundaries;

	}
 }

  if (ProblemSpecP inlet_db = db->findBlock("FlowInlet")) {
    d_inletBoundary = true;
    for (ProblemSpecP inlet_db = db->findBlock("FlowInlet");
         inlet_db != 0; inlet_db = inlet_db->findNextBlock("FlowInlet")) {
      d_flowInlets.push_back(scinew FlowInlet(total_cellTypes, d_calcVariance,
                                              d_reactingScalarSolve));
      d_flowInlets[d_numInlets]->problemSetup(inlet_db);
      // compute density and other dependent properties
      d_flowInlets[d_numInlets]->streamMixturefraction.d_initEnthalpy=true;
      d_flowInlets[d_numInlets]->streamMixturefraction.d_scalarDisp=0.0;
      d_props->computeInletProperties(
                        d_flowInlets[d_numInlets]->streamMixturefraction,
                        d_flowInlets[d_numInlets]->calcStream);
      double f = d_flowInlets[d_numInlets]->streamMixturefraction.d_mixVars[0];
      if (f > 0.0)
        d_flowInlets[d_numInlets]->fcr = d_props->getCarbonContent(f);
      if (d_calcExtraScalars) {
        ProblemSpecP extra_scalar_db = inlet_db->findBlock("ExtraScalars");
        for (int i=0; i < static_cast<int>(d_extraScalars->size()); i++) {
          double value;
          string name = d_extraScalars->at(i)->getScalarName();
          extra_scalar_db->require(name, value);
          d_extraScalarBC* bc = scinew d_extraScalarBC;
          bc->d_scalar_name = name;
          bc->d_scalarBC_value = value;
          bc->d_BC_ID = total_cellTypes;
          d_extraScalarBCs.push_back(bc);
        }
      }
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
    d_pressureBC = scinew PressureInlet(total_cellTypes, d_calcVariance,
                                        d_reactingScalarSolve);
    d_pressureBC->problemSetup(press_db);
    // compute density and other dependent properties
    d_pressureBC->streamMixturefraction.d_initEnthalpy=true;
    d_pressureBC->streamMixturefraction.d_scalarDisp=0.0;
    d_props->computeInletProperties(d_pressureBC->streamMixturefraction, 
                                    d_pressureBC->calcStream);
    if (d_calcExtraScalars) {
      ProblemSpecP extra_scalar_db = press_db->findBlock("ExtraScalars");
      for (int i=0; i < static_cast<int>(d_extraScalars->size()); i++) {
        double value;
        string name = d_extraScalars->at(i)->getScalarName();
        extra_scalar_db->require(name, value);
        d_extraScalarBC* bc = scinew d_extraScalarBC;
        bc->d_scalar_name = name;
        bc->d_scalarBC_value = value;
        bc->d_BC_ID = total_cellTypes;
        d_extraScalarBCs.push_back(bc);
      }
    }
    ++total_cellTypes;
  }
  else {
    cout << "Pressure boundary not specified" << endl;
    d_pressureBoundary = false;
  }
  
  if (ProblemSpecP outlet_db = db->findBlock("OutletBC")) {
    d_outletBoundary = true;
    d_outletBC = scinew FlowOutlet(total_cellTypes, d_calcVariance,
                                   d_reactingScalarSolve);
    d_outletBC->problemSetup(outlet_db);
    // compute density and other dependent properties
    d_outletBC->streamMixturefraction.d_initEnthalpy=true;
    d_outletBC->streamMixturefraction.d_scalarDisp=0.0;
    d_props->computeInletProperties(d_outletBC->streamMixturefraction, 
                                    d_outletBC->calcStream);
    if (d_calcExtraScalars) {
      ProblemSpecP extra_scalar_db = outlet_db->findBlock("ExtraScalars");
      for (int i=0; i < static_cast<int>(d_extraScalars->size()); i++) {
        double value;
        string name = d_extraScalars->at(i)->getScalarName();
        extra_scalar_db->require(name, value);
        d_extraScalarBC* bc = scinew d_extraScalarBC;
        bc->d_scalar_name = name;
        bc->d_scalarBC_value = value;
        bc->d_BC_ID = total_cellTypes;
        d_extraScalarBCs.push_back(bc);
      }
    }
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
  if ((d_MAlab)&&(d_intrusionBoundary))
    d_mmWallID = d_intrusionBC->d_cellTypeID;

  //adding mms access
  if (d_doMMS) {

    ProblemSpecP params_non_constant = params;
    const ProblemSpecP params_root = params_non_constant->getRootNode();
    ProblemSpecP db_mmsblock=params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("MMS");
    
    db_mmsblock->getWithDefault("whichMMS",d_mms,"constantMMS");

    if (d_mms == "constantMMS") {
      ProblemSpecP db_whichmms = db_mmsblock->findBlock("constantMMS");
      db_whichmms->getWithDefault("cu",cu,1.0);
      db_whichmms->getWithDefault("cv",cv,1.0);
      db_whichmms->getWithDefault("cw",cw,1.0);
      db_whichmms->getWithDefault("cp",cp,1.0);
      db_whichmms->getWithDefault("phi0",phi0,0.5);
    }
    else if (d_mms == "gao1MMS") {
      ProblemSpecP db_whichmms = db_mmsblock->findBlock("gao1MMS");
      db_whichmms->require("rhoair", d_airDensity);
      db_whichmms->require("rhohe", d_heDensity);
      db_whichmms->require("gravity", d_gravity);//Vector
      db_whichmms->require("viscosity",d_viscosity); 
      db_whichmms->getWithDefault("turbulentPrandtlNumber",d_turbPrNo,0.4);
      db_whichmms->getWithDefault("cu",cu,1.0);
      db_whichmms->getWithDefault("cv",cv,1.0);
      db_whichmms->getWithDefault("cw",cw,1.0);
      db_whichmms->getWithDefault("cp",cp,1.0);
      db_whichmms->getWithDefault("phi0",phi0,0.5);
    }
    else if (d_mms == "thornock1MMS") {
      ProblemSpecP db_whichmms = db_mmsblock->findBlock("thornock1MMS");
      db_whichmms->require("cu",cu);
    }
    else if (d_mms == "almgrenMMS") {
      ProblemSpecP db_whichmms = db_mmsblock->findBlock("almgrenMMS");
      db_whichmms->getWithDefault("amplitude",amp,0.0);
      db_whichmms->require("viscosity",d_viscosity);
    }
    else
      throw InvalidValue("current MMS "
                         "not supported: " + d_mms, __FILE__, __LINE__);
  }

}

//****************************************************************************
// schedule the initialization of cell types
//****************************************************************************
void 
BoundaryCondition::sched_cellTypeInit(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls)
{
  // cell type initialization
  Task* tsk = scinew Task("BoundaryCondition::cellTypeInit",
                          this, &BoundaryCondition::cellTypeInit);
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

    int celltypeval;
    // initialization for pressure boundary
    if (d_pressureBoundary) {
      int nofGeomPieces = (int)d_pressureBC->d_geomPiece.size();
      for (int ii = 0; ii < nofGeomPieces; ii++) {
        GeometryPieceP  piece = d_pressureBC->d_geomPiece[ii];
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
    // wall boundary type
    if (d_wallBoundary) {
      int nofGeomPieces = (int)d_wallBdry->d_geomPiece.size();
      for (int ii = 0; ii < nofGeomPieces; ii++) {
        GeometryPieceP  piece = d_wallBdry->d_geomPiece[ii];
        Box geomBox = piece->getBoundingBox();
        Box b = geomBox.intersect(patchBox);
        // check for another geometry
        if (!(b.degenerate())) {
          /*CellIterator iter = patch->getCellCenterIterator(b);
          IntVector idxLo = iter.begin();
          IntVector idxHi = iter.end() - IntVector(1,1,1);
          celltypeval = d_wallBdry->d_cellTypeID;
          fort_celltypeinit(idxLo, idxHi, cellType, celltypeval);*/
          for (CellIterator iter = patch->getCellCenterIterator(b);
               !iter.done(); iter++) {
            Point p = patch->cellPosition(*iter);
            if (piece->inside(p)) 
            cellType[*iter] = d_wallBdry->d_cellTypeID;
          }
        }
      }
    }
    // initialization for outlet boundary
    if (d_outletBoundary) {
      int nofGeomPieces = (int)d_outletBC->d_geomPiece.size();
      for (int ii = 0; ii < nofGeomPieces; ii++) {
        GeometryPieceP  piece = d_outletBC->d_geomPiece[ii];
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
    // set boundary type for inlet flow field
    if (d_inletBoundary) {
      for (int ii = 0; ii < d_numInlets; ii++) {
        int nofGeomPieces = (int)d_flowInlets[ii]->d_geomPiece.size();
        for (int jj = 0; jj < nofGeomPieces; jj++) {
          GeometryPieceP  piece = d_flowInlets[ii]->d_geomPiece[jj];
          Box geomBox = piece->getBoundingBox();
          Box b = geomBox.intersect(patchBox);
          // check for another geometry
          if (b.degenerate())
            continue; // continue the loop for other inlets
            // iterates thru box b, converts from geometry space to index space
            // make sure this works
          /*CellIterator iter = patch->getCellCenterIterator(b);
          IntVector idxLo = iter.begin();
          IntVector idxHi = iter.end() - IntVector(1,1,1);
          celltypeval = d_flowInlets[ii].d_cellTypeID;
          fort_celltypeinit(idxLo, idxHi, cellType, celltypeval);*/
          for (CellIterator iter = patch->getCellCenterIterator(b);
               !iter.done(); iter++) {
            Point p = patch->cellPosition(*iter);
            if (piece->inside(p)) 
              cellType[*iter] = d_flowInlets[ii]->d_cellTypeID;
          }
        }
      }
    }
    if (d_intrusionBoundary) {
      Box patchInteriorBox = patch->getInteriorBox();
      int nofGeomPieces = (int)d_intrusionBC->d_geomPiece.size();
      for (int ii = 0; ii < nofGeomPieces; ii++) {
        GeometryPieceP  piece = d_intrusionBC->d_geomPiece[ii];
        Box geomBox = piece->getBoundingBox();
        Box b = geomBox.intersect(patchInteriorBox);
        if (!(b.degenerate())) {
          for (CellIterator iter = patch->getCellCenterIterator(b);
               !iter.done(); iter++) {
            Point p = patch->cellPosition(*iter);
            if (piece->inside(p)) 
              cellType[*iter] = d_intrusionBC->d_cellTypeID;
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
BoundaryCondition::sched_mmWallCellTypeInit(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls,
                                            bool fixCellType)
{
  // cell type initialization
  Task* tsk = scinew Task("BoundaryCondition::mmWallCellTypeInit",
                          this,
                          &BoundaryCondition::mmWallCellTypeInit,
                          fixCellType);
  
  int numGhostcells = 0;

  // New DW warehouse variables to calculate cell types if we are
  // recalculating cell types and resetting void fractions

  //  double time = d_lab->d_sharedState->getElapsedTime();
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

    tsk->requires(Task::OldDW, d_lab->d_cellTypeLabel, 
                  Ghost::None, numGhostcells);
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

    //    double time = d_lab->d_sharedState->getElapsedTime();
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

      if (d_intrusionBoundary) {
        for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
          for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
            for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
              IntVector currCell(colX, colY, colZ);

              if (cellType[currCell] == d_mmWallID) {
                mmGasVolFrac[currCell] = 0.0;
                voidFracMPM[currCell] = 0.0;
              }
              else {
                mmGasVolFrac[currCell] = 1.0;
                voidFracMPM[currCell] = 1.0;
              }
            }
          }
        }
      }
      else {
        fort_mmcelltypeinit(idxLo, idxHi, mmGasVolFrac, mmCellType, d_mmWallID,
      			    d_flowfieldCellTypeVal, MM_CUTOFF_VOID_FRAC);  

        fort_mmcelltypeinit(idxLo, idxHi, voidFracMPM, mmCellTypeMPM, d_mmWallID,
      			    d_flowfieldCellTypeVal, MM_CUTOFF_VOID_FRAC);  
        if (d_cutCells)
	  fort_mmcelltypeinit(idxLo, idxHi, voidFracCutCell, mmCellTypeCutCell, d_mmWallID,
			      d_flowfieldCellTypeVal, MM_CUTOFF_VOID_FRAC);  
      }
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
    else 
      throw VariableNotFoundInGrid("cellInformation"," ", __FILE__, __LINE__);
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    
    // Get the low and high index for the variable and the patch
    IntVector domLo = cellType.getFortLowIndex();
    IntVector domHi = cellType.getFortHighIndex();
    
    // Get the geometry of the patch
    Box patchBox = patch->getBox();
    
    // Go thru the number of inlets
    for (int ii = 0; ii < d_numInlets; ii++) {
      
      // Loop thru the number of geometry pieces in each inlet
      int nofGeomPieces = (int)d_flowInlets[ii]->d_geomPiece.size();
      for (int jj = 0; jj < nofGeomPieces; jj++) {
	
	// Intersect the geometry piece with the patch box
	GeometryPieceP  piece = d_flowInlets[ii]->d_geomPiece[jj];
	Box geomBox = piece->getBoundingBox();
	Box b = geomBox.intersect(patchBox);
	// check for another geometry
	if (b.degenerate()){
	  new_dw->put(sum_vartype(0),d_flowInlets[ii]->d_area_label);
	  continue; // continue the loop for other inlets
	}
	
	// iterates thru box b, converts from geometry space to index space
	// make sure this works
	CellIterator iter = patch->getCellCenterIterator(b);
	IntVector idxLo = iter.begin();
	IntVector idxHi = iter.end() - IntVector(1,1,1);
	
	// Calculate the inlet area
	double inlet_area;
	int cellid = d_flowInlets[ii]->d_cellTypeID;
	
	bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
	bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
	bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
	bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
	bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
	bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

	fort_areain(domLo, domHi, idxLo, idxHi, cellinfo->sew, cellinfo->sns,
		    cellinfo->stb, inlet_area, cellType, cellid,
		    d_flowfieldCellTypeVal,
		    xminus, xplus, yminus, yplus, zminus, zplus);
	
	// Write the inlet area to the old_dw
	new_dw->put(sum_vartype(inlet_area),d_flowInlets[ii]->d_area_label);
      }
    }
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
  Task* tsk = scinew Task("BoundaryCondition::calculateArea",
		       this,
		       &BoundaryCondition::computeInletFlowArea);

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  // ***warning checkpointing
  //      tsk->computes(old_dw, d_lab->d_cellInfoLabel, matlIndex, patch);
  for (int ii = 0; ii < d_numInlets; ii++) 
    tsk->computes(d_flowInlets[ii]->d_area_label);

  sched->addTask(tsk, patches, matls);
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("BoundaryCondition::calculateArea",
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
			  &BoundaryCondition::setProfile);

  // This task requires cellTypeVariable and areaLabel for inlet boundary
  // Also densityIN, [u,v,w] velocityIN, scalarIN
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  for (int ii = 0; ii < d_numInlets; ii++) {
    tsk->requires(Task::NewDW, d_flowInlets[ii]->d_area_label);
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

  tsk->modifies(d_lab->d_scalarSPLabel);

  for (int ii = 0; ii < d_numInlets; ii++) 
    tsk->computes(d_flowInlets[ii]->d_flowRate_label);

  if (d_calcExtraScalars)
    for (int i=0; i < static_cast<int>(d_extraScalars->size()); i++)
      tsk->modifies(d_extraScalars->at(i)->getScalarLabel());

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Actually set flat profile at flow inlet boundary
//****************************************************************************
void 
BoundaryCondition::setProfile(const ProcessorGroup* /*pc*/,
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
    CCVariable<double> scalar;
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
    new_dw->getModifiable(scalar, d_lab->d_scalarSPLabel, matlIndex, patch);
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
        new_dw->get(area_var, d_flowInlets[indx]->d_area_label);
        double area = area_var;
	double actual_flow_rate;
      
        // Get a copy of the current flow inlet
        // check if given patch intersects with the inlet boundary of type index
        FlowInlet* fi = d_flowInlets[indx];
        //cerr << " inlet area" << area << " flowrate" << fi.flowRate << endl;
        //cerr << "density=" << fi.calcStream.d_density << endl;
        fort_profv(uVelocity, vVelocity, wVelocity, idxLo, idxHi,
		   cellType, area, fi->d_cellTypeID, fi->flowRate, fi->inletVel,
		   fi->calcStream.d_density,
		   xminus, xplus, yminus, yplus, zminus, zplus, time,
		   fi->d_ramping_inlet_flowrate, actual_flow_rate);

	d_flowInlets[indx]->flowRate = actual_flow_rate;
	new_dw->put(delt_vartype(actual_flow_rate),
		    d_flowInlets[indx]->d_flowRate_label);

        fort_profscalar(idxLo, idxHi, density, cellType,
		        fi->calcStream.d_density, fi->d_cellTypeID,
		        xminus, xplus, yminus, yplus, zminus, zplus);
        if (d_enthalpySolve)
        fort_profscalar(idxLo, idxHi, enthalpy, cellType,
		        fi->calcStream.d_enthalpy, fi->d_cellTypeID,
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

    if (d_inletBoundary) {
      for (int ii = 0; ii < d_numInlets; ii++) {
        double scalarValue = 
      	 d_flowInlets[ii]->streamMixturefraction.d_mixVars[0];
        fort_profscalar(idxLo, idxHi, scalar, cellType,
      		  scalarValue, d_flowInlets[ii]->d_cellTypeID,
      		  xminus, xplus, yminus, yplus, zminus, zplus);
        double reactScalarValue;
        if (d_reactingScalarSolve) {
          reactScalarValue = 
      	 d_flowInlets[ii]->streamMixturefraction.d_rxnVars[0];
          fort_profscalar(idxLo, idxHi, reactscalar, cellType,
      		    reactScalarValue, d_flowInlets[ii]->d_cellTypeID,
      		    xminus, xplus, yminus, yplus, zminus, zplus);
        }
      }
    }

    if (d_pressureBoundary) {
      double scalarValue = 
             d_pressureBC->streamMixturefraction.d_mixVars[0];
      fort_profscalar(idxLo, idxHi, scalar, cellType, scalarValue,
      		d_pressureBC->d_cellTypeID,
      		xminus, xplus, yminus, yplus, zminus, zplus);
      double reactScalarValue;
      if (d_reactingScalarSolve) {
        reactScalarValue = 
             d_pressureBC->streamMixturefraction.d_rxnVars[0];
        fort_profscalar(idxLo, idxHi, reactscalar, cellType,
      		  reactScalarValue, d_pressureBC->d_cellTypeID,
      		  xminus, xplus, yminus, yplus, zminus, zplus);
      }
    }

    if (d_outletBoundary) {
      double scalarValue = 
             d_outletBC->streamMixturefraction.d_mixVars[0];
      fort_profscalar(idxLo, idxHi, scalar, cellType, scalarValue,
      		d_outletBC->d_cellTypeID,
      		xminus, xplus, yminus, yplus, zminus, zplus);
      double reactScalarValue;
      if (d_reactingScalarSolve) {
        reactScalarValue = 
             d_outletBC->streamMixturefraction.d_rxnVars[0];
        fort_profscalar(idxLo, idxHi, reactscalar, cellType,
      		  reactScalarValue, d_outletBC->d_cellTypeID,
      		  xminus, xplus, yminus, yplus, zminus, zplus);
      }
    }
    uVelRhoHat.copyData(uVelocity); 
    vVelRhoHat.copyData(vVelocity); 
    wVelRhoHat.copyData(wVelocity); 

    if (d_calcExtraScalars) {
      for (int i=0; i < static_cast<int>(d_extraScalars->size()); i++) {
        string extra_scalar_name = d_extraScalars->at(i)->getScalarName();
        CCVariable<double> extra_scalar;
        new_dw->getModifiable(extra_scalar,
                              d_extraScalars->at(i)->getScalarLabel(),
                               matlIndex, patch);

        if (d_inletBoundary) {
          for (int ii = 0; ii < d_numInlets; ii++) {
            int BC_ID = d_flowInlets[ii]->d_cellTypeID;

            double extra_scalar_value;
            for (int j=0; j < static_cast<int>(d_extraScalarBCs.size()); j++)
              if ((d_extraScalarBCs[j]->d_scalar_name == extra_scalar_name)&&
                  (d_extraScalarBCs[j]->d_BC_ID) == BC_ID)
                extra_scalar_value = d_extraScalarBCs[j]->d_scalarBC_value;

            fort_profscalar(idxLo, idxHi, extra_scalar, cellType,
      		            extra_scalar_value, BC_ID,
      		            xminus, xplus, yminus, yplus, zminus, zplus);
          }
        }

        if (d_pressureBoundary) {
          int BC_ID = d_pressureBC->d_cellTypeID;

          double extra_scalar_value;
          for (int j=0; j < static_cast<int>(d_extraScalarBCs.size()); j++)
            if ((d_extraScalarBCs[j]->d_scalar_name == extra_scalar_name)&&
                (d_extraScalarBCs[j]->d_BC_ID) == BC_ID)
              extra_scalar_value = d_extraScalarBCs[j]->d_scalarBC_value;

          fort_profscalar(idxLo, idxHi, extra_scalar, cellType,
                          extra_scalar_value, BC_ID,
      		          xminus, xplus, yminus, yplus, zminus, zplus);
        }

        if (d_outletBoundary) {
          int BC_ID = d_outletBC->d_cellTypeID;

          double extra_scalar_value;
          for (int j=0; j < static_cast<int>(d_extraScalarBCs.size()); j++)
            if ((d_extraScalarBCs[j]->d_scalar_name == extra_scalar_name)&&
                (d_extraScalarBCs[j]->d_BC_ID) == BC_ID)
              extra_scalar_value = d_extraScalarBCs[j]->d_scalarBC_value;

          fort_profscalar(idxLo, idxHi, extra_scalar, cellType,
                          extra_scalar_value, BC_ID,
      		          xminus, xplus, yminus, yplus, zminus, zplus);
        }
      }
    }
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
  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getSFCXFORTLowIndex();
  IntVector idxHi = patch->getSFCXFORTHighIndex();

  // computes momentum source term due to wall
  // uses total viscosity for wall source, not just molecular viscosity
  //double molViscosity = d_physicalConsts->getMolecularViscosity();
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  fort_bcuvel(idxLo, idxHi,
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
	      constvars->cellType, wall_celltypeval, 
	      cellinfo->sewu, cellinfo->sns, cellinfo->stb,
	      constvars->viscosity,
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
  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getSFCYFORTLowIndex();
  IntVector idxHi = patch->getSFCYFORTHighIndex();

  // computes momentum source term due to wall
  // uses total viscosity for wall source, not just molecular viscosity
  //double molViscosity = d_physicalConsts->getMolecularViscosity();
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  fort_bcvvel(idxLo, idxHi,
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
	      constvars->cellType, wall_celltypeval,
	      cellinfo->sew, cellinfo->snsv, cellinfo->stb,
	      constvars->viscosity,
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
  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getSFCZFORTLowIndex();
  IntVector idxHi = patch->getSFCZFORTHighIndex();

  // computes momentum source term due to wall
  // uses total viscosity for wall source, not just molecular viscosity
  //double molViscosity = d_physicalConsts->getMolecularViscosity();
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  fort_bcwvel(idxLo, idxHi,
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
	      constvars->cellType, wall_celltypeval,
	      cellinfo->sew, cellinfo->sns, cellinfo->stbw,
	      constvars->viscosity,
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
	       d_flowInlets[ii]->d_cellTypeID,
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
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
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
  fort_bcscalar(idxLo, idxHi,
		vars->scalarCoeff[Arches::AE],
                vars->scalarCoeff[Arches::AW],
		vars->scalarCoeff[Arches::AN],
                vars->scalarCoeff[Arches::AS],
		vars->scalarCoeff[Arches::AT],
                vars->scalarCoeff[Arches::AB],
		vars->scalarNonlinearSrc, vars->scalarLinearSrc,
		vars->scalarConvectCoeff[Arches::AE],
		vars->scalarConvectCoeff[Arches::AW],
		vars->scalarConvectCoeff[Arches::AN],
		vars->scalarConvectCoeff[Arches::AS],
		vars->scalarConvectCoeff[Arches::AT],
		vars->scalarConvectCoeff[Arches::AB],
		vars->scalarDiffusionCoeff[Arches::AE],
		vars->scalarDiffusionCoeff[Arches::AW],
		vars->scalarDiffusionCoeff[Arches::AN], 
		vars->scalarDiffusionCoeff[Arches::AS],
		vars->scalarDiffusionCoeff[Arches::AT],
		vars->scalarDiffusionCoeff[Arches::AB],
		constvars->cellType, wall_celltypeval,
		xminus, xplus, yminus, yplus, zminus, zplus);
}


//****************************************************************************



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
		      vars->scalarConvectCoeff[Arches::AE], vars->scalarConvectCoeff[Arches::AW],
		      vars->scalarConvectCoeff[Arches::AN], vars->scalarConvectCoeff[Arches::AS],
		      vars->scalarConvectCoeff[Arches::AT], vars->scalarConvectCoeff[Arches::AB],
		      vars->scalarNonlinearSrc, vars->scalarLinearSrc,
		      constvars->cellType, d_mmWallID);
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
BoundaryCondition::FlowInlet::FlowInlet(int cellID, bool calcVariance,
                                        bool reactingScalarSolve):
  d_cellTypeID(cellID), d_calcVariance(calcVariance), 
  d_reactingScalarSolve(reactingScalarSolve)
{
  flowRate = 0.0;
  inletVel = 0.0;
  fcr = 0.0;
  fsr = 0.0;
  d_prefill_index = 0;
  d_ramping_inlet_flowrate = false;
  d_prefill = false;
  // add cellId to distinguish different inlets
  std::stringstream stream_cellID;
  stream_cellID << d_cellTypeID;
  d_area_label = VarLabel::create("flowarea"+stream_cellID.str(),
   ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_flowRate_label = VarLabel::create("flowRate"+stream_cellID.str(),
   ReductionVariable<double, Reductions::Min<double> >::getTypeDescription()); 
}

BoundaryCondition::FlowInlet::FlowInlet():
  d_cellTypeID(0), d_calcVariance(0), d_reactingScalarSolve(0),
  d_area_label(0), d_flowRate_label(0)
{
  flowRate = 0.0;
  inletVel = 0.0;
  fcr = 0.0;
  fsr = 0.0;
  d_prefill_index = 0;
  d_ramping_inlet_flowrate = false;
  d_prefill = false;
}

BoundaryCondition::FlowInlet::FlowInlet( const FlowInlet& copy ) :
  d_cellTypeID (copy.d_cellTypeID),
  d_calcVariance (copy.d_calcVariance),
  d_reactingScalarSolve (copy.d_reactingScalarSolve),
  flowRate(copy.flowRate),
  inletVel(copy.inletVel),
  fcr(copy.fcr),
  fsr(copy.fsr),
  d_prefill_index(copy.d_prefill_index),
  d_ramping_inlet_flowrate(copy.d_ramping_inlet_flowrate),
  streamMixturefraction(copy.streamMixturefraction),
  calcStream(copy.calcStream),
  d_area_label(copy.d_area_label),
  d_flowRate_label(copy.d_flowRate_label)
{
  for (vector<GeometryPieceP>::const_iterator it = copy.d_geomPiece.begin();
       it != copy.d_geomPiece.end(); ++it)
    d_geomPiece.push_back((*it)->clone());
  
  if (d_prefill)
    for (vector<GeometryPieceP>::const_iterator it = copy.d_prefillGeomPiece.begin();
       it != copy.d_prefillGeomPiece.end(); ++it)
      d_prefillGeomPiece.push_back((*it)->clone());

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
  d_calcVariance = copy.d_calcVariance;
  d_reactingScalarSolve = copy.d_reactingScalarSolve;
  flowRate = copy.flowRate;
  inletVel = copy.inletVel;
  fcr = copy.fcr;
  fsr = copy.fsr;
  d_prefill_index = copy.d_prefill_index;
  d_ramping_inlet_flowrate = copy.d_ramping_inlet_flowrate;
  streamMixturefraction = copy.streamMixturefraction;
  calcStream = copy.calcStream;
  d_geomPiece = copy.d_geomPiece;
  if (d_prefill)
    d_prefillGeomPiece = copy.d_prefillGeomPiece;

  return *this;
}


BoundaryCondition::FlowInlet::~FlowInlet()
{
  VarLabel::destroy(d_area_label);
  VarLabel::destroy(d_flowRate_label);
}

//****************************************************************************
// Problem Setup for BoundaryCondition::FlowInlet
//****************************************************************************
void 
BoundaryCondition::FlowInlet::problemSetup(ProblemSpecP& params)
{
  params->getWithDefault("Flow_rate", flowRate,0.0);
  params->getWithDefault("InletVelocity", inletVel,0.0);
// ramping function is the same for all inlets where ramping is on  
  params->getWithDefault("ramping_inlet_flowrate",
                         d_ramping_inlet_flowrate, false);
  // This parameter only needs to be set for fuel inlets for which
  // mixture fraction > 0, if there is an air inlet, and air has some CO2,
  // this air CO2 will be counted in the balance automatically
  // params->getWithDefault("CarbonMassFractionInFuel", fcr, 0.0);
  params->getWithDefault("SulfurMassFractionInFuel", fsr, 0.0);
  // check to see if this will work
  ProblemSpecP geomObjPS = params->findBlock("geom_object");
  GeometryPieceFactory::create(geomObjPS, d_geomPiece);
  // loop thru all the inlet geometry objects
  //for (ProblemSpecP geom_obj_ps = params->findBlock("geom_object");
  //     geom_obj_ps != 0; 
  //     geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
  //  vector<GeometryPieceP> pieces;
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
  params->require("mixture_fraction", mixfrac);
  streamMixturefraction.d_mixVars.push_back(mixfrac);
  if (d_calcVariance)
    streamMixturefraction.d_mixVarVariance.push_back(0.0);
  double reactscalar;
  if (d_reactingScalarSolve) {
    params->require("reacting_scalar", reactscalar);
    streamMixturefraction.d_rxnVars.push_back(reactscalar);
  }
  std::string d_prefill_direction;
  params->getWithDefault("prefill_direction",d_prefill_direction,"");
  if (!(d_prefill_direction == "")) {
    if (d_prefill_direction == "X") {
      d_prefill_index = 1;
      d_prefill = true;
    }
    else if (d_prefill_direction == "Y") {
    d_prefill_index = 2;
    d_prefill = true;
    }
    else if (d_prefill_direction == "Z") {
    d_prefill_index = 3;
    d_prefill = true;
    }
    else
      throw InvalidValue("Wrong prefill direction.", __FILE__, __LINE__);
  }
  if (d_prefill) {
    ProblemSpecP prefillGeomObjPS = params->findBlock("prefill_geom_object");
    GeometryPieceFactory::create(prefillGeomObjPS, d_prefillGeomPiece);
  }
 
}


//****************************************************************************
// constructor for BoundaryCondition::PressureInlet
//****************************************************************************
BoundaryCondition::PressureInlet::PressureInlet(int cellID, bool calcVariance,
                                                bool reactingScalarSolve):
  d_cellTypeID(cellID), d_calcVariance(calcVariance),
  d_reactingScalarSolve(reactingScalarSolve)
{
}

//****************************************************************************
// Problem Setup for BoundaryCondition::PressureInlet
//****************************************************************************
void 
BoundaryCondition::PressureInlet::problemSetup(ProblemSpecP& params)
{
  ProblemSpecP geomObjPS = params->findBlock("geom_object");
  GeometryPieceFactory::create(geomObjPS, d_geomPiece);
  // loop thru all the pressure inlet geometry objects
  //for (ProblemSpecP geom_obj_ps = params->findBlock("geom_object");
  //     geom_obj_ps != 0; 
  //     geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
  //  vector<GeometryPieceP> pieces;
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
  params->require("mixture_fraction", mixfrac);
  streamMixturefraction.d_mixVars.push_back(mixfrac);
  if (d_calcVariance)
    streamMixturefraction.d_mixVarVariance.push_back(0.0);
  double reactscalar;
  if (d_reactingScalarSolve) {
    params->require("reacting_scalar", reactscalar);
    streamMixturefraction.d_rxnVars.push_back(reactscalar);
  }
}

//****************************************************************************
// constructor for BoundaryCondition::FlowOutlet
//****************************************************************************
BoundaryCondition::FlowOutlet::FlowOutlet(int cellID, bool calcVariance,
                                          bool reactingScalarSolve):
  d_cellTypeID(cellID), d_calcVariance(calcVariance),
  d_reactingScalarSolve(reactingScalarSolve)
{
}

//****************************************************************************
// Problem Setup for BoundaryCondition::FlowOutlet
//****************************************************************************
void 
BoundaryCondition::FlowOutlet::problemSetup(ProblemSpecP& params)
{
  ProblemSpecP geomObjPS = params->findBlock("geom_object");
  GeometryPieceFactory::create(geomObjPS, d_geomPiece);
  // loop thru all the inlet geometry objects
  //for (ProblemSpecP geom_obj_ps = params->findBlock("geom_object");
  //     geom_obj_ps != 0; 
  //     geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
  //  vector<GeometryPieceP> pieces;
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
  params->require("mixture_fraction", mixfrac);
  streamMixturefraction.d_mixVars.push_back(mixfrac);
  if (d_calcVariance)
    streamMixturefraction.d_mixVarVariance.push_back(0.0);
  double reactscalar;
  if (d_reactingScalarSolve) {
    params->require("reacting_scalar", reactscalar);
    streamMixturefraction.d_rxnVars.push_back(reactscalar);
  }
}
//****************************************************************************
// Constructor Setup for BoundaryCondition::BCSourceInfo
//****************************************************************************
BoundaryCondition::BCSourceInfo::BCSourceInfo(bool calcVariance, bool reactingScalarSolve):
d_calcVariance(calcVariance), d_reactingScalarSolve(reactingScalarSolve)
{
		//initialize some variables
		area_x = 0.0;
		area_y = 0.0;
		area_z = 0.0;
		umom_flux = 0.0;
		vmom_flux = 0.0;
		wmom_flux = 0.0;
		f_flux = 0.0;
		h_flux = 0.0;
		totalMassFlux = 0.0;
		totalVelocity = 0.0;

}
//****************************************************************************
// Destructor Setup for BoundaryCondition::BCSourceInfo
//****************************************************************************
BoundaryCondition::BCSourceInfo::~BCSourceInfo(){
}
//****************************************************************************
// Problem Setup for BoundaryCondition::BCSourceInfo
//****************************************************************************
void 
BoundaryCondition::BCSourceInfo::problemSetup(ProblemSpecP& params)
{
	//You can pick MassFlux or Velocity
	if (ProblemSpecP massfluxchild = params->findBlock("MassFlux"))
	{
		doAreaCalc = true;	
		massfluxchild->require("massflux_value", totalMassFlux); //value of the vector

		massfluxchild->getAttribute("type",velocityType); //relative or absolute
		if (velocityType == "relative")
		{
			massfluxchild->getAttribute("relation",velocityRelation); // choose "point" or "axis"

			if (velocityRelation == "point"){
				massfluxchild->require("point_location", point);	

			}
			else if (velocityRelation == "axis"){

				massfluxchild->require("axis_start",axisStart); //Starting and ending point of the axis
				massfluxchild->require("axis_end",  axisEnd);

			}
		}
		else if (velocityType == "absolute")
		{
			massfluxchild->require("normals",normal); //since it is absolute, we need tell it what faces and how to scale it. ie v_{face,i} = n_i*V
			cout << "normal =" << normal << endl;
		}
		else
		{
			throw ParameterNotFound(" Must specify an absolute or relative attribute for the <Velocity> or <MassFlux>.",__FILE__,__LINE__); 
		}

	}
	else if (ProblemSpecP velchild = params->findBlock("Velocity"))
	{
		velchild->require("velocity_value", totalVelocity); //value of the vector

		velchild->getAttribute("type",velocityType); //relative or absolute
		if (velocityType == "relative")
		{
			velchild->getAttribute("relation",velocityRelation); // choose "point" or "axis"

			if (velocityRelation == "point"){
				velchild->require("point_location", point);	

			}
			else if (velocityRelation == "axis"){

				velchild->require("axis_start",axisStart); //Starting and ending point of the axis
				velchild->require("axis_end",  axisEnd);

			}
		}
		else if (velocityType == "absolute")
		{
			velchild->require("normals",normal); //since it is absolute, we need tell it what faces and how to scale it. ie v_{face,i} = n_i*V
			cout << "normal =" << normal << endl;
		}
		else
		{
			throw ParameterNotFound(" Must specify an absolute or relative attribute for the <Velocity> or <MassFlux>.",__FILE__,__LINE__); 
		}
	}
	else
		throw ParameterNotFound(" Please enter a MassFlux or Velocity for the <IntrusionWithBCSource> block!",__FILE__,__LINE__);
							

	// Get the scalar information
	if (ProblemSpecP mfchild = params->findBlock("MixtureFraction"))
	{
		mfchild->require("inlet_value",mixfrac_inlet);
	}
	//for getting the inlet properties
	//hard coded for now!
	d_calcVariance = false;
	d_reactingScalarSolve = false;
  	streamMixturefraction.d_mixVars.push_back(mixfrac_inlet);
  	if (d_calcVariance)
    	streamMixturefraction.d_mixVarVariance.push_back(0.0);
  	double reactscalar;
  	if (d_reactingScalarSolve) {
    	params->require("reacting_scalar", reactscalar);
    	streamMixturefraction.d_rxnVars.push_back(reactscalar);
  	}

	// Get the geometry piece(s)
	if (ProblemSpecP geomObjPS = params->findBlock("geom_object")){
		GeometryPieceFactory::create(geomObjPS, d_geomPiece);
	}
	else
		throw ParameterNotFound(" Must specify a geometry piece for BCSource.\nPlease add a <geom_object> to the <IntrusionWithBCSource> block in the inputfile.  Stopping...",__FILE__,__LINE__);

}
//****************************************************************************
// Compute surface area of inlet for BoundaryCondition::BCSourceInfo
//****************************************************************************
void
BoundaryCondition::computeInletAreaBCSource(const ProcessorGroup*,
								     		  const PatchSubset* patches,
								              const MaterialSubset*,
									  		  DataWarehouse* old_dw,
											  DataWarehouse* new_dw)
{

  //Loop through current patch...compute area	
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    Box patchInteriorBox = patch->getInteriorBox();
	
	int nofBoundaryPieces = (int)d_sourceBoundaryInfo.size();

	//assuming a uniform mesh!
	Vector dx = patch->dCell();

	for (int bp = 0; bp < nofBoundaryPieces; bp++){
		// The main loop for computing the source term
		//**get the geometry piece for this boundary source block**
		// we could have more than one geometry piece per block.
		int nofGeomPieces = (int)d_sourceBoundaryInfo[bp]->d_geomPiece.size();
		
		for (int gp = 0; gp < nofGeomPieces; gp++){

			GeometryPieceP piece = d_sourceBoundaryInfo[bp]->d_geomPiece[gp];
			Box geomBox = piece->getBoundingBox();
			Box b = geomBox.intersect(patchInteriorBox);

			if (!(b.degenerate())){
				//iterator over cells and see if a boundary source needs adding.
				for (CellIterator iter=patch->getCellCenterIterator(b); 
					!iter.done(); iter++){
					
					Point p = patch->cellPosition(*iter);
					Point p_xp = patch->cellPosition(*iter + IntVector(1,0,0));
					Point p_xm = patch->cellPosition(*iter - IntVector(1,0,0));
					Point p_yp = patch->cellPosition(*iter + IntVector(0,1,0));
					Point p_ym = patch->cellPosition(*iter - IntVector(0,1,0));
					Point p_zp = patch->cellPosition(*iter + IntVector(0,0,1));
					Point p_zm = patch->cellPosition(*iter - IntVector(0,0,1));
					
					if (piece->inside(p)){
							
						//Now check neighbors
						// x+
						if (!(piece->inside(p_xp)))
							d_sourceBoundaryInfo[bp]->area_x += dx.y()*dx.z();
						// x-
						if (!(piece->inside(p_xm)))
							d_sourceBoundaryInfo[bp]->area_x += dx.y()*dx.z();
						// y+	
						if (!(piece->inside(p_yp)))
							d_sourceBoundaryInfo[bp]->area_y += dx.x()*dx.z();
						// y-
						if (!(piece->inside(p_ym)))
							d_sourceBoundaryInfo[bp]->area_y += dx.x()*dx.z();
						// z+
						if (!(piece->inside(p_zp)))
							d_sourceBoundaryInfo[bp]->area_z += dx.x()*dx.y();
						// z-
						if (!(piece->inside(p_zm)))
							d_sourceBoundaryInfo[bp]->area_z += dx.x()*dx.y();
					}
				}
			}
		}

		//Now get the totalVelocity from the the mass flux
		//hard coded for jennifer at the moment!
		d_sourceBoundaryInfo[bp]->totalVelocity = d_sourceBoundaryInfo[bp]->totalMassFlux/(d_sourceBoundaryInfo[bp]->calcStream.d_density*
													(d_sourceBoundaryInfo[bp]->area_y+d_sourceBoundaryInfo[bp]->area_z));
		cout << "The total area for your geometry = " << d_sourceBoundaryInfo[bp]->area_y+d_sourceBoundaryInfo[bp]->area_z << " m" << endl;
	}
  }
}
// *------------------------------------------------*
// Schedule the compute of the boundary source term
// *------------------------------------------------*
void 
BoundaryCondition::sched_computeMomSourceTerm(SchedulerP& sched,
						                                      const PatchSet* patches,
                                      						  const MaterialSet* matls,
															  const TimeIntegratorLabel* timelabels)
 
{

	string taskname = "BoundaryCondition::computeMomSourceTerm" + timelabels->integrator_step_name;
	Task* tsk = scinew Task(taskname, this, &BoundaryCondition::computeMomSourceTerm, timelabels);

	tsk->modifies(d_lab->d_umomBoundarySrcLabel);
	tsk->modifies(d_lab->d_vmomBoundarySrcLabel);
	tsk->modifies(d_lab->d_wmomBoundarySrcLabel);
	
	sched->addTask(tsk, patches, matls); 	
		
}
// *------------------------------------------------*
// Carry out the compute of the boundary source term
// *------------------------------------------------*
void 
BoundaryCondition::computeMomSourceTerm(const ProcessorGroup*,
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
    Box patchInteriorBox = patch->getInteriorBox();

	SFCXVariable<double> umomSource;
	SFCYVariable<double> vmomSource;
	SFCZVariable<double> wmomSource;

	new_dw->getModifiable(umomSource, d_lab->d_umomBoundarySrcLabel, matlIndex, patch);
	new_dw->getModifiable(vmomSource, d_lab->d_vmomBoundarySrcLabel, matlIndex, patch);
	new_dw->getModifiable(wmomSource, d_lab->d_wmomBoundarySrcLabel, matlIndex, patch);
	
	umomSource.initialize(0.0);
	vmomSource.initialize(0.0);
	wmomSource.initialize(0.0);

	//assuming a uniform mesh!
	Vector dx = patch->dCell();

	IntVector idxLo = patch->getCellFORTLowIndex();
  	IntVector idxHi = patch->getCellFORTHighIndex();
  	double bc_source = 0.0; 

	int nofBoundaryPieces = (int)d_sourceBoundaryInfo.size();

	for (int bp = 0; bp < nofBoundaryPieces; bp++){
		// The main loop for computing the source term
		//**get the geometry piece for this boundary source block**
		// we could have more than one geometry piece per block.
		int nofGeomPieces = (int)d_sourceBoundaryInfo[bp]->d_geomPiece.size();
		
		for (int gp = 0; gp < nofBoundaryPieces; gp++){

			GeometryPieceP piece = d_sourceBoundaryInfo[bp]->d_geomPiece[gp];
			Box geomBox = piece->getBoundingBox();
			Box b = geomBox.intersect(patchInteriorBox);

			if (!(b.degenerate())){
				//iterator over cells and see if a boundary source needs adding.
				for (CellIterator iter=patch->getSFCXIterator(0); 
					!iter.done(); iter++){
					
					Point p = patch->cellPosition(*iter);
					Point p_xp = patch->cellPosition(*iter + IntVector(1,0,0));
					Point p_xm = patch->cellPosition(*iter - IntVector(1,0,0));
					Point p_yp = patch->cellPosition(*iter + IntVector(0,1,0));
					Point p_ym = patch->cellPosition(*iter - IntVector(0,1,0));
					Point p_zp = patch->cellPosition(*iter + IntVector(0,0,1));
					Point p_zm = patch->cellPosition(*iter - IntVector(0,0,1));
					
					if (piece->inside(p)){
						//this is a nasty embedded set of if's....will fix in the future.
						// x+
						if (!(piece->inside(p_xp))){
							if (d_sourceBoundaryInfo[bp]->velocityType == "absolute"){		
							}
							else if (d_sourceBoundaryInfo[bp]->velocityType == "relative"){
								if (d_sourceBoundaryInfo[bp]->velocityRelation == "axis"){
								}
								else if (d_sourceBoundaryInfo[bp]->velocityRelation == "point"){
								}

							}
						}
						// x-
						if (!(piece->inside(p_xm))){
							if (d_sourceBoundaryInfo[bp]->velocityType == "absolute"){		
							}
							else if (d_sourceBoundaryInfo[bp]->velocityType == "relative"){
								if (d_sourceBoundaryInfo[bp]->velocityRelation == "axis"){
								}
								else if (d_sourceBoundaryInfo[bp]->velocityRelation == "point"){
								}

							}

						}
						// y+
						if (!(piece->inside(p_yp))){
							if (d_sourceBoundaryInfo[bp]->velocityType == "absolute"){		
							}
							else if (d_sourceBoundaryInfo[bp]->velocityType == "relative"){
								if (d_sourceBoundaryInfo[bp]->velocityRelation == "axis"){
									//hard coding for jennifer for now.
									double y = p.y() - d_sourceBoundaryInfo[bp]->axisStart[1];		
									double z = p.z() - d_sourceBoundaryInfo[bp]->axisStart[2];
									double theta = atan(z/y);

									double y_comp = d_sourceBoundaryInfo[bp]->totalVelocity*cos(theta);
									vmomSource[*iter+IntVector(0,2,0)] = d_sourceBoundaryInfo[bp]->calcStream.d_density*
																		  y_comp*y_comp*dx.x()*dx.z();
								}
								else if (d_sourceBoundaryInfo[bp]->velocityRelation == "point"){
								}

							}

						}	
						// y-
						if (!(piece->inside(p_ym))){
							if (d_sourceBoundaryInfo[bp]->velocityType == "absolute"){		
							}
							else if (d_sourceBoundaryInfo[bp]->velocityType == "relative"){
								if (d_sourceBoundaryInfo[bp]->velocityRelation == "axis"){
									//hard coding for jennifer for now.
									double y = p.y() - d_sourceBoundaryInfo[bp]->axisStart[1];		
									double z = p.z() - d_sourceBoundaryInfo[bp]->axisStart[2];
									double theta = atan(z/y);

									double y_comp = -1.0*d_sourceBoundaryInfo[bp]->totalVelocity*cos(theta);
									vmomSource[*iter-IntVector(0,1,0)] = -1.0*d_sourceBoundaryInfo[bp]->calcStream.d_density*
																		  y_comp*y_comp*dx.x()*dx.z();

								}
								else if (d_sourceBoundaryInfo[bp]->velocityRelation == "point"){
								}

							}

						}
						// z+
						if (!(piece->inside(p_zp))){
							if (d_sourceBoundaryInfo[bp]->velocityType == "absolute"){		
							}
							else if (d_sourceBoundaryInfo[bp]->velocityType == "relative"){
								if (d_sourceBoundaryInfo[bp]->velocityRelation == "axis"){
									//hard coding for jennifer for now.
									double y = p.y() - d_sourceBoundaryInfo[bp]->axisStart[1];		
									double z = p.z() - d_sourceBoundaryInfo[bp]->axisStart[2];
									double theta = atan(z/y);

									double z_comp = d_sourceBoundaryInfo[bp]->totalVelocity*sin(theta);
									wmomSource[*iter+IntVector(0,0,2)] = d_sourceBoundaryInfo[bp]->calcStream.d_density*
																		  z_comp*z_comp*dx.x()*dx.y();

								}
								else if (d_sourceBoundaryInfo[bp]->velocityRelation == "point"){
								}

							}

						}
						// z-
						if (!(piece->inside(p_zm))){
							if (d_sourceBoundaryInfo[bp]->velocityType == "absolute"){		
							}
							else if (d_sourceBoundaryInfo[bp]->velocityType == "relative"){
								if (d_sourceBoundaryInfo[bp]->velocityRelation == "axis"){
									//hard coding for jennifer for now.
									double y = p.y() - d_sourceBoundaryInfo[bp]->axisStart[1];		
									double z = p.z() - d_sourceBoundaryInfo[bp]->axisStart[2];
									double theta = atan(z/y);

									double z_comp = d_sourceBoundaryInfo[bp]->totalVelocity*sin(theta);
									wmomSource[*iter-IntVector(0,0,1)] = -1.0*d_sourceBoundaryInfo[bp]->calcStream.d_density*
																		  z_comp*z_comp*dx.x()*dx.y();

								}
								else if (d_sourceBoundaryInfo[bp]->velocityRelation == "point"){
								}

							}

						}																	

					}

				}

			}
	
		
		} // end Geometry Pieces loop
	} // end of Boundary Pieces loop 
	
  }
		
}
// *--------------------------------------------------------*
// Schedule the compute of the boundary source term-scalar
// *--------------------------------------------------------*
void 
BoundaryCondition::sched_computeScalarSourceTerm(SchedulerP& sched,
				                                  const PatchSet* patches,
                          						  const MaterialSet* matls,
												  const TimeIntegratorLabel* timelabels)
{
	string taskname = "BoundaryCondition::computeScalarSourceTerm" + timelabels->integrator_step_name;
	Task* tsk = scinew Task(taskname, this, &BoundaryCondition::computeScalarSourceTerm, timelabels);

	tsk->modifies(d_lab->d_scalarBoundarySrcLabel);
	tsk->modifies(d_lab->d_enthalpyBoundarySrcLabel);
	tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::AroundCells, Arches::ONEGHOSTCELL);

	sched->addTask(tsk, patches, matls); 	
		
}
// *--------------------------------------------------------*
// Perform the compute of the boundary source term-scalar
// This includes: Mixture fraction, Enthalpy
// *--------------------------------------------------------*
void 
BoundaryCondition::computeScalarSourceTerm(const ProcessorGroup*,
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
    Box patchInteriorBox = patch->getInteriorBox();

	//assuming a uniform mesh!
	Vector dx = patch->dCell();

	CCVariable<double> scalarBoundarySrc;
	CCVariable<double> enthalpyBoundarySrc;
	constCCVariable<int> cellType;

	new_dw->getModifiable(scalarBoundarySrc, d_lab->d_scalarBoundarySrcLabel, matlIndex, patch);
	new_dw->getModifiable(enthalpyBoundarySrc, d_lab->d_enthalpyBoundarySrcLabel, matlIndex, patch);
	new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

	scalarBoundarySrc.initialize(0.0);
	enthalpyBoundarySrc.initialize(0.0);

	IntVector idxLo = patch->getCellFORTLowIndex();
  	IntVector idxHi = patch->getCellFORTHighIndex();
  	double bc_source = 0.0; 

	int nofBoundaryPieces = (int)d_sourceBoundaryInfo.size();

	for (int bp = 0; bp < nofBoundaryPieces; bp++){
		// The main loop for computing the source term
		//**get the geometry piece for this boundary source block**
		// we could have more than one geometry piece per block.
		int nofGeomPieces = (int)d_sourceBoundaryInfo[bp]->d_geomPiece.size();
		
		for (int gp = 0; gp < nofGeomPieces; gp++){

			GeometryPieceP piece = d_sourceBoundaryInfo[bp]->d_geomPiece[gp];
			Box geomBox = piece->getBoundingBox();
			Box b = geomBox.intersect(patchInteriorBox);

			if (!(b.degenerate())){
				//iterator over cells and see if a boundary source needs adding.
				for (CellIterator iter=patch->getCellCenterIterator(b); 
					!iter.done(); iter++){
					
					Point p = patch->cellPosition(*iter);
					Point p_xp = patch->cellPosition(*iter + IntVector(1,0,0));
					Point p_xm = patch->cellPosition(*iter - IntVector(1,0,0));
					Point p_yp = patch->cellPosition(*iter + IntVector(0,1,0));
					Point p_ym = patch->cellPosition(*iter - IntVector(0,1,0));
					Point p_zp = patch->cellPosition(*iter + IntVector(0,0,1));
					Point p_zm = patch->cellPosition(*iter - IntVector(0,0,1));
					
					if (cellType[*iter] == d_intrusionBC->d_cellTypeID) { 

						/*IntVector ci;
						patch->findCell(p_zm, ci);

						cout << "my cell index = " << ci << endl;*/
							
						//Now check neighbors
						// x+
						if (cellType[*iter + IntVector(1,0,0)] == d_flowfieldCellTypeVal){ 		
							// source term = \int \rho u \phi \cdot dS	
							scalarBoundarySrc[*iter + IntVector(1,0,0)] = d_sourceBoundaryInfo[bp]->calcStream.d_density*d_sourceBoundaryInfo[bp]->totalVelocity*
													   d_sourceBoundaryInfo[bp]->normal[0]*	
													   d_sourceBoundaryInfo[bp]->mixfrac_inlet*
													   dx.y()*dx.z();
							if (d_enthalpySolve)
								enthalpyBoundarySrc[*iter + IntVector(1,0,0)] = d_sourceBoundaryInfo[bp]->calcStream.d_density*d_sourceBoundaryInfo[bp]->totalVelocity*
														d_sourceBoundaryInfo[bp]->normal[0]*
														d_sourceBoundaryInfo[bp]->calcStream.d_enthalpy*
														dx.y()*dx.z();							   
						}
						// x-
						if (cellType[*iter - IntVector(1,0,0)] == d_flowfieldCellTypeVal){		
							scalarBoundarySrc[*iter - IntVector(1,0,0)] = d_sourceBoundaryInfo[bp]->calcStream.d_density*d_sourceBoundaryInfo[bp]->totalVelocity*
													   d_sourceBoundaryInfo[bp]->normal[0]*	
													   d_sourceBoundaryInfo[bp]->mixfrac_inlet*
													   dx.y()*dx.z();							
							if (d_enthalpySolve)
								enthalpyBoundarySrc[*iter - IntVector(1,0,0)] = d_sourceBoundaryInfo[bp]->calcStream.d_density*d_sourceBoundaryInfo[bp]->totalVelocity*
														d_sourceBoundaryInfo[bp]->normal[0]*
														d_sourceBoundaryInfo[bp]->calcStream.d_enthalpy*
														dx.y()*dx.z();
						}
						// y+
						if (cellType[*iter + IntVector(0,1,0)] == d_flowfieldCellTypeVal){		
							//hard coding for jennifer for now.
							double y = p.y() - d_sourceBoundaryInfo[bp]->axisStart[1];		
							double z = p.z() - d_sourceBoundaryInfo[bp]->axisStart[2];
							double theta = atan(z/y);
							double y_comp = Abs(d_sourceBoundaryInfo[bp]->totalVelocity*cos(theta));			
											
							scalarBoundarySrc[*iter + IntVector(0,1,0)] = d_sourceBoundaryInfo[bp]->calcStream.d_density*y_comp*
													   d_sourceBoundaryInfo[bp]->mixfrac_inlet*
													   dx.x()*dx.z();		
							if (d_enthalpySolve)
								enthalpyBoundarySrc[*iter + IntVector(0,1,0)] = d_sourceBoundaryInfo[bp]->calcStream.d_density*y_comp*
														d_sourceBoundaryInfo[bp]->calcStream.d_enthalpy*
														dx.x()*dx.z();							   					
						}
						// y-
						if (cellType[*iter - IntVector(0,1,0)] == d_flowfieldCellTypeVal){		
							//hard coding for jennifer for now.
							double y = p.y() - d_sourceBoundaryInfo[bp]->axisStart[1];		
							double z = p.z() - d_sourceBoundaryInfo[bp]->axisStart[2];
							double theta = atan(z/y);
							double y_comp = Abs(d_sourceBoundaryInfo[bp]->totalVelocity*cos(theta));	

							scalarBoundarySrc[*iter - IntVector(0,1,0)] = d_sourceBoundaryInfo[bp]->calcStream.d_density*y_comp*
													   d_sourceBoundaryInfo[bp]->mixfrac_inlet*
													   dx.x()*dx.z();
							if (d_enthalpySolve)
								enthalpyBoundarySrc[*iter - IntVector(0,1,0)] = d_sourceBoundaryInfo[bp]->calcStream.d_density*y_comp*
														d_sourceBoundaryInfo[bp]->calcStream.d_enthalpy*
														dx.x()*dx.z();
						}
						// z+
						if (cellType[*iter + IntVector(0,0,1)] == d_flowfieldCellTypeVal){		
							double y = p.y() - d_sourceBoundaryInfo[bp]->axisStart[1];		
							double z = p.z() - d_sourceBoundaryInfo[bp]->axisStart[2];
							double theta = atan(z/y);
							double z_comp = Abs(d_sourceBoundaryInfo[bp]->totalVelocity*sin(theta));
							scalarBoundarySrc[*iter + IntVector(0,0,1)] = d_sourceBoundaryInfo[bp]->calcStream.d_density*z_comp*
													   d_sourceBoundaryInfo[bp]->mixfrac_inlet*
													   dx.x()*dx.y();				
							if (d_enthalpySolve)
								enthalpyBoundarySrc[*iter + IntVector(0,0,1)] = d_sourceBoundaryInfo[bp]->calcStream.d_density*z_comp*
														d_sourceBoundaryInfo[bp]->calcStream.d_enthalpy*
														dx.x()*dx.y();													   			
						}
						// z-
						if (cellType[*iter - IntVector(0,0,1)] == d_flowfieldCellTypeVal){		
							double y = p.y() - d_sourceBoundaryInfo[bp]->axisStart[1];		
							double z = p.z() - d_sourceBoundaryInfo[bp]->axisStart[2];
							double theta = atan(z/y);
							double z_comp = Abs(d_sourceBoundaryInfo[bp]->totalVelocity*sin(theta));


							scalarBoundarySrc[*iter - IntVector(0,0,1)] = d_sourceBoundaryInfo[bp]->calcStream.d_density*z_comp*
													   d_sourceBoundaryInfo[bp]->mixfrac_inlet*
													   dx.x()*dx.y();
													   
							double testme = scalarBoundarySrc[*iter - IntVector(0,0,1)];
							
							if (d_enthalpySolve)	
								enthalpyBoundarySrc[*iter - IntVector(0,0,1)] = d_sourceBoundaryInfo[bp]->calcStream.d_density*z_comp*
														d_sourceBoundaryInfo[bp]->calcStream.d_enthalpy*
														dx.x()*dx.y();												   
													   				
						} 				

					}
				}
			}

		

		} // end Geometry Pieces loop
	} // end Boundary Pieces loop


  } // end patch loop
	
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
    
    throw InvalidValue("Invalid index in Source::calcVelSrc", __FILE__, __LINE__);
    
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

    throw InvalidValue("Invalid index in Source::calcVelSrc", __FILE__, __LINE__);

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

    break;
  default:
    throw InvalidValue("Invalid index in LinearSolver for velocity", __FILE__, __LINE__);
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


  fort_mm_explicit(idxLo, idxHi, vars->scalar, constvars->old_scalar,
		   constvars->scalarCoeff[Arches::AE], 
		   constvars->scalarCoeff[Arches::AW], 
		   constvars->scalarCoeff[Arches::AN], 
		   constvars->scalarCoeff[Arches::AS], 
		   constvars->scalarCoeff[Arches::AT], 
		   constvars->scalarCoeff[Arches::AB], 
		   constvars->scalarCoeff[Arches::AP], 
		   constvars->scalarNonlinearSrc, constvars->density_guess,
		   cellinfo->sew, cellinfo->sns, cellinfo->stb, 
		   delta_t,
		   constvars->cellType, d_mmWallID);

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

  fort_mm_explicit_oldvalue(idxLo, idxHi, vars->enthalpy, constvars->old_enthalpy,
			    constvars->scalarCoeff[Arches::AE], 
			    constvars->scalarCoeff[Arches::AW], 
			    constvars->scalarCoeff[Arches::AN], 
			    constvars->scalarCoeff[Arches::AS], 
			    constvars->scalarCoeff[Arches::AT], 
			    constvars->scalarCoeff[Arches::AB], 
			    constvars->scalarCoeff[Arches::AP], 
			    constvars->scalarNonlinearSrc, constvars->density_guess,
			    cellinfo->sew, cellinfo->sns, cellinfo->stb, 
			    delta_t,
			    constvars->cellType, d_mmWallID);

}

//****************************************************************************
// Set zero gradient for scalar for outlet and pressure BC
//****************************************************************************
void 
BoundaryCondition::scalarOutletPressureBC(const ProcessorGroup*,
			    const Patch* patch,
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
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);
        if ((constvars->cellType[xminusCell] == outlet_celltypeval)||
            (constvars->cellType[xminusCell] == pressure_celltypeval))
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
        if ((constvars->cellType[xplusCell] == outlet_celltypeval)||
            (constvars->cellType[xplusCell] == pressure_celltypeval))
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
        if ((constvars->cellType[yminusCell] == outlet_celltypeval)||
            (constvars->cellType[yminusCell] == pressure_celltypeval))
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
        if ((constvars->cellType[yplusCell] == outlet_celltypeval)||
            (constvars->cellType[yplusCell] == pressure_celltypeval))
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
        if ((constvars->cellType[zminusCell] == outlet_celltypeval)||
            (constvars->cellType[zminusCell] == pressure_celltypeval))
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
        if ((constvars->cellType[zplusCell] == outlet_celltypeval)||
            (constvars->cellType[zplusCell] == pressure_celltypeval))
          vars->scalar[zplusCell]= vars->scalar[currCell];
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
    FlowInlet* fi = d_flowInlets[indx];
    
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
      	  fi->d_cellTypeID, current_time,
      	  xminus, xplus, yminus, yplus, zminus, zplus,
	  fi->d_ramping_inlet_flowrate);
    
  }
}
//****************************************************************************
// Set hat velocity at the outlet
// Tangential bc's are not needed to be set for hat velocities
// Commented them out to avoid confusion
//****************************************************************************
void 
BoundaryCondition::velRhoHatOutletPressureBC(const ProcessorGroup*,
			    const Patch* patch,
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

  int sign = 0;

  if (xminus) {
    int colX = idxLo.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        if ((constvars->cellType[xminusCell] == outlet_celltypeval)||
            (constvars->cellType[xminusCell] == pressure_celltypeval)) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y())))
	    vars->uVelRhoHat[currCell] = 0.0;
	  else {
          if (constvars->cellType[xminusCell] == outlet_celltypeval)
            sign = 1;
          else
            sign = -1;
	  if (sign * constvars->old_uVelocity[currCell] < -1.0e-10)
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
        if ((constvars->cellType[xplusCell] == outlet_celltypeval)||
            (constvars->cellType[xplusCell] == pressure_celltypeval)) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y())))
	    vars->uVelRhoHat[xplusCell] = 0.0;
	  else {
          if (constvars->cellType[xplusCell] == outlet_celltypeval)
            sign = 1;
          else
            sign = -1;
	  if (sign * constvars->old_uVelocity[xplusCell] > 1.0e-10)
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
        if ((constvars->cellType[yminusCell] == outlet_celltypeval)||
            (constvars->cellType[yminusCell] == pressure_celltypeval)) {
     /*     if (!(xminus && (colX == idxLo.x())))
            vars->uVelRhoHat[yminusCell] = vars->uVelRhoHat[currCell];*/
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    vars->vVelRhoHat[currCell] = 0.0;
	  else {
          if (constvars->cellType[yminusCell] == outlet_celltypeval)
            sign = 1;
          else
            sign = -1;
	  if (sign * constvars->old_vVelocity[currCell] < -1.0e-10)
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
        if ((constvars->cellType[yplusCell] == outlet_celltypeval)||
            (constvars->cellType[yplusCell] == pressure_celltypeval)) {
      /*    if (!(xminus && (colX == idxLo.x())))
            vars->uVelRhoHat[yplusCell] = vars->uVelRhoHat[currCell];*/
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    vars->vVelRhoHat[yplusCell] = 0.0;
	  else {
          if (constvars->cellType[yplusCell] == outlet_celltypeval)
            sign = 1;
          else
            sign = -1;
	  if (sign * constvars->old_vVelocity[yplusCell] > 1.0e-10)
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
        if ((constvars->cellType[zminusCell] == outlet_celltypeval)||
            (constvars->cellType[zminusCell] == pressure_celltypeval)) {
      /*    if (!(xminus && (colX == idxLo.x())))
            vars->uVelRhoHat[zminusCell] = vars->uVelRhoHat[currCell];
          if (!(yminus && (colY == idxLo.y())))
            vars->vVelRhoHat[zminusCell] = vars->vVelRhoHat[currCell];*/
          if ((yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    vars->wVelRhoHat[currCell] = 0.0;
	  else {
          if (constvars->cellType[zminusCell] == outlet_celltypeval)
            sign = 1;
          else
            sign = -1;
	  if (sign * constvars->old_wVelocity[currCell] < -1.0e-10)
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
        if ((constvars->cellType[zplusCell] == outlet_celltypeval)||
            (constvars->cellType[zplusCell] == pressure_celltypeval)) {
       /*   if (!(xminus && (colX == idxLo.x())))
            vars->uVelRhoHat[zplusCell] = vars->uVelRhoHat[currCell];
          if (!(yminus && (colY == idxLo.y())))
            vars->vVelRhoHat[zplusCell] = vars->vVelRhoHat[currCell];*/
          if ((yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
	    vars->wVelRhoHat[zplusCell] = 0.0;
	  else {
          if (constvars->cellType[zplusCell] == outlet_celltypeval)
            sign = 1;
          else
            sign = -1;
	  if (sign * constvars->old_wVelocity[zplusCell] > 1.0e-10)
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
// Set zero gradient for tangent velocity on outlet and pressure bc
//****************************************************************************
void 
BoundaryCondition::velocityOutletPressureTangentBC(const ProcessorGroup*,
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
		throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
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
		  throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
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
		  throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
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
		throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
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
		  throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
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
		  throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
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
		throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
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
		  throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
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
		  throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
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
		throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
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
		  throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
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
		  throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
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
		throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
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
		  throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
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
		  throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
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
		throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
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
		  throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
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
		  throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
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
BoundaryCondition::addPresGradVelocityOutletPressureBC(const ProcessorGroup*,
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
				(cellinfo->dxpw[colX] * avdenlow);

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
				(cellinfo->dxpw[colX+1] * avden);

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
				(cellinfo->dyps[colY] * avdenlow);

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
				(cellinfo->dyps[colY+1] * avden);

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
				(cellinfo->dzpb[colZ] * avdenlow);

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
				(cellinfo->dzpb[colZ+1] * avden);

           vars->wVelRhoHat[zplusplusCell] = vars->wVelRhoHat[zplusCell];

        }
      }
    }
  }
  break;
  default:
   throw InvalidValue("Invalid index in addPresGradVelocityOutletBC", __FILE__, __LINE__);
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
    else 
      throw VariableNotFoundInGrid("cellInformation"," ", __FILE__, __LINE__);
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
	FlowInlet* fi = d_flowInlets[indx];
	double fout = 0.0;
	fort_inlpresbcinout(uVelocity, vVelocity, wVelocity, idxLo, idxHi,
			   density, cellType, fi->d_cellTypeID,
			   flowIN, fout, cellinfo->sew, cellinfo->sns,
			   cellinfo->stb, xminus, xplus, yminus, yplus,
			   zminus, zplus, doing_balance,
			   density, varIN, varOUT);
	if (fout > 0.0)
		throw InvalidValue("Flow comming out of inlet", __FILE__, __LINE__);

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
        FlowInlet* fi = d_flowInlets[indx];
    
        fort_inlbcs(uVelocity, vVelocity, wVelocity,
      	                 idxLo, idxHi, density, cellType, 
      	                 fi->d_cellTypeID, current_time,
      	                 xminus, xplus, yminus, yplus, zminus, zplus,
	                 fi->d_ramping_inlet_flowrate);
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
  if (d_sulfur_balance)
    tsk->requires(Task::NewDW, d_lab->d_so2INLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  if (d_enthalpySolve)
    tsk->requires(Task::NewDW, d_lab->d_enthalpySPLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->computes(d_lab->d_scalarFlowRateLabel);
  if (d_carbon_balance)
    tsk->computes(d_lab->d_CO2FlowRateLabel);
  if (d_carbon_balance_es)
    tsk->computes(d_lab->d_CO2FlowRateESLabel);
  if (d_sulfur_balance)
    tsk->computes(d_lab->d_SO2FlowRateLabel);
  if (d_sulfur_balance_es)
	tsk->computes(d_lab->d_SO2FlowRateESLabel);	  	
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
    constCCVariable<double> co2_es;
	constCCVariable<double> so2_es;
    constCCVariable<double> so2;
    constCCVariable<double> enthalpy;

    new_dw->get(constVars.cellType, d_lab->d_cellTypeLabel,
		matlIndex, patch, Ghost::AroundCells,Arches::ONEGHOSTCELL);

    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else 
      throw VariableNotFoundInGrid("cellInformation"," ", __FILE__, __LINE__);
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
    co2IN = 0.0;
    co2OUT = 0.0;
    if (d_carbon_balance_es) {

	std::vector<ExtraScalarSolver*>::iterator iss; 
	for (iss=d_extraScalars->begin(); iss!=d_extraScalars->end(); ++iss){
		bool checkCarbonBalance = (*iss)->doCarbonBalance();
		if (checkCarbonBalance){ 	
		    const VarLabel* templabel = (*iss)->getScalarLabel();
		    	
		    new_dw->get(co2_es, templabel, matlIndex,
			    patch, Ghost::None, Arches::ZEROGHOSTCELLS);
		    getVariableFlowRate(pc,patch, cellinfo, &constVars, co2_es,
			    &co2IN, &co2OUT); 
		    new_dw->put(sum_vartype(co2OUT-co2IN), d_lab->d_CO2FlowRateESLabel);
		}
	}
    }
    

    double so2IN = 0.0;
    double so2OUT = 0.0;
    if (d_sulfur_balance) {
      new_dw->get(so2, d_lab->d_so2INLabel, matlIndex,
		  patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      getVariableFlowRate(pc,patch, cellinfo, &constVars, so2, &so2IN, &so2OUT); 
      new_dw->put(sum_vartype(so2OUT-so2IN), d_lab->d_SO2FlowRateLabel);
	} 
	
    so2IN = 0.0;
    so2OUT = 0.0; 
	if (d_sulfur_balance_es) {

	std::vector<ExtraScalarSolver*>::iterator iss; 
	for (iss=d_extraScalars->begin(); iss!=d_extraScalars->end(); ++iss){
		bool checkSulfurBalance = (*iss)->doSulfurBalance();
		if (checkSulfurBalance){ 	

		    const VarLabel* templabel = (*iss)->getScalarLabel();
		    	
		    new_dw->get(so2_es, templabel, matlIndex,
			    patch, Ghost::None, Arches::ZEROGHOSTCELLS);
		    getVariableFlowRate(pc,patch, cellinfo, &constVars, so2_es,
			    &so2IN, &so2OUT); 
		    new_dw->put(sum_vartype(so2OUT-so2IN), d_lab->d_SO2FlowRateESLabel);
		}
	}
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
    tsk->requires(Task::OldDW, d_flowInlets[ii]->d_flowRate_label);
    tsk->computes(d_flowInlets[ii]->d_flowRate_label);
  }

  tsk->requires(Task::NewDW, d_lab->d_scalarFlowRateLabel);
  tsk->computes(d_lab->d_scalarEfficiencyLabel);

  if (d_carbon_balance) {
    tsk->requires(Task::NewDW, d_lab->d_CO2FlowRateLabel);
    tsk->computes(d_lab->d_carbonEfficiencyLabel);
  }
  if (d_carbon_balance_es) {
    tsk->requires(Task::NewDW, d_lab->d_CO2FlowRateESLabel);
    tsk->computes(d_lab->d_carbonEfficiencyESLabel);
  }
  if (d_sulfur_balance) {
    tsk->requires(Task::NewDW, d_lab->d_SO2FlowRateLabel);
    tsk->computes(d_lab->d_sulfurEfficiencyLabel);
  }
  if (d_sulfur_balance_es) {
    tsk->requires(Task::NewDW, d_lab->d_SO2FlowRateESLabel);
    tsk->computes(d_lab->d_sulfurEfficiencyESLabel);
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
    sum_vartype sum_scalarFlowRate, sum_CO2FlowRate, sum_SO2FlowRate, sum_enthalpyFlowRate;
    sum_vartype sum_CO2FlowRateES, sum_SO2FlowRateES;
    sum_vartype sum_totalRadSrc;
    delt_vartype flowRate;
    double scalarFlowRate = 0.0;
    double CO2FlowRate = 0.0;
    double CO2FlowRateES = 0.0;
	double SO2FlowRate = 0.0;
	double SO2FlowRateES = 0.0;
    double enthalpyFlowRate = 0.0;
    double totalFlowRate = 0.0;
    double totalCarbonFlowRate = 0.0;
    double totalCarbonFlowRateES = 0.0;
	double totalSulfurFlowRate = 0.0;
	double totalSulfurFlowRateES = 0.0;
    double totalEnthalpyFlowRate = 0.0;
    double scalarEfficiency = 0.0;
    double carbonEfficiency = 0.0;
    double carbonEfficiencyES = 0.0;
	double sulfurEfficiency = 0.0;
	double sulfurEfficiencyES = 0.0;
    double enthalpyEfficiency = 0.0;
    double totalRadSrc = 0.0;
    double normTotalRadSrc = 0.0;

    int me = pc->myrank();
    new_dw->get(sum_scalarFlowRate, d_lab->d_scalarFlowRateLabel);
    scalarFlowRate = sum_scalarFlowRate;
    if (d_carbon_balance) {
      new_dw->get(sum_CO2FlowRate, d_lab->d_CO2FlowRateLabel);
      CO2FlowRate = sum_CO2FlowRate;
    }
    if (d_carbon_balance_es) {
      new_dw->get(sum_CO2FlowRateES, d_lab->d_CO2FlowRateESLabel);
      CO2FlowRateES = sum_CO2FlowRateES;
    }

    if (d_sulfur_balance) {
      new_dw->get(sum_SO2FlowRate, d_lab->d_SO2FlowRateLabel);
      SO2FlowRate = sum_SO2FlowRate;
    }
    if (d_sulfur_balance_es) {
      new_dw->get(sum_SO2FlowRateES, d_lab->d_SO2FlowRateESLabel);
      SO2FlowRateES = sum_SO2FlowRateES;
    }
    if (d_enthalpySolve) {
      new_dw->get(sum_enthalpyFlowRate, d_lab->d_enthalpyFlowRateLabel);
      enthalpyFlowRate = sum_enthalpyFlowRate;
      new_dw->get(sum_totalRadSrc, d_lab->d_totalRadSrcLabel);
      totalRadSrc = sum_totalRadSrc;
    }
    for (int indx = 0; indx < d_numInlets; indx++) {
      FlowInlet* fi = d_flowInlets[indx];
      old_dw->get(flowRate, d_flowInlets[indx]->d_flowRate_label);
      d_flowInlets[indx]->flowRate = flowRate;
      fi->flowRate = flowRate;
      new_dw->put(flowRate, d_flowInlets[indx]->d_flowRate_label);
      double scalarValue = fi->streamMixturefraction.d_mixVars[0];
      if (scalarValue > 0.0)
	  totalFlowRate += fi->flowRate;
      if ((d_carbon_balance)&&(scalarValue > 0.0))
	    totalCarbonFlowRate += fi->flowRate * fi->fcr;
      if ((d_carbon_balance_es)&&(scalarValue > 0.0))
	    totalCarbonFlowRateES += fi->flowRate * fi->fcr;
      if ((d_sulfur_balance)&&(scalarValue > 0.0))
	    totalSulfurFlowRate += fi->flowRate * fi->fsr;
      if ((d_sulfur_balance_es)&&(scalarValue > 0.0))
	    totalSulfurFlowRateES += fi->flowRate * fi->fsr;
      if ((d_enthalpySolve)&&(scalarValue > 0.0))
	    totalEnthalpyFlowRate += fi->flowRate * fi->calcStream.getEnthalpy();
    }
    if (totalFlowRate > 0.0)
      scalarEfficiency = scalarFlowRate / totalFlowRate;
    else 
      if (me == 0)
        cout << "WARNING! No mixture fraction in the domain." << endl;
    new_dw->put(delt_vartype(scalarEfficiency), d_lab->d_scalarEfficiencyLabel);

    if (d_carbon_balance) {
      if (totalCarbonFlowRate > 0.0)
	carbonEfficiency = CO2FlowRate * 12.0/44.0 /totalCarbonFlowRate;
      else 
	throw InvalidValue("No carbon in the domain", __FILE__, __LINE__);
      new_dw->put(delt_vartype(carbonEfficiency), d_lab->d_carbonEfficiencyLabel);
    }
    if (d_carbon_balance_es) {
      if (totalCarbonFlowRateES > 0.0)
	carbonEfficiencyES = CO2FlowRateES * 12.0/44.0 /totalCarbonFlowRateES;
      else 
	throw InvalidValue("No carbon in the domain from ExtraScalar", __FILE__, __LINE__);
      new_dw->put(delt_vartype(carbonEfficiencyES), d_lab->d_carbonEfficiencyESLabel);
    }

    if (d_sulfur_balance) {
      if (totalSulfurFlowRate > 0.0)
	sulfurEfficiency = SO2FlowRate * 32.0/64.0 /totalSulfurFlowRate;
      else 
	throw InvalidValue("No sulfur in the domain", __FILE__, __LINE__);
      new_dw->put(delt_vartype(sulfurEfficiency), d_lab->d_sulfurEfficiencyLabel);
    }

    if (d_sulfur_balance_es) {
      if (totalSulfurFlowRateES > 0.0)
		sulfurEfficiencyES = SO2FlowRateES * 32.0/64.0 /totalSulfurFlowRateES;
      else 
	throw InvalidValue("No sulfur in the domain from ExtraScalar", __FILE__, __LINE__);
      new_dw->put(delt_vartype(sulfurEfficiencyES), d_lab->d_sulfurEfficiencyESLabel);
    }
    if (d_enthalpySolve) {
      if (totalEnthalpyFlowRate < 0.0) {
	enthalpyEfficiency = enthalpyFlowRate/totalEnthalpyFlowRate;
	normTotalRadSrc = totalRadSrc/totalEnthalpyFlowRate;
	enthalpyEfficiency -= normTotalRadSrc;
      }
      else 
	//throw InvalidValue("No enthalpy in the domain", __FILE__, __LINE__);
        if (me == 0)
	  cout << "No enthalpy in the domain"<<endl;
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
	FlowInlet* fi = d_flowInlets[indx];
	double varIN_inlet = 0.0;
	double varOUT_inlet = 0.0;
	fort_inlpresbcinout(constvars->uVelocity, constvars->vVelocity,
			   constvars->wVelocity, idxLo, idxHi,
			   constvars->density, constvars->cellType,
			   fi->d_cellTypeID,
			   flowIN, flowOUT, cellinfo->sew, cellinfo->sns,
			   cellinfo->stb, xminus, xplus, yminus, yplus,
			   zminus, zplus, doing_balance,
			   balance_var, varIN_inlet, varOUT_inlet);

	if (varOUT_inlet > 0.0)
		throw InvalidValue("Balance variable comming out of inlet", __FILE__, __LINE__);

	// Count balance variable comming through the air inlet
	double scalarValue = fi->streamMixturefraction.d_mixVars[0];
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
    tsk->requires(Task::OldDW, d_flowInlets[ii]->d_flowRate_label);
    tsk->computes(d_flowInlets[ii]->d_flowRate_label);
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
    FlowInlet* fi = d_flowInlets[indx];
    old_dw->get(flowRate, d_flowInlets[indx]->d_flowRate_label);
    d_flowInlets[indx]->flowRate = flowRate;
    fi->flowRate = flowRate;
    new_dw->put(flowRate, d_flowInlets[indx]->d_flowRate_label);
  }
}

//****************************************************************************
//Actually calculate the mms velocity BC
//****************************************************************************
void 
BoundaryCondition::mmsvelocityBC(const ProcessorGroup*,
			      const Patch* patch,
			      int index,
			      CellInformation* cellinfo,
			      ArchesVariables* vars,
			      ArchesConstVariables* constvars, 
			      double time_shift, 
			      double dt) 
{
  // Call the fortran routines
  switch(index) {
  case 1:
    mmsuVelocityBC(patch, cellinfo, vars, constvars, time_shift, dt);
    break;
  case 2:
    mmsvVelocityBC(patch, cellinfo, vars, constvars, time_shift, dt);
    break;
  case 3:
    mmswVelocityBC(patch, cellinfo, vars, constvars, time_shift, dt);
    break;
  default:
    cerr << "Invalid Index value" << endl;
    break;
  }
}

//****************************************************************************
// call fortran routine to calculate the MMS U Velocity BC
// Sets the uncorrected velocity values (velRhoHat)!  These should not be
// corrected after the projection so that the values persist to 
// the next time step.
//****************************************************************************
void 
BoundaryCondition::mmsuVelocityBC(const Patch* patch,
				  CellInformation* cellinfo,
				  ArchesVariables* vars,
				  ArchesConstVariables* constvars, 
				  double time_shift,
				  double dt)
{
  int wall_celltypeval = wallCellType();

  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  // Check to see if patch borders a wall
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  double time=d_lab->d_sharedState->getElapsedTime();
  double current_time = time + time_shift;
  //double current_time = time;

  //cout << "PRINTING uVelRhoHat before bc: " << endl;
  //cout << " the time shift = " << time_shift << endl;
  //cout << " current time = " << time << endl;
  //vars->uVelRhoHat.print(cerr);
  
  
  if (xminus) {
    
    int colX = idxLo.x();
    double pi = acos(-1.0);

    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector xminusCell(colX-1, colY, colZ);
	
	if (constvars->cellType[xminusCell] == wall_celltypeval){
	  // Directly set the hat velocity

	  if (d_mms == "constantMMS"){
	    vars->uVelRhoHat[currCell] = cu;
	    vars->uVelRhoHat[xminusCell] = cu;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->uVelRhoHat[currCell] = 
	      cu * cellinfo->xu[colX] + current_time;
	    vars->uVelRhoHat[xminusCell] = 
	      cu * cellinfo->xu[colX-1] + current_time;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	    vars->uVelRhoHat[currCell] = 1 - amp * cos(2.0*pi*( cellinfo->xu[colX] - current_time))
	                                         * sin(2.0*pi*( cellinfo->yy[colY] - current_time))
	                                         * exp(-2.0*d_viscosity*current_time);
	    vars->uVelRhoHat[xminusCell] = 1 - amp * cos(2.0*pi*( cellinfo->xu[colX-1] - current_time))
	                                           * sin(2.0*pi*( cellinfo->yy[colY] - current_time))
	                                         * exp(-2.0*d_viscosity*current_time);

	  }
	  
	}
      }
    }
  }

  if (xplus) {
  
    double pi = acos(-1.0);

    int colX = idxHi.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector xplusCell(colX+1, colY, colZ);
	IntVector xplusplusCell(colX+2,colY,colZ);
	
	if (constvars->cellType[xplusCell] == wall_celltypeval){

	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->uVelRhoHat[xplusCell] = cu;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->uVelRhoHat[xplusCell] = 
	      cu * cellinfo->xu[colX+1] + current_time;
	    //vars->uVelRhoHat[xplusplusCell] = 
	    //  cu * cellinfo->xu[colX+2] + current_time;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	    vars->uVelRhoHat[xplusCell] = 1 - amp * cos(2.0*pi*( cellinfo->xu[colX+1] - current_time))
	      * sin(2.0*pi*( cellinfo->yy[colY] - current_time))
	      * exp(-2.0*d_viscosity*current_time);
	  }
	}
      }
    }
  }

  if (yminus) {
    int colY = idxLo.y();
    double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector yminusCell(colX, colY-1, colZ);
	
	if (constvars->cellType[yminusCell] == wall_celltypeval){

	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->uVelRhoHat[yminusCell] = cu;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->uVelRhoHat[yminusCell] =
	      cu * cellinfo->xu[colX] + current_time;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	    vars->uVelRhoHat[yminusCell] = 1 - amp * cos(2.0*pi*( cellinfo->xu[colX] - current_time ))
	                                           * sin(2.0*pi*( cellinfo->yy[colY-1] - current_time ))
	                                           * exp(-2.0*d_viscosity*current_time);
	  }
	  
	}
      }
    }
  }

  if (yplus) {
    int colY = idxHi.y();
    double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector yplusCell(colX, colY+1, colZ);

	if (constvars->cellType[yplusCell] == wall_celltypeval){

	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->uVelRhoHat[yplusCell] = cu;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->uVelRhoHat[yplusCell] = 
	      cu * cellinfo->xu[colX] + current_time;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	    
	    vars->uVelRhoHat[yplusCell] = 1 - amp * cos(2.0*pi*( cellinfo->xu[colX] - current_time ))
	                                          * sin(2.0*pi*( cellinfo->yy[colY+1] - current_time ))
	                                          * exp(-2.0*d_viscosity*current_time);
	  }
	  
	}
      }
    }
  }


  if (zminus) {
    int colZ = idxLo.z();
    double pi = acos(-1.0);
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector zminusCell(colX, colY, colZ-1);
	
	if (constvars->cellType[zminusCell] == wall_celltypeval){

	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->uVelRhoHat[zminusCell] = cu;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->uVelRhoHat[zminusCell] = 
	      cu * cellinfo->xu[colX] + current_time;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	    
	    vars->uVelRhoHat[zminusCell] = 1 - amp * cos(2.0*pi*( cellinfo->xu[colX] - current_time ))
	                                          * sin(2.0*pi*( cellinfo->yy[colY] - current_time ))
	                                          * exp(-2.0*d_viscosity*current_time);
	  }	    
	}
      }
    }
  }

  if (zplus) {
    int colZ = idxHi.z();
    //double pi = acos(-1.0);
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector zplusCell(colX, colY, colZ+1);
	
	if (constvars->cellType[zplusCell] == wall_celltypeval){
	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->uVelRhoHat[zplusCell] = cu;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->uVelRhoHat[zplusCell] = 
	      cu * cellinfo->xu[colX] + current_time;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	    double pi = acos(-1.0);
	    vars->uVelRhoHat[zplusCell] = 1 - amp * cos(2.0*pi* ( cellinfo->xu[colX] - current_time ))
	                                          * sin(2.0*pi* ( cellinfo->yy[colY] - current_time ))
	                                          * exp(-2.0*d_viscosity*current_time);
	  }	    
	}
      }
    }
  }
  //cout << "PRINTING uVelRhoHat after bc: " << endl;
  //vars->uVelRhoHat.print(cerr);
}
  
//****************************************************************************
// call fortran routine to calculate the MMS V Velocity BC
//****************************************************************************
void 
BoundaryCondition::mmsvVelocityBC(const Patch* patch,
				  CellInformation* cellinfo,
				  ArchesVariables* vars,
				  ArchesConstVariables* constvars,
				  double time_shift, 
				  double dt) 
{
  int wall_celltypeval = wallCellType();

  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  // Check to see if patch borders a wall
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  double time=d_lab->d_sharedState->getElapsedTime();
  double current_time=time + time_shift;
  
  if (xminus) {
    int colX = idxLo.x();
    double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector xminusCell(colX-1, colY, colZ);
	
	if (constvars->cellType[xminusCell] == wall_celltypeval){
	  
	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->vVelRhoHat[xminusCell] = cv;
	  }
	  else if (d_mms == "gao1MMS"){
	    //add real function once I have the parameters...
	    vars->vVelRhoHat[xminusCell] = 
	      cv * cellinfo->yv[colY] + current_time;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	    vars->vVelRhoHat[xminusCell] = 1 + amp * sin(2.0*pi* ( cellinfo->xx[colX-1] - current_time ))
	                                           * cos(2.0*pi* ( cellinfo->yv[colY] - current_time ))
	                                           * exp(-2.0*d_viscosity*current_time);
	  }
	  
	}
      }
    }
  }
  
  if (xplus) {
    int colX = idxHi.x();
    double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector xplusCell(colX+1, colY, colZ);
	
	if (constvars->cellType[xplusCell] == wall_celltypeval){

	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->vVelRhoHat[xplusCell] = cv;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->vVelRhoHat[xplusCell] = 
	      cv * cellinfo->yv[colY] + current_time;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	    vars->vVelRhoHat[xplusCell] = 1 + amp * sin(2.0*pi*( cellinfo->xx[colX+1] - current_time ))
	                                         * cos(2.0*pi*( cellinfo->yv[colY] - current_time ))
	                                         * exp(-2.0*d_viscosity*current_time);
	  }
	  
	}
      }
    }
  }
  
  if (yminus) {
    int colY = idxLo.y();
    double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector yminusCell(colX, colY-1, colZ);
	
	if (constvars->cellType[yminusCell] == wall_celltypeval){
	  
	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->vVelRhoHat[yminusCell] = cv;
	    vars->vVelRhoHat[currCell]   = cv;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->vVelRhoHat[yminusCell] = 
	      cv * cellinfo->yv[colY-1] + current_time;
	    vars->vVelRhoHat[currCell]   = 
	      cv * cellinfo->yv[colY] + current_time;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	    vars->vVelRhoHat[yminusCell] = 1 + amp * sin(2.0*pi*( cellinfo->xx[colX] - current_time ))
	                                          * cos(2.0*pi*( cellinfo->yv[colY-1] - current_time ))
	                                          * exp(-2.0*d_viscosity*current_time);

	    vars->vVelRhoHat[currCell]   = 1 + amp * sin(2.0*pi*( cellinfo->xx[colX] - current_time ))
	                                          * cos(2.0*pi*( cellinfo->yv[colY] - current_time ))
	                                          * exp(-2.0*d_viscosity*current_time);

	  }
	  
	}
      }
    }
  }
  if (yplus) {
    int colY = idxHi.y();
    double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector yplusCell(colX, colY+1, colZ);
	IntVector yplusplusCell(colX, colY+2, colZ);
	
	if (constvars->cellType[yplusCell] == wall_celltypeval){

	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->vVelRhoHat[yplusCell] = cv;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->vVelRhoHat[yplusCell] = 
	      cv * cellinfo->yv[colY+1] + current_time;
	    //vars->vVelRhoHat[yplusplusCell] = 
	    // cv * cellinfo->yv[colY+2] + current_time;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	    vars->vVelRhoHat[yplusCell] = 1 + amp * sin(2.0*pi*( cellinfo->xx[colX] - current_time ))
	                                         * cos(2.0*pi*( cellinfo->yv[colY+1] - current_time ))
	                                         * exp(-2.0*d_viscosity*current_time);
	  }
	  
	}
      }
    }
  }
  if (zminus) {
    int colZ = idxLo.z();
    double pi = acos(-1.0);
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector zminusCell(colX, colY, colZ-1);
	
	if (constvars->cellType[zminusCell] == wall_celltypeval){

	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->vVelRhoHat[zminusCell] = cv;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->vVelRhoHat[zminusCell] = 
	      cv * cellinfo->yv[colY] + current_time;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	    vars->vVelRhoHat[zminusCell] = 1 + amp * sin(2.0*pi*( cellinfo->xx[colX] - current_time ))
	                                          * cos(2.0*pi*( cellinfo->yv[colY] - current_time ))
	                                          * exp(-2.0*d_viscosity*current_time);
	  }	    
	}
      }
    }
  }
  if (zplus) {
    int colZ = idxHi.z();
    double pi = acos(-1.0);
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector zplusCell(colX, colY, colZ+1);
	
	if (constvars->cellType[zplusCell] == wall_celltypeval){

	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->vVelRhoHat[zplusCell] = cv;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->vVelRhoHat[zplusCell] = 
	      cv * cellinfo->yv[colY] + current_time;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	    vars->vVelRhoHat[zplusCell] = 1 + amp * sin(2.0*pi*( cellinfo->xx[colX] - current_time ))
	                                         * cos(2.0*pi*( cellinfo->yv[colY] - current_time ))
	                                         * exp(-2.0*d_viscosity*current_time);
	  }	    
	}
      }
    }
  }
}

//****************************************************************************
// call fortran routine to calculate the MMS W Velocity BC
//****************************************************************************
void 
BoundaryCondition::mmswVelocityBC(const Patch* patch,
				  CellInformation* cellinfo,
				  ArchesVariables* vars,
				  ArchesConstVariables* constvars,
				  double time_shift, 
				  double dt) 
{
  int wall_celltypeval = wallCellType();

  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  // Check to see if patch borders a wall
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  double time=d_lab->d_sharedState->getElapsedTime();
  double current_time = time + time_shift;

  //currently only supporting sinemms in x-y plane
  
  if (xminus) {
    int colX = idxLo.x();
    //double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector xminusCell(colX-1, colY, colZ);
	
	if (constvars->cellType[xminusCell] == wall_celltypeval){

	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->wVelRhoHat[xminusCell] = cw;
	  }
	  else if (d_mms == "gao1MMS"){
	    //add real function once I have the parameters...
	    vars->wVelRhoHat[xminusCell] = 
	      cw * cellinfo->zw[colZ] + current_time;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	    vars->wVelRhoHat[xminusCell] = 0.0;
	  }
	  
	}
      }
    }
  }
  
  if (xplus) {
    int colX = idxHi.x();
    //double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector xplusCell(colX+1, colY, colZ);
	
	if (constvars->cellType[xplusCell] == wall_celltypeval){

	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->wVelRhoHat[xplusCell] = cw;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->wVelRhoHat[xplusCell] = 
	      cw * cellinfo->zw[colZ] + current_time;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	    vars->wVelRhoHat[xplusCell] = 0.0;
	  }
	  
	}
      }
    }
  }
  
  if (yminus) {
    int colY = idxLo.y();
    //double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector yminusCell(colX, colY-1, colZ);
	
	if (constvars->cellType[yminusCell] == wall_celltypeval){

	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->wVelRhoHat[yminusCell] = cw;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->wVelRhoHat[yminusCell] = 
	      cw * cellinfo->zw[colZ] + current_time;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	    vars->wVelRhoHat[yminusCell] = 0.0;
	  }
	  
	}
      }
    }
  }
  if (yplus) {
    int colY = idxHi.y();
    //double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector yplusCell(colX, colY+1, colZ);
	
	if (constvars->cellType[yplusCell] == wall_celltypeval){

	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->wVelRhoHat[yplusCell] = cw;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->wVelRhoHat[yplusCell] =
	      cw * cellinfo->zw[colZ] + current_time;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	    vars->wVelRhoHat[yplusCell] = 0.0;
	  }
	  
	}
      }
    }
  }
  if (zminus) {
    int colZ = idxLo.z();
    //double pi = acos(-1.0);
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector zminusCell(colX, colY, colZ-1);
	
	if (constvars->cellType[zminusCell] == wall_celltypeval){

	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->wVelRhoHat[currCell]   = cw;
	    vars->wVelRhoHat[zminusCell] = cw;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->wVelRhoHat[currCell]   = 
	      cw * cellinfo->zw[colZ] + current_time;
	    vars->wVelRhoHat[zminusCell] = 
	      cw * cellinfo->zw[colZ-1] + current_time;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	    vars->wVelRhoHat[currCell]   = 0.0;
	    vars->wVelRhoHat[zminusCell] = 0.0;
	  }	    
	}
      }
    }
  }
  if (zplus) {
    int colZ = idxHi.z();
    //double pi = acos(-1.0);
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector zplusCell(colX, colY, colZ+1);
	IntVector zplusplusCell(colX, colY, colZ+2);
	
	if (constvars->cellType[zplusCell] == wall_celltypeval){

	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->wVelRhoHat[zplusCell] = cw;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->wVelRhoHat[zplusCell] = 
	      cw * cellinfo->zw[colZ+1] + current_time;
	    //vars->wVelRhoHat[zplusplusCell] = 
	    // cw * cellinfo->zw[colZ+2] + current_time;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	    vars->wVelRhoHat[zplusCell] = 0.0;
	  }	    
	}
      }
    }
  }
}

// //****************************************************************************
// // Actually compute the MMS pressure bcs
// //****************************************************************************
void 
BoundaryCondition::mmspressureBC(const ProcessorGroup*,
			      const Patch* patch,
			      DataWarehouse* /*old_dw*/,
			      DataWarehouse* /*new_dw*/,
			      CellInformation* /*cellinfo*/,
			      ArchesVariables* vars,
			      ArchesConstVariables* constvars)
{
  //this routine is not used since Wall boundary conditions should set the 
  // pressure coefs accordingly for MMS

  // Get the low and high index for the patch
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

  if (xminus) {
    int colX = idxLo.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector xminusCell(colX-1, colY, colZ);
	
	if (constvars->cellType[xminusCell] == wall_celltypeval){

	  if (d_mms == "gao1MMS"){
	    //add real function once I have the parameters...
	    //vars->pressCoeff[Arches::AP] = 0.0;
	    //vars->pressNonlinearSrc[xminusCell] = 1.0;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	  }
	  
	}
      }
    }
  }
  
  if (xplus) {
    int colX = idxHi.x()+1;
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
	
	IntVector currCell(colX, colY, colZ);
	IntVector xplusCell(colX+1, colY, colZ);
	
	if (constvars->cellType[xplusCell] == wall_celltypeval){
	  // Directly set the hat velocity
	  if (d_mms == "gao1MMS"){
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	  }
	  
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
	
	if (constvars->cellType[yminusCell] == wall_celltypeval){
	  // Directly set the hat velocity
	  if (d_mms == "gao1MMS"){
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	  }
	  
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
	
	if (constvars->cellType[yplusCell] == wall_celltypeval){
	  // Directly set the hat velocity
	  if (d_mms == "gao1MMS"){
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	  }
	  
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
	
	if (constvars->cellType[zminusCell] == wall_celltypeval){
	  // Directly set the hat velocity
	  if (d_mms == "gao1MMS"){
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	  }	    
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
	
	if (constvars->cellType[zplusCell] == wall_celltypeval){
	  // Directly set the hat velocity
	  if (d_mms == "gao1MMS"){
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	  }	    
	}
      }
    }
  }

}

//****************************************************************************
// Actually compute the MMS scalar bcs
//****************************************************************************
void 
BoundaryCondition::mmsscalarBC(const ProcessorGroup*,
			       const Patch* patch,
			       CellInformation* cellinfo,
			       ArchesVariables* vars,
			       ArchesConstVariables* constvars,
			       double time_shift,
			       double dt)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  // Get the wall boundary and flow field codes
  int wall_celltypeval = wallCellType();

  //double time=d_lab->d_sharedState->getElapsedTime();
  //double current_time = time + time_shift;

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

	if (constvars->cellType[xminusCell] == wall_celltypeval){

	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->scalar[xminusCell] = phi0;
	  }
	  else if (d_mms == "gao1MMS"){
	    //add real function once I have the parameters...
	    vars->scalar[xminusCell] = phi0;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	  }
	  
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

	if (constvars->cellType[xplusCell] == wall_celltypeval){

	  // Directly set the hat velocity
 	  if (d_mms == "constantMMS"){
	    vars->scalar[xplusCell] = phi0;
	  }
 	  else if (d_mms == "gao1MMS"){
	    vars->scalar[xplusCell] = phi0;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	  }
	  
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
	
	if (constvars->cellType[yminusCell] == wall_celltypeval){

	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->scalar[yminusCell] = phi0;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->scalar[yminusCell] = phi0;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	  }
	  
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
	
	if (constvars->cellType[yplusCell] == wall_celltypeval){

	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->scalar[yplusCell] = phi0;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->scalar[yplusCell] = phi0;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	  }
	  
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
	
	if (constvars->cellType[zminusCell] == wall_celltypeval){

	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->scalar[zminusCell] = phi0;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->scalar[zminusCell] = phi0;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	  }	    
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
	
	if (constvars->cellType[zplusCell] == wall_celltypeval){

	  // Directly set the hat velocity
	  if (d_mms == "constantMMS"){
	    vars->scalar[zplusCell] = phi0;
	  }
	  else if (d_mms == "gao1MMS"){
	    vars->scalar[zplusCell] = phi0;
	  }
	  else if (d_mms == "thornock1MMS"){
	  }
	  else if (d_mms == "almgrenMMS"){
	  }	    
	}
      }
    }
  }
}
//****************************************************************************
// Schedule  prefill
//****************************************************************************
void 
BoundaryCondition::sched_Prefill(SchedulerP& sched, const PatchSet* patches,
				    const MaterialSet* matls)
{
  Task* tsk = scinew Task("BoundaryCondition::Prefill",
			  this,
			  &BoundaryCondition::Prefill);

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  for (int ii = 0; ii < d_numInlets; ii++) {
    tsk->requires(Task::NewDW, d_flowInlets[ii]->d_area_label);
    tsk->requires(Task::NewDW, d_flowInlets[ii]->d_flowRate_label);
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

  tsk->modifies(d_lab->d_scalarSPLabel);

  if (d_calcExtraScalars)
    for (int i=0; i < static_cast<int>(d_extraScalars->size()); i++) {
      tsk->modifies(d_extraScalars->at(i)->getScalarLabel());
    }

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Actually set flat profile at flow inlet boundary
//****************************************************************************
void 
BoundaryCondition::Prefill(const ProcessorGroup* /*pc*/,
				  const PatchSubset* patches,
				  const MaterialSubset*,
				  DataWarehouse*,
				  DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    CCVariable<double> density;
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    SFCXVariable<double> uVelRhoHat;
    SFCYVariable<double> vVelRhoHat;
    SFCZVariable<double> wVelRhoHat;
    CCVariable<double> scalar;
    CCVariable<double> reactscalar;
    CCVariable<double> enthalpy;

    new_dw->getModifiable(uVelocity, d_lab->d_uVelocitySPBCLabel, matlIndex, patch);
    new_dw->getModifiable(vVelocity, d_lab->d_vVelocitySPBCLabel, matlIndex, patch);
    new_dw->getModifiable(wVelocity, d_lab->d_wVelocitySPBCLabel, matlIndex, patch);
    new_dw->getModifiable(uVelRhoHat, d_lab->d_uVelRhoHatLabel, matlIndex, patch);
    new_dw->getModifiable(vVelRhoHat, d_lab->d_vVelRhoHatLabel, matlIndex, patch);
    new_dw->getModifiable(wVelRhoHat, d_lab->d_wVelRhoHatLabel, matlIndex, patch);
    
    new_dw->getModifiable(density, d_lab->d_densityCPLabel, matlIndex, patch);
    new_dw->getModifiable(scalar, d_lab->d_scalarSPLabel, matlIndex, patch);
    if (d_reactingScalarSolve)
      new_dw->getModifiable(reactscalar, d_lab->d_reactscalarSPLabel, matlIndex, patch);
    
    if (d_enthalpySolve) 
      new_dw->getModifiable(enthalpy, d_lab->d_enthalpySPLabel, matlIndex, patch);


    // loop thru the flow inlets to set all the components of velocity and density
    if (d_inletBoundary) {
      double time = 0.0;
      double ramping_factor;
      Box patchInteriorBox = patch->getInteriorBox();
      for (int indx = 0; indx < d_numInlets; indx++) {
        sum_vartype area_var;
        new_dw->get(area_var, d_flowInlets[indx]->d_area_label);
        double area = area_var;
        delt_vartype flow_r;
        new_dw->get(flow_r, d_flowInlets[indx]->d_flowRate_label);
	double flow_rate = flow_r;
        FlowInlet* fi = d_flowInlets[indx];
        fort_get_ramping_factor(fi->d_ramping_inlet_flowrate,
                                time, ramping_factor);
        if (fi->d_prefill) {
          int nofGeomPieces = (int)fi->d_prefillGeomPiece.size();
          for (int ii = 0; ii < nofGeomPieces; ii++) {
            GeometryPieceP  piece = fi->d_prefillGeomPiece[ii];
            Box geomBox = piece->getBoundingBox();
            Box b = geomBox.intersect(patchInteriorBox);
            if (!(b.degenerate())) {
              for (CellIterator iter = patch->getCellCenterIterator(b);
                !iter.done(); iter++) {
                Point p = patch->cellPosition(*iter);
                if (piece->inside(p)) {
                  if (fi->d_prefill_index == 1) {
                    Point p_shift = patch->cellPosition(*iter-IntVector(1,0,0));
                    if (piece->inside(p_shift))
                      uVelocity[*iter] = flow_rate/
                                       (fi->calcStream.d_density * area);
                  }
                  if (fi->d_prefill_index == 2) {
                    Point p_shift = patch->cellPosition(*iter-IntVector(0,1,0));
                    if (piece->inside(p_shift))
                      vVelocity[*iter] = flow_rate/
                                       (fi->calcStream.d_density * area);
                  }
                  if (fi->d_prefill_index == 3) {
                    Point p_shift = patch->cellPosition(*iter-IntVector(0,0,1));
                    if (piece->inside(p_shift))
                      wVelocity[*iter] = flow_rate/
                                       (fi->calcStream.d_density * area);
                  }
                  density[*iter] = fi->calcStream.d_density;
                  scalar[*iter] = fi->streamMixturefraction.d_mixVars[0];
                  if (d_enthalpySolve)
                    enthalpy[*iter] = fi->calcStream.d_enthalpy;
                  if (d_reactingScalarSolve)
                    reactscalar[*iter] = fi->streamMixturefraction.d_rxnVars[0];
                }
              }
            }
          }
        }
      }
    }
    uVelRhoHat.copyData(uVelocity); 
    vVelRhoHat.copyData(vVelocity); 
    wVelRhoHat.copyData(wVelocity); 

    if (d_calcExtraScalars)
      for (int i=0; i < static_cast<int>(d_extraScalars->size()); i++) {
        CCVariable<double> extra_scalar;
        new_dw->getModifiable(extra_scalar,
                              d_extraScalars->at(i)->getScalarLabel(), 
		              matlIndex, patch);
        string extra_scalar_name = d_extraScalars->at(i)->getScalarName();
        if (d_inletBoundary) {
          Box patchInteriorBox = patch->getInteriorBox();
          for (int indx = 0; indx < d_numInlets; indx++) {
            FlowInlet* fi = d_flowInlets[indx];
            if (fi->d_prefill) {
              int BC_ID = fi->d_cellTypeID;
              double extra_scalar_value=0.0;
              for (int j=0; j < static_cast<int>(d_extraScalarBCs.size()); j++)
                if ((d_extraScalarBCs[j]->d_scalar_name == extra_scalar_name)&&
                    (d_extraScalarBCs[j]->d_BC_ID) == BC_ID)
                  extra_scalar_value = d_extraScalarBCs[j]->d_scalarBC_value;
              int nofGeomPieces = (int)fi->d_prefillGeomPiece.size();
              for (int ii = 0; ii < nofGeomPieces; ii++) {
                GeometryPieceP  piece = fi->d_prefillGeomPiece[ii];
                Box geomBox = piece->getBoundingBox();
                Box b = geomBox.intersect(patchInteriorBox);
                if (!(b.degenerate())) {
                  for (CellIterator iter = patch->getCellCenterIterator(b);
                    !iter.done(); iter++) {
                    Point p = patch->cellPosition(*iter);
                    if (piece->inside(p)) {
                      extra_scalar[*iter] = extra_scalar_value;
                    }
                  }
                }
              }
            }
          }
        }
      }
  }
}

