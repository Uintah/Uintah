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
#include <Packages/Uintah/Core/Grid/Stencil.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/Properties.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/Reductions.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/Grid/UnionGeometryPiece.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Containers/StaticArray.h>
#include <iostream>
#include <Core/Math/MiscMath.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

#include <Packages/Uintah/CCA/Components/Arches/fortran/celltypeInit_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/areain_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/profscalar_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/calpbc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/hatvelcalpbc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/inlbcs_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/denaccum_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcinout_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/inlpresbcinout_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/outarea_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/outletbc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/outletbcenth_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/outletbcrscal_fort.h>
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
				     PhysicalConstants* phyConsts,
				     Properties* props,
				     bool calcReactScalar,
				     bool calcEnthalpy):
                                     d_lab(label), d_MAlab(MAlb),
				     d_physicalConsts(phyConsts), 
				     d_props(props),
				     d_reactingScalarSolve(calcReactScalar),
				     d_enthalpySolve(calcEnthalpy)
{
  d_nofScalars = d_props->getNumMixVars();
  MM_CUTOFF_VOID_FRAC = 0.01;
  d_wallBdry = 0;
  d_pressureBdry = 0;
  d_outletBC = 0;
}

//****************************************************************************
// Destructor
//****************************************************************************
BoundaryCondition::~BoundaryCondition()
{
  delete d_wallBdry;
  delete d_pressureBdry;
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
  for (ProblemSpecP inlet_db = db->findBlock("FlowInlet");
       inlet_db != 0; inlet_db = inlet_db->findNextBlock("FlowInlet")) {
    d_flowInlets.push_back(FlowInlet(numMixingScalars, total_cellTypes));
    d_flowInlets[d_numInlets].problemSetup(inlet_db);
    d_cellTypes.push_back(total_cellTypes);
    // compute density and other dependent properties
    d_props->computeInletProperties(
                      d_flowInlets[d_numInlets].streamMixturefraction,
		      d_flowInlets[d_numInlets].calcStream);
    ++total_cellTypes;
    ++d_numInlets;
  }
  if (ProblemSpecP wall_db = db->findBlock("WallBC")) {
    d_wallBdry = scinew WallBdry(total_cellTypes);
    d_wallBdry->problemSetup(wall_db);
    d_cellTypes.push_back(total_cellTypes);
    ++total_cellTypes;
  }
  else {
    cerr << "Wall boundary not specified" << endl;
  }
  
  if (ProblemSpecP press_db = db->findBlock("PressureBC")) {
    d_pressBoundary = true;
    d_pressureBdry = scinew PressureInlet(numMixingScalars, total_cellTypes);
    d_pressureBdry->problemSetup(press_db);
    // compute density and other dependent properties
    d_props->computeInletProperties(
                		 d_pressureBdry->streamMixturefraction, 
		        	 d_pressureBdry->calcStream);
    d_cellTypes.push_back(total_cellTypes);
    ++total_cellTypes;
  }
  else {
    d_pressBoundary = false;
  }
  
  if (ProblemSpecP outlet_db = db->findBlock("outletBC")) {
    d_outletBoundary = true;
    d_outletBC = scinew FlowOutlet(numMixingScalars, total_cellTypes);
    d_outletBC->problemSetup(outlet_db);
    // compute density and other dependent properties
    d_props->computeInletProperties(
			       d_outletBC->streamMixturefraction, 
			       d_outletBC->calcStream);
    d_cellTypes.push_back(total_cellTypes);
    ++total_cellTypes;
  }
  else {
    d_outletBoundary = false;
  }
  if (ProblemSpecP intrusion_db = db->findBlock("intrusions")) {
    d_intrusionBoundary = true;
    d_intrusionBC = scinew IntrusionBdry(total_cellTypes);
    d_intrusionBC->problemSetup(intrusion_db);
    d_cellTypes.push_back(total_cellTypes);
    ++total_cellTypes;
  }
  else {
    d_intrusionBoundary = false;
  }
  // if multimaterial then add an id for multimaterial wall
  if (d_MAlab) 
    d_mmWallID = total_cellTypes;
  else
    d_mmWallID = -9; // invalid cell type

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
 
#ifdef ARCHES_GEOM_DEBUG
    cerr << "Just before geom init" << endl;
#endif
    // initialize CCVariable to -1 which corresponds to flowfield
    int celltypeval;
    fort_celltypeinit(idxLo, idxHi, cellType, d_flowfieldCellTypeVal);
    
    // Find the geometry of the patch
    Box patchBox = patch->getBox();
#ifdef ARCHES_GEOM_DEBUG
    cerr << "Patch box = " << patchBox << endl;
#endif
    
    // wall boundary type
    {
      int nofGeomPieces = (int)d_wallBdry->d_geomPiece.size();
      for (int ii = 0; ii < nofGeomPieces; ii++) {
	GeometryPiece*  piece = d_wallBdry->d_geomPiece[ii];
	Box geomBox = piece->getBoundingBox();
#ifdef ARCHES_GEOM_DEBUG
	cerr << "Wall Geometry box = " << geomBox << endl;
#endif
	Box b = geomBox.intersect(patchBox);
#ifdef ARCHES_GEOM_DEBUG
	cerr << "Wall Intersection box = " << b << endl;
	cerr << "Just before geom wall "<< endl;
#endif
	// check for another geometry
	if (!(b.degenerate())) {
	  CellIterator iter = patch->getCellIterator(b);
	  IntVector idxLo = iter.begin();
	  IntVector idxHi = iter.end() - IntVector(1,1,1);
	  celltypeval = d_wallBdry->d_cellTypeID;
#ifdef ARCHES_GEOM_DEBUG
	  cerr << "Wall cell type val = " << celltypeval << endl;
#endif
	  fort_celltypeinit(idxLo, idxHi, cellType, celltypeval);
	}
      }
    }
    // initialization for pressure boundary
    {
      if (d_pressBoundary) {
	int nofGeomPieces = (int)d_pressureBdry->d_geomPiece.size();
	for (int ii = 0; ii < nofGeomPieces; ii++) {
	  GeometryPiece*  piece = d_pressureBdry->d_geomPiece[ii];
	  Box geomBox = piece->getBoundingBox();
#ifdef ARCHES_GEOM_DEBUG
	  cerr << "Pressure Geometry box = " << geomBox << endl;
#endif
	  Box b = geomBox.intersect(patchBox);
#ifdef ARCHES_GEOM_DEBUG
	  cerr << "Pressure Intersection box = " << b << endl;
#endif
	  // check for another geometry
	  if (!(b.degenerate())) {
	    CellIterator iter = patch->getCellIterator(b);
	    IntVector idxLo = iter.begin();
	    IntVector idxHi = iter.end() - IntVector(1,1,1);
	    celltypeval = d_pressureBdry->d_cellTypeID;
#ifdef ARCHES_GEOM_DEBUG
	    cerr << "Pressure Bdry  cell type val = " << celltypeval << endl;
#endif
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
#ifdef ARCHES_GEOM_DEBUG
	  cerr << "Outlet Geometry box = " << geomBox << endl;
#endif
	  Box b = geomBox.intersect(patchBox);
#ifdef ARCHES_GEOM_DEBUG
	  cerr << "Outlet Intersection box = " << b << endl;
#endif
	  // check for another geometry
	  if (!(b.degenerate())) {
	    CellIterator iter = patch->getCellIterator(b);
	    IntVector idxLo = iter.begin();
	    IntVector idxHi = iter.end() - IntVector(1,1,1);
	    celltypeval = d_outletBC->d_cellTypeID;
#ifdef ARCHES_GEOM_DEBUG
	    cerr << "Flow Outlet cell type val = " << celltypeval << endl;
#endif
	    fort_celltypeinit(idxLo, idxHi, cellType, celltypeval);
	  }
	}
      }
    }
    // set boundary type for inlet flow field
    for (int ii = 0; ii < d_numInlets; ii++) {
      int nofGeomPieces = (int)d_flowInlets[ii].d_geomPiece.size();
      for (int jj = 0; jj < nofGeomPieces; jj++) {
	GeometryPiece*  piece = d_flowInlets[ii].d_geomPiece[jj];
	Box geomBox = piece->getBoundingBox();
#ifdef ARCHES_GEOM_DEBUG
	cerr << "Inlet " << ii << " Geometry box = " << geomBox << endl;
#endif
	Box b = geomBox.intersect(patchBox);
#ifdef ARCHES_GEOM_DEBUG
	cerr << "Inlet " << ii << " Intersection box = " << b << endl;
#endif
	// check for another geometry
	if (b.degenerate())
	  continue; // continue the loop for other inlets
	// iterates thru box b, converts from geometry space to index space
	// make sure this works
#if 0
	CellIterator iter = patch->getCellIterator(b);
	IntVector idxLo = iter.begin();
	IntVector idxHi = iter.end() - IntVector(1,1,1);
	celltypeval = d_flowInlets[ii].d_cellTypeID;
#ifdef ARCHES_GEOM_DEBUG
	cerr << "Flow inlet " << ii << " cell type val = " << celltypeval << endl;
#endif
	fort_celltypeinit(idxLo, idxHi, cellType, celltypeval);
#endif
	for (CellIterator iter = patch->getCellIterator(b); !iter.done(); iter++) {
	  Point p = patch->cellPosition(*iter);
	  if (piece->inside(p)) 
	    cellType[*iter] = d_flowInlets[ii].d_cellTypeID;
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
	    CellIterator iter = patch->getCellIterator(b);
	    IntVector idxLo = iter.begin();
	    IntVector idxHi = iter.end() - IntVector(1,1,1);
	    celltypeval = d_intrusionBC->d_cellTypeID;
	    fort_celltypeinit(idxLo, idxHi, cellType, celltypeval);
	  }
	}
      }
    }


#ifdef ARCHES_GEOM_DEBUG
    // Testing if correct values have been put
    cerr << " In C++ (BoundaryCondition.cc) after flow inlet init " << endl;
    cerr.setf(ios_base::right, ios_base::adjustfield);
    //cerr.setf(ios_base::showpoint);
    cerr.precision(3);
    cerr.setf(ios_base::scientific, ios_base::floatfield);
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "Celltypes for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(2);
	  cerr << cellType[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif
  }
}  

// for multimaterial
//****************************************************************************
// schedule the initialization of mm wall cell types
//****************************************************************************
void 
BoundaryCondition::sched_mmWallCellTypeInit(SchedulerP& sched, const PatchSet* patches,
					    const MaterialSet* matls)
{
  // cell type initialization
  Task* tsk = scinew Task("BoundaryCondition::mmWallCellTypeInit",
			  this,
			  &BoundaryCondition::mmWallCellTypeInit);
  
  int numGhostcells = 0;
  tsk->requires(Task::NewDW, d_MAlab->void_frac_CCLabel, 
		Ghost::None, numGhostcells);
  tsk->requires(Task::OldDW, d_lab->d_cellTypeLabel, 
		Ghost::None, numGhostcells);
  tsk->computes(d_lab->d_mmcellTypeLabel);
  tsk->computes(d_lab->d_mmgasVolFracLabel);
  
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
				      DataWarehouse* new_dw)		
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    int numGhostcells = 0;
    constCCVariable<int> cellType;
    old_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch,
		Ghost::None, numGhostcells);
    constCCVariable<double> voidFrac;
    new_dw->get(voidFrac, d_MAlab->void_frac_CCLabel, matlIndex, patch,
		Ghost::None, numGhostcells);
    CCVariable<int> mmcellType;
    new_dw->allocateAndPut(mmcellType, d_lab->d_mmcellTypeLabel, matlIndex, patch);
    mmcellType.copyData(cellType);
    CCVariable<double> mmvoidFrac;
    new_dw->allocateAndPut(mmvoidFrac, d_lab->d_mmgasVolFracLabel, matlIndex, patch);
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
  tsk->computes(d_lab->d_mmgasVolFracLabel);
  
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
    new_dw->allocateAndPut(mmvoidFrac, d_lab->d_mmgasVolFracLabel, matlIndex, patch);
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
	CellIterator iter = patch->getCellIterator(b);
	IntVector idxLo = iter.begin();
	IntVector idxHi = iter.end() - IntVector(1,1,1);
	
	// Calculate the inlet area
	double inlet_area;
	int cellid = d_flowInlets[ii].d_cellTypeID;
	
	/*#define ARCHES_GEOM_DEBUG*/
#ifdef ARCHES_GEOM_DEBUG
	cerr << "Domain Lo = [" << domLo.x() << "," <<domLo.y()<< "," <<domLo.z()
	     << "] Domain hi = [" << domHi.x() << "," <<domHi.y()<< "," <<domHi.z() 
	     << "]" << endl;
	cerr << "Index Lo = [" << idxLo.x() << "," <<idxLo.y()<< "," <<idxLo.z()
	     << "] Index hi = [" << idxHi.x() << "," <<idxHi.y()<< "," <<idxHi.z()
	     << "]" << endl;
	cerr << "Cell ID = " << cellid << endl;
#endif
	bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
	bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
	bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
	bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
	bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
	bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

	fort_areain(domLo, domHi, idxLo, idxHi, cellInfo->sew, cellInfo->sns,
		    cellInfo->stb, inlet_area, cellType, cellid,
		    xminus, xplus, yminus, yplus, zminus, zplus);
	
#ifdef ARCHES_GEOM_DEBUG
	cerr << "Inlet area = " << inlet_area << endl;
#endif
	
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
  Task* tsk = scinew Task("BoundaryCondition::calcPressureBC",
			  this,
			  &BoundaryCondition::calcPressureBC);

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

  tsk->requires(Task::NewDW, d_lab->d_pressureINLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPLabel, Ghost::AroundFaces,
		Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPLabel, Ghost::AroundFaces,
		Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPLabel, Ghost::AroundFaces,
		Arches::ONEGHOSTCELL);
      // This task computes new uVelocity, vVelocity and wVelocity
  tsk->computes(d_lab->d_pressureSPBCLabel);
  tsk->computes(d_lab->d_uVelocitySPBCLabel);
  tsk->computes(d_lab->d_vVelocitySPBCLabel);
  tsk->computes(d_lab->d_wVelocitySPBCLabel);

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Actually calculate the pressure BCs
//****************************************************************************
void 
BoundaryCondition::calcPressureBC(const ProcessorGroup* ,
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
    new_dw->allocateAndPut(uVelocity, d_lab->d_uVelocitySPBCLabel, matlIndex, patch,
		     Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->allocateAndPut(vVelocity, d_lab->d_vVelocitySPBCLabel, matlIndex, patch,
		     Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->allocateAndPut(wVelocity, d_lab->d_wVelocitySPBCLabel, matlIndex, patch,
		     Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->allocateAndPut(pressure, d_lab->d_pressureSPBCLabel, matlIndex, patch);
    new_dw->copyOut(pressure, d_lab->d_pressureINLabel, matlIndex, patch);
    new_dw->copyOut(uVelocity, d_lab->d_uVelocitySPLabel, matlIndex, patch,
		     Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->copyOut(vVelocity, d_lab->d_vVelocitySPLabel, matlIndex, patch,
		     Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->copyOut(wVelocity, d_lab->d_wVelocitySPLabel, matlIndex, patch,
		     Ghost::AroundFaces, Arches::ONEGHOSTCELL);
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
		d_pressureBdry->d_cellTypeID,
		d_pressureBdry->refPressure,
		cellinfo->dxepu, cellinfo->dynpv, cellinfo->dztpw,
		xminus, xplus, yminus, yplus, zminus, zplus);
    
    // Put the calculated data into the new DW
    // allocateAndPut instead:
    /* new_dw->put(uVelocity, d_lab->d_uVelocitySPBCLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(vVelocity, d_lab->d_vVelocitySPBCLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(wVelocity, d_lab->d_wVelocitySPBCLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(pressure, d_lab->d_pressureSPBCLabel, matlIndex, patch); */;

#ifdef ARCHES_BC_DEBUG
    // Testing if correct values have been put
    cerr << " After CALPBC : " << endl;
    cerr << "Print Pressure : " << endl;
    pressure.print(cerr);
    cerr << " After CALPBC : " << endl;
    cerr << "Print U velocity : " << endl;
    uVelocity.print(cerr);
    cerr << "Print V velocity : " << endl;
    vVelocity.print(cerr);
    cerr << "Print W velocity : " << endl;
    wVelocity.print(cerr);
#endif
    
  } 
}

//****************************************************************************
// Schedule the setting of inlet velocity BC
//****************************************************************************
void 
BoundaryCondition::sched_setInletVelocityBC(SchedulerP& sched, const PatchSet* patches,
					    const MaterialSet* matls)
{
  Task* tsk = scinew Task("BoundaryCondition::setInletVelocityBC",
			  this,
			  &BoundaryCondition::setInletVelocityBC);
  
  // This task requires densityCP, [u,v,w]VelocitySP from new_dw
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  // changes to make it work for the task graph
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,  
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityINLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityINLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityINLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  
  // This task computes new density, uVelocity, vVelocity and wVelocity
  tsk->computes(d_lab->d_uVelocitySIVBCLabel);
  tsk->computes(d_lab->d_vVelocitySIVBCLabel);
  tsk->computes(d_lab->d_wVelocitySIVBCLabel);
  
  sched->addTask(tsk, patches, matls);
  
}

void
BoundaryCondition::sched_computeFlowINOUT(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls)
{
  Task* tsk = scinew Task("BoundaryCondition::computeFlowINOUT",
			  this,
			  &BoundaryCondition::computeFlowINOUT);
  
  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  
  // This task requires densityCP, [u,v,w]VelocitySP from new_dw
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::AroundCells,
		Arches::ONEGHOSTCELL);
  tsk->requires(Task::OldDW, d_lab->d_densityINLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  // changes to make it work for the task graph
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,  
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityCPBCLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityCPBCLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityCPBCLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  
  // This task computes new density, uVelocity, vVelocity and wVelocity
  tsk->computes(d_lab->d_totalflowINLabel);
  tsk->computes(d_lab->d_totalflowOUTLabel);
  tsk->computes(d_lab->d_totalflowOUToutbcLabel);
  tsk->computes(d_lab->d_totalAreaOUTLabel);
  tsk->computes(d_lab->d_denAccumLabel);
  
  sched->addTask(tsk, patches, matls);
  
}

void 
BoundaryCondition::computeFlowINOUT(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset*,
				    DataWarehouse* old_dw,
				    DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    constCCVariable<int> cellType;
    constCCVariable<double> old_density;
    constCCVariable<double> density;
    constSFCXVariable<double> uVelocity;
    constSFCYVariable<double> vVelocity;
    constSFCZVariable<double> wVelocity;

    // get cellType, velocity and density
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch,
		Ghost::AroundCells,
		Arches::ONEGHOSTCELL);
    new_dw->get(uVelocity, d_lab->d_uVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->get(vVelocity, d_lab->d_vVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->get(wVelocity, d_lab->d_wVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->get(density, d_lab->d_densityINLabel, matlIndex, patch, 
		Ghost::AroundCells,
		Arches::ONEGHOSTCELL);
    old_dw->get(old_density, d_lab->d_densityINLabel, matlIndex, patch, 
		Ghost::None,
		Arches::ZEROGHOSTCELLS);
    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();
   
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
    double flowOUT_outbc = 0.0;
    double areaOUT = 0.0;
    double denAccum = 0.0;
    fort_denaccum(idxLo, idxHi, density, old_density, denAccum, delta_t,
		  cellinfo->sew, cellinfo->sns, cellinfo->stb);
    if (xminus||xplus||yminus||yplus||zminus||zplus) {

      for (int indx = 0; indx < d_numInlets; indx++) {

	// Get a copy of the current flowinlet
	// assign flowType the value that corresponds to flow
	//CellTypeInfo flowType = FLOW;
	FlowInlet fi = d_flowInlets[indx];
	fort_bcinout(uVelocity, vVelocity, wVelocity, idxLo, idxHi, density,
		     cellType, fi.d_cellTypeID, delta_t, flowIN, flowOUT,
		     cellinfo->sew, cellinfo->sns, cellinfo->stb,
		     xminus, xplus, yminus, yplus, zminus, zplus);
      } 

      if (d_pressBoundary) {
	int press_celltypeval = d_pressureBdry->d_cellTypeID;
	fort_bcinout(uVelocity, vVelocity, wVelocity, idxLo, idxHi, density,
		     cellType, press_celltypeval, delta_t, flowIN, flowOUT,
		     cellinfo->sew, cellinfo->sns, cellinfo->stb,
		     xminus, xplus, yminus, yplus, zminus, zplus);
      }
      if (d_outletBoundary) {
	int out_celltypeval = d_outletBC->d_cellTypeID;
	fort_outarea(idxLo, idxHi, cellType, density,
		     cellinfo->sew, cellinfo->sns, cellinfo->stb,
		     areaOUT, out_celltypeval,
		     xminus, xplus, yminus, yplus, zminus, zplus);
	fort_bcinout(uVelocity, vVelocity, wVelocity, idxLo, idxHi, density,
		     cellType, out_celltypeval, delta_t, flowIN, flowOUT_outbc,
		     cellinfo->sew, cellinfo->sns, cellinfo->stb,
		     xminus, xplus, yminus, yplus, zminus, zplus);
      }
            
    }
    new_dw->put(sum_vartype(flowIN), d_lab->d_totalflowINLabel);
    new_dw->put(sum_vartype(flowOUT), d_lab->d_totalflowOUTLabel);
    new_dw->put(sum_vartype(flowOUT_outbc), d_lab->d_totalflowOUToutbcLabel);
    new_dw->put(sum_vartype(areaOUT),d_lab->d_totalAreaOUTLabel);
    new_dw->put(sum_vartype(denAccum), d_lab->d_denAccumLabel);
  }
}

void BoundaryCondition::sched_computeOMB(SchedulerP& sched,
					 const PatchSet* patches,
					 const MaterialSet* matls){

  Task* tsk = scinew Task("BoundaryCondition::computeOMB",
			  this,
			  &BoundaryCondition::computeOMB);

  tsk->requires(Task::NewDW, d_lab->d_totalflowINLabel);
  tsk->requires(Task::NewDW, d_lab->d_totalflowOUTLabel);
  tsk->requires(Task::NewDW, d_lab->d_totalflowOUToutbcLabel);
  tsk->requires(Task::NewDW, d_lab->d_totalAreaOUTLabel);
  tsk->requires(Task::NewDW, d_lab->d_denAccumLabel);
  tsk->computes(d_lab->d_uvwoutLabel);
  sched->addTask(tsk, patches, matls);
}

void 
BoundaryCondition::computeOMB(const ProcessorGroup* pc,
			      const PatchSubset*,
			      const MaterialSubset*,
			      DataWarehouse*,
			      DataWarehouse* new_dw)
{
    sum_vartype sum_totalFlowIN, sum_totalFlowOUT, sum_totalFlowOUToutbc,
                sum_totalAreaOUT, sum_denAccum;
    double totalFlowIN, totalFlowOUT, totalFlowOUT_outbc, totalAreaOUT, denAccum;
    new_dw->get(sum_totalFlowIN, d_lab->d_totalflowINLabel);
    new_dw->get(sum_totalFlowOUT, d_lab->d_totalflowOUTLabel);
    new_dw->get(sum_totalFlowOUToutbc, d_lab->d_totalflowOUToutbcLabel);
    new_dw->get(sum_totalAreaOUT, d_lab->d_totalAreaOUTLabel);
    new_dw->get(sum_denAccum, d_lab->d_denAccumLabel);
    totalFlowIN = sum_totalFlowIN;
    totalFlowOUT = sum_totalFlowOUT;
    totalFlowOUT_outbc = sum_totalFlowOUToutbc;
    totalAreaOUT = sum_totalAreaOUT;
    denAccum = sum_denAccum;

    d_overallMB = fabs((totalFlowIN - totalFlowOUT - totalFlowOUT_outbc - 
			denAccum)/totalFlowIN);

    if (d_outletBoundary) {
      if (totalAreaOUT > 0.0) {
#if 0
	d_uvwout = (totalFlowIN - denAccum - totalFlowOUT)/
	            totalAreaOUT;
#else
	d_uvwout = totalFlowOUT_outbc/totalAreaOUT;
#endif
      }
#if 0
      if (d_uvwout < 0.0) 
	d_uvwout = 0.0;
#endif
    }
    else
      d_uvwout = 0.0;
    int me = pc->myrank();
    if (me == 0) {
      if (d_overallMB > 0.0)
	cerr << "Overall Mass Balance " << log10(d_overallMB/1.e-7) << endl;
      cerr << "Total flow in " << totalFlowIN << endl;
      cerr << "Total flow out " << totalFlowOUT << endl;
      cerr << "Total flow out BC: " << totalFlowOUT_outbc << endl;
      cerr << "Overall velocity correction " << d_uvwout << endl;
      cerr << "Total Area out " << totalAreaOUT << endl;
    }
    new_dw->put(delt_vartype(d_uvwout), d_lab->d_uvwoutLabel);
}

//****************************************************************************
// Schedule the compute of Pressure BC
//****************************************************************************
void 
BoundaryCondition::sched_transOutletBC(SchedulerP& sched, 
				       const PatchSet* patches,
				       const MaterialSet* matls)
{
  Task* tsk = scinew Task("BoundaryCondition::transOutletBC",
			  this,
			  &BoundaryCondition::transOutletBC);

  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
#if 0  
  tsk->requires(Task::NewDW, d_lab->d_uVelocityINLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityINLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityINLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW,  d_lab->d_scalarINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  // changes to make it work for the task graph
  tsk->requires(Task::NewDW, d_lab->d_uVelocityCPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityCPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityCPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  for (int ii = 0; ii < d_nofScalars; ii++)
    tsk->requires(Task::NewDW, d_lab->d_scalarCPBCLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_uvwoutLabel);
  if (d_reactingScalarSolve) {
    tsk->requires(Task::NewDW, d_lab->d_reactscalarCPBCLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->computes(d_lab->d_reactscalarOUTBCLabel);
  }

  if (d_enthalpySolve) {
    tsk->requires(Task::NewDW, d_lab->d_enthalpyCPBCLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
#if 0
    tsk->requires(Task::NewDW, d_lab->d_enthalpyINLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
    tsk->computes(d_lab->d_enthalpyOUTBCLabel);
  }

  // This task computes new uVelocity, vVelocity and wVelocity
  tsk->computes(d_lab->d_netflowOUTBCLabel);
  tsk->computes(d_lab->d_uVelocityOUTBCLabel);
  tsk->computes(d_lab->d_vVelocityOUTBCLabel);
  tsk->computes(d_lab->d_wVelocityOUTBCLabel);
  for (int ii = 0; ii < d_nofScalars; ii++)
    tsk->computes(d_lab->d_scalarOUTBCLabel);
  
  sched->addTask(tsk, patches, matls);
}




//****************************************************************************
// Actually calculate the outlet BCs
//****************************************************************************
void 
BoundaryCondition::transOutletBC(const ProcessorGroup* ,
				 const PatchSubset* patches,
				 const MaterialSubset*,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw) 
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    constCCVariable<int> cellType;
    StaticArray<CCVariable<double> > scalar(d_nofScalars);
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    constSFCXVariable<double> old_uVelocity;
    constSFCYVariable<double> old_vVelocity;
    constSFCZVariable<double> old_wVelocity;
    constCCVariable<double> old_scalar;
    constCCVariable<double> density;
    constCCVariable<double> pressure;
    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->allocateAndPut(uVelocity, d_lab->d_uVelocityOUTBCLabel, matlIndex,
		     patch);
    new_dw->allocateAndPut(vVelocity, d_lab->d_vVelocityOUTBCLabel, matlIndex,
		     patch);
    new_dw->allocateAndPut(wVelocity, d_lab->d_wVelocityOUTBCLabel, matlIndex,
		     patch);
    for (int ii = 0; ii < d_nofScalars; ii++) 
      new_dw->allocateAndPut(scalar[ii], d_lab->d_scalarOUTBCLabel, matlIndex,
		       patch);

    // get cellType, pressure and velocity
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->copyOut(uVelocity, d_lab->d_uVelocityCPBCLabel, matlIndex, patch);
    new_dw->copyOut(vVelocity, d_lab->d_vVelocityCPBCLabel, matlIndex, patch);
    new_dw->copyOut(wVelocity, d_lab->d_wVelocityCPBCLabel, matlIndex, patch);
    for (int ii = 0; ii < d_nofScalars; ii++)
      new_dw->copyOut(scalar[ii], d_lab->d_scalarCPBCLabel, matlIndex, patch);

    new_dw->get(old_uVelocity, d_lab->d_uVelocityCPBCLabel,
		matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->get(old_vVelocity, d_lab->d_vVelocityCPBCLabel,
		matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->get(old_wVelocity, d_lab->d_wVelocityCPBCLabel,
		matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->get(old_scalar, d_lab->d_scalarCPBCLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);

    new_dw->get(density, d_lab->d_densityINLabel, matlIndex, patch, Ghost::AroundCells,
		Arches::ONEGHOSTCELL);

    new_dw->get(pressure, d_lab->d_pressurePSLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);

    delt_vartype uvwout_red;
    new_dw->get(uvwout_red, d_lab->d_uvwoutLabel);
    double uvwout = uvwout_red;

  // Get the low and high index for the patch and the variables

    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();

    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    double flowout = 0.0;
    if (d_outletBoundary) {

      fort_outletbc(uVelocity, vVelocity, wVelocity, scalar[0],
		    old_uVelocity, old_vVelocity, old_wVelocity, old_scalar,
		    density, pressure, cellType, d_outletBC->d_cellTypeID, uvwout, flowout,
		    idxLo, idxHi,
		    xminus, xplus, yminus, yplus, zminus, zplus, delta_t,
		    cellinfo->sew, cellinfo->sns, cellinfo->stb,
		    cellinfo->dxpwu, cellinfo->dxpw);

    }

    if (d_reactingScalarSolve) {
      CCVariable<double> reactscalar;
      constCCVariable<double> old_reactscalar;
      new_dw->get(old_reactscalar, d_lab->d_reactscalarCPBCLabel, matlIndex,
                  patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->allocateAndPut(reactscalar, d_lab->d_reactscalarOUTBCLabel, matlIndex, patch);
      new_dw->copyOut(reactscalar, d_lab->d_reactscalarCPBCLabel, matlIndex, patch);
    // assuming outlet to be pos x

      if (d_outletBoundary) {

        fort_outletbcrscal(reactscalar, old_reactscalar, density, idxLo, idxHi,
			  cellType, d_outletBC->d_cellTypeID, uvwout,
			  xminus, xplus, yminus, yplus, zminus, zplus,
			  delta_t, cellinfo->dxpw);

      }
      // allocateAndPut instead:
      /* new_dw->put(reactscalar, d_lab->d_reactscalarOUTBCLabel, matlIndex, patch); */;
    }


    if (d_enthalpySolve) {
      CCVariable<double> enthalpy;
      constCCVariable<double> old_enthalpy;
      new_dw->get(old_enthalpy, d_lab->d_enthalpyCPBCLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->allocateAndPut(enthalpy, d_lab->d_enthalpyOUTBCLabel, matlIndex,
		       patch);
      new_dw->copyOut(enthalpy, d_lab->d_enthalpyCPBCLabel, matlIndex, patch);
            
    // assuming outlet to be pos x

      if (d_outletBoundary) {

	fort_outletbcenth(enthalpy, old_enthalpy, density, idxLo, idxHi,
			  cellType, d_outletBC->d_cellTypeID, uvwout,
			  xminus, xplus, yminus, yplus, zminus, zplus,
			  delta_t, cellinfo->dxpw);

      }
	// allocateAndPut instead:
	/* new_dw->put(enthalpy, d_lab->d_enthalpyOUTBCLabel, matlIndex, patch); */;
    }

  // Put the calculated data into the new DW
    new_dw->put(sum_vartype(flowout), d_lab->d_netflowOUTBCLabel);
  }
} 



void BoundaryCondition::sched_correctOutletBC(SchedulerP& sched,
					      const PatchSet* patches,
					      const MaterialSet* matls){

  Task* tsk = scinew Task("BoundaryCondition::correctOutletBC",
			  this,
			  &BoundaryCondition::correctOutletBC);

  tsk->requires(Task::NewDW, d_lab->d_totalflowINLabel);
  tsk->requires(Task::NewDW, d_lab->d_totalflowOUTLabel);
  tsk->requires(Task::NewDW, d_lab->d_netflowOUTBCLabel);
  tsk->requires(Task::NewDW, d_lab->d_totalAreaOUTLabel);
  tsk->requires(Task::NewDW, d_lab->d_denAccumLabel);
  tsk->modifies(d_lab->d_uVelocityOUTBCLabel);
  tsk->modifies(d_lab->d_vVelocityOUTBCLabel);
  tsk->modifies(d_lab->d_wVelocityOUTBCLabel);
  sched->addTask(tsk, patches, matls);
}

void 
BoundaryCondition::correctOutletBC(const ProcessorGroup* ,
			      const PatchSubset* patches,
			      const MaterialSubset*,
			      DataWarehouse*,
			      DataWarehouse* new_dw)
{
    sum_vartype sum_totalFlowIN, sum_totalFlowOUT, sum_netflowOutbc,
                sum_totalAreaOUT, sum_denAccum;
    double totalFlowIN, totalFlowOUT, netFlowOUT_outbc, totalAreaOUT, denAccum;
    new_dw->get(sum_totalFlowIN, d_lab->d_totalflowINLabel);
    new_dw->get(sum_totalFlowOUT, d_lab->d_totalflowOUTLabel);
    new_dw->get(sum_netflowOutbc, d_lab->d_netflowOUTBCLabel);
    new_dw->get(sum_totalAreaOUT, d_lab->d_totalAreaOUTLabel);
    new_dw->get(sum_denAccum, d_lab->d_denAccumLabel);
			  
    totalFlowIN = sum_totalFlowIN;
    totalFlowOUT = sum_totalFlowOUT;
    netFlowOUT_outbc = sum_netflowOutbc;
    totalAreaOUT = sum_totalAreaOUT;
    denAccum = sum_denAccum;
    double uvwcorr;
    if (d_outletBoundary) {
      if (totalAreaOUT > 0.0) {
	uvwcorr = (totalFlowIN - denAccum - totalFlowOUT-netFlowOUT_outbc)/
	  totalAreaOUT;
#if 1
	if (uvwcorr < 0.0)
	  uvwcorr = 0.0;
#endif
      }
      
    }
    else
      uvwcorr = 0.0;
 
    for (int p = 0; p < patches->size(); p++) {
      const Patch* patch = patches->get(p);
      int archIndex = 0; // only one arches material
      int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
      SFCXVariable<double> uVelocity;
      SFCYVariable<double> vVelocity;
      SFCZVariable<double> wVelocity;
      new_dw->getModifiable(uVelocity, d_lab->d_uVelocityOUTBCLabel,
			    matlIndex, patch);
      new_dw->getModifiable(vVelocity, d_lab->d_vVelocityOUTBCLabel,
			    matlIndex, patch);
      new_dw->getModifiable(wVelocity, d_lab->d_wVelocityOUTBCLabel,
			    matlIndex, patch);

      IntVector indexLow = patch->getCellFORTLowIndex();
      IntVector indexHigh = patch->getCellFORTHighIndex();
      bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
      if (xplus) {
	for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
	  for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	    IntVector currCell(indexHigh.x()+1, colY, colZ);
	    uVelocity[currCell] += uvwcorr;   // assuming outlet is xplus
	    uVelocity[IntVector(indexHigh.x()+2, colY, colZ)] = uVelocity[currCell];
	  }
	}
      }
    }
}




//****************************************************************************
// Schedule the compute of Pressure BC
//****************************************************************************
void 
BoundaryCondition::sched_recomputePressureBC(SchedulerP& sched, 
					     const PatchSet* patches,
					     const MaterialSet* matls)
{
  Task* tsk = scinew Task("BoundaryCondition::recomputePressureBC",
			  this,
			  &BoundaryCondition::recomputePressureBC);

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
      // This task requires celltype, new density, pressure and velocity
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);

  tsk->requires(Task::NewDW, d_lab->d_pressureINLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  // changes to make it work for the task graph
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySIVBCLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySIVBCLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySIVBCLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  for (int ii = 0; ii < d_nofScalars; ii++)
    tsk->requires(Task::NewDW, d_lab->d_scalarINLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);


  if (d_enthalpySolve) {
    tsk->requires(Task::NewDW, d_lab->d_enthalpyINLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->computes(d_lab->d_enthalpyCPBCLabel);
  }
  
  if (d_reactingScalarSolve) {
    tsk->requires(Task::NewDW, d_lab->d_reactscalarINLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->computes(d_lab->d_reactscalarCPBCLabel);
  }

  // This task computes new uVelocity, vVelocity and wVelocity
  tsk->computes(d_lab->d_uVelocityCPBCLabel);
  tsk->computes(d_lab->d_vVelocityCPBCLabel);
  tsk->computes(d_lab->d_wVelocityCPBCLabel);
  tsk->computes(d_lab->d_pressurePSLabel);
  for (int ii = 0; ii < d_nofScalars; ii++)
    tsk->computes(d_lab->d_scalarCPBCLabel);
  
  sched->addTask(tsk, patches, matls);
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
  // This task requires old density, uVelocity, vVelocity and wVelocity
  tsk->requires(Task::NewDW, d_lab->d_uVelocityINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityINLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  // will only work for one scalar variable
  for (int ii = 0; ii < d_props->getNumMixVars(); ii++) 
    tsk->requires(Task::NewDW, d_lab->d_scalarINLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  if (d_enthalpySolve) {
    tsk->requires(Task::NewDW, d_lab->d_enthalpyINLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->computes(d_lab->d_enthalpySPBCLabel);
  }
  if (d_reactingScalarSolve) {
    tsk->requires(Task::NewDW, d_lab->d_reactscalarINLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->computes(d_lab->d_reactscalarSPLabel);
  }
    
  // This task computes new density, uVelocity, vVelocity and wVelocity, scalars
  tsk->modifies(d_lab->d_densityINLabel);
  tsk->computes(d_lab->d_uVelocitySPLabel);
  tsk->computes(d_lab->d_vVelocitySPLabel);
  tsk->computes(d_lab->d_wVelocitySPLabel);

  if (d_conv_scheme > 0) {
    tsk->computes(d_lab->d_maxAbsU_label);
    tsk->computes(d_lab->d_maxAbsV_label);
    tsk->computes(d_lab->d_maxAbsW_label);
  }

  for (int ii = 0; ii < d_props->getNumMixVars(); ii++) 
    tsk->computes(d_lab->d_scalarSPLabel);
  sched->addTask(tsk, patches, matls);
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
  //get Molecular Viscosity of the fluid
  double molViscosity = d_physicalConsts->getMolecularViscosity();

  // Call the fortran routines
  switch(index) {
  case 1:
    uVelocityBC(patch, molViscosity, cellinfo, vars, constvars);
    break;
  case 2:
    vVelocityBC(patch, molViscosity, cellinfo, vars, constvars);
    break;
  case 3:
    wVelocityBC(patch, molViscosity, cellinfo, vars, constvars);
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
			       double VISCOS,
			       CellInformation* cellinfo,
			       ArchesVariables* vars,
		               ArchesConstVariables* constvars)
{
  int wall_celltypeval = d_wallBdry->d_cellTypeID;
  int outlet_celltypeval = -10;
  if (d_outletBoundary)
    outlet_celltypeval = d_outletBC->d_cellTypeID;
  int press_celltypeval = d_pressureBdry->d_cellTypeID;
  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  // for no ghost cells
  
  // ** Reverted back to old ways
  // for a single patch should be equal to 1 and nx
  //IntVector idxLoU = vars->cellType.getFortLowIndex();
  //IntVector idxHiU = vars->cellType.getFortHighIndex();
  // computes momentum source term due to wall
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
	      wall_celltypeval, outlet_celltypeval, press_celltypeval, VISCOS,
	      cellinfo->sewu, cellinfo->sns, cellinfo->stb,
	      cellinfo->yy, cellinfo->yv, cellinfo->zz, cellinfo->zw,
	      xminus, xplus, yminus, yplus, zminus, zplus);

#ifdef ARCHES_BC_DEBUG
  cerr << "AFTER UVELBC_FORT" << endl;
  cerr << "Print UVel" << endl;
  vars->uVelocity.print(cerr);
  cerr << "Print UAP" << endl;
  vars->uVelocityCoeff[Arches::AP].print(cerr);
  cerr << "Print UAW" << endl;
  vars->uVelocityCoeff[Arches::AW].print(cerr);
  cerr << "Print UAE" << endl;
  vars->uVelocityCoeff[Arches::AE].print(cerr);
  cerr << "Print UAN" << endl;
  vars->uVelocityCoeff[Arches::AN].print(cerr);
  cerr << "Print UAS" << endl;
  vars->uVelocityCoeff[Arches::AS].print(cerr);
  cerr << "Print UAT" << endl;
  vars->uVelocityCoeff[Arches::AT].print(cerr);
  cerr << "Print UAB" << endl;
  vars->uVelocityCoeff[Arches::AB].print(cerr);
  cerr << "Print SU for U velocity: " << endl;
  vars->uVelNonlinearSrc.print(cerr);
  cerr << "Print SP for U velocity for:" << endl;
  vars->uVelLinearSrc.print(cerr);
#endif

}

//****************************************************************************
// call fortran routine to calculate the V Velocity BC
//****************************************************************************
void 
BoundaryCondition::vVelocityBC(const Patch* patch,
			       double VISCOS,
			       CellInformation* cellinfo,
			       ArchesVariables* vars,
		               ArchesConstVariables* constvars)
{
  int wall_celltypeval = d_wallBdry->d_cellTypeID;
  int outlet_celltypeval = -10;
  if (d_outletBoundary)
    outlet_celltypeval = d_outletBC->d_cellTypeID;
  int press_celltypeval = d_pressureBdry->d_cellTypeID;
  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  // for a single patch should be equal to 1 and nx
  //IntVector idxLoV = vars->cellType.getFortLowIndex();
  //IntVector idxHiV = vars->cellType.getFortHighIndex();
  // computes momentum source term due to wall
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  // computes remianing diffusion term and also computes source due to gravity
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
	      wall_celltypeval, outlet_celltypeval, press_celltypeval, VISCOS,
	      cellinfo->sew, cellinfo->snsv, cellinfo->stb,
	      cellinfo->xx, cellinfo->xu, cellinfo->zz, cellinfo->zw,
	      xminus, xplus, yminus, yplus, zminus, zplus);

#ifdef ARCHES_BC_DEBUG
  cerr << "AFTER VVELBC_FORT" << endl;
  cerr << "Print VVel" << endl;
  vars->vVelocity.print(cerr);
  cerr << "Print VAP" << endl;
  vars->vVelocityCoeff[Arches::AP].print(cerr);
  cerr << "Print VAW" << endl;
  vars->vVelocityCoeff[Arches::AW].print(cerr);
  cerr << "Print VAE" << endl;
  vars->vVelocityCoeff[Arches::AE].print(cerr);
  cerr << "Print VAN" << endl;
  vars->vVelocityCoeff[Arches::AN].print(cerr);
  cerr << "Print VAS" << endl;
  vars->vVelocityCoeff[Arches::AS].print(cerr);
  cerr << "Print VAT" << endl;
  vars->vVelocityCoeff[Arches::AT].print(cerr);
  cerr << "Print VAB" << endl;
  vars->vVelocityCoeff[Arches::AB].print(cerr);
  cerr << "Print SU for V velocity: " << endl;
  vars->vVelNonlinearSrc.print(cerr);
  cerr << "Print SP for V velocity for:" << endl;
  vars->vVelLinearSrc.print(cerr);
#endif

}

//****************************************************************************
// call fortran routine to calculate the W Velocity BC
//****************************************************************************
void 
BoundaryCondition::wVelocityBC(const Patch* patch,
			       double VISCOS,
			       CellInformation* cellinfo,
			       ArchesVariables* vars,
		               ArchesConstVariables* constvars)
{
  int wall_celltypeval = d_wallBdry->d_cellTypeID;
  int outlet_celltypeval = -10;
  if (d_outletBoundary)
    outlet_celltypeval = d_outletBC->d_cellTypeID;
  int press_celltypeval = d_pressureBdry->d_cellTypeID;
  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  // for no ghost cells
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
  // for a single patch should be equal to 1 and nx
  //IntVector idxLoW = vars->cellType.getFortLowIndex();
  //IntVector idxHiW = vars->cellType.getFortHighIndex();
  // computes momentum source term due to wall
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
	      wall_celltypeval, outlet_celltypeval, press_celltypeval, VISCOS,
	      cellinfo->sew, cellinfo->sns, cellinfo->stbw,
	      cellinfo->xx, cellinfo->xu, cellinfo->yy, cellinfo->yv,
	      xminus, xplus, yminus, yplus, zminus, zplus);

#ifdef ARCHES_BC_DEBUG
  cerr << "AFTER WVELBC_FORT" << endl;
  cerr << "Print WVel" << endl;
  vars->wVelocity.print(cerr);
  cerr << "Print WAP" << endl;
  vars->wVelocityCoeff[Arches::AP].print(cerr);
  cerr << "Print WAW" << endl;
  vars->wVelocityCoeff[Arches::AW].print(cerr);
  cerr << "Print WAE" << endl;
  vars->wVelocityCoeff[Arches::AE].print(cerr);
  cerr << "Print WAN" << endl;
  vars->wVelocityCoeff[Arches::AN].print(cerr);
  cerr << "Print WAS" << endl;
  vars->wVelocityCoeff[Arches::AS].print(cerr);
  cerr << "Print WAT" << endl;
  vars->wVelocityCoeff[Arches::AT].print(cerr);
  cerr << "Print WAB" << endl;
  vars->wVelocityCoeff[Arches::AB].print(cerr);
  cerr << "Print SU for W velocity: " << endl;
  vars->wVelNonlinearSrc.print(cerr);
  cerr << "Print SP for W velocity for:" << endl;
  vars->wVelLinearSrc.print(cerr);
#endif

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
  int wall_celltypeval = d_wallBdry->d_cellTypeID;
  int flow_celltypeval = d_flowfieldCellTypeVal;
  // ** WARNING ** Symmetry is hardcoded to -3
  int symmetry_celltypeval = -3;

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  //fortran call
  fort_bcpress(domLo, domHi, idxLo, idxHi, constvars->pressure,
	       vars->pressCoeff[Arches::AP],
	       vars->pressCoeff[Arches::AE], vars->pressCoeff[Arches::AW],
	       vars->pressCoeff[Arches::AN], vars->pressCoeff[Arches::AS],
	       vars->pressCoeff[Arches::AT], vars->pressCoeff[Arches::AB],
	       vars->pressNonlinearSrc, vars->pressLinearSrc,
	       constvars->cellType, wall_celltypeval, symmetry_celltypeval,
	       flow_celltypeval, xminus, xplus, yminus, yplus, zminus, zplus);

#ifdef ARCHES_BC_DEBUG
  cerr << "AFTER FORT_PRESSBC" << endl;
  cerr << "Print Pressure" << endl;
  vars->pressure.print(cerr);
  cerr << "Print PAP" << endl;
  vars->pressCoeff[Arches::AP].print(cerr);
  cerr << "Print PAW" << endl;
  vars->pressCoeff[Arches::AW].print(cerr);
  cerr << "Print PAE" << endl;
  vars->pressCoeff[Arches::AE].print(cerr);
  cerr << "Print PAN" << endl;
  vars->pressCoeff[Arches::AN].print(cerr);
  cerr << "Print PAS" << endl;
  vars->pressCoeff[Arches::AS].print(cerr);
  cerr << "Print PAT" << endl;
  vars->pressCoeff[Arches::AT].print(cerr);
  cerr << "Print PAB" << endl;
  vars->pressCoeff[Arches::AB].print(cerr);
  cerr << "Print SU for Pressure: " << endl;
  vars->pressNonlinearSrc.print(cerr);
  cerr << "Print SP for Pressure for:" << endl;
  vars->pressLinearSrc.print(cerr);
#endif
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
  int wall_celltypeval = d_wallBdry->d_cellTypeID;
  int press_celltypeval = d_pressureBdry->d_cellTypeID;
  // ** WARNING ** Symmetry/sfield/outletfield/ffield hardcoded to -3,-4,-5, -6
  //               Fmixin hardcoded to 0
  int symmetry_celltypeval = -3;
  int sfield = -4;
  int outletfield = -5;
  if (d_outletBoundary)
    outletfield = d_outletBC->d_cellTypeID;
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
		d_flowInlets[0].d_cellTypeID, press_celltypeval, ffield,
		sfield, outletfield,
		xminus, xplus, yminus, yplus, zminus, zplus);
#ifdef ARCHES_BC_DEBUG
  cerr << "AFTER FORT_SCALARBC" << endl;
  cerr << "Print Scalar" << endl;
  constvars->scalar.print(cerr);
  cerr << "Print scalar coeff, AE:" << endl;
  vars->scalarCoeff[Arches::AE].print(cerr);
#endif
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
  int wall_celltypeval = d_wallBdry->d_cellTypeID;
  int press_celltypeval = d_pressureBdry->d_cellTypeID;
  // ** WARNING ** Symmetry/sfield/outletfield/ffield hardcoded to -3,-4,-5, -6
  //               Fmixin hardcoded to 0
  int symmetry_celltypeval = -3;
  int sfield = -4;
  int outletfield = -5;
  int ffield = -1;
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  //#define enthalpySolve_debug
#ifdef enthalpySolve_debug

      // code to print all values of any variable within
      // a box, for multi-patch case

     IntVector indexLow = patch->getCellLowIndex();
     IntVector indexHigh = patch->getCellHighIndex();

      int ibot = 0;
      int itop = 0;
      int jbot = 8;
      int jtop = 8;
      int kbot = 8;
      int ktop = 8;

      // values above can be changed for each case as desired

      bool printvalues = true;
      int idloX = Max(indexLow.x(),ibot);
      int idhiX = Min(indexHigh.x()-1,itop);
      int idloY = Max(indexLow.y(),jbot);
      int idhiY = Min(indexHigh.y()-1,jtop);
      int idloZ = Max(indexLow.z(),kbot);
      int idhiZ = Min(indexHigh.z()-1,ktop);
      if ((idloX > idhiX) || (idloY > idhiY) || (idloZ > idhiZ))
	printvalues = false;
      printvalues = false;

      if (printvalues) {
	for (int ii = idloX; ii <= idhiX; ii++) {
	  for (int jj = idloY; jj <= idhiY; jj++) {
	    for (int kk = idloZ; kk <= idhiZ; kk++) {
	      cerr.width(14);
	      cerr << " point coordinates "<< ii << " " << jj << " " << kk << endl;
	      cerr << "Source after Radiation" << endl;
	      cerr << "Radiation source = " << vars->src[IntVector(ii,jj,kk)] << endl; 
	      cerr << "Nonlinear source     = " << vars->scalarNonlinearSrc[IntVector(ii,jj,kk)] << endl; 
	    }
	  }
	}
      }

#endif

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
		  press_celltypeval, ffield, sfield, outletfield,
		  xminus, xplus, yminus, yplus, zminus, zplus);

#ifdef ARCHES_BC_DEBUG
  cerr << "AFTER FORT_ENTHALPYBC" << endl;
  cerr << "Print Enthalpy" << endl;
  vars->enthalpy.print(cerr);
  cerr << "Print enthalpy coeff, AE:" << endl;
  vars->scalarCoeff[Arches::AE].print(cerr);
#endif
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
  int wall_celltypeval = d_wallBdry->d_cellTypeID;
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
// Actually set the inlet velocity BC
//****************************************************************************
void 
BoundaryCondition::setInletVelocityBC(const ProcessorGroup* ,
				      const PatchSubset* patches,
				      const MaterialSubset*,
				      DataWarehouse*,
				      DataWarehouse* new_dw) 
{
  double time = d_lab->d_sharedState->getElapsedTime();
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    constCCVariable<int> cellType;
    constCCVariable<double> density;
    constSFCXVariable<double> uVelocity_old;
    constSFCYVariable<double> vVelocity_old;
    constSFCZVariable<double> wVelocity_old;
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;

    // get cellType, velocity and density
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->get(density, d_lab->d_densityINLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);

    new_dw->allocateAndPut(uVelocity, d_lab->d_uVelocitySIVBCLabel, matlIndex, patch);
    new_dw->allocateAndPut(vVelocity, d_lab->d_vVelocitySIVBCLabel, matlIndex, patch);
    new_dw->allocateAndPut(wVelocity, d_lab->d_wVelocitySIVBCLabel, matlIndex, patch);
    new_dw->copyOut(uVelocity, d_lab->d_uVelocityINLabel, matlIndex, patch);
    new_dw->copyOut(vVelocity, d_lab->d_vVelocityINLabel, matlIndex, patch);
    new_dw->copyOut(wVelocity, d_lab->d_wVelocityINLabel, matlIndex, patch);
    
    // Get the low and high index for the patch and the variables
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
    // stores cell type info for the patch with the ghost cell type
    for (int indx = 0; indx < d_numInlets; indx++) {
      // Get a copy of the current flowinlet
      FlowInlet fi = d_flowInlets[indx];
      
      // assign flowType the value that corresponds to flow
      //CellTypeInfo flowType = FLOW;
      bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
      bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
      bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
      bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
      bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
      bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
      fort_inlbcs(uVelocity, vVelocity, wVelocity, idxLo, idxHi,
		  density, cellType, fi.d_cellTypeID, time,
		  xminus, xplus, yminus, yplus, zminus, zplus);
      
    }
    // Put the calculated data into the new DW
    // allocateAndPut instead:
    /* new_dw->put(uVelocity, d_lab->d_uVelocitySIVBCLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(vVelocity, d_lab->d_vVelocitySIVBCLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(wVelocity, d_lab->d_wVelocitySIVBCLabel, matlIndex, patch); */;

#ifdef ARCHES_BC_DEBUG
    cerr << "After presssoln" << endl;
    for(CellIterator iter = patch->getCellIterator();
	!iter.done(); iter++){
      cerr.width(10);
      cerr << "PCELL"<<*iter << ": " << cellType[*iter] << "\n" ; 
    }
    
    cerr << " After INLBCS : " << endl;
    for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
      cerr << "U velocity for ii = " << ii << endl;
      for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
	for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	  cerr.width(10);
	  cerr << uVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << " After INLBCS : " << endl;
    for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
      cerr << "V velocity for ii = " << ii << endl;
      for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
	for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	  cerr.width(10);
	  cerr << vVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << " After INLBCS : " << endl;
    for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
      cerr << "W velocity for ii = " << ii << endl;
      for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
	for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	  cerr.width(10);
	  cerr << wVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif
  }
}

template<class T> void rewindow(T& data, const IntVector& low, const IntVector& high)
{
   T newdata;
   newdata.allocate(low, high);
   newdata.copy(data, low, high);
   data=newdata;
}

//****************************************************************************
// Actually calculate the pressure BCs
//****************************************************************************
void 
BoundaryCondition::recomputePressureBC(const ProcessorGroup* ,
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
    StaticArray<CCVariable<double> > scalar(d_nofScalars);
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    new_dw->allocateAndPut(uVelocity, d_lab->d_uVelocityCPBCLabel, matlIndex, patch,
		     Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->allocateAndPut(vVelocity, d_lab->d_vVelocityCPBCLabel, matlIndex, patch,
		     Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->allocateAndPut(wVelocity, d_lab->d_wVelocityCPBCLabel, matlIndex, patch,
		     Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->allocateAndPut(pressure, d_lab->d_pressurePSLabel, matlIndex, patch);
    for (int ii = 0; ii < d_nofScalars; ii++) 
      new_dw->allocateAndPut(scalar[ii], d_lab->d_scalarCPBCLabel, matlIndex, patch);

    
    // get cellType, pressure and velocity
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->get(density, d_lab->d_densityINLabel, matlIndex, patch, 
		Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->get(viscosity, d_lab->d_viscosityINLabel, matlIndex, patch,
		Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->copyOut(pressure, d_lab->d_pressureINLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->copyOut(uVelocity, d_lab->d_uVelocitySIVBCLabel, matlIndex, patch,
		     Ghost::AroundFaces);
    new_dw->copyOut(vVelocity, d_lab->d_vVelocitySIVBCLabel, matlIndex, patch,
		     Ghost::AroundFaces);
    new_dw->copyOut(wVelocity, d_lab->d_wVelocitySIVBCLabel, matlIndex, patch,
		     Ghost::AroundFaces);
    for (int ii = 0; ii < d_nofScalars; ii++)
      new_dw->copyOut(scalar[ii], d_lab->d_scalarINLabel, matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    PerPatch<CellInformationP> cellInfoP;

    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 

      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    else {

      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    }

    CellInformation* cellinfo = cellInfoP.get().get_rep();

  // Get the low and high index for the patch and the variables
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
#ifdef ARCHES_BC_DEBUG
    cerr << "idxLo" << idxLo << " idxHi" << idxHi << endl;
    cerr << "domLo" << domLoScalar << " domHi" << domHiScalar << endl;
    cerr << "domLoU" << domLoU << " domHiU" << domHiU << endl;
    cerr << "domLoV" << domLoV << " domHiV" << domHiV << endl;
    cerr << "domLoW" << domLoW << " domHiW" << domHiW << endl;
    cerr << "pressID" << d_pressureBdry->d_cellTypeID << endl;
    for(CellIterator iter = patch->getCellIterator();
	!iter.done(); iter++){
      cerr.width(10);
      cerr << "PPP"<<*iter << ": " << pressure[*iter] << "\n" ; 
    }
#endif
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    fort_calpbc(uVelocity, vVelocity, wVelocity, idxLo, idxHi,
		pressure, density, viscosity, cellType, 
		d_pressureBdry->d_cellTypeID,
		d_pressureBdry->refPressure,
		cellinfo->dxepu, cellinfo->dynpv, cellinfo->dztpw,
		xminus, xplus, yminus, yplus, zminus, zplus);
    // set values of the scalars on the scalar boundary
#if 0
    for (int ii = 0; ii < d_nofScalars; ii++) {
      fort_profscalar(idxLo, idxHi, scalar[ii], cellType,
		      d_pressureBdry->streamMixturefraction.d_mixVars[ii],
		      d_pressureBdry->d_cellTypeID,
		      xminus, xplus, yminus, yplus, zminus, zplus);
    }
#endif
    if (d_reactingScalarSolve) {
      CCVariable<double> reactscalar;
      new_dw->allocateAndPut(reactscalar, d_lab->d_reactscalarCPBCLabel, matlIndex, patch);
      new_dw->copyOut(reactscalar, d_lab->d_reactscalarINLabel, matlIndex, patch, 
		      Ghost::None, Arches::ZEROGHOSTCELLS);
      // Get the low and high index for the patch and the variables
#if 0
      fort_profscalar(idxLo, idxHi, reactscalar, cellType,
		      d_pressureBdry->calcStream.d_reactscalar,
		      d_pressureBdry->d_cellTypeID,
		      xminus, xplus, yminus, yplus, zminus, zplus);
#endif
      // allocateAndPut instead:
      /* new_dw->put(reactscalar, d_lab->d_reactscalarCPBCLabel, matlIndex, patch); */;
    }

    if (d_enthalpySolve) {
      CCVariable<double> enthalpy;
      new_dw->allocateAndPut(enthalpy, d_lab->d_enthalpyCPBCLabel, matlIndex, patch);
      new_dw->copyOut(enthalpy, d_lab->d_enthalpyINLabel, matlIndex, patch, 
		      Ghost::None, Arches::ZEROGHOSTCELLS);
      // Get the low and high index for the patch and the variables
#if 0
      fort_profscalar(idxLo, idxHi, enthalpy, cellType,
		      d_pressureBdry->calcStream.d_enthalpy,
		      d_pressureBdry->d_cellTypeID,
		      xminus, xplus, yminus, yplus, zminus, zplus);
#endif
      // allocateAndPut instead:
      /* new_dw->put(enthalpy, d_lab->d_enthalpyCPBCLabel, matlIndex, patch); */;
    }

#ifdef ARCHES_BC_DEBUG
    cerr << "After recomputecalpbc" << endl;
    uVelocity.print(cerr);
    cerr << "print vvelocity" << endl;
    vVelocity.print(cerr);
    cerr << "print pressure" << endl;
    
    cerr << "After recomputecalpbc print pressure" << endl;
    if (patch->containsCell(IntVector(2,3,3))) {
      cerr << "[2,3,3] press[2,3,3]" << pressure[IntVector(2,3,3)] << endl;
    }
    if (patch->containsCell(IntVector(1,3,3))) {
      cerr << "[2,3,3] press[1,3,3]" << pressure[IntVector(1,3,3)] << endl;
    }
    
    //  rewindow(pressure, patch->getCellLowIndex(), patch->getCellHighIndex());
    
    cerr << "recompute calpbc: pressure=\n";
    pressure.print(cerr);
    
    cerr << "print pcell" << endl;
    cellType.print(cerr);
    cerr << "print RHO"<< endl;
    density.print(cerr);
#endif

  // Put the calculated data into the new DW
    // allocateAndPut instead:
    /* new_dw->put(uVelocity, d_lab->d_uVelocityCPBCLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(vVelocity, d_lab->d_vVelocityCPBCLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(wVelocity, d_lab->d_wVelocityCPBCLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(pressure, d_lab->d_pressurePSLabel, matlIndex, patch); */;
    for (int ii = 0; ii < d_nofScalars; ii++) 
      // allocateAndPut instead:
      /* new_dw->put(scalar[ii], d_lab->d_scalarCPBCLabel, matlIndex, patch); */;
  }
} 

// modified pressure bc
void 
BoundaryCondition::newrecomputePressureBC(const ProcessorGroup* /*pc*/,
					  const Patch* patch,
					  CellInformation* cellinfo,
					  ArchesVariables* vars)
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

    constCCVariable<double> density(vars->density);
    constCCVariable<double> viscosity(vars->viscosity);
    constCCVariable<int> cellType(vars->cellType);
    fort_hatvelcalpbc(vars->uVelRhoHat, vars->vVelRhoHat, vars->wVelRhoHat, 
		      idxLo, idxHi,
		      vars->pressure, density, cellType,
		      d_pressureBdry->d_cellTypeID,
		      d_pressureBdry->refPressure,
		      cellinfo->dxepu, cellinfo->dynpv, cellinfo->dztpw,
		      xminus, xplus, yminus, yplus, zminus, zplus);

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
    StaticArray<CCVariable<double> > scalar(d_nofScalars);
    CCVariable<double> reactscalar;
    CCVariable<double> enthalpy;

    new_dw->allocateAndPut(uVelocity, d_lab->d_uVelocitySPLabel, matlIndex, patch);
    new_dw->allocateAndPut(vVelocity, d_lab->d_vVelocitySPLabel, matlIndex, patch);
    new_dw->allocateAndPut(wVelocity, d_lab->d_wVelocitySPLabel, matlIndex, patch);
    if (d_reactingScalarSolve)
      new_dw->allocateAndPut(reactscalar, d_lab->d_reactscalarSPLabel, matlIndex, patch);
    for (int ii =0; ii < d_nofScalars; ii++) {
      new_dw->allocateAndPut(scalar[ii], d_lab->d_scalarSPLabel, matlIndex, patch);
    }
    if (d_enthalpySolve)
      new_dw->allocateAndPut(enthalpy, d_lab->d_enthalpySPBCLabel, matlIndex, patch);
    
    // get cellType, density and velocity
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->getModifiable(density, d_lab->d_densityINLabel, matlIndex, patch);
    new_dw->copyOut(uVelocity, d_lab->d_uVelocityINLabel, matlIndex, patch,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->copyOut(vVelocity, d_lab->d_vVelocityINLabel, matlIndex, patch,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->copyOut(wVelocity, d_lab->d_wVelocityINLabel, matlIndex, patch,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
    for (int ii = 0; ii < d_nofScalars; ii++) {
      new_dw->copyOut(scalar[ii], d_lab->d_scalarINLabel, matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
    }
    if (d_reactingScalarSolve) {
      new_dw->copyOut(reactscalar, d_lab->d_reactscalarINLabel, matlIndex,
		      patch, Ghost::None, Arches::ZEROGHOSTCELLS);
      // reactscalar will be zero at the boundaries, so no further calculation
      // is required.
    }
    IntVector domLoEnth;
    IntVector domHiEnth;
    
    if (d_enthalpySolve) {
      new_dw->copyOut(enthalpy, d_lab->d_enthalpyINLabel, matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
    // Get the low and high index for the patch and the variables
      domLoEnth = enthalpy.getFortLowIndex();
      domHiEnth = enthalpy.getFortHighIndex();
    }
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

    // loop thru the flow inlets to set all the components of velocity and density
    for (int indx = 0; indx < d_numInlets; indx++) {
      sum_vartype area_var;
      new_dw->get(area_var, d_flowInlets[indx].d_area_label);
      double area = area_var;
      
      // Get a copy of the current flowinlet
      // check if given patch intersects with the inlet boundary of type index
      FlowInlet fi = d_flowInlets[indx];
      //std::cerr << " inlet area" << area << " flowrate" << fi.flowRate << endl;
      //cerr << "density=" << fi.density << '\n';
      fort_profv(uVelocity, vVelocity, wVelocity, idxLo, idxHi,
		 cellType, area, fi.d_cellTypeID, fi.flowRate,
		 fi.calcStream.d_density,
		 xminus, xplus, yminus, yplus, zminus, zplus);

      fort_profscalar(idxLo, idxHi, density, cellType,
		      fi.calcStream.d_density, fi.d_cellTypeID,
		      xminus, xplus, yminus, yplus, zminus, zplus);
      if (d_enthalpySolve)
      fort_profscalar(idxLo, idxHi, enthalpy, cellType,
		      fi.calcStream.d_enthalpy, fi.d_cellTypeID,
		      xminus, xplus, yminus, yplus, zminus, zplus);

    }   
    if (d_pressureBdry) {
      // set density
      fort_profscalar(idxLo, idxHi, density, cellType,
		      d_pressureBdry->calcStream.d_density,
		      d_pressureBdry->d_cellTypeID,
		      xminus, xplus, yminus, yplus, zminus, zplus);
      if (d_enthalpySolve){
	fort_profscalar(idxLo, idxHi, enthalpy, cellType,
			d_pressureBdry->calcStream.d_enthalpy,
			d_pressureBdry->d_cellTypeID,
			xminus, xplus, yminus, yplus, zminus, zplus);
      }
    }    
    for (int indx = 0; indx < d_nofScalars; indx++) {
      for (int ii = 0; ii < d_numInlets; ii++) {
	double scalarValue = d_flowInlets[ii].streamMixturefraction.d_mixVars[indx];
	fort_profscalar(idxLo, idxHi, scalar[indx], cellType,
			scalarValue, d_flowInlets[ii].d_cellTypeID,
			xminus, xplus, yminus, yplus, zminus, zplus);
      }
      if (d_pressBoundary) {
	double scalarValue = d_pressureBdry->streamMixturefraction.d_mixVars[indx];
	fort_profscalar(idxLo, idxHi, scalar[indx], cellType, scalarValue,
			d_pressureBdry->d_cellTypeID,
			xminus, xplus, yminus, yplus, zminus, zplus);
      }
    }
    

    if (d_conv_scheme > 0) {
    double maxAbsU = 0.0;
    double maxAbsV = 0.0;
    double maxAbsW = 0.0;
    double temp_absU, temp_absV, temp_absW;
    IntVector indexLow;
    IntVector indexHigh;
    
      indexLow = patch->getSFCXFORTLowIndex();
      indexHigh = patch->getSFCXFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);

	      temp_absU = Abs(uVelocity[currCell]);
	      if (temp_absU > maxAbsU) maxAbsU = temp_absU;
          }
        }
      }
      new_dw->put(max_vartype(maxAbsU), d_lab->d_maxAbsU_label); 

      indexLow = patch->getSFCYFORTLowIndex();
      indexHigh = patch->getSFCYFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);

	      temp_absV = Abs(vVelocity[currCell]);
	      if (temp_absV > maxAbsV) maxAbsV = temp_absV;
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

	      temp_absW = Abs(wVelocity[currCell]);
	      if (temp_absW > maxAbsW) maxAbsW = temp_absW;
          }
        }
      }
      new_dw->put(max_vartype(maxAbsW), d_lab->d_maxAbsW_label); 
    }
    
#ifdef ARCHES_BC_DEBUG
    // Testing if correct values have been put
    cerr << "In set flat profile : " << endl;
    cerr << "DomLo = (" << domLo.x() << "," << domLo.y() << "," << domLo.z() << ")\n";
    cerr << "DomHi = (" << domHi.x() << "," << domHi.y() << "," << domHi.z() << ")\n";
    cerr << "DomLoU = (" << domLoU.x()<<","<<domLoU.y()<< "," << domLoU.z() << ")\n";
    cerr << "DomHiU = (" << domHiU.x()<<","<<domHiU.y()<< "," << domHiU.z() << ")\n";
    cerr << "DomLoV = (" << domLoV.x()<<","<<domLoV.y()<< "," << domLoV.z() << ")\n";
    cerr << "DomHiV = (" << domHiV.x()<<","<<domHiV.y()<< "," << domHiV.z() << ")\n";
    cerr << "DomLoW = (" << domLoW.x()<<","<<domLoW.y()<< "," << domLoW.z() << ")\n";
    cerr << "DomHiW = (" << domHiW.x()<<","<<domHiW.y()<< "," << domHiW.z() << ")\n";
    cerr << "IdxLo = (" << idxLo.x() << "," << idxLo.y() << "," << idxLo.z() << ")\n";
    cerr << "IdxHi = (" << idxHi.x() << "," << idxHi.y() << "," << idxHi.z() << ")\n";
    cerr << "IdxLoU = (" << idxLoU.x()<<","<<idxLoU.y()<< "," << idxLoU.z() << ")\n";
    cerr << "IdxHiU = (" << idxHiU.x()<<","<<idxHiU.y()<< "," << idxHiU.z() << ")\n";
    cerr << "IdxLoV = (" << idxLoV.x()<<","<<idxLoV.y()<< "," << idxLoV.z() << ")\n";
    cerr << "IdxHiV = (" << idxHiV.x()<<","<<idxHiV.y()<< "," << idxHiV.z() << ")\n";
    cerr << "IdxLoW = (" << idxLoW.x()<<","<<idxLoW.y()<< "," << idxLoW.z() << ")\n";
    cerr << "IdxHiW = (" << idxHiW.x()<<","<<idxHiW.y()<< "," << idxHiW.z() << ")\n";
    
    cerr << " After Set Flat Profile : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "Density for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << density[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << " After Set Flat Profile : " << endl;
    for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
      cerr << "U velocity for ii = " << ii << endl;
      for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
	for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	  cerr.width(10);
	  cerr << uVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << " After Set Flat Profile : " << endl;
    for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
      cerr << "V velocity for ii = " << ii << endl;
      for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
	for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	  cerr.width(10);
	  cerr << vVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << " After Set Flat Profile : " << endl;
    for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
      cerr << "W velocity for ii = " << ii << endl;
      for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
	for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	  cerr.width(10);
	  cerr << wVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << " After Set Flat Profile : " << endl;
    cerr << " Number of scalars = " << d_nofScalars << endl;
    for (int indx = 0; indx < d_nofScalars; indx++) {
      for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
	cerr << "Scalar " << indx <<" for ii = " << ii << endl;
	for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	  for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	    cerr.width(10);
	    cerr << (scalar[indx])[IntVector(ii,jj,kk)] << " " ; 
	  }
	  cerr << endl;
	}
      }
    }
#endif
  }
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
				       CCVariable<double>& temperature)
{
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	IntVector currCell = IntVector(colX, colY, colZ);	  
	if (cellType[currCell]==d_mmWallID) {
	  temperature[currCell] = solidTemp[currCell];
	  //	  temperature[currCell] = 298.0;
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
  fort_mmwallbc(idxLo, idxHi,
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
BoundaryCondition::FlowInlet::FlowInlet(int /*numMix*/, int cellID):
  d_cellTypeID(cellID)
{
  turb_lengthScale = 0.0;
  flowRate = 0.0;
  // add cellId to distinguish different inlets
  d_area_label = VarLabel::create("flowarea"+cellID,
   ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
}

BoundaryCondition::FlowInlet::FlowInlet():
  d_cellTypeID(0), d_area_label(0)
{
  turb_lengthScale = 0.0;
  flowRate = 0.0;
}

BoundaryCondition::FlowInlet::FlowInlet(const FlowInlet& copy) :
  d_cellTypeID (copy.d_cellTypeID),
  flowRate(copy.flowRate),
  streamMixturefraction(copy.streamMixturefraction),
  turb_lengthScale(copy.turb_lengthScale),
  calcStream(copy.calcStream),
  d_geomPiece(copy.d_geomPiece),
  d_area_label(copy.d_area_label)
{
  d_area_label->addReference();
}

BoundaryCondition::FlowInlet& BoundaryCondition::FlowInlet::operator=(const FlowInlet& copy)
{
  // remove reference from the old label
  VarLabel::destroy(d_area_label);
  d_area_label = copy.d_area_label;
  d_area_label->addReference();

  d_cellTypeID = copy.d_cellTypeID;
  flowRate = copy.flowRate;
  streamMixturefraction = copy.streamMixturefraction;
  turb_lengthScale = copy.turb_lengthScale;
  calcStream = copy.calcStream;
  d_geomPiece = copy.d_geomPiece;

  return *this;
}


BoundaryCondition::FlowInlet::~FlowInlet()
{
  VarLabel::destroy(d_area_label);
}

//****************************************************************************
// Problem Setup for BoundaryCondition::FlowInlet
//****************************************************************************
void 
BoundaryCondition::FlowInlet::problemSetup(ProblemSpecP& params)
{
  params->require("Flow_rate", flowRate);
  params->require("TurblengthScale", turb_lengthScale);
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
// Schedule the compute of Pressure BC
//****************************************************************************
void 
BoundaryCondition::sched_lastcomputePressureBC(SchedulerP& sched, 
					       const PatchSet* patches,
					       const MaterialSet* matls)
{
  Task* tsk = scinew Task("BoundaryCondition::lastcomputePressureBC",
			  this,
			  &BoundaryCondition::lastcomputePressureBC);

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
      // This task requires celltype, new density, pressure and velocity
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  // changes to make it work for the task graph
  tsk->modifies(d_lab->d_pressureSPBCLabel);
  tsk->modifies(d_lab->d_uVelocitySPBCLabel);
  tsk->modifies(d_lab->d_vVelocitySPBCLabel);
  tsk->modifies(d_lab->d_wVelocitySPBCLabel);
  sched->addTask(tsk, patches, matls);
}


void 
BoundaryCondition::lastcomputePressureBC(const ProcessorGroup* ,
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
    CCVariable<double> pressure;
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    new_dw->getModifiable(uVelocity, d_lab->d_uVelocitySPBCLabel,
			  matlIndex, patch);
    new_dw->getModifiable(vVelocity, d_lab->d_vVelocitySPBCLabel,
			  matlIndex, patch);
    new_dw->getModifiable(wVelocity, d_lab->d_wVelocitySPBCLabel,
			  matlIndex, patch);

    new_dw->getModifiable(pressure, d_lab->d_pressureSPBCLabel,
			  matlIndex, patch);
    
    // get cellType, pressure and velocity
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->get(density, d_lab->d_densityCPLabel, matlIndex, patch, 
		Ghost::None,
		Arches::ZEROGHOSTCELLS);
    PerPatch<CellInformationP> cellInfoP;

    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 

      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    else {

      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    }

    CellInformation* cellinfo = cellInfoP.get().get_rep();

  // Get the low and high index for the patch and the variables
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
#ifdef ARCHES_BC_DEBUG
    cerr << "idxLo" << idxLo << " idxHi" << idxHi << endl;
    cerr << "domLo" << domLoScalar << " domHi" << domHiScalar << endl;
    cerr << "domLoU" << domLoU << " domHiU" << domHiU << endl;
    cerr << "domLoV" << domLoV << " domHiV" << domHiV << endl;
    cerr << "domLoW" << domLoW << " domHiW" << domHiW << endl;
    cerr << "pressID" << d_pressureBdry->d_cellTypeID << endl;
    for(CellIterator iter = patch->getCellIterator();
	!iter.done(); iter++){
      cerr.width(10);
      cerr << "PPP"<<*iter << ": " << pressure[*iter] << "\n" ; 
    }
#endif
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    fort_hatvelcalpbc(uVelocity, vVelocity, wVelocity, idxLo, idxHi,
		pressure, density, cellType, 
		d_pressureBdry->d_cellTypeID,
		d_pressureBdry->refPressure,
		cellinfo->dxepu, cellinfo->dynpv, cellinfo->dztpw,
		xminus, xplus, yminus, yplus, zminus, zplus);
    // set values of the scalars on the scalar boundary
  }
}

//****************************************************************************
// Schedule the compute of Pressure BC
//****************************************************************************
void 
BoundaryCondition::sched_predcomputePressureBC(SchedulerP& sched, 
					       const PatchSet* patches,
					       const MaterialSet* matls)
{
  Task* tsk = scinew Task("BoundaryCondition::predcomputePressureBC",
			  this,
			  &BoundaryCondition::predcomputePressureBC);

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
      // This task requires celltype, new density, pressure and velocity
  tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  // changes to make it work for the task graph
  tsk->modifies(d_lab->d_pressurePredLabel);
  tsk->modifies(d_lab->d_uVelocityPredLabel);
  tsk->modifies(d_lab->d_vVelocityPredLabel);
  tsk->modifies(d_lab->d_wVelocityPredLabel);
  sched->addTask(tsk, patches, matls);
}


void 
BoundaryCondition::predcomputePressureBC(const ProcessorGroup* ,
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
    CCVariable<double> pressure;
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    new_dw->getModifiable(uVelocity, d_lab->d_uVelocityPredLabel,
			  matlIndex, patch);
    new_dw->getModifiable(vVelocity, d_lab->d_vVelocityPredLabel,
			  matlIndex, patch);
    new_dw->getModifiable(wVelocity, d_lab->d_wVelocityPredLabel,
			  matlIndex, patch);

    new_dw->getModifiable(pressure, d_lab->d_pressurePredLabel,
			  matlIndex, patch);
    
    // get cellType, pressure and velocity
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->get(density, d_lab->d_densityPredLabel, matlIndex, patch, 
		Ghost::None,
		Arches::ZEROGHOSTCELLS);
    PerPatch<CellInformationP> cellInfoP;

    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 

      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    else {

      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    }

    CellInformation* cellinfo = cellInfoP.get().get_rep();

  // Get the low and high index for the patch and the variables
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
#ifdef ARCHES_BC_DEBUG
    cerr << "idxLo" << idxLo << " idxHi" << idxHi << endl;
    cerr << "domLo" << domLoScalar << " domHi" << domHiScalar << endl;
    cerr << "domLoU" << domLoU << " domHiU" << domHiU << endl;
    cerr << "domLoV" << domLoV << " domHiV" << domHiV << endl;
    cerr << "domLoW" << domLoW << " domHiW" << domHiW << endl;
    cerr << "pressID" << d_pressureBdry->d_cellTypeID << endl;
    for(CellIterator iter = patch->getCellIterator();
	!iter.done(); iter++){
      cerr.width(10);
      cerr << "PPP"<<*iter << ": " << pressure[*iter] << "\n" ; 
    }
#endif
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    fort_hatvelcalpbc(uVelocity, vVelocity, wVelocity, idxLo, idxHi,
		pressure, density, cellType, 
		d_pressureBdry->d_cellTypeID,
		d_pressureBdry->refPressure,
		cellinfo->dxepu, cellinfo->dynpv, cellinfo->dztpw,
		xminus, xplus, yminus, yplus, zminus, zplus);
    // set values of the scalars on the scalar boundary
  }
}

//****************************************************************************
// Schedule the compute of Pressure BC
//****************************************************************************
void 
BoundaryCondition::sched_intermcomputePressureBC(SchedulerP& sched, 
					       const PatchSet* patches,
					       const MaterialSet* matls)
{
  Task* tsk = scinew Task("BoundaryCondition::intermcomputePressureBC",
			  this,
			  &BoundaryCondition::intermcomputePressureBC);

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
      // This task requires celltype, new density, pressure and velocity
  tsk->requires(Task::NewDW, d_lab->d_densityIntermLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  // changes to make it work for the task graph
  tsk->modifies(d_lab->d_pressureIntermLabel);
  tsk->modifies(d_lab->d_uVelocityIntermLabel);
  tsk->modifies(d_lab->d_vVelocityIntermLabel);
  tsk->modifies(d_lab->d_wVelocityIntermLabel);
  sched->addTask(tsk, patches, matls);
}


void 
BoundaryCondition::intermcomputePressureBC(const ProcessorGroup* ,
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
    CCVariable<double> pressure;
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    new_dw->getModifiable(uVelocity, d_lab->d_uVelocityIntermLabel,
			  matlIndex, patch);
    new_dw->getModifiable(vVelocity, d_lab->d_vVelocityIntermLabel,
			  matlIndex, patch);
    new_dw->getModifiable(wVelocity, d_lab->d_wVelocityIntermLabel,
			  matlIndex, patch);

    new_dw->getModifiable(pressure, d_lab->d_pressureIntermLabel,
			  matlIndex, patch);
    
    // get cellType, pressure and velocity
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    new_dw->get(density, d_lab->d_densityIntermLabel, matlIndex, patch, 
		Ghost::None,
		Arches::ZEROGHOSTCELLS);
    PerPatch<CellInformationP> cellInfoP;

    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 

      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    else {

      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    }

    CellInformation* cellinfo = cellInfoP.get().get_rep();

  // Get the low and high index for the patch and the variables
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
#ifdef ARCHES_BC_DEBUG
    cerr << "idxLo" << idxLo << " idxHi" << idxHi << endl;
    cerr << "domLo" << domLoScalar << " domHi" << domHiScalar << endl;
    cerr << "domLoU" << domLoU << " domHiU" << domHiU << endl;
    cerr << "domLoV" << domLoV << " domHiV" << domHiV << endl;
    cerr << "domLoW" << domLoW << " domHiW" << domHiW << endl;
    cerr << "pressID" << d_pressureBdry->d_cellTypeID << endl;
    for(CellIterator iter = patch->getCellIterator();
	!iter.done(); iter++){
      cerr.width(10);
      cerr << "PPP"<<*iter << ": " << pressure[*iter] << "\n" ; 
    }
#endif
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    fort_hatvelcalpbc(uVelocity, vVelocity, wVelocity, idxLo, idxHi,
		pressure, density, cellType, 
		d_pressureBdry->d_cellTypeID,
		d_pressureBdry->refPressure,
		cellinfo->dxepu, cellinfo->dynpv, cellinfo->dztpw,
		xminus, xplus, yminus, yplus, zminus, zplus);
    // set values of the scalars on the scalar boundary
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

#ifdef scalarSolve_debug

     cerr << " NEW scalar VALUES " << endl;
     for (int ii = 5; ii <= 9; ii++) {
       for (int jj = 7; jj <= 12; jj++) {
	 for (int kk = 7; kk <= 12; kk++) {
	   cerr.width(14);
	   cerr << " point coordinates "<< ii << " " << jj << " " << kk ;
	   cerr << " new scalar = " << vars->scalar[IntVector(ii,jj,kk)] ; 
	   cerr << " cellType = " << constvars->cellType[IntVector(ii,jj,kk)] ; 
	   cerr << " void fraction = " << constvars->voidFraction[IntVector(ii,jj,kk)] << endl; 
	 }
       }
     }
#endif

#ifdef ARCHES_DEBUG
    cerr << " After Scalar Explicit solve : " << endl;
    cerr << "Print Scalar: " << endl;
    vars->scalar.print(cerr);
#endif

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

  //#define enthalpySolve_debug
#ifdef enthalpySolve_debug

     cerr << " NEW enthalpy VALUES " << endl;

      // code to print all values of any variable within
      // a box, for multi-patch case

     IntVector indexLow = patch->getCellLowIndex();
     IntVector indexHigh = patch->getCellHighIndex();

      int ibot = 0;
      int itop = 0;
      int jbot = 8;
      int jtop = 8;
      int kbot = 8;
      int ktop = 8;

      // values above can be changed for each case as desired

      bool printvalues = true;
      int idloX = Max(indexLow.x(),ibot);
      int idhiX = Min(indexHigh.x()-1,itop);
      int idloY = Max(indexLow.y(),jbot);
      int idhiY = Min(indexHigh.y()-1,jtop);
      int idloZ = Max(indexLow.z(),kbot);
      int idhiZ = Min(indexHigh.z()-1,ktop);
      if ((idloX > idhiX) || (idloY > idhiY) || (idloZ > idhiZ))
	printvalues = false;

      if (printvalues) {
	for (int ii = idloX; ii <= idhiX; ii++) {
	  for (int jj = idloY; jj <= idhiY; jj++) {
	    for (int kk = idloZ; kk <= idhiZ; kk++) {
	      cerr.width(14);
	      cerr << " point coordinates "<< ii << " " << jj << " " << kk << endl;
	      cerr << "Enthalpy after solve = " << vars->enthalpy[IntVector(ii,jj,kk)] << endl;
	      cerr << "Void Fraction after solve = " << constvars->voidFraction[IntVector(ii,jj,kk)] << endl;
	      cerr << "CellType for enthalpy solve = " << constvars->cellType[IntVector(ii,jj,kk)] << endl; 
	      cerr << "Printing Coefficients and Sources" << endl;
	      cerr << "East     coefficient = " << constvars->scalarCoeff[Arches::AE][IntVector(ii,jj,kk)] << endl; 
	      cerr << "West     coefficient = " << constvars->scalarCoeff[Arches::AW][IntVector(ii,jj,kk)] << endl; 
	      cerr << "North    coefficient = " << constvars->scalarCoeff[Arches::AN][IntVector(ii,jj,kk)] << endl; 
	      cerr << "South    coefficient = " << constvars->scalarCoeff[Arches::AS][IntVector(ii,jj,kk)] << endl; 
	      cerr << "Top      coefficient = " << constvars->scalarCoeff[Arches::AT][IntVector(ii,jj,kk)] << endl; 
	      cerr << "Bottom   coefficient = " << constvars->scalarCoeff[Arches::AB][IntVector(ii,jj,kk)] << endl; 
	      cerr << "Diagonal coefficient = " << constvars->scalarCoeff[Arches::AP][IntVector(ii,jj,kk)] << endl; 
	      cerr << "Nonlinear source     = " << constvars->scalarNonlinearSrc[IntVector(ii,jj,kk)] << endl; 
	      cerr << "Old Density          = " << constvars->old_density[IntVector(ii,jj,kk)] << endl; 
	      cerr << "delta_t = " << delta_t << endl;
	      
	    }
	  }
	}
      }

#endif


#ifdef ARCHES_DEBUG
    cerr << " After Scalar Explicit solve : " << endl;
    cerr << "Print Scalar: " << endl;
    vars->enthalpy.print(cerr);
#endif

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
			    CellInformation*,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  int press_celltypeval = d_pressureBdry->d_cellTypeID;

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
        if (constvars->cellType[xminusCell] == press_celltypeval)
	  if (constvars->uVelocity[currCell] <= 0.0)
                        vars->scalar[xminusCell] = vars->scalar[currCell];
	  else vars->scalar[xminusCell] = 0.0;
      }
    }
  }
  if (xplus) {
    int colX = idxHi.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        if (constvars->cellType[xplusCell] == press_celltypeval)
	  if (constvars->uVelocity[xplusCell] >= 0.0)
                        vars->scalar[xplusCell] = vars->scalar[currCell];
	  else vars->scalar[xplusCell] = 0.0;
      }
    }
  }
  if (yminus) {
    int colY = idxLo.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);
        if (constvars->cellType[yminusCell] == press_celltypeval)
	  if (constvars->vVelocity[currCell] <= 0.0)
                        vars->scalar[yminusCell] = vars->scalar[currCell];
	  else vars->scalar[yminusCell] = 0.0;
      }
    }
  }
  if (yplus) {
    int colY = idxHi.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        if (constvars->cellType[yplusCell] == press_celltypeval)
	  if (constvars->vVelocity[yplusCell] >= 0.0)
                        vars->scalar[yplusCell] = vars->scalar[currCell];
	  else vars->scalar[yplusCell] = 0.0;
      }
    }
  }
  if (zminus) {
    int colZ = idxLo.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);
        if (constvars->cellType[zminusCell] == press_celltypeval)
	  if (constvars->wVelocity[currCell] <= 0.0)
                        vars->scalar[zminusCell] = vars->scalar[currCell];
	  else vars->scalar[zminusCell] = 0.0;
      }
    }
  }
  if (zplus) {
    int colZ = idxHi.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        if (constvars->cellType[zplusCell] == press_celltypeval)
	  if (constvars->wVelocity[zplusCell] >= 0.0)
                        vars->scalar[zplusCell] = vars->scalar[currCell];
	  else vars->scalar[zplusCell] = 0.0;
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
			    CellInformation*,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  int press_celltypeval = d_pressureBdry->d_cellTypeID;

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
        if (constvars->cellType[xminusCell] == press_celltypeval)
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
        if (constvars->cellType[xplusCell] == press_celltypeval)
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
        if (constvars->cellType[yminusCell] == press_celltypeval)
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
        if (constvars->cellType[yplusCell] == press_celltypeval)
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
        if (constvars->cellType[zminusCell] == press_celltypeval)
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
        if (constvars->cellType[zplusCell] == press_celltypeval)
                        vars->enthalpy[zplusCell] = vars->enthalpy[currCell];
      }
    }
  }
}
//****************************************************************************
// Schedule copy IN variables to OUTBC variables to go around all bc implementation
//****************************************************************************
void 
BoundaryCondition::sched_copyINtoOUT(SchedulerP& sched, const PatchSet* patches,
					    const MaterialSet* matls)
{
  Task* tsk = scinew Task("BoundaryCondition::copyINtoOUT",
			  this,
			  &BoundaryCondition::copyINtoOUT);
  
  tsk->requires(Task::NewDW, d_lab->d_uVelocityINLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityINLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityINLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_pressureINLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  for (int ii = 0; ii < d_nofScalars; ii++)
    tsk->requires(Task::NewDW, d_lab->d_scalarINLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);


  if (d_enthalpySolve) {
    tsk->requires(Task::NewDW, d_lab->d_enthalpyINLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->computes(d_lab->d_enthalpyOUTBCLabel);
  }
  
  if (d_reactingScalarSolve) {
    tsk->requires(Task::NewDW, d_lab->d_reactscalarINLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->computes(d_lab->d_reactscalarOUTBCLabel);
  }
  
  tsk->computes(d_lab->d_uVelocityOUTBCLabel);
  tsk->computes(d_lab->d_vVelocityOUTBCLabel);
  tsk->computes(d_lab->d_wVelocityOUTBCLabel);
  tsk->computes(d_lab->d_pressurePSLabel);
  for (int ii = 0; ii < d_nofScalars; ii++)
    tsk->computes(d_lab->d_scalarOUTBCLabel);
  
  sched->addTask(tsk, patches, matls);
  
}
//****************************************************************************
// Actual copy IN variables to OUTBC variables to go around all bc implementation
//****************************************************************************
void 
BoundaryCondition::copyINtoOUT(const ProcessorGroup* ,
				  const PatchSubset* patches,
				  const MaterialSubset*,
				  DataWarehouse*,
				  DataWarehouse* new_dw) 
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex =
	      d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    CCVariable<double> pressure;
    CCVariable<double> enthalpy;
    CCVariable<double> scalar;
    CCVariable<double> reactscalar;

    new_dw->allocateAndPut(uVelocity, d_lab->d_uVelocityOUTBCLabel,
			   matlIndex, patch,
			   Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->allocateAndPut(vVelocity, d_lab->d_vVelocityOUTBCLabel,
			   matlIndex, patch,
			   Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->allocateAndPut(wVelocity, d_lab->d_wVelocityOUTBCLabel,
			   matlIndex, patch,
			   Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->allocateAndPut(pressure, d_lab->d_pressurePSLabel,
			   matlIndex, patch,
			   Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->copyOut(uVelocity, d_lab->d_uVelocityINLabel, matlIndex, patch,
		     Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->copyOut(vVelocity, d_lab->d_vVelocityINLabel, matlIndex, patch,
		     Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->copyOut(wVelocity, d_lab->d_wVelocityINLabel, matlIndex, patch,
		     Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->copyOut(pressure, d_lab->d_pressureINLabel, matlIndex, patch,
		     Ghost::None, Arches::ZEROGHOSTCELLS);

    for (int ii = 0; ii < d_nofScalars; ii++)
    new_dw->allocateAndPut(scalar, d_lab->d_scalarOUTBCLabel,
			   matlIndex, patch,
			   Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->copyOut(scalar, d_lab->d_scalarINLabel, matlIndex, patch,
		     Ghost::None, Arches::ZEROGHOSTCELLS);

    if (d_enthalpySolve) {
    new_dw->allocateAndPut(enthalpy, d_lab->d_enthalpyOUTBCLabel,
			   matlIndex, patch,
			   Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->copyOut(enthalpy, d_lab->d_enthalpyINLabel, matlIndex, patch,
		     Ghost::None, Arches::ZEROGHOSTCELLS);
    }
  
    if (d_reactingScalarSolve) {
    new_dw->allocateAndPut(reactscalar, d_lab->d_reactscalarOUTBCLabel,
			   matlIndex, patch,
			   Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->copyOut(reactscalar, d_lab->d_reactscalarINLabel, matlIndex, patch,
		     Ghost::None, Arches::ZEROGHOSTCELLS);
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
			    CellInformation* cellinfo,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars,
			    const double delta_t)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  int out_celltypeval = d_outletBC->d_cellTypeID;

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
        if (constvars->cellType[xminusCell] == out_celltypeval) {
	  if (constvars->uVelocity[currCell] <= 0.0) {
           vars->scalar[xminusCell]= - delta_t * constvars->uVelocity[xplusCell] *
              (constvars->old_density[currCell]*constvars->old_scalar[currCell] -
               constvars->old_density[xminusCell]*constvars->old_scalar[xminusCell]) /
	      (cellinfo->dxep[colX-1]*constvars->old_density[xminusCell]) +
	      constvars->old_scalar[xminusCell];
           if (vars->scalar[xminusCell] > 1.0)
               vars->scalar[xminusCell] = 1.0;
           else if (vars->scalar[xminusCell] < 0.0)
               vars->scalar[xminusCell] = 0.0;
	   double prev_scalar = vars->scalar[currCell];
	   if (vars->scalar[xminusCell] > prev_scalar)
	       vars->scalar[xminusCell] = prev_scalar;
	  }
	  else vars->scalar[xminusCell] = 0.0;
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
        if (constvars->cellType[xplusCell] == out_celltypeval) {
	  if (constvars->uVelocity[xplusCell] >= 0.0) {
           vars->scalar[xplusCell]= - delta_t * constvars->uVelocity[currCell] *
              (constvars->old_density[xplusCell]*constvars->old_scalar[xplusCell] -
               constvars->old_density[currCell]*constvars->old_scalar[currCell]) /
	      (cellinfo->dxpw[colX+1]*constvars->old_density[xplusCell]) +
	      constvars->old_scalar[xplusCell];
           if (vars->scalar[xplusCell] > 1.0)
               vars->scalar[xplusCell] = 1.0;
           else if (vars->scalar[xplusCell] < 0.0)
               vars->scalar[xplusCell] = 0.0;
	   double prev_scalar = vars->scalar[currCell];
	   if (vars->scalar[xplusCell] > prev_scalar)
	       vars->scalar[xplusCell] = prev_scalar;
	  }
	  else vars->scalar[xplusCell] = 0.0;
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
        if (constvars->cellType[yminusCell] == out_celltypeval) {
	  if (constvars->vVelocity[currCell] <= 0.0) {
           vars->scalar[yminusCell]= - delta_t * constvars->vVelocity[yplusCell] *
              (constvars->old_density[currCell]*constvars->old_scalar[currCell] -
               constvars->old_density[yminusCell]*constvars->old_scalar[yminusCell]) /
	      (cellinfo->dynp[colY-1]*constvars->old_density[yminusCell]) +
	      constvars->old_scalar[yminusCell];
           if (vars->scalar[yminusCell] > 1.0)
               vars->scalar[yminusCell] = 1.0;
           else if (vars->scalar[yminusCell] < 0.0)
               vars->scalar[yminusCell] = 0.0;
	   double prev_scalar = vars->scalar[currCell];
	   if (vars->scalar[yminusCell] > prev_scalar)
	       vars->scalar[yminusCell] = prev_scalar;
	  }
	  else vars->scalar[yminusCell] = 0.0;
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
        if (constvars->cellType[yplusCell] == out_celltypeval) {
	  if (constvars->vVelocity[yplusCell] >= 0.0) {
           vars->scalar[yplusCell]= - delta_t * constvars->vVelocity[currCell] *
              (constvars->old_density[yplusCell]*constvars->old_scalar[yplusCell] -
               constvars->old_density[currCell]*constvars->old_scalar[currCell]) /
	      (cellinfo->dyps[colY+1]*constvars->old_density[yplusCell]) +
	      constvars->old_scalar[yplusCell];
           if (vars->scalar[yplusCell] > 1.0)
               vars->scalar[yplusCell] = 1.0;
           else if (vars->scalar[yplusCell] < 0.0)
               vars->scalar[yplusCell] = 0.0;
	   double prev_scalar = vars->scalar[currCell];
	   if (vars->scalar[yplusCell] > prev_scalar)
	       vars->scalar[yplusCell] = prev_scalar;
	  }
	  else vars->scalar[yplusCell] = 0.0;
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
        if (constvars->cellType[zminusCell] == out_celltypeval) {
	  if (constvars->wVelocity[currCell] <= 0.0) {
           vars->scalar[zminusCell]= - delta_t * constvars->wVelocity[zplusCell] *
              (constvars->old_density[currCell]*constvars->old_scalar[currCell] -
               constvars->old_density[zminusCell]*constvars->old_scalar[zminusCell]) /
	      (cellinfo->dztp[colZ-1]*constvars->old_density[zminusCell]) +
	      constvars->old_scalar[zminusCell];
           if (vars->scalar[zminusCell] > 1.0)
               vars->scalar[zminusCell] = 1.0;
           else if (vars->scalar[zminusCell] < 0.0)
               vars->scalar[zminusCell] = 0.0;
	   double prev_scalar = vars->scalar[currCell];
	   if (vars->scalar[zminusCell] > prev_scalar)
	       vars->scalar[zminusCell] = prev_scalar;
	  }
	  else vars->scalar[zminusCell] = 0.0;
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
        if (constvars->cellType[zplusCell] == out_celltypeval) {
	  if (constvars->wVelocity[zplusCell] >= 0.0) {
           vars->scalar[zplusCell]= - delta_t * constvars->wVelocity[currCell] *
              (constvars->old_density[zplusCell]*constvars->old_scalar[zplusCell] -
               constvars->old_density[currCell]*constvars->old_scalar[currCell]) /
	      (cellinfo->dzpb[colZ+1]*constvars->old_density[zplusCell]) +
	      constvars->old_scalar[zplusCell];
           if (vars->scalar[zplusCell] > 1.0)
               vars->scalar[zplusCell] = 1.0;
           else if (vars->scalar[zplusCell] < 0.0)
               vars->scalar[zplusCell] = 0.0;
	   double prev_scalar = vars->scalar[currCell];
	   if (vars->scalar[zplusCell] > prev_scalar)
	       vars->scalar[zplusCell] = prev_scalar;
	  }
	  else vars->scalar[zplusCell] = 0.0;
        } 
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
			    CellInformation* cellinfo,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars,
			    const double delta_t)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  int out_celltypeval = d_outletBC->d_cellTypeID;

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
        if (constvars->cellType[xminusCell] == out_celltypeval)
           vars->enthalpy[xminusCell]= - delta_t * constvars->uVelocity[xplusCell] *
              (constvars->old_density[currCell]*constvars->old_enthalpy[currCell] -
               constvars->old_density[xminusCell]*constvars->old_enthalpy[xminusCell]) /
	      (cellinfo->dxep[colX-1]*constvars->old_density[xminusCell]) +
	      constvars->old_enthalpy[xminusCell];
      }
    }
  }
  if (xplus) {
    int colX = idxHi.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        if (constvars->cellType[xplusCell] == out_celltypeval)
           vars->enthalpy[xplusCell]= - delta_t * constvars->uVelocity[currCell] *
              (constvars->old_density[xplusCell]*constvars->old_enthalpy[xplusCell] -
               constvars->old_density[currCell]*constvars->old_enthalpy[currCell]) /
	      (cellinfo->dxpw[colX+1]*constvars->old_density[xplusCell]) +
	      constvars->old_enthalpy[xplusCell];
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
        if (constvars->cellType[yminusCell] == out_celltypeval)
           vars->enthalpy[yminusCell]= - delta_t * constvars->vVelocity[yplusCell] *
              (constvars->old_density[currCell]*constvars->old_enthalpy[currCell] -
               constvars->old_density[yminusCell]*constvars->old_enthalpy[yminusCell]) /
	      (cellinfo->dynp[colY-1]*constvars->old_density[yminusCell]) +
	      constvars->old_enthalpy[yminusCell];
      }
    }
  }
  if (yplus) {
    int colY = idxHi.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        if (constvars->cellType[yplusCell] == out_celltypeval)
           vars->enthalpy[yplusCell]= - delta_t * constvars->vVelocity[currCell] *
              (constvars->old_density[yplusCell]*constvars->old_enthalpy[yplusCell] -
               constvars->old_density[currCell]*constvars->old_enthalpy[currCell]) /
	      (cellinfo->dyps[colY+1]*constvars->old_density[yplusCell]) +
	      constvars->old_enthalpy[yplusCell];
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
        if (constvars->cellType[zminusCell] == out_celltypeval)
           vars->enthalpy[zminusCell]= - delta_t * constvars->wVelocity[zplusCell] *
              (constvars->old_density[currCell]*constvars->old_enthalpy[currCell] -
               constvars->old_density[zminusCell]*constvars->old_enthalpy[zminusCell]) /
	      (cellinfo->dztp[colZ-1]*constvars->old_density[zminusCell]) +
	      constvars->old_enthalpy[zminusCell];
      }
    }
  }
  if (zplus) {
    int colZ = idxHi.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        if (constvars->cellType[zplusCell] == out_celltypeval)
           vars->enthalpy[zplusCell]= - delta_t * constvars->wVelocity[currCell] *
              (constvars->old_density[zplusCell]*constvars->old_enthalpy[zplusCell] -
               constvars->old_density[currCell]*constvars->old_enthalpy[currCell]) /
	      (cellinfo->dzpb[colZ+1]*constvars->old_density[zplusCell]) +
	      constvars->old_enthalpy[zplusCell];
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
			    CellInformation*,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)
{
  double time = d_lab->d_sharedState->getElapsedTime();
  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  // stores cell type info for the patch with the ghost cell type
  for (int indx = 0; indx < d_numInlets; indx++) {
    // Get a copy of the current flowinlet
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
      	  fi.d_cellTypeID, time,
      	  xminus, xplus, yminus, yplus, zminus, zplus);
    
  }
}
//****************************************************************************
// Set zero gradient for rho hat velocity on pressure bc
//****************************************************************************
void 
BoundaryCondition::velRhoHatPressureBC(const ProcessorGroup*,
			    const Patch* patch,
			    CellInformation*,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  int press_celltypeval = d_pressureBdry->d_cellTypeID;

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
        if (constvars->cellType[xminusCell] == press_celltypeval) {
           vars->uVelRhoHat[currCell] = vars->uVelRhoHat[xplusCell];
           vars->uVelRhoHat[xminusCell] = vars->uVelRhoHat[currCell];
        if (!(yminus && (colY == idxLo.y())))
           vars->vVelRhoHat[xminusCell] = vars->vVelRhoHat[currCell];
        if (!(zminus && (colZ == idxLo.z())))
           vars->wVelRhoHat[xminusCell] = vars->wVelRhoHat[currCell];
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
        if (constvars->cellType[xplusCell] == press_celltypeval) {
           vars->uVelRhoHat[xplusCell] = vars->uVelRhoHat[currCell];
           vars->uVelRhoHat[xplusplusCell] = vars->uVelRhoHat[xplusCell];
        if (!(yminus && (colY == idxLo.y())))
           vars->vVelRhoHat[xplusCell] = vars->vVelRhoHat[currCell];
        if (!(zminus && (colZ == idxLo.z())))
           vars->wVelRhoHat[xplusCell] = vars->wVelRhoHat[currCell];
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
        if (constvars->cellType[yminusCell] == press_celltypeval) {
        if (!(xminus && (colX == idxLo.x())))
           vars->uVelRhoHat[yminusCell] = vars->uVelRhoHat[currCell];
           vars->vVelRhoHat[currCell] = vars->vVelRhoHat[yplusCell];
           vars->vVelRhoHat[yminusCell] = vars->vVelRhoHat[currCell];
        if (!(zminus && (colZ == idxLo.z())))
           vars->wVelRhoHat[yminusCell] = vars->wVelRhoHat[currCell];
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
        if (constvars->cellType[yplusCell] == press_celltypeval) {
        if (!(xminus && (colX == idxLo.x())))
           vars->uVelRhoHat[yplusCell] = vars->uVelRhoHat[currCell];
           vars->vVelRhoHat[yplusCell] = vars->vVelRhoHat[currCell];
           vars->vVelRhoHat[yplusplusCell] = vars->vVelRhoHat[yplusCell];
        if (!(zminus && (colZ == idxLo.z())))
           vars->wVelRhoHat[yplusCell] = vars->wVelRhoHat[currCell];
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
        if (constvars->cellType[zminusCell] == press_celltypeval) {
        if (!(xminus && (colX == idxLo.x())))
           vars->uVelRhoHat[zminusCell] = vars->uVelRhoHat[currCell];
        if (!(yminus && (colY == idxLo.y())))
           vars->vVelRhoHat[zminusCell] = vars->vVelRhoHat[currCell];
           vars->wVelRhoHat[currCell] = vars->wVelRhoHat[zplusCell];
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
        if (constvars->cellType[zplusCell] == press_celltypeval) {
        if (!(xminus && (colX == idxLo.x())))
           vars->uVelRhoHat[zplusCell] = vars->uVelRhoHat[currCell];
        if (!(yminus && (colY == idxLo.y())))
           vars->vVelRhoHat[zplusCell] = vars->vVelRhoHat[currCell];
           vars->wVelRhoHat[zplusCell] = vars->wVelRhoHat[currCell];
           vars->wVelRhoHat[zplusplusCell] = vars->wVelRhoHat[zplusCell];
        }
      }
    }
  }
}
//****************************************************************************
// Set rho hat velocity at the outlet from 1d advection equation
//****************************************************************************
void 
BoundaryCondition::velRhoHatOutletBC(const ProcessorGroup*,
			    const Patch* patch,
			    CellInformation* cellinfo,
			    const double delta_t,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  int out_celltypeval = d_outletBC->d_cellTypeID;

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
        IntVector xplusCell(colX+1, colY, colZ);
        IntVector xplusyminusCell(colX+1, colY-1, colZ);
        IntVector xpluszminusCell(colX+1, colY, colZ-1);
        IntVector xminusyminusCell(colX-1, colY-1, colZ);
        IntVector xminuszminusCell(colX-1, colY, colZ-1);
        if (constvars->cellType[xminusCell] == out_celltypeval) {
           double avden = 0.5 * (constvars->old_density[xplusCell] +
			         constvars->old_density[currCell]);
           double avdenlow = 0.5 * (constvars->old_density[currCell] +
			            constvars->old_density[xminusCell]);
           double new_avdenlow = 0.5 * (constvars->new_density[currCell] +
			                constvars->new_density[xminusCell]);
	   double out_vel = constvars->uVelocity[xplusCell];

           vars->uVelRhoHat[currCell] = (- delta_t * out_vel *
            (avden*constvars->uVelocity[xplusCell] -
             avdenlow*constvars->uVelocity[currCell]) / cellinfo->dxepu[colX-1]+
	    avdenlow*constvars->uVelocity[currCell]) / new_avdenlow;

           vars->uVelRhoHat[xminusCell] = vars->uVelRhoHat[currCell];

        if (!(yminus && (colY == idxLo.y()))) {
           avden = 0.5 * (constvars->old_density[currCell] +
	                  constvars->old_density[yminusCell]);
           avdenlow = 0.5 * (constvars->old_density[xminusCell] +
			     constvars->old_density[xminusyminusCell]);
           new_avdenlow = 0.5 * (constvars->new_density[xminusCell] +
			         constvars->new_density[xminusyminusCell]);
	   out_vel = 0.5 * (constvars->uVelocity[xplusCell] +
			    constvars->uVelocity[xplusyminusCell]);

           vars->vVelRhoHat[xminusCell] = (- delta_t * out_vel *
            (avden*constvars->vVelocity[currCell] - 
	     avdenlow*constvars->vVelocity[xminusCell]) /cellinfo->dxep[colX-1]+
	    avdenlow*constvars->vVelocity[xminusCell]) / new_avdenlow;
	}
        if (!(zminus && (colZ == idxLo.z()))) {
           avden = 0.5 * (constvars->old_density[currCell] +
	                  constvars->old_density[zminusCell]);
           avdenlow = 0.5 * (constvars->old_density[xminusCell] +
			     constvars->old_density[xminuszminusCell]);
           new_avdenlow = 0.5 * (constvars->new_density[xminusCell] +
			         constvars->new_density[xminuszminusCell]);
	   out_vel = 0.5 * (constvars->uVelocity[xplusCell] +
			    constvars->uVelocity[xpluszminusCell]);

           vars->wVelRhoHat[xminusCell] = (- delta_t * out_vel *
            (avden*constvars->wVelocity[currCell] - 
	     avdenlow*constvars->wVelocity[xminusCell]) /cellinfo->dxep[colX-1]+
	    avdenlow*constvars->wVelocity[xminusCell]) / new_avdenlow;
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
	IntVector xminusCell(colX-1, colY, colZ);
	IntVector yminusCell(colX, colY-1, colZ);
	IntVector zminusCell(colX, colY, colZ-1);
        IntVector xplusCell(colX+1, colY, colZ);
        IntVector xplusyminusCell(colX+1, colY-1, colZ);
        IntVector xpluszminusCell(colX+1, colY, colZ-1);
        IntVector xplusplusCell(colX+2, colY, colZ);
        if (constvars->cellType[xplusCell] == out_celltypeval) {
           double avden = 0.5 * (constvars->old_density[xplusCell] +
			         constvars->old_density[currCell]);
           double new_avden = 0.5 * (constvars->new_density[xplusCell] +
			             constvars->new_density[currCell]);
           double avdenlow = 0.5 * (constvars->old_density[currCell] +
			            constvars->old_density[xminusCell]);
	   double out_vel = constvars->uVelocity[currCell];

           vars->uVelRhoHat[xplusCell] = (- delta_t * out_vel *
            (avden*constvars->uVelocity[xplusCell] - 
	     avdenlow*constvars->uVelocity[currCell]) / cellinfo->dxpwu[colX+1]+
	    avden*constvars->uVelocity[xplusCell]) / new_avden;

           vars->uVelRhoHat[xplusplusCell] = vars->uVelRhoHat[xplusCell];

        if (!(yminus && (colY == idxLo.y()))) {
           avden = 0.5 * (constvars->old_density[xplusCell] +
	                  constvars->old_density[xplusyminusCell]);
           new_avden = 0.5 * (constvars->new_density[xplusCell] +
	                      constvars->new_density[xplusyminusCell]);
           avdenlow = 0.5 * (constvars->old_density[currCell] +
			     constvars->old_density[yminusCell]);
	   out_vel = 0.5 * (constvars->uVelocity[currCell] +
			    constvars->uVelocity[yminusCell]);

           vars->vVelRhoHat[xplusCell] = (- delta_t * out_vel *
            (avden*constvars->vVelocity[xplusCell] - 
	     avdenlow*constvars->vVelocity[currCell]) / cellinfo->dxpw[colX+1] +
	    avden*constvars->vVelocity[xplusCell]) / new_avden;
	}

        if (!(zminus && (colZ == idxLo.z()))) {
           avden = 0.5 * (constvars->old_density[xplusCell] +
	                  constvars->old_density[xpluszminusCell]);
           new_avden = 0.5 * (constvars->new_density[xplusCell] +
	                      constvars->new_density[xpluszminusCell]);
           avdenlow = 0.5 * (constvars->old_density[currCell] +
			     constvars->old_density[zminusCell]);
	   out_vel = 0.5 * (constvars->uVelocity[currCell] +
			    constvars->uVelocity[zminusCell]);

           vars->wVelRhoHat[xplusCell] = (- delta_t * out_vel *
            (avden*constvars->wVelocity[xplusCell] - 
	     avdenlow*constvars->wVelocity[currCell]) / cellinfo->dxpw[colX+1] +
	    avden*constvars->wVelocity[xplusCell]) / new_avden;
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
        IntVector xminusCell(colX-1, colY, colZ);
	IntVector yminusCell(colX, colY-1, colZ);
	IntVector zminusCell(colX, colY, colZ-1);
        IntVector yplusCell(colX, colY+1, colZ);
        IntVector yplusxminusCell(colX-1, colY+1, colZ);
        IntVector ypluszminusCell(colX, colY+1, colZ-1);
        IntVector yminusxminusCell(colX-1, colY-1, colZ);
        IntVector yminuszminusCell(colX, colY-1, colZ-1);
        if (constvars->cellType[yminusCell] == out_celltypeval) {
           double avden = 0.5 * (constvars->old_density[yplusCell] +
			         constvars->old_density[currCell]);
           double avdenlow = 0.5 * (constvars->old_density[currCell] +
			            constvars->old_density[yminusCell]);
           double new_avdenlow = 0.5 * (constvars->new_density[currCell] +
			                constvars->new_density[yminusCell]);
	   double out_vel = constvars->vVelocity[yplusCell];

           vars->vVelRhoHat[currCell] = (- delta_t * out_vel *
            (avden*constvars->vVelocity[yplusCell] -
             avdenlow*constvars->vVelocity[currCell]) / cellinfo->dynpv[colY-1]+
	    avdenlow*constvars->vVelocity[currCell]) /new_avdenlow;

           vars->vVelRhoHat[yminusCell] = vars->vVelRhoHat[currCell];

        if (!(xminus && (colX == idxLo.x()))) {
           avden = 0.5 * (constvars->old_density[currCell] +
	                  constvars->old_density[xminusCell]);
           avdenlow = 0.5 * (constvars->old_density[yminusCell] +
			     constvars->old_density[yminusxminusCell]);
           new_avdenlow = 0.5 * (constvars->new_density[yminusCell] +
			         constvars->new_density[yminusxminusCell]);
	   out_vel = 0.5 * (constvars->vVelocity[yplusCell] +
			    constvars->vVelocity[yplusxminusCell]);

           vars->uVelRhoHat[yminusCell] = (- delta_t * out_vel *
            (avden*constvars->uVelocity[currCell] - 
	     avdenlow*constvars->uVelocity[yminusCell]) /cellinfo->dynp[colY-1]+
	    avdenlow*constvars->uVelocity[yminusCell]) / new_avdenlow;
	}
        if (!(zminus && (colZ == idxLo.z()))) {
           avden = 0.5 * (constvars->old_density[currCell] +
	                  constvars->old_density[zminusCell]);
           avdenlow = 0.5 * (constvars->old_density[yminusCell] +
			     constvars->old_density[yminuszminusCell]);
           new_avdenlow = 0.5 * (constvars->new_density[yminusCell] +
			         constvars->new_density[yminuszminusCell]);
	   out_vel = 0.5 * (constvars->vVelocity[yplusCell] +
			    constvars->vVelocity[ypluszminusCell]);

           vars->wVelRhoHat[yminusCell] = (- delta_t * out_vel *
            (avden*constvars->wVelocity[currCell] - 
	     avdenlow*constvars->wVelocity[yminusCell]) /cellinfo->dynp[colY-1]+
	    avdenlow*constvars->wVelocity[yminusCell]) / new_avdenlow;
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
	IntVector xminusCell(colX-1, colY, colZ);
	IntVector yminusCell(colX, colY-1, colZ);
	IntVector zminusCell(colX, colY, colZ-1);
        IntVector yplusCell(colX, colY+1, colZ);
        IntVector yplusxminusCell(colX-1, colY+1, colZ);
        IntVector ypluszminusCell(colX, colY+1, colZ-1);
        IntVector yplusplusCell(colX, colY+2, colZ);
        if (constvars->cellType[yplusCell] == out_celltypeval) {
           double avden = 0.5 * (constvars->old_density[yplusCell] +
			         constvars->old_density[currCell]);
           double new_avden = 0.5 * (constvars->new_density[yplusCell] +
			             constvars->new_density[currCell]);
           double avdenlow = 0.5 * (constvars->old_density[currCell] +
			            constvars->old_density[yminusCell]);
	   double out_vel = constvars->vVelocity[currCell];

           vars->vVelRhoHat[yplusCell] = (- delta_t * out_vel *
            (avden*constvars->vVelocity[yplusCell] - 
	     avdenlow*constvars->vVelocity[currCell]) / cellinfo->dypsv[colY+1]+
	    avden*constvars->vVelocity[yplusCell]) / new_avden;

           vars->vVelRhoHat[yplusplusCell] = vars->vVelRhoHat[yplusCell];

        if (!(xminus && (colX == idxLo.x()))) {
           avden = 0.5 * (constvars->old_density[yplusCell] +
	                  constvars->old_density[yplusxminusCell]);
           new_avden = 0.5 * (constvars->new_density[yplusCell] +
	                      constvars->new_density[yplusxminusCell]);
           avdenlow = 0.5 * (constvars->old_density[currCell] +
			     constvars->old_density[xminusCell]);
	   out_vel = 0.5 * (constvars->vVelocity[currCell] +
			    constvars->vVelocity[xminusCell]);

           vars->uVelRhoHat[yplusCell] = (- delta_t * out_vel *
            (avden*constvars->uVelocity[yplusCell] - 
	     avdenlow*constvars->uVelocity[currCell]) / cellinfo->dyps[colY+1] +
	    avden*constvars->uVelocity[yplusCell]) / new_avden;
	}

        if (!(zminus && (colZ == idxLo.z()))) {
           avden = 0.5 * (constvars->old_density[yplusCell] +
	                  constvars->old_density[ypluszminusCell]);
           new_avden = 0.5 * (constvars->new_density[yplusCell] +
	                      constvars->new_density[ypluszminusCell]);
           avdenlow = 0.5 * (constvars->old_density[currCell] +
			     constvars->old_density[zminusCell]);
	   out_vel = 0.5 * (constvars->vVelocity[currCell] +
			    constvars->vVelocity[zminusCell]);

           vars->wVelRhoHat[yplusCell] = (- delta_t * out_vel *
            (avden*constvars->wVelocity[yplusCell] - 
	     avdenlow*constvars->wVelocity[currCell]) / cellinfo->dyps[colY+1] +
	    avden*constvars->wVelocity[yplusCell]) / new_avden;
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
        IntVector xminusCell(colX-1, colY, colZ);
	IntVector yminusCell(colX, colY-1, colZ);
	IntVector zminusCell(colX, colY, colZ-1);
        IntVector zplusCell(colX, colY, colZ+1);
        IntVector zplusxminusCell(colX-1, colY, colZ+1);
        IntVector zplusyminusCell(colX, colY-1, colZ+1);
        IntVector zminusxminusCell(colX-1, colY, colZ-1);
        IntVector zminusyminusCell(colX, colY-1, colZ-1);
        if (constvars->cellType[zminusCell] == out_celltypeval) {
           double avden = 0.5 * (constvars->old_density[zplusCell] +
			         constvars->old_density[currCell]);
           double avdenlow = 0.5 * (constvars->old_density[currCell] +
			            constvars->old_density[zminusCell]);
           double new_avdenlow = 0.5 * (constvars->new_density[currCell] +
			                constvars->new_density[zminusCell]);
	   double out_vel = constvars->wVelocity[zplusCell];

           vars->wVelRhoHat[currCell] = (- delta_t * out_vel *
            (avden*constvars->wVelocity[zplusCell] -
             avdenlow*constvars->wVelocity[currCell]) / cellinfo->dztpw[colZ-1]+
	    avdenlow*constvars->wVelocity[currCell]) / new_avdenlow;

           vars->wVelRhoHat[zminusCell] = vars->wVelRhoHat[currCell];

        if (!(xminus && (colX == idxLo.x()))) {
           avden = 0.5 * (constvars->old_density[currCell] +
	                  constvars->old_density[xminusCell]);
           avdenlow = 0.5 * (constvars->old_density[zminusCell] +
			     constvars->old_density[zminusxminusCell]);
           new_avdenlow = 0.5 * (constvars->new_density[zminusCell] +
			         constvars->new_density[zminusxminusCell]);
	   out_vel = 0.5 * (constvars->wVelocity[zplusCell] +
			    constvars->wVelocity[zplusxminusCell]);

           vars->uVelRhoHat[zminusCell] = (- delta_t * out_vel *
            (avden*constvars->uVelocity[currCell] - 
	     avdenlow*constvars->uVelocity[zminusCell]) /cellinfo->dztp[colZ-1]+
	    avdenlow*constvars->uVelocity[zminusCell]) / new_avdenlow;
	}
        if (!(yminus && (colY == idxLo.y()))) {
           avden = 0.5 * (constvars->old_density[currCell] +
	                  constvars->old_density[yminusCell]);
           avdenlow = 0.5 * (constvars->old_density[zminusCell] +
			     constvars->old_density[zminusyminusCell]);
           new_avdenlow = 0.5 * (constvars->new_density[zminusCell] +
			         constvars->new_density[zminusyminusCell]);
	   out_vel = 0.5 * (constvars->wVelocity[zplusCell] +
			    constvars->wVelocity[zplusyminusCell]);

           vars->vVelRhoHat[zminusCell] = (- delta_t * out_vel *
            (avden*constvars->vVelocity[currCell] - 
	     avdenlow*constvars->vVelocity[zminusCell]) /cellinfo->dztp[colZ-1]+
	    avdenlow*constvars->vVelocity[zminusCell]) / new_avdenlow;
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
	IntVector xminusCell(colX-1, colY, colZ);
	IntVector yminusCell(colX, colY-1, colZ);
	IntVector zminusCell(colX, colY, colZ-1);
        IntVector zplusCell(colX, colY, colZ+1);
        IntVector zplusxminusCell(colX-1, colY, colZ+1);
        IntVector zplusyminusCell(colX, colY-1, colZ+1);
        IntVector zplusplusCell(colX, colY, colZ+2);
        if (constvars->cellType[zplusCell] == out_celltypeval) {
           double avden = 0.5 * (constvars->old_density[zplusCell] +
			         constvars->old_density[currCell]);
           double new_avden = 0.5 * (constvars->new_density[zplusCell] +
			             constvars->new_density[currCell]);
           double avdenlow = 0.5 * (constvars->old_density[currCell] +
			            constvars->old_density[zminusCell]);
	   double out_vel = constvars->wVelocity[currCell];

           vars->wVelRhoHat[zplusCell] = (- delta_t * out_vel *
            (avden*constvars->wVelocity[zplusCell] - 
	     avdenlow*constvars->wVelocity[currCell]) / cellinfo->dzpbw[colZ+1]+
	    avden*constvars->wVelocity[zplusCell]) / new_avden;

           vars->wVelRhoHat[zplusplusCell] = vars->wVelRhoHat[zplusCell];

        if (!(xminus && (colX == idxLo.x()))) {
           avden = 0.5 * (constvars->old_density[zplusCell] +
	                  constvars->old_density[zplusxminusCell]);
           new_avden = 0.5 * (constvars->new_density[zplusCell] +
	                      constvars->new_density[zplusxminusCell]);
           avdenlow = 0.5 * (constvars->old_density[currCell] +
			     constvars->old_density[xminusCell]);
	   out_vel = 0.5 * (constvars->wVelocity[currCell] +
			    constvars->wVelocity[xminusCell]);

           vars->uVelRhoHat[zplusCell] = (- delta_t * out_vel *
            (avden*constvars->uVelocity[zplusCell] - 
	     avdenlow*constvars->uVelocity[currCell]) / cellinfo->dzpb[colZ+1] +
	    avden*constvars->uVelocity[zplusCell]) / new_avden;
	}

        if (!(yminus && (colY == idxLo.y()))) {
           avden = 0.5 * (constvars->old_density[zplusCell] +
	                  constvars->old_density[zplusyminusCell]);
           new_avden = 0.5 * (constvars->new_density[zplusCell] +
	                      constvars->new_density[zplusyminusCell]);
           avdenlow = 0.5 * (constvars->old_density[currCell] +
			     constvars->old_density[yminusCell]);
	   out_vel = 0.5 * (constvars->wVelocity[currCell] +
			    constvars->wVelocity[yminusCell]);

           vars->vVelRhoHat[zplusCell] = (- delta_t * out_vel *
            (avden*constvars->vVelocity[zplusCell] - 
	     avdenlow*constvars->vVelocity[currCell]) / cellinfo->dzpb[colZ+1] +
	    avden*constvars->vVelocity[zplusCell]) / new_avden;
	}
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
			    CellInformation*,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  int press_celltypeval = d_pressureBdry->d_cellTypeID;

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
        if (constvars->cellType[xminusCell] == press_celltypeval) {
          switch (index) {
           case Arches::XDIR:
           vars->uVelRhoHat[currCell] = vars->uVelRhoHat[xplusCell];
           vars->uVelRhoHat[xminusCell] = vars->uVelRhoHat[currCell];
           break;
           case Arches::YDIR:
        if (!(yminus && (colY == idxLo.y())))
           vars->vVelRhoHat[xminusCell] = vars->vVelRhoHat[currCell];
           break;
           case Arches::ZDIR:
        if (!(zminus && (colZ == idxLo.z())))
           vars->wVelRhoHat[xminusCell] = vars->wVelRhoHat[currCell];
           break;
           default:
		throw InvalidValue("Invalid index in velocityPressureBC");
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
        IntVector xplusplusCell(colX+2, colY, colZ);
        if (constvars->cellType[xplusCell] == press_celltypeval) {
          switch (index) {
           case Arches::XDIR:
           vars->uVelRhoHat[xplusCell] = vars->uVelRhoHat[currCell];
           vars->uVelRhoHat[xplusplusCell] = vars->uVelRhoHat[xplusCell];
           break;
           case Arches::YDIR:
        if (!(yminus && (colY == idxLo.y())))
           vars->vVelRhoHat[xplusCell] = vars->vVelRhoHat[currCell];
           break;
           case Arches::ZDIR:
        if (!(zminus && (colZ == idxLo.z())))
           vars->wVelRhoHat[xplusCell] = vars->wVelRhoHat[currCell];
           break;
           default:
		throw InvalidValue("Invalid index in velocityPressureBC");
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
        IntVector yplusCell(colX, colY+1, colZ);
        if (constvars->cellType[yminusCell] == press_celltypeval) {
          switch (index) {
           case Arches::XDIR:
        if (!(xminus && (colX == idxLo.x())))
           vars->uVelRhoHat[yminusCell] = vars->uVelRhoHat[currCell];
           break;
           case Arches::YDIR:
           vars->vVelRhoHat[currCell] = vars->vVelRhoHat[yplusCell];
           vars->vVelRhoHat[yminusCell] = vars->vVelRhoHat[currCell];
           break;
           case Arches::ZDIR:
        if (!(zminus && (colZ == idxLo.z())))
           vars->wVelRhoHat[yminusCell] = vars->wVelRhoHat[currCell];
           break;
           default:
		throw InvalidValue("Invalid index in velocityPressureBC");
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
        IntVector yplusplusCell(colX, colY+2, colZ);
        if (constvars->cellType[yplusCell] == press_celltypeval) {
          switch (index) {
           case Arches::XDIR:
        if (!(xminus && (colX == idxLo.x())))
           vars->uVelRhoHat[yplusCell] = vars->uVelRhoHat[currCell];
           break;
           case Arches::YDIR:
           vars->vVelRhoHat[yplusCell] = vars->vVelRhoHat[currCell];
           vars->vVelRhoHat[yplusplusCell] = vars->vVelRhoHat[yplusCell];
           break;
           case Arches::ZDIR:
        if (!(zminus && (colZ == idxLo.z())))
           vars->wVelRhoHat[yplusCell] = vars->wVelRhoHat[currCell];
           break;
           default:
		throw InvalidValue("Invalid index in velocityPressureBC");
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
        IntVector zplusCell(colX, colY, colZ+1);
        if (constvars->cellType[zminusCell] == press_celltypeval) {
          switch (index) {
           case Arches::XDIR:
        if (!(xminus && (colX == idxLo.x())))
           vars->uVelRhoHat[zminusCell] = vars->uVelRhoHat[currCell];
           break;
           case Arches::YDIR:
        if (!(yminus && (colY == idxLo.y())))
           vars->vVelRhoHat[zminusCell] = vars->vVelRhoHat[currCell];
           break;
           case Arches::ZDIR:
           vars->wVelRhoHat[currCell] = vars->wVelRhoHat[zplusCell];
           vars->wVelRhoHat[zminusCell] = vars->wVelRhoHat[currCell];
           break;
           default:
		throw InvalidValue("Invalid index in velocityPressureBC");
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
        IntVector zplusplusCell(colX, colY, colZ+2);
        if (constvars->cellType[zplusCell] == press_celltypeval) {
          switch (index) {
           case Arches::XDIR:
        if (!(xminus && (colX == idxLo.x())))
           vars->uVelRhoHat[zplusCell] = vars->uVelRhoHat[currCell];
           break;
           case Arches::YDIR:
        if (!(yminus && (colY == idxLo.y())))
           vars->vVelRhoHat[zplusCell] = vars->vVelRhoHat[currCell];
           break;
           case Arches::ZDIR:
           vars->wVelRhoHat[zplusCell] = vars->wVelRhoHat[currCell];
           vars->wVelRhoHat[zplusplusCell] = vars->wVelRhoHat[zplusCell];
           break;
           default:
		throw InvalidValue("Invalid index in velocityPressureBC");
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

  int out_celltypeval = d_outletBC->d_cellTypeID;

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
        if (constvars->cellType[xminusCell] == out_celltypeval) {
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
        if (constvars->cellType[xplusCell] == out_celltypeval) {
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
        if (constvars->cellType[yminusCell] == out_celltypeval) {
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
        if (constvars->cellType[yplusCell] == out_celltypeval) {
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
        if (constvars->cellType[zminusCell] == out_celltypeval) {
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
        if (constvars->cellType[zplusCell] == out_celltypeval) {
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

  tsk->requires(Task::NewDW, timelabels->density_out, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, timelabels->uvelocity_out,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, timelabels->vvelocity_out,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, timelabels->wvelocity_out,
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

    new_dw->get(density, timelabels->density_out, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(uVelocity, timelabels->uvelocity_out, matlIndex,
		patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(vVelocity, timelabels->vvelocity_out, matlIndex,
		patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(wVelocity, timelabels->wvelocity_out, matlIndex,
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

    for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
      for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
        for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
	  IntVector currCell(ii,jj,kk);
	  denAccum += filterdrhodt[currCell];
        }
      }
    }

    if (xminus||xplus||yminus||yplus||zminus||zplus) {

      for (int indx = 0; indx < d_numInlets; indx++) {

	// Get a copy of the current flowinlet
	// assign flowType the value that corresponds to flow
	//CellTypeInfo flowType = FLOW;
	FlowInlet fi = d_flowInlets[indx];
	double fout = 0.0;
	fort_inlpresbcinout(uVelocity, vVelocity, wVelocity, idxLo, idxHi,
			   density, cellType, fi.d_cellTypeID,
			   flowIN, fout, cellinfo->sew, cellinfo->sns,
			   cellinfo->stb, xminus, xplus, yminus, yplus,
			   zminus, zplus);
	if (fout > 0.0)
		throw InvalidValue("Flow comming out of inlet");
      } 

      if (d_pressBoundary) {
	int press_celltypeval = d_pressureBdry->d_cellTypeID;
	fort_inlpresbcinout(uVelocity, vVelocity, wVelocity, idxLo, idxHi,
			    density, cellType, press_celltypeval,
			    flowIN, flowOUT, cellinfo->sew, cellinfo->sns,
			    cellinfo->stb, xminus, xplus, yminus, yplus,
			    zminus, zplus);
      }
      if (d_outletBoundary) {
	int out_celltypeval = d_outletBC->d_cellTypeID;
	if (xminus) {
	  int colX = idxLo.x();
  	  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
	    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
	      IntVector currCell(colX, colY, colZ);
      	      IntVector xminusCell(colX-1, colY, colZ);

	      if (cellType[xminusCell] == out_celltypeval) {
                 double avdenlow = 0.5 * (density[currCell] +
	      		            density[xminusCell]);
     	    	 floutbc += avdenlow*uVelocity[currCell] *
	          	     cellinfo->sns[colY] * cellinfo->stb[colZ];
    	         areaOUT += avdenlow *
	          	     cellinfo->sns[colY] * cellinfo->stb[colZ];
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

	      if (cellType[xplusCell] == out_celltypeval) {
	         double avden = 0.5 * (density[xplusCell] +
	      		         density[currCell]);
	         floutbc += avden*uVelocity[xplusCell] *
	       	     cellinfo->sns[colY] * cellinfo->stb[colZ];
    	         areaOUT += avden *
	       	     cellinfo->sns[colY] * cellinfo->stb[colZ];
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

 	      if (cellType[yminusCell] == out_celltypeval) {
 	         double avdenlow = 0.5 * (density[currCell] +
 	      		            density[yminusCell]);
 	         floutbc += avdenlow*vVelocity[currCell] *
	                   cellinfo->sew[colX] * cellinfo->stb[colZ];
    	         areaOUT += avdenlow *
	                   cellinfo->sew[colX] * cellinfo->stb[colZ];
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

 	      if (cellType[yplusCell] == out_celltypeval) {
 	         double avden = 0.5 * (density[yplusCell] +
 	      		         density[currCell]);
 	         floutbc += avden*vVelocity[yplusCell] *
 	          	     cellinfo->sew[colX] * cellinfo->stb[colZ];
    	         areaOUT += avden *
 	          	     cellinfo->sew[colX] * cellinfo->stb[colZ];
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

 	      if (cellType[zminusCell] == out_celltypeval) {
 	         double avdenlow = 0.5 * (density[currCell] +
 	      		            density[zminusCell]);
 	         floutbc += avdenlow*wVelocity[currCell] *
 	          	     cellinfo->sew[colX] * cellinfo->sns[colY];
    	         areaOUT += avdenlow *
 	          	     cellinfo->sew[colX] * cellinfo->sns[colY];
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

 	      if (cellType[zplusCell] == out_celltypeval) {
 	         double avden = 0.5 * (density[zplusCell] +
 	      		         density[currCell]);
 	         floutbc += avden*wVelocity[zplusCell] *
 	          	     cellinfo->sew[colX] * cellinfo->sns[colY];
    	         areaOUT += avden *
 	          	     cellinfo->sew[colX] * cellinfo->sns[colY];
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
// Schedule outlet velocity correction
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

    tsk->modifies(timelabels->uvelocity_out);
    tsk->modifies(timelabels->vvelocity_out);
    tsk->modifies(timelabels->wvelocity_out);

  if (timelabels->integrator_last_step)
    tsk->computes(d_lab->d_uvwoutLabel);

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Correct outlet velocity
//****************************************************************************
void 
BoundaryCondition::correctVelocityOutletBC(const ProcessorGroup* pc,
			      const PatchSubset* patches,
			      const MaterialSubset*,
			      DataWarehouse*,
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
    double uvwcorr;

    d_overallMB = fabs((totalFlowIN - denAccum - totalFlowOUT - 
			 netFlowOUT_outbc)/totalFlowIN);

    int me = pc->myrank();

    if (d_outletBoundary) {
     if (totalAreaOUT > 0.0) {
	uvwcorr = (totalFlowIN - denAccum - totalFlowOUT - netFlowOUT_outbc)/
	           totalAreaOUT;
#ifdef discard_negative_velocity_correction
	if (uvwcorr < 0.0) {
         if (me == 0) {
          cerr << "Negative velocity correction " << uvwcorr
	       << " , discarding it" << endl;
         }
	 uvwcorr = 0.0;
        }
#endif
      }
    }
    else
      uvwcorr = 0.0;

    if (me == 0) {
      if (d_overallMB > 0.0)
	cerr << "Overall Mass Balance " << log10(d_overallMB/1.e-7) << endl;
      cerr << "Total flow in " << totalFlowIN << endl;
      cerr << "Total flow out " << totalFlowOUT << endl;
      cerr << "Total flow out BC: " << netFlowOUT_outbc << endl;
      cerr << "Overall velocity correction " << uvwcorr << endl;
      cerr << "Total Area out " << totalAreaOUT << endl;
    }
    if (timelabels->integrator_last_step)
      new_dw->put(delt_vartype(uvwcorr), d_lab->d_uvwoutLabel);
 
    for (int p = 0; p < patches->size(); p++) {
      const Patch* patch = patches->get(p);
      int archIndex = 0; // only one arches material
      int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
      SFCXVariable<double> uVelocity;
      SFCYVariable<double> vVelocity;
      SFCZVariable<double> wVelocity;
      new_dw->getModifiable(uVelocity, timelabels->uvelocity_out, matlIndex,
		  	    patch);
      new_dw->getModifiable(vVelocity, timelabels->vvelocity_out, matlIndex,
		  	    patch);
      new_dw->getModifiable(wVelocity, timelabels->wvelocity_out, matlIndex,
		  	    patch);

// Assuming outlet is xplus
      IntVector indexLow = patch->getCellFORTLowIndex();
      IntVector indexHigh = patch->getCellFORTHighIndex();
      bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
      if (xplus) {
        int colX = indexHigh.x()+1;
	for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
	  for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	    IntVector currCell(colX, colY, colZ);
	    IntVector xplusCell(colX+1, colY, colZ);
	    
	    uVelocity[currCell] += uvwcorr;
// Negative velocity limiter, as requested by Rajesh
	    if (uVelocity[currCell] < 0.0) uVelocity[currCell] = 0.0;
	    uVelocity[xplusCell] = uVelocity[currCell];
	  }
	}
      }
    }
}
