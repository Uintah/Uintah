//----- BoundaryCondition.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesFort.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/Core/Grid/Stencil.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Properties.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Core/Util/NotFinished.h>
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
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/TypeUtils.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

//****************************************************************************
// Constructor for BoundaryCondition
//****************************************************************************
BoundaryCondition::BoundaryCondition(const ArchesLabel* label,
				     const MPMArchesLabel* MAlb,
				     TurbulenceModel* turb_model,
				     Properties* props):
                                     d_lab(label), d_MAlab(MAlb),
				     d_turbModel(turb_model), 
				     d_props(props)
{
  d_nofScalars = d_props->getNumMixVars();
  MM_CUTOFF_VOID_FRAC = 0.01;
}

//****************************************************************************
// Destructor
//****************************************************************************
BoundaryCondition::~BoundaryCondition()
{
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void 
BoundaryCondition::problemSetup(const ProblemSpecP& params)
{

  ProblemSpecP db = params->findBlock("BoundaryConditions");
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
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;

    // cell type initialization
    Task* tsk = scinew Task("BoundaryCondition::cellTypeInit",
			 patch, old_dw, new_dw, this,
			 &BoundaryCondition::cellTypeInit);
    int matlIndex = 0;
    tsk->computes(new_dw, d_lab->d_cellTypeLabel, matlIndex, patch);
    sched->addTask(tsk);
  }
#endif
}

//****************************************************************************
// Actual initialization of celltype
//****************************************************************************
void 
BoundaryCondition::cellTypeInit(const ProcessorGroup*,
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    CCVariable<int> cellType;
    new_dw->allocate(cellType, d_lab->d_cellTypeLabel, matlIndex, patch);

    IntVector domLo = cellType.getFortLowIndex();
    IntVector domHi = cellType.getFortHighIndex();
    IntVector idxLo = domLo;
    IntVector idxHi = domHi;
 
#ifdef ARCHES_GEOM_DEBUG
    cerr << "Just before geom init" << endl;
#endif
    // initialize CCVariable to -1 which corresponds to flowfield
    int celltypeval;
    d_flowfieldCellTypeVal = -1;
    FORT_CELLTYPEINIT(domLo.get_pointer(), domHi.get_pointer(), 
		      idxLo.get_pointer(), idxHi.get_pointer(),
		      cellType.getPointer(), &d_flowfieldCellTypeVal);
    
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
	  FORT_CELLTYPEINIT(domLo.get_pointer(), domHi.get_pointer(), 
			    idxLo.get_pointer(), idxHi.get_pointer(),
			    cellType.getPointer(), &celltypeval);
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
	    FORT_CELLTYPEINIT(domLo.get_pointer(), domHi.get_pointer(),
			      idxLo.get_pointer(), idxHi.get_pointer(),
			      cellType.getPointer(), &celltypeval);
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
	    FORT_CELLTYPEINIT(domLo.get_pointer(), domHi.get_pointer(),
			      idxLo.get_pointer(), idxHi.get_pointer(),
			      cellType.getPointer(), &celltypeval);
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
	CellIterator iter = patch->getCellIterator(b);
	IntVector idxLo = iter.begin();
	IntVector idxHi = iter.end() - IntVector(1,1,1);
	celltypeval = d_flowInlets[ii].d_cellTypeID;
#ifdef ARCHES_GEOM_DEBUG
	cerr << "Flow inlet " << ii << " cell type val = " << celltypeval << endl;
#endif
	FORT_CELLTYPEINIT(domLo.get_pointer(), domHi.get_pointer(),
			  idxLo.get_pointer(), idxHi.get_pointer(),
			  cellType.getPointer(), &celltypeval);
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
    
    new_dw->put(cellType, d_lab->d_cellTypeLabel, matlIndex, patch);
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
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;

    // cell type initialization
    Task* tsk = scinew Task("BoundaryCondition::mmWallCellTypeInit",
			 patch, old_dw, new_dw, this,
			 &BoundaryCondition::mmWallCellTypeInit);
    int matlIndex = 0;
    int numGhostcells = 0;
    tsk->requires(new_dw, d_MAlab->void_frac_CCLabel, matlIndex, patch,
		  Ghost::None, numGhostcells);
    tsk->requires(old_dw, d_lab->d_cellTypeLabel, matlIndex, patch,
		  Ghost::None, numGhostcells);
    tsk->computes(new_dw, d_lab->d_mmcellTypeLabel, matlIndex, patch);
    tsk->computes(new_dw, d_lab->d_mmgasVolFracLabel, matlIndex, patch);
    
    sched->addTask(tsk);
  }
#endif
}

//****************************************************************************
// Actual initialization of celltype
//****************************************************************************
void 
BoundaryCondition::mmWallCellTypeInit(const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset* matls,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw)		
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    int numGhostcells = 0;
    CCVariable<int> cellType;
    old_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch,
		Ghost::None, numGhostcells);
    CCVariable<double> voidFrac;
    new_dw->get(voidFrac, d_MAlab->void_frac_CCLabel, matlIndex, patch,
		Ghost::None, numGhostcells);
    CCVariable<int> mmcellType;
    new_dw->allocate(mmcellType, d_lab->d_mmcellTypeLabel, matlIndex, patch);
    IntVector domLo = cellType.getFortLowIndex();
    IntVector domHi = cellType.getFortHighIndex();
    IntVector idxLo = domLo;
    IntVector idxHi = domHi;
    // resets old mmwall type back to flow field and sets cells with void fraction
    // of less than .01 to mmWall
    FORT_MMCELLTYPEINIT(domLo.get_pointer(), domHi.get_pointer(), 
			idxLo.get_pointer(), idxHi.get_pointer(),
			voidFrac.getPointer(),
			cellType.getPointer(),
			&d_mmWallID, &d_flowfieldCellTypeVal,
			&MM_CUTOFF_VOID_FRAC);
    
    new_dw->put(cellType, d_lab->d_mmcellTypeLabel, matlIndex, patch);
    // save in arches label
    new_dw->put(voidFrac, d_lab->d_mmgasVolFracLabel, matlIndex, patch);
  }  
}


    
//****************************************************************************
// Actual initialization of celltype
//****************************************************************************
void 
BoundaryCondition::computeInletFlowArea(const ProcessorGroup*,
					const PatchSubset* patches,
					const MaterialSubset* matls,
					DataWarehouse* old_dw,
					DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Create the cellType variable
    CCVariable<int> cellType;
    
    // Get the cell type data from the old_dw
    // **WARNING** numGhostcells, Ghost::None may change in the future
    int numGhostCells = 0;
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch,
		Ghost::None, numGhostCells);
    
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
	int xminus = patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1;
	int xplus =  patch->getBCType(Patch::xplus) == Patch::Neighbor?0:1;
	int yminus = patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1;
	int yplus =  patch->getBCType(Patch::yplus) == Patch::Neighbor?0:1;
	int zminus = patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1;
	int zplus =  patch->getBCType(Patch::zplus) == Patch::Neighbor?0:1;
	
	FORT_AREAIN(domLo.get_pointer(), domHi.get_pointer(),
		    idxLo.get_pointer(), idxHi.get_pointer(),
		    cellInfo->sew.get_objs(),
		    cellInfo->sns.get_objs(), cellInfo->stb.get_objs(),
		    &inlet_area, cellType.getPointer(), &cellid,
		    &xminus, &xplus, &yminus, &yplus,
		    &zminus, &zplus);
	
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
  int numGhostCells = 0;
  // This task requires the pressure
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		Ghost::None, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_pressureINLabel, Ghost::None,
		numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::None,
		numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPLabel, Ghost::None,
		numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPLabel, Ghost::None,
		numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPLabel, Ghost::None,
		numGhostCells);
      // This task computes new uVelocity, vVelocity and wVelocity
  tsk->computes(d_lab->d_pressureSPBCLabel);
  tsk->computes(d_lab->d_uVelocitySPBCLabel);
  tsk->computes(d_lab->d_vVelocitySPBCLabel);
  tsk->computes(d_lab->d_wVelocitySPBCLabel);

  sched->addTask(tsk, patches, matls);
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      //copies old db to new_db and then uses non-linear
      //solver to compute new values
      Task* tsk = scinew Task("BoundaryCondition::calcPressureBC",patch,
			      old_dw, new_dw, this,
			      &BoundaryCondition::calcPressureBC);

      int numGhostCells = 0;
      int matlIndex = 0;
      
      // This task requires the pressure
      tsk->requires(old_dw, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_lab->d_pressureINLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_lab->d_densityCPLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_lab->d_uVelocitySPLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_lab->d_vVelocitySPLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_lab->d_wVelocitySPLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      // This task computes new uVelocity, vVelocity and wVelocity
      tsk->computes(new_dw, d_lab->d_pressureSPBCLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_uVelocitySPBCLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_vVelocitySPBCLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_wVelocitySPBCLabel, matlIndex, patch);

      sched->addTask(tsk);
    }
  }
#endif
}

//****************************************************************************
// Actually calculate the pressure BCs
//****************************************************************************
void 
BoundaryCondition::calcPressureBC(const ProcessorGroup* ,
				  const PatchSubset* patches,
				  const MaterialSubset* matls,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw) 
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    CCVariable<int> cellType;
    CCVariable<double> density;
    CCVariable<double> pressure;
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    int nofGhostCells = 0;

    // get cellType, pressure and velocity
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    new_dw->get(density, d_lab->d_densityCPLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    new_dw->get(pressure, d_lab->d_pressureINLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    new_dw->get(uVelocity, d_lab->d_uVelocitySPLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    new_dw->get(vVelocity, d_lab->d_vVelocitySPLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    new_dw->get(wVelocity, d_lab->d_wVelocitySPLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    
    // Get the low and high index for the patch and the variables
    //  IntVector domLoScalar = density.getFortLowIndex();
    //  IntVector domHiScalar = density.getFortHighIndex();
    IntVector domLoDen = density.getFortLowIndex();
    IntVector domHiDen = density.getFortHighIndex();
    IntVector domLoPress = pressure.getFortLowIndex();
    IntVector domHiPress = pressure.getFortHighIndex();
    IntVector domLoCT = cellType.getFortLowIndex();
    IntVector domHiCT = cellType.getFortHighIndex();
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
    IntVector domLoU = uVelocity.getFortLowIndex();
    IntVector domHiU = uVelocity.getFortHighIndex();
    IntVector domLoV = vVelocity.getFortLowIndex();
    IntVector domHiV = vVelocity.getFortHighIndex();
    IntVector domLoW = wVelocity.getFortLowIndex();
    IntVector domHiW = wVelocity.getFortHighIndex();
    
    int xminus = patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1;
    int xplus =  patch->getBCType(Patch::xplus) == Patch::Neighbor?0:1;
    int yminus = patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1;
    int yplus =  patch->getBCType(Patch::yplus) == Patch::Neighbor?0:1;
    int zminus = patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1;
    int zplus =  patch->getBCType(Patch::zplus) == Patch::Neighbor?0:1;
    
    
    FORT_CALPBC(domLoU.get_pointer(), domHiU.get_pointer(), 
		uVelocity.getPointer(),
		domLoV.get_pointer(), domHiV.get_pointer(), 
		vVelocity.getPointer(),
		domLoW.get_pointer(), domHiW.get_pointer(), 
		wVelocity.getPointer(),
		domLoDen.get_pointer(), domHiDen.get_pointer(), 
		domLoPress.get_pointer(), domHiPress.get_pointer(), 
		domLoCT.get_pointer(), domHiCT.get_pointer(), 
		idxLo.get_pointer(), idxHi.get_pointer(), 
		pressure.getPointer(),
		density.getPointer(), 
		cellType.getPointer(),
		&(d_pressureBdry->d_cellTypeID),
		&(d_pressureBdry->refPressure),
		&xminus, &xplus, &yminus, &yplus,
		&zminus, &zplus);
    
    // Put the calculated data into the new DW
    new_dw->put(uVelocity, d_lab->d_uVelocitySPBCLabel, matlIndex, patch);
    new_dw->put(vVelocity, d_lab->d_vVelocitySPBCLabel, matlIndex, patch);
    new_dw->put(wVelocity, d_lab->d_wVelocitySPBCLabel, matlIndex, patch);
    new_dw->put(pressure, d_lab->d_pressureSPBCLabel, matlIndex, patch);

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
  
  int numGhostCells = 0;
  // This task requires densityCP, [u,v,w]VelocitySP from new_dw
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None,
		numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, Ghost::None,
		numGhostCells);
  // changes to make it work for the task graph
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,  
		Ghost::AroundCells, numGhostCells+2);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityINLabel, Ghost::None,
		numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityINLabel, Ghost::None,
		numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityINLabel, Ghost::None,
		numGhostCells);
  
  // This task computes new density, uVelocity, vVelocity and wVelocity
  tsk->computes(d_lab->d_uVelocitySIVBCLabel);
  tsk->computes(d_lab->d_vVelocitySIVBCLabel);
  tsk->computes(d_lab->d_wVelocitySIVBCLabel);
  
  sched->addTask(tsk, patches, matls);
  
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("BoundaryCondition::setInletVelocityBC",
			      patch, old_dw, new_dw, this,
			      &BoundaryCondition::setInletVelocityBC);

      int numGhostCells = 0;
      int matlIndex = 0;

      // This task requires densityCP, [u,v,w]VelocitySP from new_dw
      tsk->requires(old_dw, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_lab->d_densityINLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      // changes to make it work for the task graph
      tsk->requires(new_dw, d_lab->d_densityINLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells+2);
      tsk->requires(new_dw, d_lab->d_uVelocityINLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_lab->d_vVelocityINLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_lab->d_wVelocityINLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);

      // This task computes new density, uVelocity, vVelocity and wVelocity
      tsk->computes(new_dw, d_lab->d_uVelocitySIVBCLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_vVelocitySIVBCLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_wVelocitySIVBCLabel, matlIndex, patch);

      sched->addTask(tsk);
    }
  }
#endif
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
  int numGhostCells = 0;
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		Ghost::None, numGhostCells);
      // This task requires celltype, new density, pressure and velocity
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel, Ghost::AroundCells,
		numGhostCells+2);
  tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel, Ghost::None,
		numGhostCells);
  // changes to make it work for the task graph
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySIVBCLabel,
		Ghost::AroundCells, numGhostCells+2);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySIVBCLabel,
		Ghost::AroundCells, numGhostCells+2);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySIVBCLabel,
		Ghost::AroundCells, numGhostCells+2);
  for (int ii = 0; ii < d_nofScalars; ii++)
    tsk->requires(Task::NewDW, d_lab->d_scalarINLabel, 
		  Ghost::None, numGhostCells);
  
  
  // This task computes new uVelocity, vVelocity and wVelocity
  tsk->computes(d_lab->d_uVelocityCPBCLabel);
  tsk->computes(d_lab->d_vVelocityCPBCLabel);
  tsk->computes(d_lab->d_wVelocityCPBCLabel);
  tsk->computes(d_lab->d_pressureSPBCLabel);
  for (int ii = 0; ii < d_nofScalars; ii++)
    tsk->computes(d_lab->d_scalarCPBCLabel);
  
  sched->addTask(tsk, patches, matls);
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("BoundaryCondition::recomputePressureBC",
			      patch, old_dw, new_dw, this,
			      &BoundaryCondition::recomputePressureBC);

      int numGhostCells = 0;
      int matlIndex = 0;
      tsk->requires(old_dw, d_lab->d_cellTypeLabel, matlIndex, patch,
		    Ghost::None, numGhostCells);
      // This task requires celltype, new density, pressure and velocity
      tsk->requires(new_dw, d_lab->d_densityINLabel, matlIndex, patch, Ghost::AroundCells,
		    numGhostCells+2);
      tsk->requires(new_dw, d_lab->d_pressurePSLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      // changes to make it work for the task graph
      tsk->requires(new_dw, d_lab->d_uVelocitySIVBCLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells+2);
      tsk->requires(new_dw, d_lab->d_vVelocitySIVBCLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells+2);
      tsk->requires(new_dw, d_lab->d_wVelocitySIVBCLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells+2);
      for (int ii = 0; ii < d_nofScalars; ii++)
	tsk->requires(new_dw, d_lab->d_scalarINLabel, ii, patch,
		      Ghost::None, numGhostCells);
      

      // This task computes new uVelocity, vVelocity and wVelocity
      tsk->computes(new_dw, d_lab->d_uVelocityCPBCLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_vVelocityCPBCLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_wVelocityCPBCLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_pressureSPBCLabel, matlIndex, patch);
      for (int ii = 0; ii < d_nofScalars; ii++)
	tsk->computes(new_dw, d_lab->d_scalarCPBCLabel, ii, patch);

      sched->addTask(tsk);
    }
  }
#endif
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
  int numGhostCells = 0;
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None,
		numGhostCells);
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
      int numGhostCells = 0;
      tsk->requires(old_dw, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
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
  int numGhostCells = 0;
  // This task requires cellTypeVariable and areaLabel for inlet boundary
  // Also densityIN, [u,v,w] velocityIN, scalarIN
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::None, numGhostCells);
  for (int ii = 0; ii < d_numInlets; ii++) {
    tsk->requires(Task::NewDW, d_flowInlets[ii].d_area_label);
  }
  // This task requires old density, uVelocity, vVelocity and wVelocity
  tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		Ghost::None, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_uVelocityINLabel, 
		Ghost::None, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_vVelocityINLabel,
		Ghost::None, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_wVelocityINLabel, 
		Ghost::None, numGhostCells);
  // will only work for one scalar variable
  for (int ii = 0; ii < d_props->getNumMixVars(); ii++) 
    tsk->requires(Task::NewDW, d_lab->d_scalarINLabel,
		  Ghost::None, numGhostCells);
  // This task computes new density, uVelocity, vVelocity and wVelocity, scalars
  tsk->computes(d_lab->d_densitySPLabel);
  tsk->computes(d_lab->d_uVelocitySPLabel);
  tsk->computes(d_lab->d_vVelocitySPLabel);
  tsk->computes(d_lab->d_wVelocitySPLabel);
  for (int ii = 0; ii < d_props->getNumMixVars(); ii++) 
    tsk->computes(d_lab->d_scalarSPLabel);
  sched->addTask(tsk, patches, matls);
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("BoundaryCondition::setProfile",
			      patch, old_dw, new_dw, this,
			      &BoundaryCondition::setFlatProfile);
      int numGhostCells = 0;
      int matlIndex = 0;

      // This task requires cellTypeVariable and areaLabel for inlet boundary
      // Also densityIN, [u,v,w] velocityIN, scalarIN
      tsk->requires(old_dw, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      for (int ii = 0; ii < d_numInlets; ii++) {
	tsk->requires(old_dw, d_flowInlets[ii].d_area_label);
      }
      // This task requires old density, uVelocity, vVelocity and wVelocity
      tsk->requires(old_dw, d_lab->d_densityINLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_lab->d_uVelocityINLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_lab->d_vVelocityINLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_lab->d_wVelocityINLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      for (int ii = 0; ii < d_props->getNumMixVars(); ii++) 
	tsk->requires(old_dw, d_lab->d_scalarINLabel, ii, patch, Ghost::None,
		      numGhostCells);
      // This task computes new density, uVelocity, vVelocity and wVelocity, scalars
      tsk->computes(new_dw, d_lab->d_densitySPLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_uVelocitySPLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_vVelocitySPLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_wVelocitySPLabel, matlIndex, patch);
      for (int ii = 0; ii < d_props->getNumMixVars(); ii++) 
	tsk->computes(new_dw, d_lab->d_scalarSPLabel, ii, patch);
      sched->addTask(tsk);
    }
  }
#endif
}

//****************************************************************************
// Actually calculate the velocity BC
//****************************************************************************
void 
BoundaryCondition::velocityBC(const ProcessorGroup* pc,
			      const Patch* patch,
			      int index,
			      CellInformation* cellinfo,
			      ArchesVariables* vars) 
{
  //get Molecular Viscosity of the fluid
  double molViscosity = d_turbModel->getMolecularViscosity();

  // Call the fortran routines
  switch(index) {
  case 1:
    uVelocityBC(patch,
		&molViscosity, cellinfo, vars);
    break;
  case 2:
    vVelocityBC(patch,
		&molViscosity, cellinfo, vars);
    break;
  case 3:
    wVelocityBC(patch,
		&molViscosity, cellinfo, vars);
    break;
  default:
    cerr << "Invalid Index value" << endl;
    break;
  }
  // Calculate the velocity wall BC
  // For Arches::PRESSURE
  //  inputs : densityCP, [u,v,w]VelocitySIVBC, [u,v,w]VelCoefPBLM
  //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
  //  outputs: [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM, 
  //           [u,v,w]VelNonLinSrcPBLM
  // For Arches::MOMENTUM
  //  inputs : densityCP, [u,v,w]VelocitySIVBC, [u,v,w]VelCoefMBLM
  //           [u,v,w]VelLinSrcMBLM, [u,v,w]VelNonLinSrcMBLM
  //  outputs: [u,v,w]VelCoefMBLM, [u,v,w]VelLinSrcMBLM, 
  //           [u,v,w]VelNonLinSrcMBLM
  //  d_turbModel->calcVelocityWallBC(pc, patch, old_dw, new_dw, index, eqnType);
}

//****************************************************************************
// call fortran routine to calculate the U Velocity BC
//****************************************************************************
void 
BoundaryCondition::uVelocityBC(const Patch* patch,
			       const double* VISCOS,
			       CellInformation* cellinfo,
			       ArchesVariables* vars)
{
  int wall_celltypeval = d_wallBdry->d_cellTypeID;
  int flow_celltypeval = d_flowfieldCellTypeVal;
  int press_celltypeval = d_pressureBdry->d_cellTypeID;
  // Get the low and high index for the patch and the variables
  IntVector domLoU = vars->uVelocity.getFortLowIndex();
  IntVector domHiU = vars->uVelocity.getFortHighIndex();
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  IntVector domLoV = vars->vVelocity.getFortLowIndex();
  IntVector domHiV = vars->vVelocity.getFortHighIndex();
  IntVector idxLoV = patch->getSFCYFORTLowIndex();
  IntVector idxHiV = patch->getSFCYFORTHighIndex();
  IntVector domLoW = vars->wVelocity.getFortLowIndex();
  IntVector domHiW = vars->wVelocity.getFortHighIndex();
  IntVector idxLoW = patch->getSFCZFORTLowIndex();
  IntVector idxHiW = patch->getSFCZFORTHighIndex();
  IntVector domLo = vars->cellType.getFortLowIndex();
  IntVector domHi = vars->cellType.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  // for no ghost cells
  IntVector domLoUng = vars->uVelLinearSrc.getFortLowIndex();
  IntVector domHiUng = vars->uVelLinearSrc.getFortHighIndex();
  
  // ** Reverted back to old ways
  // for a single patch should be equal to 1 and nx
  //IntVector idxLoU = vars->cellType.getFortLowIndex();
  //IntVector idxHiU = vars->cellType.getFortHighIndex();
  // computes momentum source term due to wall
  int xminus = patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1;
  int xplus =  patch->getBCType(Patch::xplus) == Patch::Neighbor?0:1;
  int yminus = patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1;
  int yplus =  patch->getBCType(Patch::yplus) == Patch::Neighbor?0:1;
  int zminus = patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1;
  int zplus =  patch->getBCType(Patch::zplus) == Patch::Neighbor?0:1;

  FORT_BCUVEL(domLoU.get_pointer(), domHiU.get_pointer(), 
	      domLoUng.get_pointer(), domHiUng.get_pointer(), 
	      idxLoU.get_pointer(), idxHiU.get_pointer(),
	      vars->uVelocity.getPointer(),
	      vars->uVelocityCoeff[Arches::AP].getPointer(), 
	      vars->uVelocityCoeff[Arches::AE].getPointer(), 
	      vars->uVelocityCoeff[Arches::AW].getPointer(), 
	      vars->uVelocityCoeff[Arches::AN].getPointer(), 
	      vars->uVelocityCoeff[Arches::AS].getPointer(), 
	      vars->uVelocityCoeff[Arches::AT].getPointer(), 
	      vars->uVelocityCoeff[Arches::AB].getPointer(), 
	      vars->uVelNonlinearSrc.getPointer(), vars->uVelLinearSrc.getPointer(),
	      domLoV.get_pointer(), domHiV.get_pointer(),
	      idxLoV.get_pointer(), idxHiV.get_pointer(),
	      vars->vVelocity.getPointer(),
	      domLoW.get_pointer(), domHiW.get_pointer(),
	      idxLoW.get_pointer(), idxHiW.get_pointer(),
	      vars->wVelocity.getPointer(),
	      domLo.get_pointer(), domHi.get_pointer(),
	      idxLo.get_pointer(), idxHi.get_pointer(),
	      vars->cellType.getPointer(),
	      &wall_celltypeval, &flow_celltypeval, &press_celltypeval,
	      VISCOS, 
	      cellinfo->sewu.get_objs(), cellinfo->sns.get_objs(), 
	      cellinfo->stb.get_objs(),
	      cellinfo->yy.get_objs(), cellinfo->yv.get_objs(), 
	      cellinfo->zz.get_objs(),
	      cellinfo->zw.get_objs(),
	      &xminus, &xplus, &yminus, &yplus,
	      &zminus, &zplus);

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
			       const double* VISCOS,
			       CellInformation* cellinfo,
			       ArchesVariables* vars)
{
  int wall_celltypeval = d_wallBdry->d_cellTypeID;
  int flow_celltypeval = d_flowfieldCellTypeVal;
  int press_celltypeval = d_pressureBdry->d_cellTypeID;
  // Get the low and high index for the patch and the variables
  IntVector domLoU = vars->uVelocity.getFortLowIndex();
  IntVector domHiU = vars->uVelocity.getFortHighIndex();
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  IntVector domLoV = vars->vVelocity.getFortLowIndex();
  IntVector domHiV = vars->vVelocity.getFortHighIndex();
  IntVector idxLoV = patch->getSFCYFORTLowIndex();
  IntVector idxHiV = patch->getSFCYFORTHighIndex();
  IntVector domLoW = vars->wVelocity.getFortLowIndex();
  IntVector domHiW = vars->wVelocity.getFortHighIndex();
  IntVector idxLoW = patch->getSFCZFORTLowIndex();
  IntVector idxHiW = patch->getSFCZFORTHighIndex();
  IntVector domLo = vars->cellType.getFortLowIndex();
  IntVector domHi = vars->cellType.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  // for no ghost cells
  IntVector domLoVng = vars->vVelLinearSrc.getFortLowIndex();
  IntVector domHiVng = vars->vVelLinearSrc.getFortHighIndex();
  // for a single patch should be equal to 1 and nx
  //IntVector idxLoV = vars->cellType.getFortLowIndex();
  //IntVector idxHiV = vars->cellType.getFortHighIndex();
  // computes momentum source term due to wall
  int xminus = patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1;
  int xplus =  patch->getBCType(Patch::xplus) == Patch::Neighbor?0:1;
  int yminus = patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1;
  int yplus =  patch->getBCType(Patch::yplus) == Patch::Neighbor?0:1;
  int zminus = patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1;
  int zplus =  patch->getBCType(Patch::zplus) == Patch::Neighbor?0:1;

  // computes remianing diffusion term and also computes source due to gravity
  FORT_BCVVEL(domLoV.get_pointer(), domHiV.get_pointer(), 
	      domLoVng.get_pointer(), domHiVng.get_pointer(), 
	      idxLoV.get_pointer(), idxHiV.get_pointer(),
	      vars->vVelocity.getPointer(),   
	      vars->vVelocityCoeff[Arches::AP].getPointer(), 
	      vars->vVelocityCoeff[Arches::AE].getPointer(), 
	      vars->vVelocityCoeff[Arches::AW].getPointer(), 
	      vars->vVelocityCoeff[Arches::AN].getPointer(), 
	      vars->vVelocityCoeff[Arches::AS].getPointer(), 
	      vars->vVelocityCoeff[Arches::AT].getPointer(), 
	      vars->vVelocityCoeff[Arches::AB].getPointer(), 
	      vars->vVelNonlinearSrc.getPointer(), vars->vVelLinearSrc.getPointer(),
	      domLoU.get_pointer(), domHiU.get_pointer(),
	      idxLoU.get_pointer(), idxHiU.get_pointer(),
	      vars->uVelocity.getPointer(),
	      domLoW.get_pointer(), domHiW.get_pointer(),
	      idxLoW.get_pointer(), idxHiW.get_pointer(),
	      vars->wVelocity.getPointer(),
	      domLo.get_pointer(), domHi.get_pointer(),
	      idxLo.get_pointer(), idxHi.get_pointer(),
	      vars->cellType.getPointer(),
	      &wall_celltypeval, &flow_celltypeval, &press_celltypeval,
	      VISCOS, 
	      cellinfo->sew.get_objs(), cellinfo->snsv.get_objs(), 
	      cellinfo->stb.get_objs(),
	      cellinfo->xx.get_objs(), cellinfo->xu.get_objs(), 
	      cellinfo->zz.get_objs(),
	      cellinfo->zw.get_objs(), 
	      &xminus, &xplus, &yminus, &yplus,
	      &zminus, &zplus);

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
			       const double* VISCOS,
			       CellInformation* cellinfo,
			       ArchesVariables* vars)
{
  int wall_celltypeval = d_wallBdry->d_cellTypeID;
  int flow_celltypeval = d_flowfieldCellTypeVal;
  int press_celltypeval = d_pressureBdry->d_cellTypeID;
  // Get the low and high index for the patch and the variables
  IntVector domLoU = vars->uVelocity.getFortLowIndex();
  IntVector domHiU = vars->uVelocity.getFortHighIndex();
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  IntVector domLoV = vars->vVelocity.getFortLowIndex();
  IntVector domHiV = vars->vVelocity.getFortHighIndex();
  IntVector idxLoV = patch->getSFCYFORTLowIndex();
  IntVector idxHiV = patch->getSFCYFORTHighIndex();
  IntVector domLoW = vars->wVelocity.getFortLowIndex();
  IntVector domHiW = vars->wVelocity.getFortHighIndex();
  IntVector idxLoW = patch->getSFCZFORTLowIndex();
  IntVector idxHiW = patch->getSFCZFORTHighIndex();
  IntVector domLo = vars->cellType.getFortLowIndex();
  IntVector domHi = vars->cellType.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  // for no ghost cells
  IntVector domLoWng = vars->wVelLinearSrc.getFortLowIndex();
  IntVector domHiWng = vars->wVelLinearSrc.getFortHighIndex();
  int xminus = patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1;
  int xplus =  patch->getBCType(Patch::xplus) == Patch::Neighbor?0:1;
  int yminus = patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1;
  int yplus =  patch->getBCType(Patch::yplus) == Patch::Neighbor?0:1;
  int zminus = patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1;
  int zplus =  patch->getBCType(Patch::zplus) == Patch::Neighbor?0:1;
  // for a single patch should be equal to 1 and nx
  //IntVector idxLoW = vars->cellType.getFortLowIndex();
  //IntVector idxHiW = vars->cellType.getFortHighIndex();
  // computes momentum source term due to wall
  FORT_BCWVEL(domLoW.get_pointer(), domHiW.get_pointer(), 
	      domLoWng.get_pointer(), domHiWng.get_pointer(), 
	      idxLoW.get_pointer(), idxHiW.get_pointer(),
	      vars->wVelocity.getPointer(),   
	      vars->wVelocityCoeff[Arches::AP].getPointer(), 
	      vars->wVelocityCoeff[Arches::AE].getPointer(), 
	      vars->wVelocityCoeff[Arches::AW].getPointer(), 
	      vars->wVelocityCoeff[Arches::AN].getPointer(), 
	      vars->wVelocityCoeff[Arches::AS].getPointer(), 
	      vars->wVelocityCoeff[Arches::AT].getPointer(), 
	      vars->wVelocityCoeff[Arches::AB].getPointer(), 
	      vars->wVelNonlinearSrc.getPointer(), vars->wVelLinearSrc.getPointer(),
	      domLoU.get_pointer(), domHiU.get_pointer(),
	      idxLoU.get_pointer(), idxHiU.get_pointer(),
	      vars->uVelocity.getPointer(),
	      domLoV.get_pointer(), domHiV.get_pointer(),
	      idxLoV.get_pointer(), idxHiV.get_pointer(),
	      vars->vVelocity.getPointer(),
	      domLo.get_pointer(), domHi.get_pointer(),
	      idxLo.get_pointer(), idxHi.get_pointer(),
	      vars->cellType.getPointer(),
	      &wall_celltypeval, &flow_celltypeval, &press_celltypeval,
	      VISCOS, 
	      cellinfo->sew.get_objs(), cellinfo->sns.get_objs(), 
	      cellinfo->stbw.get_objs(),
	      cellinfo->xx.get_objs(), cellinfo->xu.get_objs(), 
	      cellinfo->yy.get_objs(),
	      cellinfo->yv.get_objs(),
	      &xminus, &xplus, &yminus, &yplus,
	      &zminus, &zplus);

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
			      ArchesVariables* vars)
{
  // Get the low and high index for the patch
  IntVector domLo = vars->cellType.getFortLowIndex();
  IntVector domHi = vars->cellType.getFortHighIndex();
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

  int xminus = patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1;
  int xplus =  patch->getBCType(Patch::xplus) == Patch::Neighbor?0:1;
  int yminus = patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1;
  int yplus =  patch->getBCType(Patch::yplus) == Patch::Neighbor?0:1;
  int zminus = patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1;
  int zplus =  patch->getBCType(Patch::zplus) == Patch::Neighbor?0:1;

  //fortran call
  FORT_PRESSBC(domLo.get_pointer(), domHi.get_pointer(),
	       domLong.get_pointer(), domHing.get_pointer(),
	       idxLo.get_pointer(), idxHi.get_pointer(),
	       vars->pressure.getPointer(), 
	       vars->pressCoeff[Arches::AE].getPointer(),
	       vars->pressCoeff[Arches::AW].getPointer(),
	       vars->pressCoeff[Arches::AN].getPointer(),
	       vars->pressCoeff[Arches::AS].getPointer(),
	       vars->pressCoeff[Arches::AT].getPointer(),
	       vars->pressCoeff[Arches::AB].getPointer(),
	       vars->pressNonlinearSrc.getPointer(),
	       vars->pressLinearSrc.getPointer(),
	       vars->cellType.getPointer(),
	       &wall_celltypeval, &symmetry_celltypeval,
	       &flow_celltypeval,
	       &xminus, &xplus, &yminus, &yplus,
	       &zminus, &zplus);

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
			    ArchesVariables* vars)
{
  // Get the low and high index for the patch
  IntVector domLo = vars->density.getFortLowIndex();
  IntVector domHi = vars->density.getFortHighIndex();
  IntVector domLong = vars->scalar.getFortLowIndex();
  IntVector domHing = vars->scalar.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  IntVector domLoU = vars->uVelocity.getFortLowIndex();
  IntVector domHiU = vars->uVelocity.getFortHighIndex();
  IntVector domLoV = vars->vVelocity.getFortLowIndex();
  IntVector domHiV = vars->vVelocity.getFortHighIndex();
  IntVector domLoW = vars->wVelocity.getFortLowIndex();
  IntVector domHiW = vars->wVelocity.getFortHighIndex();

  // Get the wall boundary and flow field codes
  int wall_celltypeval = d_wallBdry->d_cellTypeID;
  int press_celltypeval = d_pressureBdry->d_cellTypeID;
  // ** WARNING ** Symmetry/sfield/outletfield/ffield hardcoded to -3,-4,-5, -6
  //               Fmixin hardcoded to 0
  int symmetry_celltypeval = -3;
  int sfield = -4;
  int outletfield = -5;
  int ffield = -1;
  double fmixin = 0.0;
  int xminus = patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1;
  int xplus =  patch->getBCType(Patch::xplus) == Patch::Neighbor?0:1;
  int yminus = patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1;
  int yplus =  patch->getBCType(Patch::yplus) == Patch::Neighbor?0:1;
  int zminus = patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1;
  int zplus =  patch->getBCType(Patch::zplus) == Patch::Neighbor?0:1;

  //fortran call
  FORT_SCALARBC(domLo.get_pointer(), domHi.get_pointer(),
		domLong.get_pointer(), domHing.get_pointer(),
		idxLo.get_pointer(), idxHi.get_pointer(),
		vars->scalar.getPointer(), 
		vars->scalarCoeff[Arches::AE].getPointer(),
		vars->scalarCoeff[Arches::AW].getPointer(),
		vars->scalarCoeff[Arches::AN].getPointer(),
		vars->scalarCoeff[Arches::AS].getPointer(),
		vars->scalarCoeff[Arches::AT].getPointer(),
		vars->scalarCoeff[Arches::AB].getPointer(),
		vars->scalarNonlinearSrc.getPointer(),
		vars->scalarLinearSrc.getPointer(),
		vars->density.getPointer(),
		&fmixin,
		domLoU.get_pointer(), domHiU.get_pointer(),
		vars->uVelocity.getPointer(), 
		domLoV.get_pointer(), domHiV.get_pointer(),
		vars->vVelocity.getPointer(), 
		domLoW.get_pointer(), domHiW.get_pointer(),
		vars->wVelocity.getPointer(), 
		cellinfo->sew.get_objs(), cellinfo->sns.get_objs(), 
		cellinfo->stb.get_objs(),
		vars->cellType.getPointer(),
		&wall_celltypeval, &symmetry_celltypeval,
		&d_flowInlets[0].d_cellTypeID, &press_celltypeval,
		&ffield, &sfield, &outletfield,
		&xminus, &xplus, &yminus, &yplus,
		&zminus, &zplus);

#ifdef ARCHES_BC_DEBUG
  cerr << "AFTER FORT_SCALARBC" << endl;
  cerr << "Print Scalar" << endl;
  vars->scalar.print(cerr);
  cerr << "Print scalar coeff, AE:" << endl;
  vars->scalarCoeff[Arches::AE].print(cerr);
#endif
}


//****************************************************************************
// Actually set the inlet velocity BC
//****************************************************************************
void 
BoundaryCondition::setInletVelocityBC(const ProcessorGroup* ,
				       const PatchSubset* patches,
				      const MaterialSubset* matls,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw) 
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    CCVariable<int> cellType;
    CCVariable<double> density;
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    int nofGhostCells = 0;

    // get cellType, velocity and density
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    new_dw->get(uVelocity, d_lab->d_uVelocityINLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    new_dw->get(vVelocity, d_lab->d_vVelocityINLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    new_dw->get(wVelocity, d_lab->d_wVelocityINLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    new_dw->get(density, d_lab->d_densityINLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    
    // Get the low and high index for the patch and the variables
    IntVector domLo = density.getFortLowIndex();
    IntVector domHi = density.getFortHighIndex();
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
    IntVector domLoU = uVelocity.getFortLowIndex();
    IntVector domHiU = uVelocity.getFortHighIndex();
    IntVector domLoV = vVelocity.getFortLowIndex();
    IntVector domHiV = vVelocity.getFortHighIndex();
    IntVector domLoW = wVelocity.getFortLowIndex();
    IntVector domHiW = wVelocity.getFortHighIndex();
    // stores cell type info for the patch with the ghost cell type
    for (int indx = 0; indx < d_numInlets; indx++) {
      // Get a copy of the current flowinlet
      FlowInlet fi = d_flowInlets[indx];
      
      // assign flowType the value that corresponds to flow
      //CellTypeInfo flowType = FLOW;
      int xminus = patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1;
      int xplus =  patch->getBCType(Patch::xplus) == Patch::Neighbor?0:1;
      int yminus = patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1;
      int yplus =  patch->getBCType(Patch::yplus) == Patch::Neighbor?0:1;
      int zminus = patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1;
      int zplus =  patch->getBCType(Patch::zplus) == Patch::Neighbor?0:1;
      FORT_INLBCS(domLoU.get_pointer(), domHiU.get_pointer(), 
		  uVelocity.getPointer(),
		  domLoV.get_pointer(), domHiV.get_pointer(), 
		  vVelocity.getPointer(),
		  domLoW.get_pointer(), domHiW.get_pointer(), 
		  wVelocity.getPointer(),
		  domLo.get_pointer(), domHi.get_pointer(), 
		  idxLo.get_pointer(), idxHi.get_pointer(), 
		  density.getPointer(),
		  cellType.getPointer(),
		  &fi.d_cellTypeID,
		  &xminus, &xplus, &yminus, &yplus, &zminus, &zplus); 
      //      &fi.flowRate, area, &fi.density, 
      //	&fi.inletType,
      //	flowType);
      
    }
    // Put the calculated data into the new DW
    new_dw->put(uVelocity, d_lab->d_uVelocitySIVBCLabel, matlIndex, patch);
    new_dw->put(vVelocity, d_lab->d_vVelocitySIVBCLabel, matlIndex, patch);
    new_dw->put(wVelocity, d_lab->d_wVelocitySIVBCLabel, matlIndex, patch);

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
				       const MaterialSubset* matls,
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw) 
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    CCVariable<int> cellType;
    CCVariable<double> density;
    CCVariable<double> pressure;
    vector<CCVariable<double> > scalar(d_nofScalars);
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    int nofGhostCells = 0;

    // get cellType, pressure and velocity
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    new_dw->get(density, d_lab->d_densityINLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    new_dw->get(pressure, d_lab->d_pressurePSLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    new_dw->get(uVelocity, d_lab->d_uVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    new_dw->get(vVelocity, d_lab->d_vVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    new_dw->get(wVelocity, d_lab->d_wVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    for (int ii = 0; ii < d_nofScalars; ii++)
      new_dw->get(scalar[ii], d_lab->d_scalarINLabel, matlIndex, patch, Ghost::None,
		  nofGhostCells);

  // Get the low and high index for the patch and the variables
    IntVector domLoScalar = scalar[0].getFortLowIndex();
    IntVector domHiScalar = scalar[0].getFortHighIndex();
    IntVector domLoDen = density.getFortLowIndex();
    IntVector domHiDen = density.getFortHighIndex();
    IntVector domLoPress = pressure.getFortLowIndex();
    IntVector domHiPress = pressure.getFortHighIndex();
    IntVector domLoCT = cellType.getFortLowIndex();
    IntVector domHiCT = cellType.getFortHighIndex();
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
    IntVector domLoU = uVelocity.getFortLowIndex();
    IntVector domHiU = uVelocity.getFortHighIndex();
    IntVector domLoV = vVelocity.getFortLowIndex();
    IntVector domHiV = vVelocity.getFortHighIndex();
    IntVector domLoW = wVelocity.getFortLowIndex();
    IntVector domHiW = wVelocity.getFortHighIndex();
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
    //rewindow(pressure, patch->getCellLowIndex(), patch->getCellHighIndex());
#if 0
    rewindow(pressure, patch->getCellLowIndex(), patch->getCellHighIndex());
    IntVector n1 (4,0,0);
    if(patch->containsCell(n1))
      cerr << "4,0,0: " << pressure[n1] << '\n';
    IntVector n2 (4,0,1);
    if(patch->containsCell(n2))
      cerr << "4,0,1: " << pressure[n2] << '\n';
    
    cerr << "4,0: pressure: " << pressure.getLowIndex() << ", " << pressure.getHighIndex() << '\n';
#endif
    int xminus = patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1;
    int xplus =  patch->getBCType(Patch::xplus) == Patch::Neighbor?0:1;
    int yminus = patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1;
    int yplus =  patch->getBCType(Patch::yplus) == Patch::Neighbor?0:1;
    int zminus = patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1;
    int zplus =  patch->getBCType(Patch::zplus) == Patch::Neighbor?0:1;
    
    FORT_CALPBC(domLoU.get_pointer(), domHiU.get_pointer(), 
		uVelocity.getPointer(),
		domLoV.get_pointer(), domHiV.get_pointer(), 
		vVelocity.getPointer(),
		domLoW.get_pointer(), domHiW.get_pointer(), 
		wVelocity.getPointer(),
		domLoDen.get_pointer(), domHiDen.get_pointer(), 
		domLoPress.get_pointer(), domHiPress.get_pointer(), 
		domLoCT.get_pointer(), domHiCT.get_pointer(), 
		idxLo.get_pointer(), idxHi.get_pointer(), 
		pressure.getPointer(),
		density.getPointer(), 
		cellType.getPointer(),
		&(d_pressureBdry->d_cellTypeID),
		&(d_pressureBdry->refPressure),
		&xminus, &xplus, &yminus, &yplus,
		&zminus, &zplus);
#if 0
    if(patch->containsCell(n1))
      cerr << "4,0,0: " << pressure[n1] << '\n';
    if(patch->containsCell(n2))
      cerr << "4,0,1: " << pressure[n2] << '\n';
#endif
    
    // set values of the scalars on the scalar boundary
    for (int ii = 0; ii < d_nofScalars; ii++) {
      FORT_PROFSCALAR(domLoScalar.get_pointer(), domHiScalar.get_pointer(),
		      domLoCT.get_pointer(), domHiCT.get_pointer(),
		      idxLo.get_pointer(), idxHi.get_pointer(),
		      scalar[ii].getPointer(), cellType.getPointer(),
		      &(d_pressureBdry->streamMixturefraction.d_mixVars[ii]),
		      &(d_pressureBdry->d_cellTypeID),
		      &xminus, &xplus, &yminus, &yplus,
		      &zminus, &zplus);
    }
#if 0
    if(patch->containsCell(n1))
      cerr << "4,0,0: " << pressure[n1] << '\n';
    if(patch->containsCell(n2))
      cerr << "4,0,1: " << pressure[n2] << '\n';
#endif
    
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
    for(CellIterator iter = patch->getCellIterator();
	!iter.done(); iter++){
      cerr.width(10);
      cerr << "RHO"<<*iter << ": " << density[*iter] << "\n" ; 
    }
#endif

  // Put the calculated data into the new DW
    new_dw->put(uVelocity, d_lab->d_uVelocityCPBCLabel, matlIndex, patch);
    new_dw->put(vVelocity, d_lab->d_vVelocityCPBCLabel, matlIndex, patch);
    new_dw->put(wVelocity, d_lab->d_wVelocityCPBCLabel, matlIndex, patch);
    new_dw->put(pressure, d_lab->d_pressureSPBCLabel, matlIndex, patch);
    for (int ii = 0; ii < d_nofScalars; ii++) 
      new_dw->put(scalar[ii], d_lab->d_scalarCPBCLabel, matlIndex, patch);
  }
} 

//****************************************************************************
// Actually set flat profile at flow inlet boundary
//****************************************************************************
void 
BoundaryCondition::setFlatProfile(const ProcessorGroup* /*pc*/,
				  const PatchSubset* patches,
				  const MaterialSubset* matls,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    CCVariable<int> cellType;
    CCVariable<double> density;
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    vector<CCVariable<double> > scalar(d_nofScalars);
    int nofGhostCells = 0;
    
    // get cellType, density and velocity
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    new_dw->get(density, d_lab->d_densityINLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    new_dw->get(uVelocity, d_lab->d_uVelocityINLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    new_dw->get(vVelocity, d_lab->d_vVelocityINLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    new_dw->get(wVelocity, d_lab->d_wVelocityINLabel, matlIndex, patch, Ghost::None,
		nofGhostCells);
    for (int ii = 0; ii < d_nofScalars; ii++) {
      new_dw->get(scalar[ii], d_lab->d_scalarINLabel, matlIndex, patch, Ghost::None,
		  nofGhostCells);
    }

    // Get the low and high index for the patch and the variables
    IntVector domLo = density.getFortLowIndex();
    IntVector domHi = density.getFortHighIndex();
    IntVector domLoCT = cellType.getFortLowIndex();
    IntVector domHiCT = cellType.getFortHighIndex();
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
    IntVector domLoU = uVelocity.getFortLowIndex();
    IntVector domHiU = uVelocity.getFortHighIndex();
    IntVector idxLoU = patch->getSFCXFORTLowIndex();
    IntVector idxHiU = patch->getSFCXFORTHighIndex();
    IntVector domLoV = vVelocity.getFortLowIndex();
    IntVector domHiV = vVelocity.getFortHighIndex();
    IntVector idxLoV = patch->getSFCYFORTLowIndex();
    IntVector idxHiV = patch->getSFCYFORTHighIndex();
    IntVector domLoW = wVelocity.getFortLowIndex();
    IntVector domHiW = wVelocity.getFortHighIndex();
    IntVector idxLoW = patch->getSFCZFORTLowIndex();
    IntVector idxHiW = patch->getSFCZFORTHighIndex();
    int xminus = patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1;
    int xplus =  patch->getBCType(Patch::xplus) == Patch::Neighbor?0:1;
    int yminus = patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1;
    int yplus =  patch->getBCType(Patch::yplus) == Patch::Neighbor?0:1;
    int zminus = patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1;
    int zplus =  patch->getBCType(Patch::zplus) == Patch::Neighbor?0:1;
    
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
      FORT_PROFV(domLoU.get_pointer(), domHiU.get_pointer(),
		 idxLoU.get_pointer(), idxHiU.get_pointer(),
		 uVelocity.getPointer(), 
		 domLoV.get_pointer(), domHiV.get_pointer(),
		 idxLoV.get_pointer(), idxHiV.get_pointer(),
		 vVelocity.getPointer(),
		 domLoW.get_pointer(), domHiW.get_pointer(),
		 idxLoW.get_pointer(), idxHiW.get_pointer(),
		 wVelocity.getPointer(),
		 domLoCT.get_pointer(), domHiCT.get_pointer(),
		 idxLo.get_pointer(), idxHi.get_pointer(),
		 cellType.getPointer(), 
		 &area, &fi.d_cellTypeID, &fi.flowRate, 
		 &fi.calcStream.d_density, &xminus, &xplus, &yminus, &yplus,
		 &zminus, &zplus);

      FORT_PROFSCALAR(domLo.get_pointer(), domHi.get_pointer(),
		      domLoCT.get_pointer(), domHiCT.get_pointer(),
		      idxLo.get_pointer(), idxHi.get_pointer(),
		      density.getPointer(), 
		      cellType.getPointer(),
		      &fi.calcStream.d_density, &fi.d_cellTypeID, &xminus, &xplus, &yminus, &yplus,
		      &zminus, &zplus);
    }   
    if (d_pressureBdry) {
      // set density
      FORT_PROFSCALAR(domLo.get_pointer(), domHi.get_pointer(), 
		      domLoCT.get_pointer(), domHiCT.get_pointer(), 
		      idxLo.get_pointer(), idxHi.get_pointer(),
		      density.getPointer(), cellType.getPointer(),
		      &d_pressureBdry->calcStream.d_density, 
		      &d_pressureBdry->d_cellTypeID, 
		      &xminus, &xplus, &yminus, &yplus,
		      &zminus, &zplus);
    }    
    for (int indx = 0; indx < d_nofScalars; indx++) {
      for (int ii = 0; ii < d_numInlets; ii++) {
	double scalarValue = d_flowInlets[ii].streamMixturefraction.d_mixVars[indx];
	FORT_PROFSCALAR(domLo.get_pointer(), domHi.get_pointer(), 
			domLoCT.get_pointer(), domHiCT.get_pointer(), 
			idxLo.get_pointer(), idxHi.get_pointer(),
			scalar[indx].getPointer(), cellType.getPointer(),
			&scalarValue, &d_flowInlets[ii].d_cellTypeID,
			&xminus, &xplus, &yminus, &yplus,
			&zminus, &zplus);
      }
      if (d_pressBoundary) {
	double scalarValue = d_pressureBdry->streamMixturefraction.d_mixVars[indx];
	FORT_PROFSCALAR(domLo.get_pointer(), domHi.get_pointer(),
			domLoCT.get_pointer(), domHiCT.get_pointer(),
			idxLo.get_pointer(), idxHi.get_pointer(),
			scalar[indx].getPointer(), cellType.getPointer(),
			&scalarValue, &d_pressureBdry->d_cellTypeID, 
			&xminus, &xplus, &yminus, &yplus,
			&zminus, &zplus);
      }
    }
    
    // Put the calculated data into the new DW
    new_dw->put(density, d_lab->d_densitySPLabel, matlIndex, patch);
    new_dw->put(uVelocity, d_lab->d_uVelocitySPLabel, matlIndex, patch);
    new_dw->put(vVelocity, d_lab->d_vVelocitySPLabel, matlIndex, patch);
    new_dw->put(wVelocity, d_lab->d_wVelocitySPLabel, matlIndex, patch);
    for (int ii =0; ii < d_nofScalars; ii++) {
      new_dw->put(scalar[ii], d_lab->d_scalarSPLabel, matlIndex, patch);
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
				   CellInformation* cellinfo,			  
				   ArchesVariables* vars)
{
  // Get the patch and variable indices
  IntVector domLoU, domHiU;
  IntVector domLoUng, domHiUng;
  IntVector idxLoU, idxHiU;
  IntVector domLo = vars->cellType.getFortLowIndex();
  IntVector domHi = vars->cellType.getFortHighIndex();
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
	    vars->pressGradUSu[currCell] *= vars->voidFraction[currCell];
	  }
	}
      }
    }
    FORT_ADDPRESSUREGRAD(domLoU.get_pointer(), domHiU.get_pointer(),
			 domLoUng.get_pointer(), domHiUng.get_pointer(),
			 idxLoU.get_pointer(), idxHiU.get_pointer(),
			 vars->pressGradUSu.getPointer(),
			 vars->uVelNonlinearSrc.getPointer(), 
			 domLo.get_pointer(), domHi.get_pointer(),
			 vars->cellType.getPointer(), &d_mmWallID,
			 &ioff, &joff, &koff);
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
	    vars->pressGradVSu[currCell] *= vars->voidFraction[currCell];
	  }
	}
      }
    }
    FORT_ADDPRESSUREGRAD(domLoU.get_pointer(), domHiU.get_pointer(),
			 domLoUng.get_pointer(), domHiUng.get_pointer(),
			 idxLoU.get_pointer(), idxHiU.get_pointer(),
			 vars->pressGradVSu.getPointer(),
			 vars->vVelNonlinearSrc.getPointer(), 
			 domLo.get_pointer(), domHi.get_pointer(),
			 vars->cellType.getPointer(), &d_mmWallID,
			 &ioff, &joff, &koff);
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
	    vars->pressGradWSu[currCell] *= vars->voidFraction[currCell];
	  }
	}
      }
    }
    FORT_ADDPRESSUREGRAD(domLoU.get_pointer(), domHiU.get_pointer(),
			 domLoUng.get_pointer(), domHiUng.get_pointer(),
			 idxLoU.get_pointer(), idxHiU.get_pointer(),
			 vars->pressGradWSu.getPointer(),
			 vars->wVelNonlinearSrc.getPointer(), 
			 domLo.get_pointer(), domHi.get_pointer(),
			 vars->cellType.getPointer(), &d_mmWallID,
			 &ioff, &joff, &koff);
    break;
  default:
    throw InvalidValue("Invalid index in BoundaryCondition::addPressGrad");
  }
}

      // compute multimaterial wall bc
void 
BoundaryCondition::mmvelocityBC(const ProcessorGroup*,
				const Patch* patch,
				int index, CellInformation* cellinfo,
				ArchesVariables* vars) 
{
    // Call the fortran routines
  switch(index) {
  case 1:
    mmuVelocityBC(patch,
		  vars);
    break;
  case 2:
    mmvVelocityBC(patch,
		  vars);
    break;
  case 3:
    mmwVelocityBC(patch,
		  vars);
    break;
  default:
    cerr << "Invalid Index value" << endl;
    break;

  // add kumar's mmvelbc
  }
}

void 
BoundaryCondition::mmuVelocityBC(const Patch* patch,
				 ArchesVariables* vars)
{
  // Get the low and high index for the patch and the variables
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  //***warning need two layers of ghost cells
  IntVector domLo = vars->cellType.getFortLowIndex();
  IntVector domHi = vars->cellType.getFortHighIndex();
  // for no ghost cells
  IntVector domLoUng = vars->uVelLinearSrc.getFortLowIndex();
  IntVector domHiUng = vars->uVelLinearSrc.getFortHighIndex();
  
  int ioff = 1;
  int joff = 0;
  int koff = 0;

  FORT_MMBCVELOCITY(domLoUng.get_pointer(), domHiUng.get_pointer(), 
		    idxLoU.get_pointer(), idxHiU.get_pointer(),
		    vars->uVelocityCoeff[Arches::AE].getPointer(), 
		    vars->uVelocityCoeff[Arches::AW].getPointer(), 
		    vars->uVelocityCoeff[Arches::AN].getPointer(), 
		    vars->uVelocityCoeff[Arches::AS].getPointer(), 
		    vars->uVelocityCoeff[Arches::AT].getPointer(), 
		    vars->uVelocityCoeff[Arches::AB].getPointer(), 
		    vars->uVelNonlinearSrc.getPointer(), 
		    vars->uVelLinearSrc.getPointer(),
		    domLo.get_pointer(), domHi.get_pointer(),
		    vars->cellType.getPointer(),
		    &d_mmWallID,
		    &ioff, &joff, &koff);

}

void 
BoundaryCondition::mmvVelocityBC(const Patch* patch,
				 ArchesVariables* vars)
{
  // Get the low and high index for the patch and the variables
  IntVector idxLoU = patch->getSFCYFORTLowIndex();
  IntVector idxHiU = patch->getSFCYFORTHighIndex();
  //***warning need two layers of ghost cells
  IntVector domLo = vars->cellType.getFortLowIndex();
  IntVector domHi = vars->cellType.getFortHighIndex();
  // for no ghost cells
  IntVector domLoUng = vars->vVelLinearSrc.getFortLowIndex();
  IntVector domHiUng = vars->vVelLinearSrc.getFortHighIndex();
  
  int ioff = 0;
  int joff = 1;
  int koff = 0;

  FORT_MMBCVELOCITY(domLoUng.get_pointer(), domHiUng.get_pointer(), 
		    idxLoU.get_pointer(), idxHiU.get_pointer(),
		    vars->vVelocityCoeff[Arches::AN].getPointer(), 
		    vars->vVelocityCoeff[Arches::AS].getPointer(), 
		    vars->vVelocityCoeff[Arches::AT].getPointer(), 
		    vars->vVelocityCoeff[Arches::AB].getPointer(), 
		    vars->vVelocityCoeff[Arches::AE].getPointer(), 
		    vars->vVelocityCoeff[Arches::AW].getPointer(), 
		    vars->vVelNonlinearSrc.getPointer(), 
		    vars->vVelLinearSrc.getPointer(),
		    domLo.get_pointer(), domHi.get_pointer(),
		    vars->cellType.getPointer(),
		    &d_mmWallID,
		    &ioff, &joff, &koff);

}

void 
BoundaryCondition::mmwVelocityBC( const Patch* patch,
				  ArchesVariables* vars) {
  // Get the low and high index for the patch and the variables
  IntVector idxLoU = patch->getSFCZFORTLowIndex();
  IntVector idxHiU = patch->getSFCZFORTHighIndex();
  //***warning need two layers of ghost cells
  IntVector domLo = vars->cellType.getFortLowIndex();
  IntVector domHi = vars->cellType.getFortHighIndex();
  // for no ghost cells
  IntVector domLoUng = vars->wVelLinearSrc.getFortLowIndex();
  IntVector domHiUng = vars->wVelLinearSrc.getFortHighIndex();
  
  int ioff = 0;
  int joff = 0;
  int koff = 1;

  FORT_MMBCVELOCITY(domLoUng.get_pointer(), domHiUng.get_pointer(), 
		    idxLoU.get_pointer(), idxHiU.get_pointer(),
		    vars->wVelocityCoeff[Arches::AT].getPointer(), 
		    vars->wVelocityCoeff[Arches::AB].getPointer(), 
		    vars->wVelocityCoeff[Arches::AE].getPointer(), 
		    vars->wVelocityCoeff[Arches::AW].getPointer(), 
		    vars->wVelocityCoeff[Arches::AN].getPointer(), 
		    vars->wVelocityCoeff[Arches::AS].getPointer(), 
		    vars->wVelNonlinearSrc.getPointer(), 
		    vars->wVelLinearSrc.getPointer(),
		    domLo.get_pointer(), domHi.get_pointer(),
		    vars->cellType.getPointer(),
		    &d_mmWallID,
		    &ioff, &joff, &koff);

}

void 
BoundaryCondition::mmpressureBC(const ProcessorGroup*,
				const Patch* patch,
				CellInformation* cellinfo,
				ArchesVariables* vars) {
  // Get the low and high index for the patch
  IntVector domLo = vars->cellType.getFortLowIndex();
  IntVector domHi = vars->cellType.getFortHighIndex();
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
  FORT_MMWALLBC(domLo.get_pointer(), domHi.get_pointer(),
		domLong.get_pointer(), domHing.get_pointer(),
		idxLo.get_pointer(), idxHi.get_pointer(),
		vars->pressCoeff[Arches::AE].getPointer(),
		vars->pressCoeff[Arches::AW].getPointer(),
		vars->pressCoeff[Arches::AN].getPointer(),
		vars->pressCoeff[Arches::AS].getPointer(),
		vars->pressCoeff[Arches::AT].getPointer(),
		vars->pressCoeff[Arches::AB].getPointer(),
		vars->pressNonlinearSrc.getPointer(),
		vars->pressLinearSrc.getPointer(),
		vars->cellType.getPointer(),
		&d_mmWallID);
}
// applies multimaterial bc's for scalars and pressure
void
BoundaryCondition::mmscalarWallBC( const ProcessorGroup*,
				   const Patch* patch,
				   CellInformation* cellinfo,
				   ArchesVariables* vars) {
  // Get the low and high index for the patch
  IntVector domLo = vars->cellType.getFortLowIndex();
  IntVector domHi = vars->cellType.getFortHighIndex();
  IntVector domLong = vars->scalarCoeff[Arches::AB].getFortLowIndex();
  IntVector domHing = vars->scalarCoeff[Arches::AB].getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  //fortran call
  FORT_MMWALLBC(domLo.get_pointer(), domHi.get_pointer(),
		domLong.get_pointer(), domHing.get_pointer(),
		idxLo.get_pointer(), idxHi.get_pointer(),
		vars->scalarCoeff[Arches::AE].getPointer(),
		vars->scalarCoeff[Arches::AW].getPointer(),
		vars->scalarCoeff[Arches::AN].getPointer(),
		vars->scalarCoeff[Arches::AS].getPointer(),
		vars->scalarCoeff[Arches::AT].getPointer(),
		vars->scalarCoeff[Arches::AB].getPointer(),
		vars->scalarNonlinearSrc.getPointer(),
		vars->scalarLinearSrc.getPointer(),
		vars->cellType.getPointer(),
		&d_mmWallID);

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
  // loop thru all the wall bdry geometry objects
  //for (ProblemSpecP geom_obj_ps = params->findBlock("geom_object");
    // geom_obj_ps != 0; 
    // geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
    //vector<GeometryPiece> pieces;
    //GeometryPieceFactory::create(geom_obj_ps, pieces);
    //if(pieces.size() == 0){
    //  throw ParameterNotFound("No piece specified in geom_object");
    //} else if(pieces.size() > 1){
    //  d_geomPiece = scinew UnionGeometryPiece(pieces);
    //} else {
    //  d_geomPiece = pieces[0];
    //}
  //}
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
  d_area_label = scinew VarLabel("flowarea"+cellID,
   ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
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

