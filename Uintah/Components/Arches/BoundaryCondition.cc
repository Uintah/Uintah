//----- BoundaryCondition.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/CellInformation.h>
#include <Uintah/Components/Arches/CellInformationP.h>
#include <Uintah/Components/Arches/StencilMatrix.h>
#include <Uintah/Components/Arches/ArchesFort.h>
#include <Uintah/Components/Arches/Discretization.h>
#include <Uintah/Grid/Stencil.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <Uintah/Components/Arches/Properties.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/SFCXVariable.h>
#include <Uintah/Grid/SFCYVariable.h>
#include <Uintah/Grid/SFCZVariable.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/PerPatch.h>
#include <Uintah/Grid/CellIterator.h>
#include <Uintah/Grid/SoleVariable.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/Reductions.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/GeometryPieceFactory.h>
#include <Uintah/Grid/UnionGeometryPiece.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/IntVector.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Exceptions/ParameterNotFound.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Grid/TypeUtils.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
using namespace std;
using namespace Uintah;
using namespace Uintah::ArchesSpace;
using SCICore::Geometry::Vector;

//****************************************************************************
// Constructor for BoundaryCondition
//****************************************************************************
BoundaryCondition::BoundaryCondition(const ArchesLabel* label,
				     TurbulenceModel* turb_model,
				     Properties* props):
                                     d_lab(label), 
				     d_turbModel(turb_model), 
				     d_props(props)
{
  d_nofScalars = d_props->getNumMixVars();
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
    d_flowInlets[d_numInlets].density = 
      d_props->computeInletProperties(d_flowInlets[d_numInlets].streamMixturefraction);
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
    d_pressureBdry->density = 
      d_props->computeInletProperties(d_pressureBdry->streamMixturefraction);
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
    d_outletBC->density = 
      d_props->computeInletProperties(d_outletBC->streamMixturefraction);
    d_cellTypes.push_back(total_cellTypes);
    ++total_cellTypes;
  }
  else {
    d_outletBoundary = false;
  }

}

//****************************************************************************
// schedule the initialization of cell types
//****************************************************************************
void 
BoundaryCondition::sched_cellTypeInit(const LevelP& level,
				      SchedulerP& sched,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw)
{
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
}

//****************************************************************************
// Actual initialization of celltype
//****************************************************************************
void 
BoundaryCondition::cellTypeInit(const ProcessorGroup*,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP&)
{
  CCVariable<int> cellType;
  int matlIndex = 0;

  old_dw->allocate(cellType, d_lab->d_cellTypeLabel, matlIndex, patch);

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

  old_dw->put(cellType, d_lab->d_cellTypeLabel, matlIndex, patch);
}  
    
//****************************************************************************
// Actual initialization of celltype
//****************************************************************************
void 
BoundaryCondition::computeInletFlowArea(const ProcessorGroup*,
					const Patch* patch,
					DataWarehouseP& old_dw,
					DataWarehouseP&)
{
  // Create the cellType variable
  CCVariable<int> cellType;

  // Get the cell type data from the old_dw
  // **WARNING** numGhostcells, Ghost::NONE may change in the future
  int matlIndex = 0;
  int numGhostCells = 0;
  old_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch,
	      Ghost::None, numGhostCells);

  // Get the PerPatch CellInformation data
  PerPatch<CellInformationP> cellInfoP;
  if (old_dw->exists(d_lab->d_cellInfoLabel, patch)) 
    old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
  else {
    cellInfoP.setData(scinew CellInformation(patch));
    old_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
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
	 old_dw->put(sum_vartype(0),d_flowInlets[ii].d_area_label);
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
      old_dw->put(sum_vartype(inlet_area),d_flowInlets[ii].d_area_label);
    }
  }
}
    
//****************************************************************************
// Schedule the computation of the presures bcs
//****************************************************************************
void 
BoundaryCondition::sched_computePressureBC(const LevelP& level,
				    SchedulerP& sched,
				    DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw)
{
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
}

//****************************************************************************
// Actually calculate the pressure BCs
//****************************************************************************
void 
BoundaryCondition::calcPressureBC(const ProcessorGroup* ,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw) 
{
  CCVariable<int> cellType;
  CCVariable<double> density;
  CCVariable<double> pressure;
  SFCXVariable<double> uVelocity;
  SFCYVariable<double> vVelocity;
  SFCZVariable<double> wVelocity;
  int matlIndex = 0;
  int nofGhostCells = 0;

  // get cellType, pressure and velocity
  old_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(density, d_lab->d_densityCPLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(pressure, d_lab->d_pressureINLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(uVelocity, d_lab->d_uVelocitySPLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(vVelocity, d_lab->d_vVelocitySPLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(wVelocity, d_lab->d_wVelocitySPLabel, matlIndex, patch, Ghost::None,
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
  for (int ii = domLoScalar.x(); ii <= domHiScalar.x(); ii++) {
    cerr << "Pressure for ii = " << ii << endl;
    for (int jj = domLoScalar.y(); jj <= domHiScalar.y(); jj++) {
      for (int kk = domLoScalar.z(); kk <= domHiScalar.z(); kk++) {
	cerr.width(10);
	cerr << pressure[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << " After CALPBC : " << endl;
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
  cerr << " After CALPBC : " << endl;
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
  cerr << " After CALPBC : " << endl;
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

//****************************************************************************
// Schedule the setting of inlet velocity BC
//****************************************************************************
void 
BoundaryCondition::sched_setInletVelocityBC(const LevelP& level,
					    SchedulerP& sched,
					    DataWarehouseP& old_dw,
					    DataWarehouseP& new_dw)
{
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
}

//****************************************************************************
// Schedule the compute of Pressure BC
//****************************************************************************
void 
BoundaryCondition::sched_recomputePressureBC(const LevelP& level,
					   SchedulerP& sched,
					   DataWarehouseP& old_dw,
					   DataWarehouseP& new_dw)
{
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
      tsk->requires(new_dw, d_lab->d_densityINLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_lab->d_pressurePSLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_lab->d_uVelocitySIVBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells+1);
      tsk->requires(new_dw, d_lab->d_vVelocitySIVBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells+1);
      tsk->requires(new_dw, d_lab->d_wVelocitySIVBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells+1);
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
}

//****************************************************************************
// Schedule computes inlet areas
// computes inlet area for inlet bc
//****************************************************************************
void 
BoundaryCondition::sched_calculateArea(const LevelP& level,
				       SchedulerP& sched,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw)
{
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
      for (int ii = 0; ii < d_numInlets; ii++) {
	// make it simple by adding matlindex for reduction vars
	tsk->computes(old_dw, d_flowInlets[ii].d_area_label);
      }
      sched->addTask(tsk);
    }
  }
}

//****************************************************************************
// Schedule set profile
// assigns flat velocity profiles for primary and secondary inlets
// Also sets flat profiles for density
//****************************************************************************
void 
BoundaryCondition::sched_setProfile(const LevelP& level,
				    SchedulerP& sched,
				    DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw)
{
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
}

//****************************************************************************
// Actually calculate the velocity BC
//****************************************************************************
void 
BoundaryCondition::velocityBC(const ProcessorGroup* pc,
			      const Patch* patch,
			      DataWarehouseP& old_dw,
			      DataWarehouseP& new_dw,
			      int index, int eqnType,
			      CellInformation* cellinfo,
			      ArchesVariables* vars) 
{
  //get Molecular Viscosity of the fluid
  double molViscosity = d_turbModel->getMolecularViscosity();

  // Call the fortran routines
  switch(index) {
  case 1:
    uVelocityBC(new_dw, patch,
		&molViscosity, cellinfo, vars);
    break;
  case 2:
    vVelocityBC(new_dw, patch,
		&molViscosity, cellinfo, vars);
    break;
  case 3:
    wVelocityBC(new_dw, patch,
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
BoundaryCondition::uVelocityBC(DataWarehouseP& new_dw,
			       const Patch* patch,
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
  for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
    cerr << "U velocity for ii = " << ii << endl;
    for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
      for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	cerr.width(10);
	cerr << vars->uVelocity[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER UVELBC_FORT" << endl;
  for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
    cerr << "AP for U velocity for ii = " << ii << endl;
    for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
      for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	cerr.width(10);
	cerr << (vars->uVelocityCoeff[Arches::AP])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER UVELBC_FORT" << endl;
  for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
    cerr << "AE for U velocity for ii = " << ii << endl;
    for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
      for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	cerr.width(10);
	cerr << (vars->uVelocityCoeff[Arches::AE])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER UVELBC_FORT" << endl;
  for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
    cerr << "AW for U velocity for ii = " << ii << endl;
    for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
      for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	cerr.width(10);
	cerr << (vars->uVelocityCoeff[Arches::AW])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER UVELBC_FORT" << endl;
  for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
    cerr << "AN for U velocity for ii = " << ii << endl;
    for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
      for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	cerr.width(10);
	cerr << (vars->uVelocityCoeff[Arches::AN])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER UVELBC_FORT" << endl;
  for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
    cerr << "AS for U velocity for ii = " << ii << endl;
    for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
      for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	cerr.width(10);
	cerr << (vars->uVelocityCoeff[Arches::AS])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER UVELBC_FORT" << endl;
  for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
    cerr << "AT for U velocity for ii = " << ii << endl;
    for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
      for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	cerr.width(10);
	cerr << (vars->uVelocityCoeff[Arches::AT])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER UVELBC_FORT" << endl;
  for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
    cerr << "AB for U velocity for ii = " << ii << endl;
    for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
      for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	cerr.width(10);
	cerr << (vars->uVelocityCoeff[Arches::AB])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER UVELBC_FORT" << endl;
  for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
    cerr << "SU for U velocity for ii = " << ii << endl;
    for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
      for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	cerr.width(10);
	cerr << vars->uVelNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER UVELBC_FORT" << endl;
  for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
    cerr << "SP for U velocity for ii = " << ii << endl;
    for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
      for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	cerr.width(10);
	cerr << vars->uVelLinearSrc[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
#endif

}

//****************************************************************************
// call fortran routine to calculate the V Velocity BC
//****************************************************************************
void 
BoundaryCondition::vVelocityBC(DataWarehouseP& new_dw,
			       const Patch* patch,
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
  for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
    cerr << "V velocity for ii = " << ii << endl;
    for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
      for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	cerr.width(10);
	cerr << vars->vVelocity[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER VVELBC_FORT" << endl;
  for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
    cerr << "AP for V velocity for ii = " << ii << endl;
    for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
      for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	cerr.width(10);
	cerr << (vars->vVelocityCoeff[Arches::AP])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER VVELBC_FORT" << endl;
  for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
    cerr << "AE for V velocity for ii = " << ii << endl;
    for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
      for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	cerr.width(10);
	cerr << (vars->vVelocityCoeff[Arches::AE])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER VVELBC_FORT" << endl;
  for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
    cerr << "AW for V velocity for ii = " << ii << endl;
    for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
      for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	cerr.width(10);
	cerr << (vars->vVelocityCoeff[Arches::AW])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER VVELBC_FORT" << endl;
  for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
    cerr << "AN for V velocity for ii = " << ii << endl;
    for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
      for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	cerr.width(10);
	cerr << (vars->vVelocityCoeff[Arches::AN])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER VVELBC_FORT" << endl;
  for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
    cerr << "AS for V velocity for ii = " << ii << endl;
    for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
      for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	cerr.width(10);
	cerr << (vars->vVelocityCoeff[Arches::AS])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER VVELBC_FORT" << endl;
  for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
    cerr << "AT for V velocity for ii = " << ii << endl;
    for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
      for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	cerr.width(10);
	cerr << (vars->vVelocityCoeff[Arches::AT])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER VVELBC_FORT" << endl;
  for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
    cerr << "AB for V velocity for ii = " << ii << endl;
    for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
      for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	cerr.width(10);
	cerr << (vars->vVelocityCoeff[Arches::AB])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER VVELBC_FORT" << endl;
  for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
    cerr << "SU for V velocity for ii = " << ii << endl;
    for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
      for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	cerr.width(10);
	cerr << vars->vVelNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER VVELBC_FORT" << endl;
  for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
    cerr << "SP for V velocity for ii = " << ii << endl;
    for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
      for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	cerr.width(10);
	cerr << vars->vVelLinearSrc[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
#endif

}

//****************************************************************************
// call fortran routine to calculate the W Velocity BC
//****************************************************************************
void 
BoundaryCondition::wVelocityBC(DataWarehouseP& new_dw,
			       const Patch* patch,
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
  for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
    cerr << "W velocity for ii = " << ii << endl;
    for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
      for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	cerr.width(10);
	cerr << vars->wVelocity[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER WVELBC_FORT" << endl;
  for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
    cerr << "AP for W velocity for ii = " << ii << endl;
    for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
      for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	cerr.width(10);
	cerr << (vars->wVelocityCoeff[Arches::AP])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER WVELBC_FORT" << endl;
  for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
    cerr << "AE for W velocity for ii = " << ii << endl;
    for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
      for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	cerr.width(10);
	cerr << (vars->wVelocityCoeff[Arches::AE])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER WVELBC_FORT" << endl;
  for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
    cerr << "AW for W velocity for ii = " << ii << endl;
    for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
      for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	cerr.width(10);
	cerr << (vars->wVelocityCoeff[Arches::AW])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER WVELBC_FORT" << endl;
  for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
    cerr << "AN for W velocity for ii = " << ii << endl;
    for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
      for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	cerr.width(10);
	cerr << (vars->wVelocityCoeff[Arches::AN])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER WVELBC_FORT" << endl;
  for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
    cerr << "AS for W velocity for ii = " << ii << endl;
    for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
      for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	cerr.width(10);
	cerr << (vars->wVelocityCoeff[Arches::AS])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER WVELBC_FORT" << endl;
  for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
    cerr << "AT for W velocity for ii = " << ii << endl;
    for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
      for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	cerr.width(10);
	cerr << (vars->wVelocityCoeff[Arches::AT])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER WVELBC_FORT" << endl;
  for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
    cerr << "AB for W velocity for ii = " << ii << endl;
    for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
      for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	cerr.width(10);
	cerr << (vars->wVelocityCoeff[Arches::AB])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER WVELBC_FORT" << endl;
  for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
    cerr << "SU for W velocity for ii = " << ii << endl;
    for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
      for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	cerr.width(10);
	cerr << vars->wVelNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER WVELBC_FORT" << endl;
  for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
    cerr << "SP for W velocity for ii = " << ii << endl;
    for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
      for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	cerr.width(10);
	cerr << vars->wVelLinearSrc[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
#endif

}

//****************************************************************************
// Actually compute the pressure bcs
//****************************************************************************
void 
BoundaryCondition::pressureBC(const ProcessorGroup*,
			      const Patch* patch,
			      DataWarehouseP& /*old_dw*/,
			      DataWarehouseP& /*new_dw*/,
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
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "Pressure for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << vars->pressure[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER FORT_PRESSBC" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "AE for Pressure for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (vars->pressCoeff[Arches::AE])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER FORT_PRESSBC" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "AW for Pressure for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (vars->pressCoeff[Arches::AW])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER FORT_PRESSBC" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "AN for Pressure for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (vars->pressCoeff[Arches::AN])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER FORT_PRESSBC" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "AS for Pressure for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (vars->pressCoeff[Arches::AS])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER FORT_PRESSBC" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "AT for Pressure for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (vars->pressCoeff[Arches::AT])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER FORT_PRESSBC" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "AB for Pressure for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (vars->pressCoeff[Arches::AB])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER FORT_PRESSBC" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "SU for Pressure for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << vars->pressNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER FORT_PRESSBC" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "SP for Pressure for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << vars->pressLinearSrc[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
#endif
}

//****************************************************************************
// Actually compute the scalar bcs
//****************************************************************************
void 
BoundaryCondition::scalarBC(const ProcessorGroup*,
			    const Patch* patch,
			    DataWarehouseP& /*old_dw*/,
			    DataWarehouseP& /*new_dw*/,
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
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "Scalar " << index << " for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << vars->scalar[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER FORT_SCALARBC" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "AE for Scalar " << index << " for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (vars->scalarCoeff[Arches::AE])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER FORT_SCALARBC" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "AW for Scalar " << index << " for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (vars->scalarCoeff[Arches::AW])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER FORT_SCALARBC" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "AN for Scalar " << index << " for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (vars->scalarCoeff[Arches::AN])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER FORT_SCALARBC" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "AS for Scalar " << index << " for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (vars->scalarCoeff[Arches::AS])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER FORT_SCALARBC" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "AT for Scalar " << index << " for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (vars->scalarCoeff[Arches::AT])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER FORT_SCALARBC" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "AB for Scalar " << index << " for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (vars->scalarCoeff[Arches::AB])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER FORT_SCALARBC" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "SU for Scalar " << index << " for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << vars->scalarNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "AFTER FORT_SCALARBC" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "SP for Scalar " << index << " for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << vars->scalarLinearSrc[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
#endif
}


//****************************************************************************
// Actually set the inlet velocity BC
//****************************************************************************
void 
BoundaryCondition::setInletVelocityBC(const ProcessorGroup* ,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw) 
{
  CCVariable<int> cellType;
  CCVariable<double> density;
  SFCXVariable<double> uVelocity;
  SFCYVariable<double> vVelocity;
  SFCZVariable<double> wVelocity;

  int matlIndex = 0;
  int nofGhostCells = 0;

  // get cellType, velocity and density
  old_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
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
				    const Patch* patch,
				    DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw) 
{
  CCVariable<int> cellType;
  CCVariable<double> density;
  CCVariable<double> pressure;
  vector<CCVariable<double> > scalar(d_nofScalars);
  SFCXVariable<double> uVelocity;
  SFCYVariable<double> vVelocity;
  SFCZVariable<double> wVelocity;
  int matlIndex = 0;
  int nofGhostCells = 0;

  // get cellType, pressure and velocity
  old_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
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
    new_dw->get(scalar[ii], d_lab->d_scalarINLabel, ii, patch, Ghost::None,
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
		    &(d_pressureBdry->streamMixturefraction[ii]),
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
    new_dw->put(scalar[ii], d_lab->d_scalarCPBCLabel, ii, patch);
} 

//****************************************************************************
// Actually set flat profile at flow inlet boundary
//****************************************************************************
void 
BoundaryCondition::setFlatProfile(const ProcessorGroup* /*pc*/,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw)
{
  CCVariable<int> cellType;
  CCVariable<double> density;
  SFCXVariable<double> uVelocity;
  SFCYVariable<double> vVelocity;
  SFCZVariable<double> wVelocity;
  vector<CCVariable<double> > scalar(d_nofScalars);
  int matlIndex = 0;
  int nofGhostCells = 0;

  // get cellType, density and velocity
  old_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(density, d_lab->d_densityINLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(uVelocity, d_lab->d_uVelocityINLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(vVelocity, d_lab->d_vVelocityINLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(wVelocity, d_lab->d_wVelocityINLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  for (int ii = 0; ii < d_nofScalars; ii++) {
    old_dw->get(scalar[ii], d_lab->d_scalarINLabel, ii, patch, Ghost::None,
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
    old_dw->get(area_var, d_flowInlets[indx].d_area_label);
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
	       &fi.density, &xminus, &xplus, &yminus, &yplus,
	       &zminus, &zplus);
    FORT_PROFSCALAR(domLo.get_pointer(), domHi.get_pointer(),
		    domLoCT.get_pointer(), domHiCT.get_pointer(),
		    idxLo.get_pointer(), idxHi.get_pointer(),
		    density.getPointer(), 
		    cellType.getPointer(),
		    &fi.density, &fi.d_cellTypeID, &xminus, &xplus, &yminus, &yplus,
		    &zminus, &zplus);
  }   
  if (d_pressureBdry) {
    // set density
    FORT_PROFSCALAR(domLo.get_pointer(), domHi.get_pointer(), 
		    domLoCT.get_pointer(), domHiCT.get_pointer(), 
		    idxLo.get_pointer(), idxHi.get_pointer(),
		    density.getPointer(), cellType.getPointer(),
		    &d_pressureBdry->density, &d_pressureBdry->d_cellTypeID, 
		    &xminus, &xplus, &yminus, &yplus,
		    &zminus, &zplus);
  }    
  for (int indx = 0; indx < d_nofScalars; indx++) {
    for (int ii = 0; ii < d_numInlets; ii++) {
      double scalarValue = d_flowInlets[ii].streamMixturefraction[indx];
      FORT_PROFSCALAR(domLo.get_pointer(), domHi.get_pointer(), 
		      domLoCT.get_pointer(), domHiCT.get_pointer(), 
		      idxLo.get_pointer(), idxHi.get_pointer(),
		      scalar[indx].getPointer(), cellType.getPointer(),
		      &scalarValue, &d_flowInlets[ii].d_cellTypeID,
		      &xminus, &xplus, &yminus, &yplus,
		      &zminus, &zplus);
    }
    if (d_pressBoundary) {
      double scalarValue = d_pressureBdry->streamMixturefraction[indx];
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
    new_dw->put(scalar[ii], d_lab->d_scalarSPLabel, ii, patch);
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
  density = 0.0;
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
    streamMixturefraction.push_back(mixfrac);
  }
 
}


//****************************************************************************
// constructor for BoundaryCondition::PressureInlet
//****************************************************************************
BoundaryCondition::PressureInlet::PressureInlet(int /*numMix*/, int cellID):
  d_cellTypeID(cellID)
{
  //  streamMixturefraction.setsize(numMix-1);
  density = 0.0;
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
    streamMixturefraction.push_back(mixfrac);
  }
}

//****************************************************************************
// constructor for BoundaryCondition::FlowOutlet
//****************************************************************************
BoundaryCondition::FlowOutlet::FlowOutlet(int /*numMix*/, int cellID):
  d_cellTypeID(cellID)
{
  //  streamMixturefraction.setsize(numMix-1);
  density = 0.0;
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
    streamMixturefraction.push_back(mixfrac);
  }
}

//
// $Log$
// Revision 1.72  2000/12/10 09:06:00  sparker
// Merge from csafe_risky1
//
// Revision 1.71  2000/11/21 23:54:35  guilkey
// Removed references to MPM namespace which existed only because of
// the use of GeometryPiece.
//
// Revision 1.59.4.2  2000/10/20 04:41:57  sparker
// Temporarily require 2 ghost cells for all *VelocitySIVBCLabel's, due
//  to limitation in risky scheduler.  Put this back when the scheduler
//  gets fixed ASV.
//
// Revision 1.59.4.1  2000/10/19 05:17:27  sparker
// Merge changes from main branch into csafe_risky1
//
// Revision 1.70  2000/10/14 17:11:05  sparker
// Changed PerPatch<CellInformation*> to PerPatch<CellInformationP>
// to get rid of memory leak
//
// Revision 1.69  2000/10/13 19:48:02  sparker
// Commenting out chatter
//
// Revision 1.68  2000/10/12 20:08:32  sparker
// Made multipatch work for several timesteps
// Cleaned up print statements
//
// Revision 1.67  2000/10/11 21:14:03  rawat
// fixed a bug in scalar solver
//
// Revision 1.66  2000/10/11 17:40:28  sparker
// Added rewindow hack to trim ghost cells from variables
// fixed compiler warnings
//
// Revision 1.65  2000/10/11 16:37:29  rawat
// modified calpbc for ghost cells
//
// Revision 1.63  2000/10/07 21:40:49  rawat
// fixed pressure norm
//
// Revision 1.62  2000/10/07 05:37:49  sparker
// Fixed warnings under g++
//
// Revision 1.61  2000/10/06 23:07:47  rawat
// fixed some more bc routines for mulit-patch
//
// Revision 1.60  2000/10/05 16:39:46  rawat
// modified bcs for multi-patch
//
// Revision 1.59  2000/09/26 19:59:17  sparker
// Work on MPI petsc
//
// Revision 1.58  2000/09/26 04:35:27  rawat
// added some more multi-patch support
//
// Revision 1.57  2000/09/21 22:45:41  sparker
// Towards compiling petsc stuff
//
// Revision 1.56  2000/09/07 23:07:17  rawat
// fixed some bugs in bc and added pressure solver using petsc
//
// Revision 1.55  2000/08/23 06:20:51  bbanerje
// 1) Results now correct for pressure solve.
// 2) Modified BCU, BCV, BCW to add stuff for pressure BC.
// 3) Removed some bugs in BCU, V, W.
// 4) Coefficients for MOM Solve not computed correctly yet.
//
// Revision 1.54  2000/08/19 16:36:35  rawat
// fixed some bugs in scalarcoef calculations
//
// Revision 1.53  2000/08/19 05:53:43  bbanerje
// Changed code so that output looks more like fortran output.
//
// Revision 1.52  2000/08/10 00:56:33  rawat
// added pressure bc for scalar and changed discretization option for velocity
//
// Revision 1.51  2000/08/09 03:17:56  jas
// Changed new to scinew and added deletes to some of the destructors.
//
// Revision 1.50  2000/08/04 03:02:01  bbanerje
// Add some inits.
//
// Revision 1.49  2000/08/04 02:14:32  bbanerje
// Added debug statements.
//
// Revision 1.48  2000/07/30 22:21:21  bbanerje
// Added bcscalar.F (originally bcf.f in Kumar's code) needs more work
// in C++ side.
//
// Revision 1.47  2000/07/28 02:30:59  rawat
// moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
// coeffecients
//
// Revision 1.44  2000/07/17 22:06:58  rawat
// modified momentum source
//
// Revision 1.43  2000/07/14 03:45:45  rawat
// completed velocity bc and fixed some bugs
//
// Revision 1.42  2000/07/13 06:32:09  bbanerje
// Labels are once more consistent for one iteration.
//
// Revision 1.41  2000/07/13 04:51:32  bbanerje
// Added pressureBC (bcp) .. now called bcpress.F (bcp.F removed)
//
// Revision 1.40  2000/07/12 23:59:21  rawat
// added wall bc for u-velocity
//
// Revision 1.39  2000/07/11 15:46:27  rawat
// added setInitialGuess in PicardNonlinearSolver and also added uVelSrc
//
// Revision 1.38  2000/07/08 23:42:53  bbanerje
// Moved all enums to Arches.h and made corresponding changes.
//
// Revision 1.37  2000/07/08 08:03:33  bbanerje
// Readjusted the labels upto uvelcoef, removed bugs in CellInformation,
// made needed changes to uvelcoef.  Changed from StencilMatrix::AE etc
// to Arches::AE .. doesn't like enums in templates apparently.
//
// Revision 1.36  2000/07/07 23:07:44  rawat
// added inlet bc's
//
// Revision 1.35  2000/07/03 05:30:13  bbanerje
// Minor changes for inlbcs dummy code to compile and work. densitySIVBC is no more.
//
// Revision 1.34  2000/07/02 05:47:29  bbanerje
// Uncommented all PerPatch and CellInformation stuff.
// Updated array sizes in inlbcs.F
//
// Revision 1.33  2000/06/30 22:41:18  bbanerje
// Corrected behavior of profv and profscalar
//
// Revision 1.32  2000/06/30 06:29:42  bbanerje
// Got Inlet Area to be calculated correctly .. but now two CellInformation
// variables are being created (Rawat ... check that).
//
// Revision 1.31  2000/06/30 04:19:16  rawat
// added turbulence model and compute properties
//
// Revision 1.30  2000/06/29 06:22:47  bbanerje
// Updated FCVariable to SFCX, SFCY, SFCZVariables and made corresponding
// changes to profv.  Code is broken until the changes are reflected
// thru all the files.
//
// Revision 1.29  2000/06/28 08:14:52  bbanerje
// Changed the init routines a bit.
//
// Revision 1.28  2000/06/22 23:06:33  bbanerje
// Changed velocity related variables to FCVariable type.
// ** NOTE ** We may need 3 types of FCVariables (one for each direction)
//
// Revision 1.27  2000/06/21 07:50:59  bbanerje
// Corrected new_dw, old_dw problems, commented out intermediate dw (for now)
// and made the stuff go through schedule_time_advance.
//
// Revision 1.26  2000/06/21 06:49:21  bbanerje
// Straightened out some of the problems in data location .. still lots to go.
//
// Revision 1.25  2000/06/20 20:42:36  rawat
// added some more boundary stuff and modified interface to IntVector. Before
// compiling the code you need to update /SCICore/Geometry/IntVector.h
//
// Revision 1.24  2000/06/19 18:00:29  rawat
// added function to compute velocity and density profiles and inlet bc.
// Fixed bugs in CellInformation.cc
//
// Revision 1.23  2000/06/18 01:20:14  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
// Revision 1.22  2000/06/17 07:06:22  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.21  2000/06/16 21:50:47  bbanerje
// Changed the Varlabels so that sequence in understood in init stage.
// First cycle detected in task graph.
//
// Revision 1.20  2000/06/16 07:06:16  bbanerje
// Added init of props, pressure bcs and turbulence model in Arches.cc
// Changed duplicate task names (setProfile) in BoundaryCondition.cc
// Commented out nolinear_dw creation in PicardNonlinearSolver.cc
//
// Revision 1.19  2000/06/15 23:47:56  rawat
// modified Archesfort to fix function call
//
// Revision 1.18  2000/06/15 22:13:22  rawat
// modified boundary stuff
//
// Revision 1.17  2000/06/15 08:48:12  bbanerje
// Removed most commented stuff , added StencilMatrix, tasks etc.  May need some
// more work
//
//
