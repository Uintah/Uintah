//----- BoundaryCondition.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/CellInformation.h>
#include <Uintah/Components/Arches/StencilMatrix.h>
#include <Uintah/Components/Arches/ArchesFort.h>
#include <Uintah/Components/Arches/Discretization.h>
#include <Uintah/Grid/Stencil.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <Uintah/Components/Arches/Properties.h>
#include <Uintah/Components/Arches/CellInformation.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/FCVariable.h>
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
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Exceptions/ParameterNotFound.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Grid/TypeUtils.h>
#include <iostream>
using namespace std;
using namespace Uintah::MPM;
using namespace Uintah::ArchesSpace;
using SCICore::Geometry::Vector;

//****************************************************************************
// Default constructor for BoundaryCondition
//****************************************************************************
BoundaryCondition::BoundaryCondition()
{
}

//****************************************************************************
// Actual constructor for BoundaryCondition
//****************************************************************************
BoundaryCondition::BoundaryCondition(TurbulenceModel* turb_model,
				     Properties* props) :
                                     d_turbModel(turb_model), d_props(props)
{
  //** WARNING ** Velocity is a FC Variable : change all velocity related stuff
  // to FCVariable type and delete this comment.

  // The input labels first
  d_cellTypeLabel = scinew VarLabel("CellType", 
				    CCVariable<int>::getTypeDescription() );
  d_pressureLabel = scinew VarLabel("pressure", 
				   CCVariable<double>::getTypeDescription() );
  d_densityLabel = scinew VarLabel("density", 
				   CCVariable<double>::getTypeDescription() );
  d_viscosityLabel = scinew VarLabel("viscosity", 
				   CCVariable<double>::getTypeDescription() );
  d_uVelocityLabel = scinew VarLabel("uVelocity", 
				    CCVariable<double>::getTypeDescription() );
  d_vVelocityLabel = scinew VarLabel("vVelocity", 
				    CCVariable<double>::getTypeDescription() );
  d_wVelocityLabel = scinew VarLabel("wVelocity", 
				    CCVariable<double>::getTypeDescription() );

  // The internal labels for computations
  // 1) The labels computed by setProfile (SP)
  d_densitySPLabel = scinew VarLabel("densitySP", 
				   CCVariable<double>::getTypeDescription() );
  d_uVelocitySPLabel = scinew VarLabel("uVelocitySP", 
				    CCVariable<double>::getTypeDescription() );
  d_vVelocitySPLabel = scinew VarLabel("vVelocitySP", 
				    CCVariable<double>::getTypeDescription() );
  d_wVelocitySPLabel = scinew VarLabel("wVelocitySP", 
				    CCVariable<double>::getTypeDescription() );



  d_uVelCoefLabel = scinew VarLabel("uVelCoef", 
				    CCVariable<double>::getTypeDescription() );
  d_vVelCoefLabel = scinew VarLabel("vVelCoef", 
				    CCVariable<double>::getTypeDescription() );
  d_wVelCoefLabel = scinew VarLabel("wVelCoef", 
				    CCVariable<double>::getTypeDescription() );
  //** WARNING ** Velocity is a FC Variable
  d_uVelLinSrcLabel = scinew VarLabel("uVelLinSrc", 
				    CCVariable<double>::getTypeDescription() );
  //** WARNING ** Velocity is a FC Variable
  d_vVelLinSrcLabel = scinew VarLabel("vVelLinSrc", 
				    CCVariable<double>::getTypeDescription() );
  //** WARNING ** Velocity is a FC Variable
  d_wVelLinSrcLabel = scinew VarLabel("wVelLinSrc", 
				    CCVariable<double>::getTypeDescription() );
  //** WARNING ** Velocity is a FC Variable
  d_uVelNonLinSrcLabel = scinew VarLabel("uVelNonLinSrc", 
				    CCVariable<double>::getTypeDescription() );
  //** WARNING ** Velocity is a FC Variable
  d_vVelNonLinSrcLabel = scinew VarLabel("vVelNonLinSrc", 
				    CCVariable<double>::getTypeDescription() );
  //** WARNING ** Velocity is a FC Variable
  d_wVelNonLinSrcLabel = scinew VarLabel("wVelNonLinSrc", 
				    CCVariable<double>::getTypeDescription() );
  d_presCoefLabel = scinew VarLabel("presCoef", 
				    CCVariable<double>::getTypeDescription() );
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
    d_cellTypes.push_back(total_cellTypes);
    ++total_cellTypes;
  }
  else {
    d_outletBoundary = false;
  }

}

//****************************************************************************
// Actual initialization of celltype
//****************************************************************************
void 
BoundaryCondition::cellTypeInit(const ProcessorContext*,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP&)
{
  CCVariable<int> cellType;
  int matlIndex = 0;

  old_dw->allocate(cellType, d_cellTypeLabel, matlIndex, patch);
  // initialize CCVariable to -1 which corresponds to flowfield
  // ** WARNING **  this needs to be changed soon (6/9/2000)
  // IntVector domainLow = patch->getCellLowIndex();
  // IntVector domainHigh = patch->getCellHighIndex();
  // IntVector indexLow = patch->getCellLowIndex();
  // IntVector indexHigh = patch->getCellHighIndex();
  int domainLow[3], domainHigh[3];
  int indexLow[3], indexHigh[3];
  domainLow[2] = (patch->getCellLowIndex()).x()+1;
  domainLow[1] = (patch->getCellLowIndex()).y()+1;
  domainLow[0] = (patch->getCellLowIndex()).z()+1;
  domainHigh[2] = (patch->getCellHighIndex()).x();
  domainHigh[1] = (patch->getCellHighIndex()).y();
  domainHigh[0] = (patch->getCellHighIndex()).z();
  for (int ii = 0; ii < 3; ii++) {
    indexLow[ii] = domainLow[ii]+1;
    indexHigh[ii] = domainHigh[ii]-1;
  }
 
  int celltypeval = -1;
  cerr << "Just before geom init" << endl;
  FORT_CELLTYPEINIT(domainLow, domainHigh, indexLow, indexHigh,
		    cellType.getPointer(), &celltypeval);
  // set boundary type for inlet flow field
  cerr << "Just after fortran call "<< endl;
  Box patchBox = patch->getBox();
  // wall boundary type
  GeometryPiece*  piece = d_wallBdry->d_geomPiece;
  Box geomBox = piece->getBoundingBox();
  Box b = geomBox.intersect(patchBox);
  cerr << "Just before geom wall "<< endl;
  // check for another geometry
  if (!(b.degenerate())) {
    CellIterator iter = patch->getCellIterator(b);
    domainLow[0] = (iter.begin()).x()+1;
    domainLow[1] = (iter.begin()).y()+1;
    domainLow[2] = (iter.begin()).z()+1;
    domainHigh[0] = (iter.end()).x();
    domainHigh[1] = (iter.end()).y();
    domainHigh[2] = (iter.end()).z();
    for (int indx = 0; indx < 3; indx++) {
      indexLow[indx] = domainLow[indx]+1;
      indexHigh[indx] = domainHigh[indx]-1;
    }
    celltypeval = d_wallBdry->d_cellTypeID;
    FORT_CELLTYPEINIT(domainLow, domainHigh, indexLow, indexHigh,
		      cellType.getPointer(), &celltypeval);
  }

  for (int ii = 0; ii < d_numInlets; ii++) {
    GeometryPiece*  piece = d_flowInlets[ii].d_geomPiece;
    Box geomBox = piece->getBoundingBox();
    Box b = geomBox.intersect(patchBox);
    // check for another geometry
    if (b.degenerate())
      continue; // continue the loop for other inlets
    // iterates thru box b, converts from geometry space to index space
    // make sure this works
    CellIterator iter = patch->getCellIterator(b);
    domainLow[0] = (iter.begin()).x()+1;
    domainLow[1] = (iter.begin()).y()+1;
    domainLow[2] = (iter.begin()).z()+1;
    domainHigh[0] = (iter.end()).x();
    domainHigh[1] = (iter.end()).y();
    domainHigh[2] = (iter.end()).z();
    for (int indx = 0; indx < 3; indx++) {
      indexLow[indx] = domainLow[indx]+1;
      indexHigh[indx] = domainHigh[indx]-1;
    }
    celltypeval = d_flowInlets[ii].d_cellTypeID;
    FORT_CELLTYPEINIT(domainLow, domainHigh, indexLow, indexHigh,
		      cellType.getPointer(), &celltypeval);
  }
  // initialization for pressure boundary
  if (d_pressBoundary) {
    GeometryPiece*  piece = d_pressureBdry->d_geomPiece;
    Box geomBox = piece->getBoundingBox();
    Box b = geomBox.intersect(patchBox);
    // check for another geometry
    if (!(b.degenerate())) {
      CellIterator iter = patch->getCellIterator(b);
      domainLow[0] = (iter.begin()).x()+1;
      domainLow[1] = (iter.begin()).y()+1;
      domainLow[2] = (iter.begin()).z()+1;
      domainHigh[0] = (iter.end()).x();
      domainHigh[1] = (iter.end()).y();
      domainHigh[2] = (iter.end()).z();
      for (int indx = 0; indx < 3; indx++) {
	indexLow[indx] = domainLow[indx]+1;
	indexHigh[indx] = domainHigh[indx]-1;
      }
      celltypeval = d_pressureBdry->d_cellTypeID;
      FORT_CELLTYPEINIT(domainLow, domainHigh, indexLow, indexHigh,
			cellType.getPointer(), &celltypeval);
    }
  }
  // initialization for outlet boundary
  if (d_outletBoundary) {
    GeometryPiece*  piece = d_outletBC->d_geomPiece;
    Box geomBox = piece->getBoundingBox();
    Box b = geomBox.intersect(patchBox);
    // check for another geometry
    if (!(b.degenerate())) {
      CellIterator iter = patch->getCellIterator(b);
      domainLow[0] = (iter.begin()).x()+1;
      domainLow[1] = (iter.begin()).y()+1;
      domainLow[2] = (iter.begin()).z()+1;
      domainHigh[0] = (iter.end()).x();
      domainHigh[1] = (iter.end()).y();
      domainHigh[2] = (iter.end()).z();
      for (int indx = 0; indx < 3; indx++) {
	indexLow[indx] = domainLow[indx]+1;
	indexHigh[indx] = domainHigh[indx]-1;
      }
      celltypeval = d_outletBC->d_cellTypeID;
      FORT_CELLTYPEINIT(domainLow, domainHigh, indexLow, indexHigh,
			cellType.getPointer(), &celltypeval);
    }
  }
  old_dw->put(cellType, d_cellTypeLabel, matlIndex, patch);
}  
    
//****************************************************************************
// Actual initialization of celltype
//****************************************************************************
void 
BoundaryCondition::computeInletFlowArea(const ProcessorContext*,
				 const Patch* patch,
				 DataWarehouseP& old_dw,
				 DataWarehouseP&)
{
  CCVariable<int> cellType;
  int matlIndex = 0;
  int numGhostCells = 0;
  old_dw->get(cellType, d_cellTypeLabel, matlIndex, patch,
	      Ghost::None, numGhostCells);
  // initialize CCVariable to -1 which corresponds to flowfield
  // ** WARNING **  this needs to be changed soon (6/9/2000)
  // IntVector domainLow = patch->getCellLowIndex();
  // IntVector domainHigh = patch->getCellHighIndex();
  // IntVector indexLow = patch->getCellLowIndex();
  // IntVector indexHigh = patch->getCellHighIndex();
  int domainLow[3], domainHigh[3];
  int indexLow[3], indexHigh[3];
  domainLow[2] = (patch->getCellLowIndex()).x()+1;
  domainLow[1] = (patch->getCellLowIndex()).y()+1;
  domainLow[0] = (patch->getCellLowIndex()).z()+1;
  domainHigh[2] = (patch->getCellHighIndex()).x();
  domainHigh[1] = (patch->getCellHighIndex()).y();
  domainHigh[0] = (patch->getCellHighIndex()).z();
  for (int ii = 0; ii < 3; ii++) {
    indexLow[ii] = domainLow[ii]+1;
    indexHigh[ii] = domainHigh[ii]-1;
  }
 
  Box patchBox = patch->getBox();
  for (int ii = 0; ii < d_numInlets; ii++) {
    GeometryPiece*  piece = d_flowInlets[ii].d_geomPiece;
    Box geomBox = piece->getBoundingBox();
    Box b = geomBox.intersect(patchBox);
    // check for another geometry
    if (b.degenerate())
      continue; // continue the loop for other inlets
    // iterates thru box b, converts from geometry space to index space
    // make sure this works
    CellIterator iter = patch->getCellIterator(b);
    domainLow[0] = (iter.begin()).x()+1;
    domainLow[1] = (iter.begin()).y()+1;
    domainLow[2] = (iter.begin()).z()+1;
    domainHigh[0] = (iter.end()).x();
    domainHigh[1] = (iter.end()).y();
    domainHigh[2] = (iter.end()).z();
    for (int indx = 0; indx < 3; indx++) {
      indexLow[indx] = domainLow[indx]+1;
      indexHigh[indx] = domainHigh[indx]-1;
    }
  
#ifdef WONT_COMPILE_YET
  // using chain of responsibility pattern for getting cell information
    DataWarehouseP top_dw = new_dw->getTop();
    PerPatch<CellInformation*> cellinfop;
    if(top_dw->exists("cellinfo", patch)){
      top_dw->get(cellinfop, "cellinfo", patch);
    } else {
      cellinfop.setData(scinew CellInformation(patch));
      top_dw->put(cellinfop, "cellinfo", patch);
    } 
    CellInformation* cellinfo = cellinfop;
#endif
  // ** WARNING ** this is just for compilation purposes
    CellInformation* cellinfo = scinew CellInformation(patch);
    double inlet_area;
    int cellid = d_flowInlets[ii].d_cellTypeID;
    FORT_AREAIN(domainLow, domainHigh, indexLow, indexHigh,
		cellinfo->sew.get_objs(),
		cellinfo->sns.get_objs(), cellinfo->stb.get_objs(),
		&inlet_area, cellType.getPointer(), &cellid);
    old_dw->put(sum_vartype(inlet_area),d_flowInlets[ii].d_area_label);
  }
}
    
//****************************************************************************
// Schedule the calculation of the velocity BCs
//****************************************************************************
void 
BoundaryCondition::sched_velocityBC(const LevelP& level,
				    SchedulerP& sched,
				    DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw,
				    int index)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("BoundaryCondition::VelocityBC",
			      patch, old_dw, new_dw, this,
			      &BoundaryCondition::velocityBC,
			      index);

      int numGhostCells = 0;
      int matlIndex = 0;

      // This task requires old velocity, density and viscosity
      // for all cases
      tsk->requires(old_dw, d_uVelocityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_vVelocityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_wVelocityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_densityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_viscosityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);

      // For the three cases u or v or w are required based on index
      switch(index) {
      case 1:
	tsk->requires(old_dw, d_uVelCoefLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
	tsk->requires(old_dw, d_uVelLinSrcLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
	tsk->requires(old_dw, d_uVelNonLinSrcLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
	break;
      case 2:
	tsk->requires(old_dw, d_vVelCoefLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
	tsk->requires(old_dw, d_vVelLinSrcLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
	tsk->requires(old_dw, d_vVelNonLinSrcLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
	break;
      case 3:
	tsk->requires(old_dw, d_wVelCoefLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
	tsk->requires(old_dw, d_wVelLinSrcLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
	tsk->requires(old_dw, d_wVelNonLinSrcLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
	break;
      default:
	throw InvalidValue("Invalid component for velocity" +index);
      }

      // This task computes new u, v, w coefs, lin & non lin src terms
      switch(index) {
      case 1:
	tsk->computes(new_dw, d_uVelCoefLabel, matlIndex, patch);
	tsk->computes(new_dw, d_uVelLinSrcLabel, matlIndex, patch);
	tsk->computes(new_dw, d_uVelNonLinSrcLabel, matlIndex, patch);
	break;
      case 2:
	tsk->computes(new_dw, d_vVelCoefLabel, matlIndex, patch);
	tsk->computes(new_dw, d_vVelLinSrcLabel, matlIndex, patch);
	tsk->computes(new_dw, d_vVelNonLinSrcLabel, matlIndex, patch);
	break;
      case 3:
	tsk->computes(new_dw, d_wVelCoefLabel, matlIndex, patch);
	tsk->computes(new_dw, d_wVelLinSrcLabel, matlIndex, patch);
	tsk->computes(new_dw, d_wVelNonLinSrcLabel, matlIndex, patch);
	break;
      default:
	throw InvalidValue("Invalid component for velocity" +index);
      }
      sched->addTask(tsk);
    }
  }
}

//****************************************************************************
// Schedule the computation of the presures bcs
//****************************************************************************
void 
BoundaryCondition::sched_pressureBC(const LevelP& level,
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
      Task* tsk = scinew Task("BoundaryCondition::PressureBC",patch,
			      old_dw, new_dw, this,
			      &BoundaryCondition::pressureBC);

      int numGhostCells = 0;
      int matlIndex = 0;

      // This task requires the pressure and the pressure stencil coeffs
      tsk->requires(old_dw, d_pressureLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_presCoefLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);

      // This task computes new uVelocity, vVelocity and wVelocity
      tsk->computes(new_dw, d_presCoefLabel, matlIndex, patch);

      sched->addTask(tsk);
    }
  }
}

//****************************************************************************
// Schedule the computation of scalar BCS
//****************************************************************************
void 
BoundaryCondition::sched_scalarBC(const LevelP& level,
				  SchedulerP& sched,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw,
				  int index)
{
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

      // This task requires old uVelocity, vVelocity and wVelocity
      tsk->requires(old_dw, d_uVelocityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_vVelocityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_wVelocityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);

      // This task computes new uVelocity, vVelocity and wVelocity
      tsk->computes(new_dw, d_uVelocityLabel, matlIndex, patch);
      tsk->computes(new_dw, d_vVelocityLabel, matlIndex, patch);
      tsk->computes(new_dw, d_wVelocityLabel, matlIndex, patch);

      sched->addTask(tsk);
    }
  }
}

//****************************************************************************
// Schedule the compute of Pressure BC
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
      Task* tsk = scinew Task("BoundaryCondition::calculatePressBC",
			      patch, old_dw, new_dw, this,
			      &BoundaryCondition::calculatePressBC);

      int numGhostCells = 0;
      int matlIndex = 0;

      // This task requires old density, pressure and velocity
      tsk->requires(old_dw, d_densityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_pressureLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_uVelocityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_vVelocityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_wVelocityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);

      // This task computes new uVelocity, vVelocity and wVelocity
      tsk->computes(new_dw, d_uVelocityLabel, matlIndex, patch);
      tsk->computes(new_dw, d_vVelocityLabel, matlIndex, patch);
      tsk->computes(new_dw, d_wVelocityLabel, matlIndex, patch);

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

      // This task requires old density, uVelocity, vVelocity and wVelocity
      tsk->requires(old_dw, d_densityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_uVelocityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_vVelocityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_wVelocityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);

      // This task computes new density, uVelocity, vVelocity and wVelocity
      tsk->computes(new_dw, d_densitySPLabel, matlIndex, patch);
      tsk->computes(new_dw, d_uVelocitySPLabel, matlIndex, patch);
      tsk->computes(new_dw, d_vVelocitySPLabel, matlIndex, patch);
      tsk->computes(new_dw, d_wVelocitySPLabel, matlIndex, patch);

      sched->addTask(tsk);
    }
  }
}

//****************************************************************************
// Actually calculate the velocity BC
//****************************************************************************
void 
BoundaryCondition::velocityBC(const ProcessorContext* pc,
			      const Patch* patch,
			      DataWarehouseP& old_dw,
			      DataWarehouseP& new_dw,
			      int index) 
{
  CCVariable<int> cellType;
  CCVariable<double> density;
  // ** WARNING** velocity is a FC variable
  CCVariable<double> uVelocity;
  CCVariable<double> vVelocity;
  CCVariable<double> wVelocity;

  int matlIndex = 0;
  int nofGhostCells = 0;

  // get cellType and velocity
  old_dw->get(cellType, d_cellTypeLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(density, d_densityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(uVelocity, d_uVelocityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(vVelocity, d_vVelocityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(wVelocity, d_wVelocityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);

#ifdef WONT_COMPILE_YET
  // using chain of responsibility pattern for getting cell information
  DataWarehouseP top_dw = new_dw->getTop();
  PerPatch<CellInformation*> cellinfop;
  if(top_dw->exists("cellinfo", patch)){
    top_dw->get(cellinfop, "cellinfo", patch);
  } else {
    cellinfop.setData(scinew CellInformation(patch));
    top_dw->put(cellinfop, "cellinfo", patch);
  } 
  CellInformation* cellinfo = cellinfop;
#endif
  // ** WARNING ** this is just for compilation purposes
  CellInformation* cellinfo = scinew CellInformation(patch);

  // Get the low and high index for the patch
  IntVector indexLow = patch->getCellLowIndex();
  IntVector indexHigh = patch->getCellHighIndex();

  // Put the calculated data into the new DW
  new_dw->put(density, d_densityLabel, matlIndex, patch);
  new_dw->put(uVelocity, d_uVelocityLabel, matlIndex, patch);
  new_dw->put(vVelocity, d_vVelocityLabel, matlIndex, patch);
  new_dw->put(wVelocity, d_wVelocityLabel, matlIndex, patch);

#ifdef WONT_COMPILE_YET
  //get Molecular Viscosity of the fluid
  SoleVariable<double> VISCOS_CONST;
  top_dw->get(VISCOS_CONST, "viscosity"); 
  double VISCOS = VISCOS_CONST;
#endif
  // ** WARNING ** temporarily assigning VISCOS
  double VISCOS = 1.0;

  // Call the fortran routines
  switch(index) {
  case 1:
    uVelocityBC(new_dw, patch, indexLow, indexHigh, &cellType, &uVelocity, 
		&vVelocity, &wVelocity, &density, &VISCOS,
		cellinfo);
    break;
  case 2:
    vVelocityBC(new_dw, patch, indexLow, indexHigh, &cellType, &uVelocity, 
		&vVelocity, &wVelocity, &density, &VISCOS,
		cellinfo);
    break;
  case 3:
    wVelocityBC(new_dw, patch, indexLow, indexHigh, &cellType, &uVelocity, 
		&vVelocity, &wVelocity, &density, &VISCOS,
		cellinfo);
    break;
  default:
    cerr << "Invalid Index value" << endl;
    break;
  }
  d_turbModel->calcVelocityWallBC(pc, patch, old_dw, new_dw, index);
}

//****************************************************************************
// call fortran routine to calculate the U Velocity BC
//****************************************************************************
void 
BoundaryCondition::uVelocityBC(DataWarehouseP& new_dw,
			       const Patch* patch,
			       const IntVector& indexLow, 
			       const IntVector& indexHigh,
			       CCVariable<int>* cellType,
			       CCVariable<double>* uVelocity, 
			       CCVariable<double>* vVelocity, 
			       CCVariable<double>* wVelocity, 
			       CCVariable<double>* density,
			       const double* VISCOS,
			       CellInformation*)
{
  int matlIndex = 0;
  int numGhostCells = 0;
  int nofStencils = 7;

  // ** WARNING ** velocity is a FCVariable
  StencilMatrix<CCVariable<double> > velocityCoeff;
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(velocityCoeff[ii], d_uVelCoefLabel, ii, patch, Ghost::None,
		numGhostCells);
  }

  // SP term in Arches (** WARNING ** should be FCvariable)
  CCVariable<double> linearSrc;
  new_dw->get(linearSrc, d_uVelLinSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  // SU term in Arches (** WARNING ** should be FCvariable)
  CCVariable<double> nonlinearSrc;
  new_dw->get(nonlinearSrc, d_uVelNonLinSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  int ioff = 1;
  int joff = 0;
  int koff = 0;

#ifdef WONT_COMPILE_YET
    // 3-d array for volume - fortran uses it for temporary storage
    Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
    // computes remianing diffusion term and also computes source due to gravity
    FORT_BCUVEL(velocityCoeff, linearSrc, nonlinearSrc, velocity,  
		density, VISCOS, ioff, joff, koff, lowIndex, highIndex,
		cellType,
		cellinfo->ceeu, cellinfo->cweu, cellinfo->cwwu,
		cellinfo->cnn, cellinfo->csn, cellinfo->css,
		cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
		cellinfo->sewu, cellinfo->sns, cellinfo->stb,
		cellinfo->dxepu, cellinfo->dynp, cellinfo->dztp,
		cellinfo->dxpw, cellinfo->fac1u, cellinfo->fac2u,
		cellinfo->fac3u, cellinfo->fac4u,cellinfo->iesdu,
		cellinfo->iwsdu, cellinfo->enfac, cellinfo->sfac,
		cellinfo->tfac, cellinfo->bfac, volume);
#endif

  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(velocityCoeff[ii], d_uVelCoefLabel, ii, patch);
  }
  new_dw->put(linearSrc, d_uVelLinSrcLabel, matlIndex, patch);
  new_dw->put(nonlinearSrc, d_uVelNonLinSrcLabel, matlIndex, patch);
}

//****************************************************************************
// call fortran routine to calculate the V Velocity BC
//****************************************************************************
void 
BoundaryCondition::vVelocityBC(DataWarehouseP& new_dw,
			       const Patch* patch,
			       const IntVector& indexLow, 
			       const IntVector& indexHigh,
			       CCVariable<int>* cellType,
			       CCVariable<double>* uVelocity, 
			       CCVariable<double>* vVelocity, 
			       CCVariable<double>* wVelocity, 
			       CCVariable<double>* density,
			       const double* VISCOS,
			       CellInformation*)
{
  int matlIndex = 0;
  int numGhostCells = 0;
  int nofStencils = 7;

  // ** WARNING ** velocity is a FCVariable
  StencilMatrix<CCVariable<double> > velocityCoeff;
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(velocityCoeff[ii], d_vVelCoefLabel, ii, patch, Ghost::None,
		numGhostCells);
  }

  // SP term in Arches (** WARNING ** should be FCvariable)
  CCVariable<double> linearSrc;
  new_dw->get(linearSrc, d_vVelLinSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  // SU term in Arches (** WARNING ** should be FCvariable)
  CCVariable<double> nonlinearSrc;
  new_dw->get(nonlinearSrc, d_vVelNonLinSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  int ioff = 1;
  int joff = 0;
  int koff = 0;

#ifdef WONT_COMPILE_YET
    // 3-d array for volume - fortran uses it for temporary storage
    Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
    // computes remianing diffusion term and also computes source due to gravity
    FORT_BCVVEL(velocityCoeff, linearSrc, nonlinearSrc, velocity,  
		density, VISCOS, ioff, joff, koff, lowIndex, highIndex,
		cellType,
		cellinfo->ceeu, cellinfo->cweu, cellinfo->cwwu,
		cellinfo->cnn, cellinfo->csn, cellinfo->css,
		cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
		cellinfo->sewu, cellinfo->sns, cellinfo->stb,
		cellinfo->dxepu, cellinfo->dynp, cellinfo->dztp,
		cellinfo->dxpw, cellinfo->fac1u, cellinfo->fac2u,
		cellinfo->fac3u, cellinfo->fac4u,cellinfo->iesdu,
		cellinfo->iwsdu, cellinfo->enfac, cellinfo->sfac,
		cellinfo->tfac, cellinfo->bfac, volume);
#endif

  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(velocityCoeff[ii], d_vVelCoefLabel, ii, patch);
  }
  new_dw->put(linearSrc, d_vVelLinSrcLabel, matlIndex, patch);
  new_dw->put(nonlinearSrc, d_vVelNonLinSrcLabel, matlIndex, patch);
}

//****************************************************************************
// call fortran routine to calculate the W Velocity BC
//****************************************************************************
void 
BoundaryCondition::wVelocityBC(DataWarehouseP& new_dw,
			       const Patch* patch,
			       const IntVector& indexLow, 
			       const IntVector& indexHigh,
			       CCVariable<int>* cellType,
			       CCVariable<double>* uVelocity, 
			       CCVariable<double>* vVelocity, 
			       CCVariable<double>* wVelocity, 
			       CCVariable<double>* density,
			       const double* VISCOS,
			       CellInformation*)
{
  int matlIndex = 0;
  int numGhostCells = 0;
  int nofStencils = 7;

  // ** WARNING ** velocity is a FCVariable
  StencilMatrix<CCVariable<double> > velocityCoeff;
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(velocityCoeff[ii], d_wVelCoefLabel, ii, patch, Ghost::None,
		numGhostCells);
  }

  // SP term in Arches (** WARNING ** should be FCvariable)
  CCVariable<double> linearSrc;
  new_dw->get(linearSrc, d_wVelLinSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  // SU term in Arches (** WARNING ** should be FCvariable)
  CCVariable<double> nonlinearSrc;
  new_dw->get(nonlinearSrc, d_wVelNonLinSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  int ioff = 1;
  int joff = 0;
  int koff = 0;

#ifdef WONT_COMPILE_YET
    // 3-d array for volume - fortran uses it for temporary storage
    Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
    // computes remianing diffusion term and also computes source due to gravity
    FORT_BCWVEL(velocityCoeff, linearSrc, nonlinearSrc, velocity,  
		density, VISCOS, ioff, joff, koff, lowIndex, highIndex,
		cellType,
		cellinfo->ceeu, cellinfo->cweu, cellinfo->cwwu,
		cellinfo->cnn, cellinfo->csn, cellinfo->css,
		cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
		cellinfo->sewu, cellinfo->sns, cellinfo->stb,
		cellinfo->dxepu, cellinfo->dynp, cellinfo->dztp,
		cellinfo->dxpw, cellinfo->fac1u, cellinfo->fac2u,
		cellinfo->fac3u, cellinfo->fac4u,cellinfo->iesdu,
		cellinfo->iwsdu, cellinfo->enfac, cellinfo->sfac,
		cellinfo->tfac, cellinfo->bfac, volume);
#endif

  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(velocityCoeff[ii], d_wVelCoefLabel, ii, patch);
  }
  new_dw->put(linearSrc, d_wVelLinSrcLabel, matlIndex, patch);
  new_dw->put(nonlinearSrc, d_wVelNonLinSrcLabel, matlIndex, patch);
}

//****************************************************************************
// Actually compute the pressure bcs
//****************************************************************************
void 
BoundaryCondition::pressureBC(const ProcessorContext*,
			      const Patch* patch,
			      DataWarehouseP& old_dw,
			      DataWarehouseP& new_dw)
{
  CCVariable<int> cellType;
  CCVariable<double> pressure;
  StencilMatrix<CCVariable<double> > presCoef;

  int matlIndex = 0;
  int nofGhostCells = 0;
  int nofStencils = 7;

  // get cellType, pressure and pressure stencil coeffs
  old_dw->get(cellType, d_cellTypeLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(pressure, d_pressureLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(presCoef[ii], d_presCoefLabel, ii, patch, Ghost::None,
		nofGhostCells);
  }

  // Get the low and high index for the patch
  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

#ifdef WONT_COMPILE_YET
  // using chain of responsibility pattern for getting cell information
  DataWarehouseP top_dw = new_dw->getTop();
  PerPatch<CellInformation*> cellinfop;
  if(top_dw->exists("cellinfo", patch)){
    top_dw->get(cellinfop, "cellinfo", patch);
  } else {
    cellinfop.setData(scinew CellInformation(patch));
    top_dw->put(cellinfop, "cellinfo", patch);
  } 
  CellInformation* cellinfo = cellinfop;

  //fortran call
  FORT_PRESSBC(pressCoeff, pressure, 
	       lowIndex, highIndex, cellType);
#endif

  // Put the calculated data into the new DW
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(presCoef[ii], d_presCoefLabel, ii, patch);
  }
}

//****************************************************************************
// Actually compute the scalar bcs
//****************************************************************************
void 
BoundaryCondition::scalarBC(const ProcessorContext*,
			    const Patch* patch,
			    DataWarehouseP& old_dw,
			    DataWarehouseP& new_dw,
			    int index)
{
}


//****************************************************************************
// Actually set the inlet velocity BC
//****************************************************************************
void 
BoundaryCondition::setInletVelocityBC(const ProcessorContext* pc,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw) 
{
  CCVariable<int> cellType;
  CCVariable<double> density;
  // ** WARNING** velocity is a FC variable
  CCVariable<double> uVelocity;
  CCVariable<double> vVelocity;
  CCVariable<double> wVelocity;

  int matlIndex = 0;
  int nofGhostCells = 0;

  // get cellType and velocity
  old_dw->get(cellType, d_cellTypeLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(uVelocity, d_uVelocityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(vVelocity, d_vVelocityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(wVelocity, d_wVelocityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);

  // Get the low and high index for the patch
  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

  // stores cell type info for the patch with the ghost cell type
  for (int indx = 0; indx < d_numInlets; indx++) {

    // Get a copy of the current flowinlet
    FlowInlet fi = d_flowInlets[indx];

#ifdef WONT_COMPILE_YET
    // assign flowType the value that corresponds to flow
    CellTypeInfo flowType = FLOW;
    FORT_INLBCS(domainLow, domianHigh, indexLow, indexHigh,
		uVelocity, vVelocity, wVelocity,
		density, 
		&fi.cellTypeID, &fi.flowRate, &fi.area, &fi.density, 
		&fi.inletType,
		flowType);
#endif

    // Put the calculated data into the new DW
    new_dw->put(density, d_densityLabel, matlIndex, patch);
    new_dw->put(uVelocity, d_uVelocityLabel, matlIndex, patch);
    new_dw->put(vVelocity, d_vVelocityLabel, matlIndex, patch);
    new_dw->put(wVelocity, d_wVelocityLabel, matlIndex, patch);
  }
}

//****************************************************************************
// Actually calculate the pressure BCs
//****************************************************************************
void 
BoundaryCondition::calculatePressBC(const ProcessorContext* pc,
				    const Patch* patch,
				    DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw) 
{
  CCVariable<int> cellType;
  CCVariable<double> pressure;
  // ** WARNING** velocity is a FC variable
  CCVariable<double> uVelocity;
  CCVariable<double> vVelocity;
  CCVariable<double> wVelocity;

  int matlIndex = 0;
  int nofGhostCells = 0;

  // get cellType, pressure and velocity
  old_dw->get(cellType, d_cellTypeLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(pressure, d_pressureLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(uVelocity, d_uVelocityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(vVelocity, d_vVelocityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(wVelocity, d_wVelocityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);

  // Get the low and high index for the patch
  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

#ifdef WONT_COMPILE_YET
  FORT_CALPBC(domainLow, domainHigh, indexLow, indexHigh,
	      uVelocity, vVelocity, wVelocity, 
	      pressure, density, 
	      &(d_pressureBdry->d_cellTypeID),
	      &(d_pressureBdry->refPressure), 
	      &(d_pressureBdry->area),
	      &(d_pressureBdry->density), 
	      &(d_pressureBdry->inletType));
#endif

  // Put the calculated data into the new DW
  new_dw->put(uVelocity, d_uVelocityLabel, matlIndex, patch);
  new_dw->put(vVelocity, d_vVelocityLabel, matlIndex, patch);
  new_dw->put(wVelocity, d_wVelocityLabel, matlIndex, patch);
} 

//****************************************************************************
// Actually set flat profile
//****************************************************************************
void 
BoundaryCondition::setFlatProfile(const ProcessorContext* pc,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw)
{
  CCVariable<int> cellType;
  CCVariable<double> density;
  // ** WARNING** velocity is a FC variable
  CCVariable<double> uVelocity;
  CCVariable<double> vVelocity;
  CCVariable<double> wVelocity;

  int matlIndex = 0;
  int nofGhostCells = 0;

  // get cellType, density and velocity
  old_dw->get(cellType, d_cellTypeLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(density, d_densityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(uVelocity, d_uVelocityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(vVelocity, d_vVelocityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  old_dw->get(wVelocity, d_wVelocityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);

  // Get the low and high index for the patch
  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

  // loop thru the flow inlets
  for (int indx = 0; indx < d_numInlets; indx++) {

    // Get a copy of the current flowinlet
    FlowInlet fi = d_flowInlets[indx];

#ifdef WONT_COMPILE_YET
    FORT_PROFV(domainLow, domainHigh, indexLow, indexHigh,
	       uVelocity, vVelocity, wVelocity, density, 
	       &fi.d_cellTypeID, &fi.flowRate, &fi.area, &fi.density, 
	       &fi.inletType);
#endif

    // Put the calculated data into the new DW
    new_dw->put(density, d_densitySPLabel, matlIndex, patch);
    new_dw->put(uVelocity, d_uVelocitySPLabel, matlIndex, patch);
    new_dw->put(vVelocity, d_vVelocitySPLabel, matlIndex, patch);
    new_dw->put(wVelocity, d_wVelocitySPLabel, matlIndex, patch);
  }
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
  // loop thru all the wall bdry geometry objects
  for (ProblemSpecP geom_obj_ps = params->findBlock("geom_object");
       geom_obj_ps != 0; 
       geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
    vector<GeometryPiece*> pieces;
    GeometryPieceFactory::create(geom_obj_ps, pieces);
    if(pieces.size() == 0){
      throw ParameterNotFound("No piece specified in geom_object");
    } else if(pieces.size() > 1){
      d_geomPiece = scinew UnionGeometryPiece(pieces);
    } else {
      d_geomPiece = pieces[0];
    }
  }
}




//****************************************************************************
// constructor for BoundaryCondition::FlowInlet
//****************************************************************************
BoundaryCondition::FlowInlet::FlowInlet(int numMix, int cellID):
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
  double mixfrac;
  // loop thru all the inlet geometry objects
  for (ProblemSpecP geom_obj_ps = params->findBlock("geom_object");
       geom_obj_ps != 0; 
       geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
    vector<GeometryPiece*> pieces;
    GeometryPieceFactory::create(geom_obj_ps, pieces);
    if(pieces.size() == 0){
      throw ParameterNotFound("No piece specified in geom_object");
    } else if(pieces.size() > 1){
      d_geomPiece = scinew UnionGeometryPiece(pieces);
    } else {
      d_geomPiece = pieces[0];
    }
  }

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
BoundaryCondition::PressureInlet::PressureInlet(int numMix, int cellID):
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
  // loop thru all the pressure inlet geometry objects
  for (ProblemSpecP geom_obj_ps = params->findBlock("geom_object");
       geom_obj_ps != 0; 
       geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
    vector<GeometryPiece*> pieces;
    GeometryPieceFactory::create(geom_obj_ps, pieces);
    if(pieces.size() == 0){
      throw ParameterNotFound("No piece specified in geom_object");
    } else if(pieces.size() > 1){
      d_geomPiece = scinew UnionGeometryPiece(pieces);
    } else {
      d_geomPiece = pieces[0];
    }
  }
  double mixfrac;
  for (ProblemSpecP mixfrac_db = params->findBlock("MixtureFraction");
       mixfrac_db != 0; 
       mixfrac_db = mixfrac_db->findNextBlock("MixtureFraction")) {
    mixfrac_db->require("Mixfrac", mixfrac);
    streamMixturefraction.push_back(mixfrac);
  }
  // check to see if this will work
  // params->require("Mixturefraction", streamMixturefraction[0]);
}

//****************************************************************************
// constructor for BoundaryCondition::FlowOutlet
//****************************************************************************
BoundaryCondition::FlowOutlet::FlowOutlet(int numMix, int cellID):
  d_cellTypeID(cellID)
{
  //  streamMixturefraction.setsize(numMix-1);
  density = 0.0;
  turb_lengthScale = 0.0;
}

//****************************************************************************
// Problem Setup for BoundaryCondition::FlowInlet
//****************************************************************************
void 
BoundaryCondition::FlowOutlet::problemSetup(ProblemSpecP& params)
{
  params->require("TurblengthScale", turb_lengthScale);
  // loop thru all the inlet geometry objects
  for (ProblemSpecP geom_obj_ps = params->findBlock("geom_object");
       geom_obj_ps != 0; 
       geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
    vector<GeometryPiece*> pieces;
    GeometryPieceFactory::create(geom_obj_ps, pieces);
    if(pieces.size() == 0){
      throw ParameterNotFound("No piece specified in geom_object");
    } else if(pieces.size() > 1){
      d_geomPiece = scinew UnionGeometryPiece(pieces);
    } else {
      d_geomPiece = pieces[0];
    }
  }
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
