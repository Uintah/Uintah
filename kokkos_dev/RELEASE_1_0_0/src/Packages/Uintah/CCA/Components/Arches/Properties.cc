//----- Properties.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Properties.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ColdflowMixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/PDFMixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/InletStream.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Core/Util/NotFinished.h>

using namespace Uintah;

//****************************************************************************
// Default constructor for Properties
//****************************************************************************
Properties::Properties(const ArchesLabel* label, const MPMArchesLabel* MAlb):
  d_lab(label), d_MAlab(MAlb)
{
  d_bc = 0;
}

//****************************************************************************
// Destructor
//****************************************************************************
Properties::~Properties()
{
}

//****************************************************************************
// Problem Setup for Properties
//****************************************************************************
void 
Properties::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("Properties");
  db->require("denUnderrelax", d_denUnderrelax);
  db->require("ref_point", d_denRef);

  // read type of mixing model
  string mixModel;
  db->require("mixing_model",mixModel);
  if (mixModel == "coldFlowMixingModel")
    d_mixingModel = scinew ColdflowMixingModel();
  else if (mixModel == "pdfMixingModel")
    d_mixingModel = scinew PDFMixingModel();
  else
    throw InvalidValue("Mixing Model not supported" + mixModel);
  d_mixingModel->problemSetup(db);
  // Read the mixing variable streams, total is noofStreams 0 
  d_numMixingVars = d_mixingModel->getNumMixVars();
}

//****************************************************************************
// compute density for inlet streams: only for cold streams
//****************************************************************************

void
Properties::computeInletProperties(const InletStream& inStream, 
				   Stream& outStream)
{
  d_mixingModel->computeProps(inStream, outStream);
}
  
//****************************************************************************
// Schedule the computation of properties
//****************************************************************************
void 
Properties::sched_computeProps(SchedulerP& sched, const PatchSet* patches,
			       const MaterialSet* matls)
{
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("Properties::ComputeProps",
			      patch, old_dw, new_dw, this,
			      &Properties::computeProps);

      int numGhostCells = 0;
      int matlIndex = 0;
      // requires scalars
      tsk->requires(old_dw, d_lab->d_densitySPLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      for (int ii = 0; ii < d_numMixingVars; ii++) 
	tsk->requires(old_dw, d_lab->d_scalarSPLabel, ii, patch, Ghost::None,
			 numGhostCells);
      tsk->computes(old_dw, d_lab->d_refDensity_label);
      tsk->computes(new_dw, d_lab->d_densityCPLabel, matlIndex, patch);
      sched->addTask(tsk);
    }
  }
#else
  NOT_FINISHED("new task stuff");
#endif
}

//****************************************************************************
// Schedule the recomputation of properties
//****************************************************************************
void 
Properties::sched_reComputeProps(SchedulerP& sched, const PatchSet* patches,
				 const MaterialSet* matls)
{
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("Properties::ReComputeProps",
			      patch, old_dw, new_dw, this,
			      &Properties::reComputeProps);

      int numGhostCells = 0;
      int matlIndex = 0;
      // requires scalars
      tsk->requires(new_dw, d_lab->d_densityINLabel, matlIndex, patch, 
		    Ghost::AroundCells,
		    numGhostCells+2);
      if (d_MAlab)
	tsk->requires(new_dw, d_lab->d_mmgasVolFracLabel, matlIndex, patch,
		      Ghost::None, numGhostCells);
      for (int ii = 0; ii < d_numMixingVars; ii++) 
	tsk->requires(new_dw, d_lab->d_scalarSPLabel, ii, patch, Ghost::None,
			 numGhostCells);
      tsk->computes(new_dw, d_lab->d_refDensity_label);
      tsk->computes(new_dw, d_lab->d_densityCPLabel, matlIndex, patch);
      sched->addTask(tsk);
    }
  }
#else
  NOT_FINISHED("new task stuff");
#endif
}

//****************************************************************************
// Actually compute the properties here
//****************************************************************************
void 
Properties::computeProps(const ProcessorGroup*,
			 const Patch* patch,
			 DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw)
{
  // Get the cellType and density from the old datawarehouse
  int matlIndex = 0;
  int nofGhostCells = 0;
  CCVariable<int> cellType;
  old_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, 
	      Ghost::None, nofGhostCells);
  CCVariable<double> density;
  std::vector<CCVariable<double> > scalar(d_numMixingVars);
  old_dw->get(density, d_lab->d_densitySPLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  //cerr << "number of mixing vars" << d_numMixingVars << endl;
  for (int ii = 0; ii < d_numMixingVars; ii++)
    old_dw->get(scalar[ii], d_lab->d_scalarSPLabel, ii, patch, Ghost::None,
		nofGhostCells);
  // get multimaterial vars
#ifdef multimaterialform
  MultiMaterialVars* mmVars = 0;
  if (d_mmInterface) 
    mmVars = d_mmInterface->getMMVars();
#endif

  //CCVariable<double> new_density;
  //new_dw->allocate(new_density, d_densityCPLabel, matlIndex, patch);
  IntVector indexLow = patch->getCellLowIndex();
  IntVector indexHigh = patch->getCellHighIndex();
  // set density for the whole domain
  for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
    for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
      for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {

	// Store current cell
	IntVector currCell(colX, colY, colZ);

	// for combustion calculations mixingmodel will be called
	// this is similar to prcf.f
	// construct an InletStream for input to the computeProps of mixingModel
	InletStream inStream(d_numMixingVars, d_mixingModel->getNumMixStatVars(),
			     d_mixingModel->getNumRxnVars());
	for (int ii = 0; ii < d_numMixingVars; ii++ ) {
	  inStream.d_mixVars[ii] = (scalar[ii])[currCell];
	}
	// after computing variance get that too, for the time being setting the 
	// value to zero
	inStream.d_mixVarVariance[0] = 0.0;
	// currently not using any reaction progress variables
	if (!d_mixingModel->isAdiabatic())
	  // get absolute enthalpy from enthalpy eqn
	  cerr << "No eqn for enthalpy yet" << '/n';
	Stream outStream;
	d_mixingModel->computeProps(inStream, outStream);
	double local_den = outStream.getDensity();
#ifdef multimaterialform
	if (d_mmInterface) 
	  local_den *= mmVars->voidFraction[currCell];
#endif
	if (d_bc == 0)
	  throw InvalidValue("BoundaryCondition pointer not assigned");
	if (cellType[currCell] != d_bc->wallCellType()) 
	  density[currCell] = d_denUnderrelax*local_den +
	                      (1.0-d_denUnderrelax)*density[currCell];
      }
    }
  }
#ifdef ARCHES_DEBUG
  // Testing if correct values have been put
  cerr << " AFTER COMPUTE PROPERTIES " << endl;
  IntVector domLo = density.getFortLowIndex();
  IntVector domHi = density.getFortHighIndex();
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
#endif
  if (patch->containsCell(d_denRef)) {
    double den_ref = density[d_denRef];
#ifdef ARCHES_DEBUG
    cerr << "density_ref " << den_ref << endl;
#endif
    old_dw->put(sum_vartype(den_ref),d_lab->d_refDensity_label);
  }
  else
    old_dw->put(sum_vartype(0), d_lab->d_refDensity_label);

  // Write the computed density to the new data warehouse
  new_dw->put(density,d_lab->d_densityCPLabel, matlIndex, patch);
}

//****************************************************************************
// Actually recompute the properties here
//****************************************************************************
void 
Properties::reComputeProps(const ProcessorGroup*,
			   const Patch* patch,
			   DataWarehouseP& /* old_dw*/,
			   DataWarehouseP& new_dw)
{
  // Get the CCVariable (density) from the old datawarehouse
  // just write one function for computing properties
  CCVariable<double> density;
  CCVariable<double> voidFraction;
  std::vector<CCVariable<double> > scalar(d_numMixingVars);
  int matlIndex = 0;
  int nofGhostCells = 0;
  new_dw->get(density, d_lab->d_densityINLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);
  if (d_MAlab)
    new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, matlIndex, patch, 
		Ghost::None, nofGhostCells);
  for (int ii = 0; ii < d_numMixingVars; ii++)
    new_dw->get(scalar[ii], d_lab->d_scalarSPLabel, ii, patch, Ghost::None,
		nofGhostCells);

#ifdef multimaterialform
  MultiMaterialVars* mmVars = 0;
  if (d_mmInterface) 
    mmVars = d_mmInterface->getMMVars();
#endif
  //CCVariable<double> new_density;
  //new_dw->allocate(new_density, d_densityCPLabel, matlIndex, patch);
  IntVector indexLow = patch->getCellLowIndex();
  IntVector indexHigh = patch->getCellHighIndex();
  // set density for the whole domain
  for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
    for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
      for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {
	// for combustion calculations mixingmodel will be called
	// this is similar to prcf.f
	// construct an InletStream for input to the computeProps of mixingModel
	IntVector currCell(colX, colY, colZ);
	InletStream inStream(d_numMixingVars, d_mixingModel->getNumMixStatVars(),
			     d_mixingModel->getNumRxnVars());
	for (int ii = 0; ii < d_numMixingVars; ii++ ) {
	  inStream.d_mixVars[ii] = (scalar[ii])[currCell];
	}
	// after computing variance get that too, for the time being setting the 
	// value to zero
	inStream.d_mixVarVariance[0] = 0.0;
	// currently not using any reaction progress variables
	if (!d_mixingModel->isAdiabatic())
	  // get absolute enthalpy from enthalpy eqn
	  cerr << "No eqn for enthalpy yet" << '/n';
	Stream outStream;
	d_mixingModel->computeProps(inStream, outStream);
	double local_den = outStream.getDensity();
#ifdef multimaterialform
	if (d_mmInterface) 
	  local_den *= mmVars->voidFraction[currCell];
#endif
	if (d_MAlab)
	  local_den *= voidFraction[currCell];
	
	density[IntVector(colX, colY, colZ)] = d_denUnderrelax*local_den +
                 	  (1.0-d_denUnderrelax)*density[IntVector(colX, colY, colZ)];
      }
    }
  }
  // Write the computed density to the new data warehouse
#ifdef ARCHES_PRES_DEBUG
  // Testing if correct values have been put
  cerr << " AFTER COMPUTE PROPERTIES " << endl;
  IntVector domLo = density.getFortLowIndex();
  IntVector domHi = density.getFortHighIndex();
  density.print(cerr);
#endif
  if (patch->containsCell(d_denRef)) {
    double den_ref = density[d_denRef];
    cerr << "density_ref " << den_ref << endl;
    new_dw->put(sum_vartype(den_ref),d_lab->d_refDensity_label);
  }
  else
    new_dw->put(sum_vartype(0), d_lab->d_refDensity_label);

  new_dw->put(density, d_lab->d_densityCPLabel, matlIndex, patch);
}



