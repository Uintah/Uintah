//----- Properties.cc --------------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/Properties.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Exceptions/InvalidValue.h>

using namespace Uintah::ArchesSpace;

//****************************************************************************
// Default constructor for Properties
//****************************************************************************
Properties::Properties()
{
  d_scalarSPLabel = scinew VarLabel("scalarSP", 
				    CCVariable<double>::getTypeDescription() );
  d_densitySPLabel = scinew VarLabel("densitySP",
				   CCVariable<double>::getTypeDescription() );
  d_densityCPLabel = scinew VarLabel("densityCP",
				   CCVariable<double>::getTypeDescription() );
  d_densityRCPLabel = scinew VarLabel("densityRCP",
				   CCVariable<double>::getTypeDescription() );
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

  // Read the mixing variable streams, total is noofStreams 0 
  d_numMixingVars = 0;
  for (ProblemSpecP stream_db = db->findBlock("Stream");
       stream_db != 0; stream_db = stream_db->findNextBlock("Stream")) {

    // Create the stream and add it to the vector
    d_streams.push_back(Stream());
    d_streams[d_numMixingVars].problemSetup(stream_db);
    ++d_numMixingVars;
  }
}

//****************************************************************************
// compute density for inlet streams: only for cold streams
//****************************************************************************

double 
Properties::computeInletProperties(const std::vector<double>&
				   mixfractionStream) 
{
  double invDensity = 0;
  double mixfracSum = 0.0;
  int ii;
  for (ii = 0 ; ii < mixfractionStream.size(); ii++) {
    invDensity += mixfractionStream[ii]/d_streams[ii].d_density;
    mixfracSum += mixfractionStream[ii];
  }
  invDensity += (1.0 - mixfracSum)/d_streams[ii].d_density;
  if (invDensity <= 0.0)
    throw InvalidValue("Computed zero density for inlet stream" + ii );
  else
    return (1.0/invDensity);
}
  
//****************************************************************************
// Schedule the computation of properties
//****************************************************************************
void 
Properties::sched_computeProps(const LevelP& level,
			       SchedulerP& sched, 
			       DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw)
{
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
      tsk->requires(old_dw, d_densitySPLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      for (int ii = 0; ii < d_numMixingVars; ii++) 
	tsk->requires(old_dw, d_scalarSPLabel, ii, patch, Ghost::None,
			 numGhostCells);
      tsk->computes(new_dw, d_densityCPLabel, matlIndex, patch);
      sched->addTask(tsk);
    }
  }
}

//****************************************************************************
// Schedule the recomputation of properties
//****************************************************************************
void 
Properties::sched_reComputeProps(const LevelP& level,
			       SchedulerP& sched, 
			       DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw)
{
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
      tsk->requires(old_dw, d_densityCPLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->computes(new_dw, d_densityRCPLabel, matlIndex, patch);
      sched->addTask(tsk);
    }
  }
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
  // Get the CCVariable (density) from the old datawarehouse
  CCVariable<double> density;
  std::vector<CCVariable<double> > scalar(d_numMixingVars);
  int matlIndex = 0;
  int nofGhostCells = 0;
  old_dw->get(density, d_densitySPLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);

  for (int ii = 0; ii < d_numMixingVars; ii++)
    old_dw->get(scalar[ii], d_scalarSPLabel, ii, patch, Ghost::None,
		nofGhostCells);

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
	double local_den = 0.0;
	double mixFracSum = 0.0;
	for (int ii = 0; ii < d_numMixingVars; ii++ ) {
	  local_den += 
	    (scalar[ii])[IntVector(colX, colY, colZ)]/d_streams[ii].d_density;
	  mixFracSum += (scalar[ii])[IntVector(colX, colY, colZ)];
	}
	local_den += (1.0 - mixFracSum)/d_streams[d_numMixingVars-1].d_density;
	std::cerr << "local_den " << local_den << endl;
	if (local_den <= 0.0)
	  throw InvalidValue("Computed zero density in props" );
	else
	  local_den = (1.0/local_den);
	density[IntVector(colX, colY, colZ)] = d_denUnderrelax*local_den +
                 	  (1.0-d_denUnderrelax)*density[IntVector(colX, colY, colZ)];
      }
    }
  }
  cout << "After compute properties \n";
  for (int kk = indexLow.z(); kk < indexHigh.z(); kk++) 
    for (int jj = indexLow.y(); jj < indexHigh.y(); jj++) 
      for (int ii = indexLow.x(); ii < indexHigh.x(); ii++) 
	cout << "(" << ii << "," << jj << "," << kk << ") : "
	     << " DEN = " << density[IntVector(ii,jj,kk)] << endl;
  
  // Write the computed density to the new data warehouse
  new_dw->put(density, d_densityCPLabel, matlIndex, patch);
}

//****************************************************************************
// Actually recompute the properties here
//****************************************************************************
void 
Properties::reComputeProps(const ProcessorGroup*,
			   const Patch* patch,
			   DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw)
{
  // Get the CCVariable (density) from the old datawarehouse
  CCVariable<double> density;
  int matlIndex = 0;
  int nofGhostCells = 0;
  old_dw->get(density, d_densityCPLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);

  // Create the CCVariable for storing the computed density
  CCVariable<double> new_density;
  new_dw->allocate(new_density, d_densityRCPLabel, matlIndex, patch);

#ifdef WONT_COMPILE_YET
  // Get the low and high index for the patch
  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

  // Calculate the properties
  // this calculation will be done by MixingModel class
  // for more complicated cases...this will only work for
  // helium plume
  FORT_COLDPROPS(new_density, density,
		 lowIndex, highIndex, d_denUnderrelax);
#endif

  // Write the computed density to the new data warehouse
  new_dw->put(new_density, d_densityRCPLabel, matlIndex, patch);
}

//****************************************************************************
// Default constructor for Properties::Stream
//****************************************************************************
Properties::Stream::Stream()
{
}

//****************************************************************************
// Problem Setup for Properties::Stream
//****************************************************************************
void 
Properties::Stream::problemSetup(ProblemSpecP& params)
{
  params->require("density", d_density);
  params->require("temperature", d_temperature);
}

//
// $Log$
// Revision 1.22  2000/07/03 05:30:15  bbanerje
// Minor changes for inlbcs dummy code to compile and work. densitySIVBC is no more.
//
// Revision 1.21  2000/07/01 05:20:59  bbanerje
// Changed CellInformation calcs for Turbulence model requirements ..
// CellInformation still needs work.
//
// Revision 1.20  2000/06/30 04:19:17  rawat
// added turbulence model and compute properties
//
// Revision 1.19  2000/06/21 06:12:12  bbanerje
// Added missing VarLabel* mallocs .
//
// Revision 1.18  2000/06/21 05:43:51  bbanerje
// nofMixingVars init changed from -1 to 0 to avoid seg violation.
// Was calling a vector with -1 as index.
//
// Revision 1.17  2000/06/19 18:00:30  rawat
// added function to compute velocity and density profiles and inlet bc.
// Fixed bugs in CellInformation.cc
//
// Revision 1.16  2000/06/18 01:20:16  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
// Revision 1.15  2000/06/17 07:06:25  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.14  2000/06/16 21:50:48  bbanerje
// Changed the Varlabels so that sequence in understood in init stage.
// First cycle detected in task graph.
//
// Revision 1.13  2000/06/15 21:56:58  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.12  2000/06/14 20:40:49  rawat
// modified boundarycondition for physical boundaries and
// added CellInformation class
//
// Revision 1.11  2000/06/07 06:13:55  bbanerje
// Changed CCVariable<Vector> to CCVariable<double> for most cases.
// Some of these variables may not be 3D Vectors .. they may be Stencils
// or more than 3D arrays. Need help here.
//
// Revision 1.10  2000/05/31 23:44:52  rawat
// modified arches and properties
//
// Revision 1.9  2000/05/31 20:11:30  bbanerje
// Cocoon stuff, tasks added to SmagorinskyModel, TurbulenceModel.
// Added schedule compute of properties and TurbModel to Arches.
//
// Revision 1.8  2000/05/31 08:12:45  bbanerje
// Added Cocoon stuff to Properties, added VarLabels, changed task, requires,
// computes, get etc.in Properties, changed fixed size Mixing Var array to
// vector.
//
//
