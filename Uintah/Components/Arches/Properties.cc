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

using namespace Uintah::ArchesSpace;

//****************************************************************************
// Default constructor for Properties
//****************************************************************************
Properties::Properties()
{
  d_densitySPLabel = scinew VarLabel("densitySP",
				   CCVariable<double>::getTypeDescription() );
  d_densityCPLabel = scinew VarLabel("densityCP",
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

  // Read the mixing variable streams
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
      tsk->computes(new_dw, d_densityCPLabel, matlIndex, patch);
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
  int matlIndex = 0;
  int nofGhostCells = 0;
  old_dw->get(density, d_densitySPLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);

  // Create the CCVariable for storing the computed density
  CCVariable<double> new_density;
  new_dw->allocate(new_density, d_densityCPLabel, matlIndex, patch);

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
  new_dw->put(new_density, d_densityCPLabel, matlIndex, patch);
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
