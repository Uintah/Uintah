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
  d_densityLabel = scinew VarLabel("density",
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
      tsk->requires(old_dw, d_densityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->computes(new_dw, d_densityLabel, matlIndex, patch);
      sched->addTask(tsk);
    }
  }
}

//****************************************************************************
// Actually compute the properties here
//****************************************************************************
void 
Properties::computeProps(const ProcessorContext*,
			 const Patch* patch,
			 DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw)
{
  // Get the CCVariable (density) from the old datawarehouse
  CCVariable<double> density;
  int matlIndex = 0;
  int nofGhostCells = 0;
  old_dw->get(density, d_densityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);

  // Get the low and high index for the patch
  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

  // Create the CCVariable for storing the computed density
  CCVariable<double> new_density;
  new_dw->allocate(new_density, d_densityLabel, matlIndex, patch);

#ifdef WONT_COMPILE_YET
  // Calculate the properties
  // this calculation will be done by MixingModel class
  // for more complicated cases...this will only work for
  // helium plume
  FORT_COLDPROPS(new_density, density,
		 lowIndex, highIndex, d_denUnderrelax);
#endif

  // Write the computed density to the new data warehouse
  new_dw->put(new_density, d_densityLabel, matlIndex, patch);
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
