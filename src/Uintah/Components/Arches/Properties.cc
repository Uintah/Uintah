/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/Properties.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/DataWarehouse.h>

using namespace Uintah::ArchesSpace;

Properties::Properties()
{
}

Properties::~Properties()
{
}

void Properties::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("Properties");
  db->require("denUnderrelax", d_denUnderrelax);
  d_numMixingVars = 0;
  for (ProblemSpecP stream_db = db->findBlock("Stream");
       stream_db != 0; stream_db = params->findNextBlock("Stream")) {
    d_streams[d_numMixingVars].problemSetup(stream_db);
    ++d_numMixingVars;
  }
}

#if 0
void Properties::sched_computeProps(const LevelP& level,
				    SchedulerP&, DataWarehouseP& old,
				    DataWarehouseP& new)
{
  for(Level::const_regionIterator iter=level->regionsBegin();
      iter != level->regionsEnd(); iter++){
    const Region* region=*iter;
    {
      Task* tsk = new Task("Properties::ComputeProps",
			   region, old_dw, new_dw, this,
			   Properties::computeProps);
      tsk->requires(old_dw, "density", region, 0,
		    CCVariable<Vector>::getTypeDescription());
      tsk->computes(new_dw, "density", region, 0,
		    CCVariable<Vector>::getTypeDescription());
      sched->addTask(tsk);
    }

  }
}

void Properties::computeProps(const ProcessorContext* pc,
			      const Region* region,
			      const DataWarehouseP& old_dw,
			      DataWarehouseP& new_dw)
{
  CCVariable<double> density;
  old_dw->get(density, "density", region, 1);
  Array3Index lowIndex = region->getLowIndex();
  Array3Index highIndex = region->getHighIndex();
  CCVariable<double> new_density;
  // this calculation will be done by MixingModel class
  // for more complicated cases...this will only work for
  // helium plume
  FORT_COLDPROPS(new_density, density,
		 lowIndex, highIndex, d_denUnderrelax);
  new_dw->put(new_density, "density", region);
}

#endif

Properties::Stream::Stream()
{
}

void Properties::Stream::problemSetup(ProblemSpecP& params)
{
  params->require("Density", d_density);
  params->require("Temperature", d_temperature);
}
