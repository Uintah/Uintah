#include <Uintah/Components/Arches/Source.h>
#include <SCICore/Util/NotFinished.h>

Source::Source()
{
}

Source::~Source()
{
}

void Source::sched_calculateVelocitySource(const int index,
					  const LevelP& level,
					  SchedulerP& sched,
					  const DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw)
{
  for(Level::const_regionIterator iter=level->regionsBegin();
      iter != level->regionsEnd(); iter++){
    const Region* region=*iter;
    {
      Task* tsk = new Task("Source::VelocitySource",index,
			   region, old_dw, new_dw, this,
			   Source::calculateVelocitySource);
      tsk->requires(old_dw, "velocity", region, 1,
		    FCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "density", region, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(old_dw, "viscosity", region, 1,
		    CCVariable<double>::getTypeDescription());
      if (index == 1) {
	tsk->computes(new_dw, "uLinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "uNonlinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
      }
      else if (index == 2) {
	tsk->computes(new_dw, "vLinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "vNonlinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
      }
      else if (index == 3) {
	tsk->computes(new_dw, "wLinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "wNonlinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
      }
      else 
	throw InvalidValue("Invalid componenet for velocity" +index);
      
      sched->addTask(tsk);
    }

  }
}

void Source::shed_calculatePressureSource(const LevelP& level,
					  const Region* region,
					  const DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw)
{
}



void Source::sched_calculateScalarSource(const int index,
					 const LevelP& level,
					 SchedulerP& sched,
					 const DataWarehouseP& old_dw,
					 DataWarehouseP& new_dw)
{
}

