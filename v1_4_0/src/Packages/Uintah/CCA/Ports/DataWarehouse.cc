
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Geometry/Vector.h>

#include <iostream>

using namespace Uintah;
using namespace SCIRun;

using std::cerr;

bool DataWarehouse::show_warnings = true;


DataWarehouse::DataWarehouse(const ProcessorGroup* myworld,
			     const Scheduler* scheduler,
			     int generation)
  : d_myworld(myworld), d_scheduler(scheduler), d_generation(generation)
{
}

DataWarehouse::~DataWarehouse()
{
}

void DataWarehouse::copyOut(ParticleVariableBase& var, const VarLabel* label,
			    ParticleSubset* pset)
{
  constParticleVariableBase* constVar = var.cloneConstType();
  this->get(*constVar, label, pset);
  var.copyData(&constVar->getBaseRep());
  delete constVar;
}

void DataWarehouse::getCopy(ParticleVariableBase& var, const VarLabel* label,
			    ParticleSubset* pset)
{
  constParticleVariableBase* constVar = var.cloneConstType();
  this->get(*constVar, label, pset);
  var.allocate(pset);
  var.copyData(&constVar->getBaseRep());
  delete constVar;
}
