
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Geometry/Vector.h>

#include <iostream>

using namespace Uintah;
using namespace SCIRun;

using std::cerr;

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

void DataWarehouse::copyOutGridData(Variable* var, const VarLabel* label,
				    int matlIndex, const Patch* patch,
				    Ghost::GhostType gtype,
				    int numGhostCells)
{
  union {
    NCVariableBase* nc;
    CCVariableBase* cc;
    SFCXVariableBase* sfcx;
    SFCYVariableBase* sfcy;
    SFCZVariableBase* sfcz;
  } castVar;
  
  if ((castVar.nc = dynamic_cast<NCVariableBase*>(var)) != NULL)
    copyOut(*castVar.nc, label, matlIndex, patch, gtype, numGhostCells);
  else if ((castVar.cc = dynamic_cast<CCVariableBase*>(var)) != NULL)
    copyOut(*castVar.cc, label, matlIndex, patch, gtype, numGhostCells);
  else if ((castVar.sfcx=dynamic_cast<SFCXVariableBase*>(var)) != NULL)
    copyOut(*castVar.sfcx, label, matlIndex, patch, gtype, numGhostCells);
  else if ((castVar.sfcy=dynamic_cast<SFCYVariableBase*>(var)) != NULL)
    copyOut(*castVar.sfcy, label, matlIndex, patch, gtype, numGhostCells);
  else if ((castVar.sfcz=dynamic_cast<SFCZVariableBase*>(var)) != NULL)
    copyOut(*castVar.sfcz, label, matlIndex, patch, gtype, numGhostCells);
  else
    throw InternalError("OnDemandDataWarehouse::copyOutGridData: Not a grid variable type");
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
