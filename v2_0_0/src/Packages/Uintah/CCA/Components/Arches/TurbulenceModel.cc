#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SoleVariable.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>

using namespace Uintah;

TurbulenceModel::TurbulenceModel(const ArchesLabel* label, 
				 const MPMArchesLabel* MAlb):
                                 d_lab(label), d_MAlab(MAlb)
{
#ifdef PetscFilter
  d_filter = 0;
#endif
}

TurbulenceModel::~TurbulenceModel()
{
#ifdef PetscFilter
  if (d_filter)
    delete d_filter;
#endif
}
#ifdef PetscFilter
void 
TurbulenceModel::sched_initFilterMatrix(const LevelP& level,
					SchedulerP& sched, 
					const PatchSet* patches,
					const MaterialSet* matls)
{
  d_filter->sched_buildFilterMatrix(level, sched);
  Task* tsk = scinew Task("TurbulenceModel::initFilterMatrix",
			  this,
			  &TurbulenceModel::initFilterMatrix);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  sched->addTask(tsk, patches, matls);


}

void
TurbulenceModel::initFilterMatrix(const ProcessorGroup* pg,
				  const PatchSubset* patches,
				  const MaterialSubset*,
				  DataWarehouse*,
				  DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    PerPatch<CellInformationP> cellInfoP;
    constCCVariable<int> cellType;
    new_dw->get(cellType, d_lab->d_cellTypeLabel,
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    d_filter->setFilterMatrix(pg, patch, cellinfo, cellType);
  }
}
#endif



