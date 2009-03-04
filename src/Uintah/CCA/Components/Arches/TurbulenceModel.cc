/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Variables/SoleVariable.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Exceptions/VariableNotFoundInGrid.h>

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
//______________________________________________________________________
//
void 
TurbulenceModel::sched_initFilterMatrix(const LevelP& level,
                                        SchedulerP& sched, 
                                        const PatchSet* patches,
                                        const MaterialSet* matls)
{
  d_filter->sched_buildFilterMatrix(level, sched);
  Task* tsk = scinew Task("TurbulenceModel::initFilterMatrix",this,
                          &TurbulenceModel::initFilterMatrix);
                                              
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::AroundCells, 1);
  sched->addTask(tsk, patches, matls);
}
//______________________________________________________________________
//
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
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    constCCVariable<int> cellType;
    
    new_dw->get(cellType, d_lab->d_cellTypeLabel,indx, patch, Ghost::AroundCells, 1);

    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, indx, patch)){ 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    }else{ 
      throw VariableNotFoundInGrid("cellInformation"," ", __FILE__, __LINE__);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    d_filter->setFilterMatrix(pg, patch, cellinfo, cellType);
  }
}
#endif



