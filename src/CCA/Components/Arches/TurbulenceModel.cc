/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/Arches/TurbulenceModel.h>
#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <Core/Grid/Level.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>

using namespace Uintah;

TurbulenceModel::TurbulenceModel(const ArchesLabel* label,
                                 const MPMArchesLabel* MAlb):
                                 d_lab(label), d_MAlab(MAlb)
{
  d_filter = 0;
  d_dissipationRateLabel = VarLabel::create(
    "dissipationRate", CCVariable<double>::getTypeDescription() );
}

TurbulenceModel::~TurbulenceModel()
{
  if (d_filter)
    delete d_filter;
  VarLabel::destroy( d_dissipationRateLabel );
}

void
TurbulenceModel::problemSetupCommon( const ProblemSpecP& params )
{

  ProblemSpecP db = params;
  ProblemSpecP db_turb;

  d_filter_type = "moin98";
  d_filter_width = 3;

  if ( db->findBlock("Turbulence") ){
    db_turb = db->findBlock("Turbulence");

    //setup the filter:
    d_use_old_filter = true;
    if ( db_turb->findBlock("ignore_filter_bc") ) {
      //Will not adjust filter weights for the presence of any BC.
      d_use_old_filter = false;
    }

    if ( db_turb->findBlock("filter_type")){
      db_turb->getWithDefault("filter_type", d_filter_type, "moin98");
    }

    if ( db_turb->findBlock("filter_width")){
      db_turb->getWithDefault("filter_width",d_filter_width,3);
    }
  }

  d_filter = scinew Filter( d_use_old_filter, d_filter_type, d_filter_width );

}
void TurbulenceModel::sched_computeFilterVol( SchedulerP& sched,
                                              const LevelP& level,
                                              const MaterialSet* matls )
{

  Task* tsk = scinew Task( "TurbulenceModel::computeFilterVol",this, &TurbulenceModel::computeFilterVol);
  tsk->computes( d_lab->d_filterVolumeLabel );
  tsk->requires( Task::NewDW, d_lab->d_cellTypeLabel, Ghost::AroundCells, 1 );
  //hacking in the dissipation rate here under the assumption that we are
  // recoding this turbulence models under the kokkos design
  tsk->computes( d_dissipationRateLabel );

  sched->addTask(tsk, level->eachPatch(), matls);

}

void
TurbulenceModel::computeFilterVol( const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset*,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw )
{

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double>   filter_volume;
    CCVariable<double>   dissipation_rate;
    constCCVariable<int> cell_type;

    new_dw->get( cell_type, d_lab->d_cellTypeLabel, indx, patch, Ghost::AroundCells, 1 );
    new_dw->allocateAndPut( filter_volume, d_lab->d_filterVolumeLabel, indx, patch );
    new_dw->allocateAndPut( dissipation_rate, d_dissipationRateLabel, indx, patch );

    filter_volume.initialize(0.0);
    dissipation_rate.initialize(0.0);

    d_filter->computeFilterVolume( patch, cell_type, filter_volume );

  }
}

void TurbulenceModel::sched_carryForwardFilterVol( SchedulerP& sched,
                                                   const PatchSet* patches,
                                                   const MaterialSet* matls )
{
  Task* tsk = scinew Task( "TurbulenceModel::carryForwardFilterVol",
      this, &TurbulenceModel::carryForwardFilterVol);
  tsk->computes( d_lab->d_filterVolumeLabel );
  tsk->computes( d_dissipationRateLabel );
  tsk->requires( Task::OldDW, d_lab->d_filterVolumeLabel, Ghost::None, 0 );

  sched->addTask(tsk, patches, matls);
}

void
TurbulenceModel::carryForwardFilterVol( const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset*,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw )
{

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int indx = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double>   filter_vol;
    constCCVariable<double> old_filter_vol;
    CCVariable<double> dissipation_rate;

    new_dw->allocateAndPut( filter_vol, d_lab->d_filterVolumeLabel, indx, patch );
    new_dw->allocateAndPut( dissipation_rate, d_dissipationRateLabel, indx, patch );
    old_dw->get( old_filter_vol,  d_lab->d_filterVolumeLabel, indx, patch, Ghost::None, 0 );

    filter_vol.copyData(old_filter_vol);
    // The specific turbulence model will compute this.
    // Just initializing it to zero for convenience.
    dissipation_rate.initialize(0.0);

  }
}
