/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

//----- Properties.cc --------------------------------------------------
#include <CCA/Components/Arches/Properties.h>
#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#if HAVE_TABPROPS
# include <CCA/Components/Arches/ChemMix/TabPropsInterface.h>
#endif
# include <CCA/Components/Arches/ChemMix/ClassicTableInterface.h>
# include <CCA/Components/Arches/ChemMix/ColdFlow.h>
# include <CCA/Components/Arches/ChemMix/ConstantProps.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/CellInformationP.h>
#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>

#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/SimulationState.h>

#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Math/MiscMath.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>


#include <iostream>
using namespace std;
using namespace Uintah;

//****************************************************************************
// Default constructor for Properties
//****************************************************************************
Properties::Properties(ArchesLabel* label,
                       const MPMArchesLabel* MAlb,
                       PhysicalConstants* phys_const,
                       const ProcessorGroup* myworld):
                       d_lab(label), d_MAlab(MAlb),
                       d_physicalConsts(phys_const),
                       d_myworld(myworld)
{
}

//****************************************************************************
// Destructor
//****************************************************************************
Properties::~Properties()
{
  if ( mixModel == "TabProps" || mixModel == "ClassicTable"
      || mixModel == "ColdFlow" || mixModel == "ConstantProps" ){
    delete d_mixingRxnTable;
  }
}

//****************************************************************************
// Problem Setup for Properties
//****************************************************************************
void
Properties::problemSetup(const ProblemSpecP& params)
{

  ProblemSpecP db = params->findBlock("Properties");

  if ( db == nullptr ){
    throw ProblemSetupException("Error: Please specify a <Properties> section in <Arches>.", __FILE__, __LINE__); 
  }

  db->getWithDefault("filter_drhodt",          d_filter_drhodt,          false);
  db->getWithDefault("first_order_drhodt",     d_first_order_drhodt,     true);
  db->getWithDefault("inverse_density_average",d_inverse_density_average,false);

  d_denRef = d_physicalConsts->getRefPoint();

  // check to see if gas is adiabatic and (if DQMOM) particles are not:
  d_adiabGas_nonadiabPart = false;
  if (params->findBlock("DQMOM")) {
    ProblemSpecP db_dqmom = params->findBlock("DQMOM");
    db_dqmom->getWithDefault("adiabGas_nonadiabPart", d_adiabGas_nonadiabPart, false);
  }

//   // read type of mixing model
//   mixModel = "NA";
//   if (db->findBlock("ClassicTable"))
//     mixModel = "ClassicTable";
//   else if (db->findBlock("ColdFlow"))
//     mixModel = "ColdFlow";
//   else if (db->findBlock("ConstantProps"))
//     mixModel = "ConstantProps";
// #if HAVE_TABPROPS
//   else if (db->findBlock("TabProps"))
//     mixModel = "TabProps";
// #endif
//   else
//     throw InvalidValue("ERROR!: No mixing/reaction table specified! If you are attempting to use the new TabProps interface, ensure that you configured properly with TabProps and Boost libs.",__FILE__,__LINE__);
//
//   SimulationStateP& sharedState = d_lab->d_sharedState;
//
//   if (mixModel == "ClassicTable") {
//
//     // New Classic interface
//     d_mixingRxnTable = scinew ClassicTableInterface( sharedState );
//     d_mixingRxnTable->problemSetup( db );
//   } else if (mixModel == "ColdFlow") {
//     d_mixingRxnTable = scinew ColdFlow( sharedState );
//     d_mixingRxnTable->problemSetup( db );
//   } else if (mixModel == "ConstantProps" ) {
//     d_mixingRxnTable = scinew ConstantProps( sharedState );
//     d_mixingRxnTable->problemSetup( db );
//   }
// #if HAVE_TABPROPS
//   else if (mixModel == "TabProps") {
//     // New TabPropsInterface stuff...
//     d_mixingRxnTable = scinew TabPropsInterface( sharedState );
//     d_mixingRxnTable->problemSetup( db );
//   }
// #endif
//   else{
//     throw InvalidValue("Mixing Model not supported: " + mixModel, __FILE__, __LINE__);
//   }
}

//****************************************************************************
// Schedule the averaging of properties for Runge-Kutta step
//****************************************************************************
void
Properties::sched_averageRKProps( SchedulerP& sched, const PatchSet* patches,
                                  const MaterialSet* matls,
                                  const TimeIntegratorLabel* timelabels )
{
  string taskname =  "Properties::averageRKProps" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &Properties::averageRKProps,
                          timelabels);

  Ghost::GhostType  gn = Ghost::None;
  tsk->requires(Task::OldDW, d_lab->d_densityCPLabel,     gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_densityTempLabel,   gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,     gn, 0);
  tsk->modifies(d_lab->d_densityGuessLabel);



  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually average the Runge-Kutta properties here
//****************************************************************************
void
Properties::averageRKProps( const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset*,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw,
                            const TimeIntegratorLabel* timelabels )
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex();

    constCCVariable<double> old_density;
    constCCVariable<double> rho1_density;
    constCCVariable<double> new_density;
    CCVariable<double> density_guess;

    Ghost::GhostType  gn = Ghost::None;
    old_dw->get(old_density,  d_lab->d_densityCPLabel,    indx, patch, gn, 0);
    new_dw->get(rho1_density, d_lab->d_densityTempLabel,  indx, patch, gn, 0);
    new_dw->get(new_density,  d_lab->d_densityCPLabel,    indx, patch, gn, 0);
    new_dw->getModifiable(density_guess, d_lab->d_densityGuessLabel, indx, patch);

    double factor_old, factor_new, factor_divide;
    factor_old = timelabels->factor_old;
    factor_new = timelabels->factor_new;
    factor_divide = timelabels->factor_divide;

    IntVector indexLow  = patch->getExtraCellLowIndex();
    IntVector indexHigh = patch->getExtraCellHighIndex();

    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);

          if (new_density[currCell] > 0.0) {

            double predicted_density;

            if (old_density[currCell] > 0.0) {

              //predicted_density = rho1_density[currCell];
              if (d_inverse_density_average)
                predicted_density = 1.0/((factor_old/old_density[currCell] + factor_new/new_density[currCell])/factor_divide);
              else
                predicted_density = (factor_old*old_density[currCell] + factor_new*new_density[currCell])/factor_divide;

            } else {

              predicted_density = new_density[currCell];

            }

            density_guess[currCell] = predicted_density;
          }

        }
      }
    }
  }
}

//****************************************************************************
// Schedule saving of temp density
//****************************************************************************
void
Properties::sched_saveTempDensity(SchedulerP& sched,
                                  const PatchSet* patches,
                                  const MaterialSet* matls,
                                  const TimeIntegratorLabel* timelabels)
{
  string taskname =  "Properties::saveTempDensity" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &Properties::saveTempDensity,
                          timelabels);

  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::None, 0);
  tsk->modifies(d_lab->d_densityTempLabel);
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually save temp density here
//****************************************************************************
void
Properties::saveTempDensity(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset*,
                            DataWarehouse*,
                            DataWarehouse* new_dw,
                            const TimeIntegratorLabel* )
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex();

    CCVariable<double> temp_density;

    new_dw->getModifiable(temp_density, d_lab->d_densityTempLabel,indx, patch);
    new_dw->copyOut(temp_density,      d_lab->d_densityCPLabel,   indx, patch);
  }
}
//****************************************************************************
// Schedule the computation of drhodt
//****************************************************************************
void
Properties::sched_computeDrhodt(SchedulerP& sched,
                                const PatchSet* patches,
                                const MaterialSet* matls,
                                const TimeIntegratorLabel* timelabels)
{
  string taskname =  "Properties::computeDrhodt" +
                     timelabels->integrator_step_name;

  Task* tsk = scinew Task(taskname, this,
                          &Properties::computeDrhodt,
                          timelabels);

  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = Task::ParentOldDW;
  }else{
    parent_old_dw = Task::OldDW;
  }

  Ghost::GhostType  gn = Ghost::None;
  Ghost::GhostType  ga = Ghost::AroundCells;

  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);
  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
  tsk->requires(parent_old_dw, d_lab->d_oldDeltaTLabel);

  tsk->requires(Task::NewDW   , d_lab->d_densityCPLabel    , gn , 0);
  tsk->requires(parent_old_dw , d_lab->d_densityCPLabel    , gn , 0);
  tsk->requires(Task::NewDW   , d_lab->d_filterVolumeLabel , ga , 1);
  tsk->requires(Task::NewDW   , d_lab->d_cellTypeLabel     , ga , 1);

  //tsk->requires(Task::NewDW, VarLabel::find("mixture_fraction"), gn, 0);

  if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::First ) {
    tsk->computes(d_lab->d_filterdrhodtLabel);
    tsk->computes(d_lab->d_oldDeltaTLabel);
  }
  else{
    tsk->modifies(d_lab->d_filterdrhodtLabel);
  }
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Compute drhodt
//****************************************************************************
void
Properties::computeDrhodt(const ProcessorGroup* pc,
                          const PatchSubset* patches,
                          const MaterialSubset*,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          const TimeIntegratorLabel* timelabels)
{
  DataWarehouse* parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  }else{
   parent_old_dw = old_dw;
  }

  int drhodt_1st_order = 1;
  int current_step = d_lab->d_sharedState->getCurrentTopLevelTimeStep();
  if (d_MAlab){
    drhodt_1st_order = 2;
  }
  delt_vartype delT, old_delT;
  parent_old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );


  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
    new_dw->put(delT, d_lab->d_oldDeltaTLabel);
  }
  double delta_t = delT;
  delta_t *= timelabels->time_multiplier;
  delta_t *= timelabels->time_position_multiplier_after_average;

  parent_old_dw->get(old_delT, d_lab->d_oldDeltaTLabel);
  double  old_delta_t = old_delT;

  //__________________________________
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex();

    constCCVariable<double> new_density;
    constCCVariable<double> old_density;
    constCCVariable<double> old_old_density;
    //constCCVariable<double> mf;
    CCVariable<double> drhodt;
    CCVariable<double> filterdrhodt;
    constCCVariable<double> filterVolume;
    constCCVariable<int> cellType;
    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  ga = Ghost::AroundCells;

    parent_old_dw->get(old_density,     d_lab->d_densityCPLabel,     indx, patch,gn, 0);

    new_dw->get( cellType, d_lab->d_cellTypeLabel, indx, patch, ga, 1 );
    new_dw->get( filterVolume, d_lab->d_filterVolumeLabel, indx, patch, ga, 1 );
    //new_dw->get( mf, VarLabel::find("mixture_fraction"), indx, patch, gn, 0);

    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);

    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(new_density, d_lab->d_densityCPLabel,  indx, patch, gn, 0);

    if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::First ){
      new_dw->allocateAndPut(filterdrhodt, d_lab->d_filterdrhodtLabel, indx, patch);
    }else{
      new_dw->getModifiable(filterdrhodt, d_lab->d_filterdrhodtLabel,  indx, patch);
    }
    filterdrhodt.initialize(0.0);

    // Get the patch and variable indices
    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();
    // compute drhodt and its filtered value
    drhodt.allocate(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    drhodt.initialize(0.0);

    //__________________________________
    if ((d_first_order_drhodt)||(current_step <= drhodt_1st_order)) {
// 1st order drhodt
      for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
        for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
          for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
            IntVector currcell(ii,jj,kk);

            double vol =cellinfo->sns[jj]*cellinfo->stb[kk]*cellinfo->sew[ii];
            //double rho_f = 1.18;
            //double rho_ox = 0.5;
            //double newnewrho = mf[currcell]/rho_ox + (1.0 - mf[currcell])/rho_f;
            drhodt[currcell] = (new_density[currcell] -
                                old_density[currcell])*vol/delta_t;
            //drhodt[currcell] = (newnewrho - old_density[currcell]*vol/delta_t);
          }
        }
      }
    }
    else {
// 2nd order drhodt, assuming constant volume
      double factor = 1.0 + old_delta_t/delta_t;
      double new_factor = factor * factor - 1.0;
      double old_factor = factor * factor;
      for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
        for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
          for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
            IntVector currcell(ii,jj,kk);

            double vol =cellinfo->sns[jj]*cellinfo->stb[kk]*cellinfo->sew[ii];
            drhodt[currcell] = (new_factor*new_density[currcell] -
                                old_factor*old_density[currcell] +
                                old_old_density[currcell])*vol /
                               (old_delta_t*factor);
            //double rho_f = 1.18;
            //double rho_ox = 0.5;
            //double newnewrho = mf[currcell]/rho_f + (1.0 - mf[currcell])/rho_ox;
          }
        }
      }
    }

    if ((d_filter_drhodt)&&(!(d_3d_periodic))) {
    // filtering for periodic case is not implemented
    // if it needs to be then drhodt will require 1 layer of boundary cells to be computed
      d_filter->applyFilter_noPetsc<CCVariable<double> >(pc, patch, drhodt, filterVolume, cellType, filterdrhodt);
    }else{
      filterdrhodt.copy(drhodt, drhodt.getLowIndex(),
                      drhodt.getHighIndex());
    }

    for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
      for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
        for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
          IntVector currcell(ii,jj,kk);

          double vol =cellinfo->sns[jj]*cellinfo->stb[kk]*cellinfo->sew[ii];
          if (Abs(filterdrhodt[currcell]/vol) < 1.0e-9){
              filterdrhodt[currcell] = 0.0;
          }
        }
      }
    }
  }
}

void
Properties::sched_computeProps( const LevelP& level,
                                SchedulerP& sched,
                                const bool initialize,
                                const bool modify_ref_den,
                                const int time_substep )
{
  // this method is temporary while we get rid of properties.cc
  d_mixingRxnTable->sched_getState( level, sched, time_substep, initialize, modify_ref_den );

}

void
Properties::sched_checkTableBCs( const LevelP& level,
                                SchedulerP& sched )
{
  d_mixingRxnTable->sched_checkTableBCs( level, sched );
}

void
Properties::addLookupSpecies( ){

  ChemHelper& helper = ChemHelper::self();
  std::vector<std::string> sps;
  sps = helper.model_req_species;

  if ( mixModel == "ClassicTable"  || mixModel == "TabProps"
      || "ColdFlow" || "ConstantProps" ) {
    for ( vector<string>::iterator i = sps.begin(); i != sps.end(); i++ ){
      bool test = d_mixingRxnTable->insertIntoMap( *i );
      if ( !test ){
        throw InvalidValue("Error: Cannot locate the following variable for lookup in the table: "+*i, __FILE__, __LINE__ );
      }
    }
  }

  std::vector<std::string> old_sps;
  old_sps = helper.model_req_old_species;
  if ( mixModel == "ClassicTable"  || mixModel == "TabProps"
      || "ColdFlow" || "ConstantProps") {
    for ( vector<string>::iterator i = old_sps.begin(); i != old_sps.end(); i++ ){
      d_mixingRxnTable->insertOldIntoMap( *i );
    }
  }
}

void
Properties::doTableMatching(){

  d_mixingRxnTable->tableMatching();

}
