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

//----- CompDynamicProcedure.cc --------------------------------------------------

#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/CompDynamicProcedure.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/StencilMatrix.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>

#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Util/Timers/Timers.hpp>

using namespace std;

using namespace Uintah;

// flag to enable filter check
// need even grid size, unfiltered values are +-1; filtered value should be 0
// #define FILTER_CHECK
#ifdef FILTER_CHECK
#include <Core/Math/MiscMath.h>
#endif

//****************************************************************************
// Default constructor for CompDynamicProcedure
//****************************************************************************
CompDynamicProcedure::CompDynamicProcedure(const ArchesLabel* label,
    const MPMArchesLabel* MAlb,
    PhysicalConstants* phyConsts,
    BoundaryCondition* bndry_cond):
  TurbulenceModel(label, MAlb),
  d_physicalConsts(phyConsts),
  d_boundaryCondition(bndry_cond)
{
}

//****************************************************************************
// Destructor
//****************************************************************************
CompDynamicProcedure::~CompDynamicProcedure()
{
}

//****************************************************************************
//  Get the molecular viscosity from the Physical Constants object
//****************************************************************************
double
CompDynamicProcedure::getMolecularViscosity() const {
  return d_physicalConsts->getMolecularViscosity();
}

//****************************************************************************
// Problem Setup
//****************************************************************************
  void
CompDynamicProcedure::problemSetup(const ProblemSpecP& params)
{
  problemSetupCommon( params );
  ProblemSpecP db = params->findBlock("Turbulence");

  db->getWithDefault("filter_cs_squared",d_filter_cs_squared,false);

}

//****************************************************************************
// Schedule recomputation of the turbulence sub model
//****************************************************************************
  void
CompDynamicProcedure::sched_reComputeTurbSubmodel( SchedulerP& sched,
                                                   const LevelP& level,
                                                   const MaterialSet* matls,
                                                   const TimeIntegratorLabel* timelabels)
{
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gaf = Ghost::AroundFaces;
  Ghost::GhostType  gn = Ghost::None;
  Task::MaterialDomainSpec oams = Task::OutOfDomain;  //outside of arches matlSet.

  {
    string taskname =  "CompDynamicProcedure::reComputeTurbSubmodel" +
                        timelabels->integrator_step_name;

    Task* tsk = scinew Task(taskname, this,
        &CompDynamicProcedure::reComputeTurbSubmodel,
        timelabels);


    d_denRefArrayLabel = VarLabel::find("denRefArray");

    // Requires
    tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,     gac, 2);
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,      gac, 2);
    tsk->requires(Task::NewDW, d_lab->d_volFractionLabel,   gac, 2);
    tsk->requires(Task::NewDW, d_lab->d_filterVolumeLabel, gn);
    tsk->requires(Task::NewDW, d_denRefArrayLabel, Ghost::None, 0);

    // Computes
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->computes(d_lab->d_filterRhoULabel);
      tsk->computes(d_lab->d_filterRhoVLabel);
      tsk->computes(d_lab->d_filterRhoWLabel);
      tsk->computes(d_lab->d_filterRhoLabel);
    } else {
      tsk->modifies(d_lab->d_filterRhoULabel);
      tsk->modifies(d_lab->d_filterRhoVLabel);
      tsk->modifies(d_lab->d_filterRhoWLabel);
      tsk->modifies(d_lab->d_filterRhoLabel);
    }

    sched->addTask(tsk, level->eachPatch(), matls);
  }

  //__________________________________
  {
    string taskname =  "CompDynamicProcedure::reComputeStrainRateTensors" +
                        timelabels->integrator_step_name;
    Task* tsk = scinew Task( taskname, this,
                             &CompDynamicProcedure::reComputeStrainRateTensors,
                             timelabels);
    // Requires
    // Assuming one layer of ghost cells
    // initialize with the value of zero at the physical bc's
    // construct a stress tensor and stored as a array with the following order
    // {t11, t12, t13, t21, t22, t23, t31, t23, t33}
    tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_CCVelocityLabel,gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_filterRhoULabel,    gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_filterRhoVLabel,    gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_filterRhoWLabel,    gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_filterRhoLabel,     gac, 1);

    // Computes
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->computes(d_lab->d_strainTensorCompLabel,
          d_lab->d_symTensorMatl, oams);
      tsk->computes(d_lab->d_filterStrainTensorCompLabel,
          d_lab->d_symTensorMatl, oams);
    } else {
      tsk->modifies(d_lab->d_strainTensorCompLabel,
          d_lab->d_symTensorMatl, oams);
      tsk->modifies(d_lab->d_filterStrainTensorCompLabel,
          d_lab->d_symTensorMatl, oams);
    }
    sched->addTask(tsk, level->eachPatch(), matls);
  }

  //__________________________________
  {
    string taskname =  "CompDynamicProcedure::reComputeFilterValues" +
                       timelabels->integrator_step_name;
    Task* tsk = scinew Task( taskname, this,
                             &CompDynamicProcedure::reComputeFilterValues,
                             timelabels);

    // Requires
    // Assuming one layer of ghost cells
    // initialize with the value of zero at the physical bc's
    // construct a stress tensor and stored as a array with the following order
    // {t11, t12, t13, t21, t22, t23, t31, t23, t33}
    tsk->requires(Task::NewDW, d_lab->d_CCVelocityLabel, gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,      gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_filterRhoLabel,      gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_filterVolumeLabel, gn);
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, gac, 2);
    tsk->requires(Task::NewDW, d_lab->d_volFractionLabel, gac, 1);

    tsk->requires(Task::NewDW, d_lab->d_strainTensorCompLabel,
        d_lab->d_symTensorMatl, oams,gac, 1);

    tsk->requires(Task::NewDW, d_lab->d_filterStrainTensorCompLabel,
        d_lab->d_symTensorMatl, oams,gac, 1);

    // Computes
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->computes(d_lab->d_strainMagnitudeLabel);
      tsk->computes(d_lab->d_strainMagnitudeMLLabel);
      tsk->computes(d_lab->d_strainMagnitudeMMLabel);
    }
    else {
      tsk->modifies(d_lab->d_strainMagnitudeLabel);
      tsk->modifies(d_lab->d_strainMagnitudeMLLabel);
      tsk->modifies(d_lab->d_strainMagnitudeMMLabel);
    }

    sched->addTask(tsk, level->eachPatch(), matls);
  }

  //__________________________________
  {
    string taskname =  "CompDynamicProcedure::reComputeSmagCoeff" +
                        timelabels->integrator_step_name;
    Task* tsk = scinew Task( taskname, this,
                             &CompDynamicProcedure::reComputeSmagCoeff,
                             timelabels);

    // Requires
    // Assuming one layer of ghost cells
    // initialize with the value of zero at the physical bc's
    // construct a stress tensor and stored as an array with the following order
    // {t11, t12, t13, t21, t22, t23, t31, t23, t33}
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,         gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_strainMagnitudeLabel,   gn, 0);
    tsk->requires(Task::NewDW, d_lab->d_strainMagnitudeMLLabel, gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_strainMagnitudeMMLabel, gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,          gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_filterVolumeLabel, gn);
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_volFractionLabel, gac, 1);

    // Computes
    tsk->modifies(d_lab->d_viscosityCTSLabel);
    tsk->modifies(d_lab->d_turbViscosLabel);

    // modifies
    tsk->modifies(d_dissipationRateLabel);

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->computes(d_lab->d_CsLabel);
    }
    else {
      tsk->modifies(d_lab->d_CsLabel);
    }

    sched->addTask(tsk, level->eachPatch(), matls);
  }
}


//****************************************************************************
// Actual recompute
//****************************************************************************
  void
CompDynamicProcedure::reComputeTurbSubmodel(const ProcessorGroup* pc,
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse*,
    DataWarehouse* new_dw,
    const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    constSFCXVariable<double> uVel;
    constSFCYVariable<double> vVel;
    constSFCZVariable<double> wVel;
    constCCVariable<double> density;
    constCCVariable<int> cellType;
    constCCVariable<double> filterVolume;
    constCCVariable<double> vol_fraction;

    // Get the velocity
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    new_dw->get(uVel, d_lab->d_uVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(vVel, d_lab->d_vVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(wVel, d_lab->d_wVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(density, d_lab->d_densityCPLabel,  indx, patch, gac, 2);
    new_dw->get(filterVolume, d_lab->d_filterVolumeLabel, indx, patch, Ghost::None, 0);

    new_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch, gac, 2);
    new_dw->get(vol_fraction, d_lab->d_volFractionLabel, indx, patch, gac, 2);

    SFCXVariable<double> filterRhoU;
    SFCYVariable<double> filterRhoV;
    SFCZVariable<double> filterRhoW;
    CCVariable<double> filterRho;

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->allocateAndPut(filterRhoU, d_lab->d_filterRhoULabel, indx, patch);
      new_dw->allocateAndPut(filterRhoV, d_lab->d_filterRhoVLabel, indx, patch);
      new_dw->allocateAndPut(filterRhoW, d_lab->d_filterRhoWLabel, indx, patch);
      new_dw->allocateAndPut(filterRho,  d_lab->d_filterRhoLabel,  indx, patch);

    } else {
      new_dw->getModifiable(filterRhoU, d_lab->d_filterRhoULabel, indx, patch);
      new_dw->getModifiable(filterRhoV, d_lab->d_filterRhoVLabel, indx, patch);
      new_dw->getModifiable(filterRhoW, d_lab->d_filterRhoWLabel, indx, patch);
      new_dw->getModifiable(filterRho,  d_lab->d_filterRhoLabel,  indx, patch);
    }
    filterRhoU.initialize(0.0);
    filterRhoV.initialize(0.0);
    filterRhoW.initialize(0.0);
    filterRho.initialize(0.0);

    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;

    IntVector low;
    if ( xminus ){
      low  = patch->getCellLowIndex()+IntVector(1,0,0);
    }else{
      low  = patch->getCellLowIndex();
    }
    IntVector high = patch->getCellHighIndex();

    CellIterator iter = CellIterator(low,high);

    double dim = 0;
    d_filter->applyFilter( pc, iter, uVel, density, filterVolume, vol_fraction, filterRhoU, dim );

    if ( yminus ){
      low = patch->getCellLowIndex()+IntVector(0,1,0);
    } else {
      low = patch->getCellLowIndex();
    }

    iter = CellIterator(low,high);
    dim = 1;
    d_filter->applyFilter( pc, iter, vVel, density, filterVolume, vol_fraction, filterRhoV, dim );

    if ( zminus ){
      low = patch->getCellLowIndex()+IntVector(0,0,1);
    } else {
      low = patch->getCellLowIndex();
    }

    iter = CellIterator(low,high);
    dim = 2;
    d_filter->applyFilter( pc, iter, wVel, density, filterVolume, vol_fraction, filterRhoW, dim );

    d_filter->applyFilter<constCCVariable<double> >(pc, patch, density, filterVolume, vol_fraction, filterRho);

    // making filterRho nonzero
    int mmWallID = d_boundaryCondition->getMMWallId();
    if (mmWallID > 0) {

      constCCVariable<double> ref_density;
      new_dw->get(ref_density, d_denRefArrayLabel, indx, patch, Ghost::None, 0);
      IntVector idxLo = patch->getExtraCellLowIndex();
      IntVector idxHi = patch->getExtraCellHighIndex();
      for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
        for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
          for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
            IntVector currCell(colX, colY, colZ);
            if (filterRho[currCell] < 1.0e-15)
              filterRho[currCell]=ref_density[currCell];
          }
        }
      }
    }

//    constCCVariable<double> ref_density;
//    new_dw->get(ref_density, d_denRefArrayLabel, indx, patch, Ghost::None, 0);
//
//    IntVector idxLo = patch->getExtraCellLowIndex();
//    IntVector idxHi = patch->getExtraCellHighIndex();
//
//    for (CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
//
//      IntVector c = *iter;
//
//      //assuming a 0 or 1 volume fraction
//      filterRho[c] = (1.0-vol_fraction[c]) * ref_density[c] +
//                      vol_fraction[c] * filterRho[c];
//
//    }
  }
}
//****************************************************************************
// Actual recompute
//****************************************************************************
  void
CompDynamicProcedure::reComputeStrainRateTensors(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse*,
    DataWarehouse* new_dw,
    const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    // Variables
    constSFCXVariable<double> uVel;
    constSFCYVariable<double> vVel;
    constSFCZVariable<double> wVel;
    constCCVariable<Vector> VelCC;
    constSFCXVariable<double> filterRhoU;
    constSFCYVariable<double> filterRhoV;
    constSFCZVariable<double> filterRhoW;
    constCCVariable<double> filterRho;
    constCCVariable<double> filterRhoF;
    constCCVariable<double> filterRhoE;
    constCCVariable<double> filterRhoRF;

    // Get the velocity
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    new_dw->get(uVel,       d_lab->d_uVelocitySPBCLabel,  indx, patch, gaf, 1);
    new_dw->get(vVel,       d_lab->d_vVelocitySPBCLabel,  indx, patch, gaf, 1);
    new_dw->get(wVel,       d_lab->d_wVelocitySPBCLabel,  indx, patch, gaf, 1);
    new_dw->get(VelCC,      d_lab->d_CCVelocityLabel,     indx, patch, gac, 1);
    new_dw->get(filterRhoU, d_lab->d_filterRhoULabel,     indx, patch, gaf, 1);
    new_dw->get(filterRhoV, d_lab->d_filterRhoVLabel,     indx, patch, gaf, 1);
    new_dw->get(filterRhoW, d_lab->d_filterRhoWLabel,     indx, patch, gaf, 1);
    new_dw->get(filterRho,  d_lab->d_filterRhoLabel,      indx, patch, gac, 1);

    // Get the patch and variable details
    // compatible with fortran index
    StencilMatrix<CCVariable<double> > SIJ;    //6 point tensor
    StencilMatrix<CCVariable<double> > filterSIJ;    //6 point tensor
    for (int ii = 0; ii < d_lab->d_symTensorMatl->size(); ii++) {
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
        new_dw->allocateAndPut(SIJ[ii],       d_lab->d_strainTensorCompLabel,       ii, patch);
        new_dw->allocateAndPut(filterSIJ[ii], d_lab->d_filterStrainTensorCompLabel, ii, patch);
      }else {
        new_dw->getModifiable(SIJ[ii],        d_lab->d_strainTensorCompLabel,       ii, patch);
        new_dw->getModifiable(filterSIJ[ii],  d_lab->d_filterStrainTensorCompLabel, ii, patch);
      }
      SIJ[ii].initialize(0.0);
      filterSIJ[ii].initialize(0.0);
    }

    IntVector indexLow = patch->getFortranCellLowIndex();
    IntVector indexHigh = patch->getFortranCellHighIndex();


    Vector Dx = patch->dCell();
    double dx = Dx.x();
    double dy = Dx.y();
    double dz = Dx.z();
    double uep, uwp, unp, usp, utp, ubp;
    double vnp, vsp, vep, vwp, vtp, vbp;
    double wtp, wbp, wep, wwp, wnp, wsp;
    double fuep, fuwp, funp, fusp, futp, fubp;
    double fvnp, fvsp, fvep, fvwp, fvtp, fvbp;
    double fwtp, fwbp, fwep, fwwp, fwnp, fwsp;

    for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;
      IntVector ce = c + IntVector(1,0,0);
      IntVector cw = c + IntVector(-1,0,0);
      IntVector cn = c + IntVector(0,1,0);
      IntVector cs = c + IntVector(0,-1,0);
      IntVector ct = c + IntVector(0,0,1);
      IntVector cb = c + IntVector(0,0,-1);
      IntVector cne = c + IntVector(1,1,0);
      IntVector cnw = c + IntVector(-1,1,0);
      IntVector cse = c + IntVector(1,-1,0);
      IntVector csw = c + IntVector(-1,-1,0);
      IntVector cte = c + IntVector(1,0,1);
      IntVector ctw = c + IntVector(-1,0,1);
      IntVector cbe = c + IntVector(1,0,-1);
      IntVector cbw = c + IntVector(-1,0,-1);
      IntVector ctn = c + IntVector(0,1,1);
      IntVector cbn = c + IntVector(0,1,-1);
      IntVector cts = c + IntVector(0,-1,1);
      IntVector cbs = c + IntVector(0,-1,-1);

      uep = uVel[ce];
      uwp = uVel[c];
      unp = 0.5*VelCC[cn].x();
      usp = 0.5*VelCC[cs].x();
      utp = 0.5*VelCC[ct].x();
      ubp = 0.5*VelCC[cb].x();

      vnp = vVel[cn];
      vsp = vVel[c];
      vep = 0.5*VelCC[ce].y();
      vwp = 0.5*VelCC[cw].y();
      vtp = 0.5*VelCC[ct].y();
      vbp = 0.5*VelCC[cb].y();

      wtp = wVel[ct];
      wbp = wVel[c];
      wep = 0.5*VelCC[ce].z();
      wwp = 0.5*VelCC[cw].z();
      wnp = 0.5*VelCC[cn].z();
      wsp = 0.5*VelCC[cs].z();

      //SIJ: grid strain rate tensor
      (SIJ[0])[c] = (uep-uwp) / dx;
      (SIJ[1])[c] = (vnp-vsp) / dy;
      (SIJ[2])[c] = (wtp-wbp) / dz;
      (SIJ[3])[c] = 0.5*((unp-usp) / dy +
                         (vep-vwp) / dx );
      (SIJ[4])[c] = 0.5*((utp-ubp) / dz +
                         (wep-wwp) / dx );
      (SIJ[5])[c] = 0.5*((vtp-vbp) / dz +
                         (wnp-wsp) / dy );

      fuep = filterRhoU[ce] /
             (0.5 * (filterRho[c] + filterRho[ce]));

      fuwp = filterRhoU[c]/
             (0.5 * (filterRho[c] + filterRho[cw]));

      //note: we have removed the (1/2) from the denom. because
      //we are multiplying by (1/2) for Sij
      funp = ( 0.5 * filterRhoU[cne] /
             ( (filterRho[cn] + filterRho[cne]))
             + 0.5 * filterRhoU[cn] /
             ( (filterRho[cn] + filterRho[IntVector(cnw)])));

      fusp = ( 0.5 * filterRhoU[cse] /
             ( (filterRho[cs] + filterRho[cse]) )
             + 0.5 * filterRhoU[cs] /
             ( (filterRho[cs] + filterRho[csw])));

      futp = ( 0.5 * filterRhoU[cte] /
             ( (filterRho[ct] + filterRho[cte]) )
             + 0.5 * filterRhoU[ct] /
             ( (filterRho[ct] + filterRho[ctw])));

      fubp = ( 0.5 * filterRhoU[cbe] /
             ( ( filterRho[cb] + filterRho[cbe]))
             + 0.5 * filterRhoU[cb] /
             ( (filterRho[cb] + filterRho[cbw])));

      fvnp = filterRhoV[cn] /
             ( 0.5 * (filterRho[c] + filterRho[cn]));

      fvsp = filterRhoV[c] /
             ( 0.5 * (filterRho[c] + filterRho[cs]));

      fvep = ( 0.5 * filterRhoV[cne]/
             ( (filterRho[ce] +filterRho[cne]))
             + 0.5 * filterRhoV[ce]/
             ( (filterRho[ce] + filterRho[cse])));

      fvwp = ( 0.5 * filterRhoV[cnw]/
             ( (filterRho[cw] + filterRho[cnw]))
             + 0.5 * filterRhoV[cw]/
             ( (filterRho[cw] + filterRho[csw])));

      fvtp = ( 0.5 * filterRhoV[ctn] /
             ( (filterRho[ct] + filterRho[ctn]))
             + 0.5 * filterRhoV[ct] /
             ( (filterRho[ct] + filterRho[cts])));

      fvbp = ( 0.5 * filterRhoV[cbn]/
             ( (filterRho[cb] + filterRho[cbn]))
             + 0.5 * filterRhoV[cb] /
             ( (filterRho[cb] + filterRho[cbs])));

      fwtp = filterRhoW[ct] /
             ( 0.5 * (filterRho[c] + filterRho[ct]));

      fwbp = filterRhoW[c] /
             ( 0.5 * (filterRho[c] + filterRho[cb]));

      fwep = ( 0.5 * filterRhoW[cte] /
             ( (filterRho[ce] + filterRho[cte]))
             + 0.5 * filterRhoW[ce] /
             ( (filterRho[ce] + filterRho[cbe])));

      fwwp = ( 0.5 * filterRhoW[ctw] /
             ( (filterRho[cw] + filterRho[ctw]))
             + 0.5 * filterRhoW[cw] /
             ( (filterRho[cw] + filterRho[cbw])));

      fwnp = ( 0.5 * filterRhoW[ctn]/
             ( (filterRho[cn] + filterRho[ctn]))
             + 0.5 * filterRhoW[cn] /
             ( (filterRho[cn] + filterRho[cbn])));

      fwsp = ( 0.5 * filterRhoW[cts]/
             ( (filterRho[cs] + filterRho[cts]))
             + 0.5 * filterRhoW[cs]/
             ( (filterRho[cs] + filterRho[cbs])));

      //calculate the filtered strain rate tensor
      (filterSIJ[0])[c] = (fuep-fuwp)/dx;
      (filterSIJ[1])[c] = (fvnp-fvsp)/dy;
      (filterSIJ[2])[c] = (fwtp-fwbp)/dz;
      (filterSIJ[3])[c] = 0.5*((funp-fusp)/dy + (fvep-fvwp)/dx);
      (filterSIJ[4])[c] = 0.5*((futp-fubp)/dz + (fwep-fwwp)/dx);
      (filterSIJ[5])[c] = 0.5*((fvtp-fvbp)/dz + (fwnp-fwsp)/dy);


    }
  }
}



//****************************************************************************
// Actual recompute
//****************************************************************************
  void
CompDynamicProcedure::reComputeFilterValues(const ProcessorGroup* pc,
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse*,
    DataWarehouse* new_dw,
    const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    // Variables
    constCCVariable<Vector> ccVel;
    constCCVariable<double> den;
    constCCVariable<double> filterRho;
    constCCVariable<double> filterRhoF;
    constCCVariable<double> filterRhoE;
    constCCVariable<double> filterRhoRF;
    constCCVariable<double> filterVolume;
    constCCVariable<int> cellType;
    constCCVariable<double> vol_fraction;


    // Get the velocity and density
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(filterVolume, d_lab->d_filterVolumeLabel, indx, patch, Ghost::None, 0);
    new_dw->get(cellType,     d_lab->d_cellTypeLabel,     indx, patch, gac, 2);
    new_dw->get(vol_fraction, d_lab->d_volFractionLabel,     indx, patch, gac, 1);
    new_dw->get(ccVel,     d_lab->d_CCVelocityLabel, indx, patch, gac, 1);
    new_dw->get(den,       d_lab->d_densityCPLabel,      indx, patch, gac, 1);
    new_dw->get(filterRho, d_lab->d_filterRhoLabel,      indx, patch, gac, 1);


    IntVector idxLo = patch->getExtraCellLowIndex(Arches::ONEGHOSTCELL);
    IntVector idxHi = patch->getExtraCellHighIndex(Arches::ONEGHOSTCELL);

    StencilMatrix<constCCVariable<double> > SIJ; //6 point tensor
    StencilMatrix<constCCVariable<double> > SHATIJ; //6 point tensor
    for (int ii = 0; ii < d_lab->d_symTensorMatl->size(); ii++) {
      new_dw->get(SIJ[ii],    d_lab->d_strainTensorCompLabel,      ii, patch,gac, 1);
      new_dw->get(SHATIJ[ii], d_lab->d_filterStrainTensorCompLabel,ii, patch, gac, 1);
    }

    StencilMatrix<Array3<double> > betaIJ;    //6 point tensor
    StencilMatrix<Array3<double> > betaHATIJ; //6 point tensor
    //  0-> 11, 1->22, 2->33, 3 ->12, 4->13, 5->23
    for (int ii = 0; ii < d_lab->d_symTensorMatl->size(); ii++) {
      betaIJ[ii].resize(idxLo, idxHi);
      betaIJ[ii].initialize(0.0);
      betaHATIJ[ii].resize(idxLo, idxHi);
      betaHATIJ[ii].initialize(0.0);
    }  // allocate stress tensor coeffs

    CCVariable<double> IsImag;
    CCVariable<double> MLI;
    CCVariable<double> MMI;
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->allocateAndPut(IsImag, d_lab->d_strainMagnitudeLabel,   indx, patch);
      new_dw->allocateAndPut(MLI,    d_lab->d_strainMagnitudeMLLabel, indx, patch);
      new_dw->allocateAndPut(MMI,    d_lab->d_strainMagnitudeMMLabel, indx, patch);
    }
    else {
      new_dw->getModifiable(IsImag,
          d_lab->d_strainMagnitudeLabel, indx, patch);
      new_dw->getModifiable(MLI,
          d_lab->d_strainMagnitudeMLLabel, indx, patch);
      new_dw->getModifiable(MMI,
          d_lab->d_strainMagnitudeMMLabel, indx, patch);
    }
    IsImag.initialize(0.0);
    MLI.initialize(0.0);
    MMI.initialize(0.0);

    Array3<double> IsI(idxLo, idxHi); // magnitude of strain rate
    Array3<double> rhoU(idxLo, idxHi);
    Array3<double> rhoV(idxLo, idxHi);
    Array3<double> rhoW(idxLo, idxHi);
    Array3<double> rhoUU(idxLo, idxHi);
    Array3<double> rhoUV(idxLo, idxHi);
    Array3<double> rhoUW(idxLo, idxHi);
    Array3<double> rhoVV(idxLo, idxHi);
    Array3<double> rhoVW(idxLo, idxHi);
    Array3<double> rhoWW(idxLo, idxHi);
    IsI.initialize(0.0);
    rhoU.initialize(0.0);
    rhoV.initialize(0.0);
    rhoW.initialize(0.0);
    rhoUU.initialize(0.0);
    rhoUV.initialize(0.0);
    rhoUW.initialize(0.0);
    rhoVV.initialize(0.0);
    rhoVW.initialize(0.0);
    rhoWW.initialize(0.0);
    Array3<double> rhoFU;
    Array3<double> rhoFV;
    Array3<double> rhoFW;
    Array3<double> rhoEU;
    Array3<double> rhoEV;
    Array3<double> rhoEW;
    Array3<double> rhoRFU;
    Array3<double> rhoRFV;
    Array3<double> rhoRFW;
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    int startZ = idxLo.z();
    if (zminus) startZ++;
    int endZ = idxHi.z();
    if (zplus) endZ--;
    int startY = idxLo.y();
    if (yminus) startY++;
    int endY = idxHi.y();
    if (yplus) endY--;
    int startX = idxLo.x();
    if (xminus) startX++;
    int endX = idxHi.x();
    if (xplus) endX--;

    for (int colZ = startZ; colZ < endZ; colZ ++) {
      for (int colY = startY; colY < endY; colY ++) {
        for (int colX = startX; colX < endX; colX ++) {

          IntVector currCell(colX, colY, colZ);

          // calculate absolute value of the grid strain rate
          // computes for the ghost cells too
          // trace has been neglected
          double sij0 = (SIJ[0])[currCell];
          double sij1 = (SIJ[1])[currCell];
          double sij2 = (SIJ[2])[currCell];
          double sij3 = (SIJ[3])[currCell];
          double sij4 = (SIJ[4])[currCell];
          double sij5 = (SIJ[5])[currCell];
          double isi_cur = sqrt(2.0*(sij0*sij0 + sij1*sij1 + sij2*sij2 +
                2.0*(sij3*sij3 + sij4*sij4 + sij5*sij5)));

          double uvel_cur = ccVel[currCell].x();
          double vvel_cur = ccVel[currCell].y();
          double wvel_cur = ccVel[currCell].z();
          double den_cur = den[currCell];

          IsI[currCell] = isi_cur;

          //    calculate the grid filtered stress tensor, beta
          (betaIJ[0])[currCell] = den_cur*isi_cur*(sij0);
          (betaIJ[1])[currCell] = den_cur*isi_cur*(sij1);
          (betaIJ[2])[currCell] = den_cur*isi_cur*(sij2);
          (betaIJ[3])[currCell] = den_cur*isi_cur*sij3;
          (betaIJ[4])[currCell] = den_cur*isi_cur*sij4;
          (betaIJ[5])[currCell] = den_cur*isi_cur*sij5;

          // required to compute Leonard term
          rhoUU[currCell] = den_cur*uvel_cur*uvel_cur;
          rhoUV[currCell] = den_cur*uvel_cur*vvel_cur;
          rhoUW[currCell] = den_cur*uvel_cur*wvel_cur;
          rhoVV[currCell] = den_cur*vvel_cur*vvel_cur;
          rhoVW[currCell] = den_cur*vvel_cur*wvel_cur;
          rhoWW[currCell] = den_cur*wvel_cur*wvel_cur;
          rhoU[currCell] = den_cur*uvel_cur;
          rhoV[currCell] = den_cur*vvel_cur;
          rhoW[currCell] = den_cur*wvel_cur;

        }
      }
    }

    Array3<double> filterRhoUU(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRhoUU.initialize(0.0);
    Array3<double> filterRhoUV(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRhoUV.initialize(0.0);
    Array3<double> filterRhoUW(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRhoUW.initialize(0.0);
    Array3<double> filterRhoVV(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRhoVV.initialize(0.0);
    Array3<double> filterRhoVW(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRhoVW.initialize(0.0);
    Array3<double> filterRhoWW(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRhoWW.initialize(0.0);
    Array3<double> filterRhoU(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRhoU.initialize(0.0);
    Array3<double> filterRhoV(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRhoV.initialize(0.0);
    Array3<double> filterRhoW(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRhoW.initialize(0.0);

    Array3<double> filterRhoFU;
    Array3<double> filterRhoFV;
    Array3<double> filterRhoFW;
    Array3<double> filterRhoEU;
    Array3<double> filterRhoEV;
    Array3<double> filterRhoEW;
    Array3<double> filterRhoRFU;
    Array3<double> filterRhoRFV;
    Array3<double> filterRhoRFW;

    IntVector indexLow = patch->getFortranCellLowIndex();
    IntVector indexHigh = patch->getFortranCellHighIndex();

    Timers::Simple timer;
    timer.start();

    d_filter->applyFilter<Array3<double> >(pc, patch, rhoU,   filterVolume, vol_fraction, filterRhoU);
    d_filter->applyFilter<Array3<double> >(pc, patch, rhoV,   filterVolume, vol_fraction, filterRhoV);
    d_filter->applyFilter<Array3<double> >(pc, patch, rhoW,   filterVolume, vol_fraction, filterRhoW);
    d_filter->applyFilter<Array3<double> >(pc, patch, rhoUU,  filterVolume, vol_fraction,  filterRhoUU);
    d_filter->applyFilter<Array3<double> >(pc, patch, rhoUV,  filterVolume, vol_fraction,  filterRhoUV);
    d_filter->applyFilter<Array3<double> >(pc, patch, rhoUW,  filterVolume, vol_fraction,  filterRhoUW);
    d_filter->applyFilter<Array3<double> >(pc, patch, rhoVV,  filterVolume, vol_fraction,  filterRhoVV);
    d_filter->applyFilter<Array3<double> >(pc, patch, rhoVW,  filterVolume, vol_fraction,  filterRhoVW);
    d_filter->applyFilter<Array3<double> >(pc, patch, rhoWW,  filterVolume, vol_fraction,  filterRhoWW);

    for (int ii = 0; ii < d_lab->d_symTensorMatl->size(); ii++) {
      d_filter->applyFilter<Array3<double> >(pc, patch, betaIJ[ii], filterVolume, vol_fraction, betaHATIJ[ii]);
    }

    string msg = "Time for the Filter operation in Turbulence Model: (patch: ";
    proc0cerr << msg << p << ") " << timer().seconds() << " seconds\n";

    Vector Dx = patch->dCell();
    double fhat = 3.0;
    double filter = pow(Dx.x()*Dx.y()*Dx.z(),1.0/3.0);
    double filter2 = filter*filter;

    for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;

      double shatij0 = (SHATIJ[0])[c];
      double shatij1 = (SHATIJ[1])[c];
      double shatij2 = (SHATIJ[2])[c];
      double shatij3 = (SHATIJ[3])[c];
      double shatij4 = (SHATIJ[4])[c];
      double shatij5 = (SHATIJ[5])[c];
      double IshatIcur = sqrt( 2.0*(shatij0*shatij0 + shatij1*shatij1 + shatij2*shatij2
                             + 2.0*(shatij3*shatij3 + shatij4*shatij4 + shatij5*shatij5)));
      double filterDencur = filterRho[c];

      //ignoring the trace
      IsImag[c] = IsI[c];

      double MIJ0cur = 2.0 * filter2 *
                       ( (betaHATIJ[0])[c]-
                         2.0*fhat*filterDencur*IshatIcur*(shatij0));
      double MIJ1cur = 2.0 * filter2 *
                       ( (betaHATIJ[1])[c]-
                         2.0*fhat*filterDencur*IshatIcur*(shatij1));
      double MIJ2cur = 2.0 * filter2 *
                       ( (betaHATIJ[2])[c]-
                         2.0*fhat*filterDencur*IshatIcur*(shatij2));
      double MIJ3cur = 2.0 * filter2 *
                       ( (betaHATIJ[3])[c]-
                         2.0*fhat*filterDencur*IshatIcur*shatij3);
      double MIJ4cur = 2.0 * filter2 *
                       ( (betaHATIJ[4])[c]-
                         2.0*fhat*filterDencur*IshatIcur*shatij4);
      double MIJ5cur = 2.0 * filter2 *
                       ( (betaHATIJ[5])[c]-
                         2.0*fhat*filterDencur*IshatIcur*shatij5);


      // compute Leonard stress tensor
      double filterRhoUcur = filterRhoU[c];
      double filterRhoVcur = filterRhoV[c];
      double filterRhoWcur = filterRhoW[c];
      double LIJ0cur = filterRhoUU[c] -
                       filterRhoUcur*filterRhoUcur/filterDencur;
      double LIJ1cur = filterRhoVV[c] -
                       filterRhoVcur*filterRhoVcur/filterDencur;
      double LIJ2cur = filterRhoWW[c] -
                       filterRhoWcur*filterRhoWcur/filterDencur;
      double LIJ3cur = filterRhoUV[c] -
                       filterRhoUcur*filterRhoVcur/filterDencur;
      double LIJ4cur = filterRhoUW[c] -
                       filterRhoUcur*filterRhoWcur/filterDencur;
      double LIJ5cur = filterRhoVW[c] -
                       filterRhoVcur*filterRhoWcur/filterDencur;

      //Again, ignoring the trace
      LIJ0cur = LIJ0cur;
      LIJ1cur = LIJ1cur;
      LIJ2cur = LIJ2cur;

      // compute the magnitude of ML and MM
      MLI[c] = MIJ0cur * LIJ0cur +
                      MIJ1cur * LIJ1cur +
                      MIJ2cur * LIJ2cur +
                      2.0 * (
                      MIJ3cur * LIJ3cur +
                      MIJ4cur * LIJ4cur +
                      MIJ5cur * LIJ5cur
                      );

      // calculate absolute value of the grid strain rate
      MMI[c] = MIJ0cur * MIJ0cur +
                      MIJ1cur * MIJ1cur +
                      MIJ2cur * MIJ2cur +
                      2.0 * (
                      MIJ3cur * MIJ3cur +
                      MIJ4cur * MIJ4cur +
                      MIJ5cur * MIJ5cur
                      );

    }
  }
}



//______________________________________________________________________
//
  void
CompDynamicProcedure::reComputeSmagCoeff(const ProcessorGroup* pc,
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse*,
    DataWarehouse* new_dw,
    const TimeIntegratorLabel* timelabels)
{

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    // Variables
    constCCVariable<double> IsI;
    constCCVariable<double> MLI;
    constCCVariable<double> MMI;
    CCVariable<double> Cs;
    constCCVariable<double> den;
    constCCVariable<int> cellType;
    constCCVariable<double> vol_fraction;
    CCVariable<double> viscosity;
    CCVariable<double> turbViscosity;
    CCVariable<double> dissipation_rate;
    constCCVariable<double> filterVolume;
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->allocateAndPut(Cs, d_lab->d_CsLabel, indx, patch);
    }
    else {
      new_dw->getModifiable(Cs, d_lab->d_CsLabel, indx, patch);
    }
    Cs.initialize(0.0);

    new_dw->getModifiable(viscosity, d_lab->d_viscosityCTSLabel, indx, patch);
    new_dw->getModifiable(turbViscosity, d_lab->d_turbViscosLabel, indx, patch);

    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;

    new_dw->get(IsI, d_lab->d_strainMagnitudeLabel,   indx, patch,   gn, 0);
    new_dw->getModifiable( dissipation_rate, d_dissipationRateLabel, indx, patch );
    new_dw->get(MLI, d_lab->d_strainMagnitudeMLLabel, indx, patch, gac, 1);
    new_dw->get(MMI, d_lab->d_strainMagnitudeMMLabel, indx, patch, gac, 1);
    new_dw->get(filterVolume, d_lab->d_filterVolumeLabel, indx, patch, gn, 0);
    new_dw->get(den, d_lab->d_densityCPLabel, indx, patch,gac, 1);
    new_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch, gac, 1);
    new_dw->get(vol_fraction, d_lab->d_volFractionLabel, indx, patch, gac, 1);

    // get physical constants
    double viscos; // molecular viscosity
    viscos = d_physicalConsts->getMolecularViscosity();

    // compute test filtered velocities, density and product
    Array3<double> MLHatI(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex()); // magnitude of strain rate
    MLHatI.initialize(0.0);
    Array3<double> MMHatI(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex()); // magnitude of test filter strain rate
    MLHatI.initialize(0.0);
    IntVector indexLow = patch->getFortranCellLowIndex();
    IntVector indexHigh = patch->getFortranCellHighIndex();

    d_filter->applyFilter<constCCVariable<double> >(pc, patch, MLI, filterVolume, vol_fraction, MLHatI);
    d_filter->applyFilter<constCCVariable<double> >(pc, patch, MMI, filterVolume, vol_fraction, MMHatI);

    CCVariable<double> tempCs;
    tempCs.allocate(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    tempCs.initialize(0.0);

    Vector Dx = patch->dCell();
    double filter = pow(Dx.x()*Dx.y()*Dx.z(),1.0/3.0);
    double filter2 = filter*filter;

    if ( d_filter_cs_squared ){
      //FILTER CS^2
      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        double value;

        if ( MMHatI[c] < 1.0e-10 || MLHatI[c] < 1.0e-7 ){
          value = 0.0;
        } else {
          value = MLHatI[c] / MMHatI[c];
        }

        tempCs[c] = value;
      }

      d_filter->applyFilter<Array3<double> >(pc, patch, tempCs, filterVolume, vol_fraction, Cs );

      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){

        IntVector c = *iter;

        Cs[c] = Min(tempCs[c],10.0);
        viscosity[c] =  ( Cs[c] * filter2 * IsI[c] * den[c] + viscos ) * vol_fraction[c];
        turbViscosity[c] = viscosity[c] - viscos;
        //estimate the dissipation rate
        // see: https://gitlab.chpc.utah.edu/ccmscteam/stokes_analysis_for_les
        dissipation_rate[c] = (IsI[c]*IsI[c])/(2. * den[c] + 1.e-16) * turbViscosity[c] * vol_fraction[c];

      }

    } else {

      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){

        IntVector c = *iter;
        double value;

        if ( MMHatI[c] < 1.0e-10 || MLHatI[c] < 1.0e-7 ){
          value = 0.0;
        } else {
          value = MLHatI[c] / MMHatI[c];
        }

        Cs[c] = Min(value,10.0);
        viscosity[c] =  ( Cs[c] * filter2 * IsI[c] * den[c] + viscos ) * vol_fraction[c];
        turbViscosity[c] = viscosity[c] - viscos;

        //estimate the dissipation rate
        // see: https://gitlab.chpc.utah.edu/ccmscteam/stokes_analysis_for_les
        dissipation_rate[c] = (IsI[c]*IsI[c])/( 2. * den[c] + 1.e-16) * turbViscosity[c] * vol_fraction[c];

      }
    }

    apply_zero_neumann( patch, turbViscosity, viscosity, vol_fraction );

  }
}
