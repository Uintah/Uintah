/*

   The MIT License

   Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


//----- CompDynamicProcedure.cc --------------------------------------------------

#include <TauProfilerForSCIRun.h>
#include <CCA/Components/Arches/CompDynamicProcedure.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/StencilMatrix.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Parallel/ProcessorGroup.h>

#include <Core/Thread/Time.h>

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
  ProblemSpecP db = params->findBlock("Turbulence");
  if (d_calcVariance) {

    proc0cout << "Scale similarity type model with Favre filter will be used"<<endl;
    proc0cout << "to model variance" << endl;

    db->require("variance_coefficient",d_CFVar); // const reqd by variance eqn
    db->getWithDefault("mixture_fraction_label",d_mix_frac_label_name, "scalarSP"); 
    d_mf_label = VarLabel::find( d_mix_frac_label_name );
    proc0cout << "Using " << *d_mf_label << " to compute scalar variance." << endl;

    db->getWithDefault("filter_variance_limit_scalar",d_filter_var_limit_scalar,true);
    if( d_filter_var_limit_scalar) {
      proc0cout << "Scalar for variance limit will be Favre filtered" << endl;
    }
    else {
      proc0cout << "WARNING! Scalar for variance limit will NOT be filtered" << endl;
      proc0cout << "possibly causing high variance values" << endl;
    }
  }

  // actually, Shmidt number, not Prandtl number
  d_turbPrNo = 0.0;
  if (db->findBlock("turbulentPrandtlNumber")) {
    db->getWithDefault("turbulentPrandtlNumber",d_turbPrNo,0.4);
  }

  db->getWithDefault("dynamicScalarModel",d_dynScalarModel,false);
  if (d_dynScalarModel){
    d_turbPrNo = 1.0; 
  }

  db->getWithDefault("filter_cs_squared",d_filter_cs_squared,false);

}

//****************************************************************************
// Schedule recomputation of the turbulence sub model 
//****************************************************************************
  void 
CompDynamicProcedure::sched_reComputeTurbSubmodel(SchedulerP& sched, 
    const PatchSet* patches,
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


    // Requires
    tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,     gac, 2);
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,      gac, 2);
    tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);
    tsk->requires(Task::NewDW, d_lab->d_filterVolumeLabel, gn); 

    //__________________________________
    if (d_dynScalarModel) {
      if (d_calcScalar){
        tsk->requires(Task::NewDW, d_mf_label,      gac, 1);
      }
      if (d_calcEnthalpy){
        tsk->requires(Task::NewDW, d_lab->d_enthalpySPLabel,    gac, 1);
      }
    }

    int mmWallID = d_boundaryCondition->getMMWallId();
    if (mmWallID > 0)
      tsk->requires(Task::NewDW, timelabels->ref_density);

    // Computes
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->computes(d_lab->d_filterRhoULabel);
      tsk->computes(d_lab->d_filterRhoVLabel);
      tsk->computes(d_lab->d_filterRhoWLabel);
      tsk->computes(d_lab->d_filterRhoLabel);
      if (d_dynScalarModel) {
        if (d_calcScalar)
          tsk->computes(d_lab->d_filterRhoFLabel);
        if (d_calcEnthalpy)
          tsk->computes(d_lab->d_filterRhoELabel);
      }
    }
    else {
      tsk->modifies(d_lab->d_filterRhoULabel);
      tsk->modifies(d_lab->d_filterRhoVLabel);
      tsk->modifies(d_lab->d_filterRhoWLabel);
      tsk->modifies(d_lab->d_filterRhoLabel);
      if (d_dynScalarModel) {
        if (d_calcScalar)
          tsk->modifies(d_lab->d_filterRhoFLabel);
        if (d_calcEnthalpy)
          tsk->modifies(d_lab->d_filterRhoELabel);
      }
    }  

    sched->addTask(tsk, patches, matls);
  }
  //__________________________________
  {
    string taskname =  "CompDynamicProcedure::reComputeStrainRateTensors" +
      timelabels->integrator_step_name;
    Task* tsk = scinew Task(taskname, this,
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
    tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);

    if (d_dynScalarModel) {
      if (d_calcScalar) {
        tsk->requires(Task::NewDW, d_mf_label,   gac, 1);
        tsk->requires(Task::NewDW, d_lab->d_filterRhoFLabel, gac, 1);
      }
      if (d_calcEnthalpy) {
        tsk->requires(Task::NewDW, d_lab->d_enthalpySPLabel, gac, 1);
        tsk->requires(Task::NewDW, d_lab->d_filterRhoELabel, gac, 1);
      }
    }

    // Computes
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->computes(d_lab->d_strainTensorCompLabel,
          d_lab->d_symTensorMatl, oams);
      tsk->computes(d_lab->d_filterStrainTensorCompLabel,
          d_lab->d_symTensorMatl, oams);
      if (d_dynScalarModel) {
        if (d_calcScalar) {
          tsk->computes(d_lab->d_scalarGradientCompLabel,
              d_lab->d_vectorMatl, oams);
          tsk->computes(d_lab->d_filterScalarGradientCompLabel,
              d_lab->d_vectorMatl, oams);
        }
        if (d_calcEnthalpy) {
          tsk->computes(d_lab->d_enthalpyGradientCompLabel,
              d_lab->d_vectorMatl, oams);
          tsk->computes(d_lab->d_filterEnthalpyGradientCompLabel,
              d_lab->d_vectorMatl, oams);
        }
      }  
    }
    else {
      tsk->modifies(d_lab->d_strainTensorCompLabel,
          d_lab->d_symTensorMatl, oams);
      tsk->modifies(d_lab->d_filterStrainTensorCompLabel,
          d_lab->d_symTensorMatl, oams);
      if (d_dynScalarModel) {
        if (d_calcScalar) {
          tsk->modifies(d_lab->d_scalarGradientCompLabel,
              d_lab->d_vectorMatl, oams);
          tsk->modifies(d_lab->d_filterScalarGradientCompLabel,
              d_lab->d_vectorMatl, oams);
        }
        if (d_calcEnthalpy) {
          tsk->modifies(d_lab->d_enthalpyGradientCompLabel,
              d_lab->d_vectorMatl, oams);
          tsk->modifies(d_lab->d_filterEnthalpyGradientCompLabel,
              d_lab->d_vectorMatl, oams);
        }
      }  
    }  
    sched->addTask(tsk, patches, matls);
  }
  //__________________________________
  {
    string taskname =  "CompDynamicProcedure::reComputeFilterValues" +
      timelabels->integrator_step_name;
    Task* tsk = scinew Task(taskname, this,
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
    tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);
    tsk->requires(Task::NewDW, d_lab->d_filterVolumeLabel, gn); 
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, gac, 2);

    tsk->requires(Task::NewDW, d_lab->d_strainTensorCompLabel,
        d_lab->d_symTensorMatl, oams,gac, 1);

    tsk->requires(Task::NewDW, d_lab->d_filterStrainTensorCompLabel,
        d_lab->d_symTensorMatl, oams,gac, 1);

    //__________________________________
    if (d_dynScalarModel) {
      if (d_calcScalar) {
        tsk->requires(Task::NewDW, d_mf_label,   gac, 1);
        tsk->requires(Task::NewDW, d_lab->d_filterRhoFLabel, gac, 1);
        tsk->requires(Task::NewDW, d_lab->d_scalarGradientCompLabel,
            d_lab->d_vectorMatl, oams, gac, 1);
        tsk->requires(Task::NewDW, d_lab->d_filterScalarGradientCompLabel,
            d_lab->d_vectorMatl, oams, gac, 1);
      }
      if (d_calcEnthalpy) {
        tsk->requires(Task::NewDW, d_lab->d_enthalpySPLabel,  gac, 1);
        tsk->requires(Task::NewDW, d_lab->d_filterRhoELabel,  gac, 1);
        tsk->requires(Task::NewDW, d_lab->d_enthalpyGradientCompLabel,
            d_lab->d_vectorMatl, oams, gac, 1);
        tsk->requires(Task::NewDW, d_lab->d_filterEnthalpyGradientCompLabel,
            d_lab->d_vectorMatl, oams, gac, 1);
      }
    }  

    // Computes
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->computes(d_lab->d_strainMagnitudeLabel);
      tsk->computes(d_lab->d_strainMagnitudeMLLabel);
      tsk->computes(d_lab->d_strainMagnitudeMMLabel);
      if (d_dynScalarModel) {
        if (d_calcScalar) {
          tsk->computes(d_lab->d_scalarNumeratorLabel);
          tsk->computes(d_lab->d_scalarDenominatorLabel);
        }
        if (d_calcEnthalpy) {
          tsk->computes(d_lab->d_enthalpyNumeratorLabel);
          tsk->computes(d_lab->d_enthalpyDenominatorLabel);
        }
      }      
    }
    else {
      tsk->modifies(d_lab->d_strainMagnitudeLabel);
      tsk->modifies(d_lab->d_strainMagnitudeMLLabel);
      tsk->modifies(d_lab->d_strainMagnitudeMMLabel);
      if (d_dynScalarModel) {
        if (d_calcScalar) {
          tsk->modifies(d_lab->d_scalarNumeratorLabel);
          tsk->modifies(d_lab->d_scalarDenominatorLabel);
        }
        if (d_calcEnthalpy) {
          tsk->modifies(d_lab->d_enthalpyNumeratorLabel);
          tsk->modifies(d_lab->d_enthalpyDenominatorLabel);
        }
      }      
    }

    sched->addTask(tsk, patches, matls);
  }
  //__________________________________
  {
    string taskname =  "CompDynamicProcedure::reComputeSmagCoeff" +
      timelabels->integrator_step_name;
    Task* tsk = scinew Task(taskname, this,
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
    tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);
    tsk->requires(Task::NewDW, d_lab->d_filterVolumeLabel, gn); 
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, gac, 1);

    if (d_dynScalarModel) {
      if (d_calcScalar) {
        tsk->requires(Task::NewDW, d_lab->d_scalarNumeratorLabel,    gac, 1);
        tsk->requires(Task::NewDW, d_lab->d_scalarDenominatorLabel,  gac, 1);
      }
      if (d_calcEnthalpy) {
        tsk->requires(Task::NewDW, d_lab->d_enthalpyNumeratorLabel,     gac, 1);
        tsk->requires(Task::NewDW, d_lab->d_enthalpyDenominatorLabel,   gac, 1);
      }
    }      

    // for multimaterial
    if (d_MAlab){
      tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, gn, 0);
    }

    // Computes
    tsk->modifies(d_lab->d_viscosityCTSLabel);
    tsk->modifies(d_lab->d_turbViscosLabel); 
    if (d_dynScalarModel) {
      if (d_calcScalar) {
        tsk->modifies(d_lab->d_scalarDiffusivityLabel);
      }
      if (d_calcEnthalpy) {
        tsk->modifies(d_lab->d_enthalpyDiffusivityLabel);
      }
    }      

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->computes(d_lab->d_CsLabel);
      if (d_dynScalarModel) {
        if (d_calcScalar) {
          tsk->computes(d_lab->d_ShFLabel);
        }
        if (d_calcEnthalpy) {
          tsk->computes(d_lab->d_ShELabel);
        }
      }      
    }
    else {
      tsk->modifies(d_lab->d_CsLabel);
      if (d_dynScalarModel) {
        if (d_calcScalar) {
          tsk->modifies(d_lab->d_ShFLabel);
        }
        if (d_calcEnthalpy) {
          tsk->modifies(d_lab->d_ShELabel);
        }
      }      
    }

    sched->addTask(tsk, patches, matls);
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
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constSFCXVariable<double> uVel;
    constSFCYVariable<double> vVel;
    constSFCZVariable<double> wVel;
    constCCVariable<double> density;
    constCCVariable<double> scalar;
    constCCVariable<double> enthalpy;
    constCCVariable<int> cellType;
    constCCVariable<double> filterVolume; 

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    // Get the velocity
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    new_dw->get(uVel, d_lab->d_uVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(vVel, d_lab->d_vVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(wVel, d_lab->d_wVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(density, d_lab->d_densityCPLabel,  indx, patch, gac, 2);
    new_dw->get(filterVolume, d_lab->d_filterVolumeLabel, indx, patch, Ghost::None, 0); 

    if (d_dynScalarModel) {
      if (d_calcScalar){
        new_dw->get(scalar,       d_mf_label,     indx, patch, gac, 1);
      }
      if (d_calcEnthalpy){
        new_dw->get(enthalpy,     d_lab->d_enthalpySPLabel,   indx, patch, gac, 1);
      }
    }

    new_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch, gac, 2);


    SFCXVariable<double> filterRhoU;
    SFCYVariable<double> filterRhoV;
    SFCZVariable<double> filterRhoW;
    CCVariable<double> filterRho;
    CCVariable<double> filterRhoF;
    CCVariable<double> filterRhoE;
    CCVariable<double> filterRhoRF;
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->allocateAndPut(filterRhoU, d_lab->d_filterRhoULabel, indx, patch);
      new_dw->allocateAndPut(filterRhoV, d_lab->d_filterRhoVLabel, indx, patch);
      new_dw->allocateAndPut(filterRhoW, d_lab->d_filterRhoWLabel, indx, patch);
      new_dw->allocateAndPut(filterRho,  d_lab->d_filterRhoLabel,  indx, patch);

      if (d_dynScalarModel) {
        if (d_calcScalar)
          new_dw->allocateAndPut(filterRhoF,  d_lab->d_filterRhoFLabel, indx, patch);
        if (d_calcEnthalpy)
          new_dw->allocateAndPut(filterRhoE,  d_lab->d_filterRhoELabel, indx, patch);
      }
    }
    else {
      new_dw->getModifiable(filterRhoU, d_lab->d_filterRhoULabel, indx, patch);
      new_dw->getModifiable(filterRhoV, d_lab->d_filterRhoVLabel, indx, patch);
      new_dw->getModifiable(filterRhoW, d_lab->d_filterRhoWLabel, indx, patch);
      new_dw->getModifiable(filterRho,  d_lab->d_filterRhoLabel,  indx, patch);
      if (d_dynScalarModel) {
        if (d_calcScalar)
          new_dw->getModifiable(filterRhoF, d_lab->d_filterRhoFLabel, indx, patch);
        if (d_calcEnthalpy)
          new_dw->getModifiable(filterRhoE, d_lab->d_filterRhoELabel, indx, patch);
      }
    }
    filterRhoU.initialize(0.0);
    filterRhoV.initialize(0.0);
    filterRhoW.initialize(0.0);
    filterRho.initialize(0.0);
    if (d_dynScalarModel) {
      if (d_calcScalar)
        filterRhoF.initialize(0.0);
      if (d_calcEnthalpy)
        filterRhoE.initialize(0.0);
    }

    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

    IntVector indexLowU = patch->getSFCXFORTLowIndex__Old();
    IntVector indexHighU = patch->getSFCXFORTHighIndex__Old();
    IntVector indexLowV = patch->getSFCYFORTLowIndex__Old();
    IntVector indexHighV = patch->getSFCYFORTHighIndex__Old();
    IntVector indexLowW = patch->getSFCZFORTLowIndex__Old();
    IntVector indexHighW = patch->getSFCZFORTHighIndex__Old();

    if (xminus) indexLowU -= IntVector(1,0,0); 
    if (yminus) indexLowV -= IntVector(0,1,0); 
    if (zminus) indexLowW -= IntVector(0,0,1); 
    if (xplus) indexHighU += IntVector(1,0,0); 
    if (yplus) indexHighV += IntVector(0,1,0); 
    if (zplus) indexHighW += IntVector(0,0,1); 

    int flowID = d_boundaryCondition->flowCellType();
    int mmWallID = d_boundaryCondition->getMMWallId();
    for (int colZ = indexLowU.z(); colZ <= indexHighU.z(); colZ ++) {
      for (int colY = indexLowU.y(); colY <= indexHighU.y(); colY ++) {
        for (int colX = indexLowU.x(); colX <= indexHighU.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          IntVector shift(0,0,0);
          if ((xplus)&&((colX == indexHighU.x())||(colX == indexHighU.x()-1)))
            shift = IntVector(-1,0,0);
          int bndry_count=0;
          if  (!(cellType[currCell + shift - IntVector(1,0,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift + IntVector(1,0,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift - IntVector(0,1,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift + IntVector(0,1,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift - IntVector(0,0,1)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift + IntVector(0,0,1)] == flowID))
            bndry_count++;
          bool corner = (bndry_count==3);
          double totalVol = 0.0;
          if ((cellType[currCell+shift] == flowID)&&
              (cellType[currCell+shift-IntVector(1,0,0)] != mmWallID)) {
            for (int kk = -1; kk <= 1; kk ++) {
              for (int jj = -1; jj <= 1; jj ++) {
                for (int ii = -1; ii <= 1; ii ++) {
                  IntVector filterCell = IntVector(colX+ii,colY+jj,colZ+kk);
                  // on the boundary
                  if (cellType[filterCell+shift] != flowID) {
                    // intrusion
                    if (filterCell+shift == currCell+shift) {
                      // do nothing here, assuming intrusion velocity is 0
                    }
                  }
                  // inside the domain
                  else
                    if (cellType[filterCell+shift-IntVector(1,0,0)] != mmWallID) {
                      double vol = cellinfo->sewu[colX+ii]*
                        cellinfo->sns[colY+jj]*
                        cellinfo->stb[colZ+kk];
                      if (!(corner)) vol *= (1.0-0.5*abs(ii))*
                        (1.0-0.5*abs(jj))*(1.0-0.5*abs(kk));
                      filterRhoU[currCell] += vol*uVel[filterCell]*
                        0.5*(density[filterCell]+
                            density[filterCell-IntVector(1,0,0)]);
                      totalVol += vol;
                    }
                }
              }
            }
            filterRhoU[currCell] /= totalVol;
          }
        }
      }
    }
    // assuming SFCY still stored with z being outer index and x inner index
    for (int colZ = indexLowV.z(); colZ <= indexHighV.z(); colZ ++) {
      for (int colY = indexLowV.y(); colY <= indexHighV.y(); colY ++) {
        for (int colX = indexLowV.x(); colX <= indexHighV.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          IntVector shift(0,0,0);
          if ((yplus)&&((colY == indexHighV.y())||(colY == indexHighV.y()-1)))
            shift = IntVector(0,-1,0);
          int bndry_count=0;
          if  (!(cellType[currCell + shift - IntVector(1,0,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift + IntVector(1,0,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift - IntVector(0,1,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift + IntVector(0,1,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift - IntVector(0,0,1)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift + IntVector(0,0,1)] == flowID))
            bndry_count++;
          bool corner = (bndry_count==3);
          double totalVol = 0.0;
          if ((cellType[currCell+shift] == flowID)&&
              (cellType[currCell+shift-IntVector(0,1,0)] != mmWallID)) {
            for (int kk = -1; kk <= 1; kk ++) {
              for (int jj = -1; jj <= 1; jj ++) {
                for (int ii = -1; ii <= 1; ii ++) {
                  IntVector filterCell = IntVector(colX+ii,colY+jj,colZ+kk);
                  // on the boundary
                  if (cellType[filterCell+shift] != flowID) {
                    // intrusion
                    if (filterCell+shift == currCell+shift) {
                      // do nothing here, assuming intrusion velocity is 0
                    }
                  }
                  // inside the domain
                  else
                    if (cellType[filterCell+shift-IntVector(0,1,0)] != mmWallID) {
                      double vol = cellinfo->sew[colX+ii]*
                        cellinfo->snsv[colY+jj]*
                        cellinfo->stb[colZ+kk];
                      if (!(corner)) vol *= (1.0-0.5*abs(ii))*
                        (1.0-0.5*abs(jj))*(1.0-0.5*abs(kk));
                      filterRhoV[currCell] += vol*vVel[filterCell]*
                        0.5*(density[filterCell]+
                            density[filterCell-IntVector(0,1,0)]);
                      totalVol += vol;
                    }
                }
              }
            }

            filterRhoV[currCell] /= totalVol;
          }
        }
      }
    }
    // assuming SFCZ still stored with z being outer index and x inner index
    for (int colZ = indexLowW.z(); colZ <= indexHighW.z(); colZ ++) {
      for (int colY = indexLowW.y(); colY <= indexHighW.y(); colY ++) {
        for (int colX = indexLowW.x(); colX <= indexHighW.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          IntVector shift(0,0,0);
          if ((zplus)&&((colZ == indexHighW.z())||(colZ == indexHighW.z()-1))) 
            shift = IntVector(0,0,-1);
          int bndry_count=0;
          if  (!(cellType[currCell + shift - IntVector(1,0,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift + IntVector(1,0,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift - IntVector(0,1,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift + IntVector(0,1,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift - IntVector(0,0,1)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift + IntVector(0,0,1)] == flowID))
            bndry_count++;
          bool corner = (bndry_count==3);
          double totalVol = 0.0;
          if ((cellType[currCell+shift] == flowID)&&
              (cellType[currCell+shift-IntVector(0,0,1)] != mmWallID)) {
            for (int kk = -1; kk <= 1; kk ++) {
              for (int jj = -1; jj <= 1; jj ++) {
                for (int ii = -1; ii <= 1; ii ++) {
                  IntVector filterCell = IntVector(colX+ii,colY+jj,colZ+kk);
                  // on the boundary
                  if (cellType[filterCell+shift] != flowID) {
                    // intrusion
                    if (filterCell+shift == currCell+shift) {
                      // do nothing here, assuming intrusion velocity is 0
                    }
                  }
                  // inside the domain
                  else
                    if (cellType[filterCell+shift-IntVector(0,0,1)] != mmWallID) {
                      double vol = cellinfo->sew[colX+ii]*
                        cellinfo->sns[colY+jj]*
                        cellinfo->stbw[colZ+kk];
                      if (!(corner)) vol *= (1.0-0.5*abs(ii))*
                        (1.0-0.5*abs(jj))*(1.0-0.5*abs(kk));
                      filterRhoW[currCell] += vol*wVel[filterCell]*
                        0.5*(density[filterCell]+
                            density[filterCell-IntVector(0,0,1)]);
                      totalVol += vol;
                    }
                }
              }
            }

            filterRhoW[currCell] /= totalVol;
          }
        }
      }
    }

    int ngc = 1;
    IntVector idxLo = patch->getExtraCellLowIndex(ngc);
    IntVector idxHi = patch->getExtraCellHighIndex(ngc);
    Array3<double> rhoF(idxLo, idxHi);
    Array3<double> rhoE(idxLo, idxHi);
    Array3<double> rhoRF(idxLo, idxHi);
    rhoF.initialize(0.0);
    rhoE.initialize(0.0);
    rhoRF.initialize(0.0);

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

          if (d_dynScalarModel) {
            if (d_calcScalar)
              rhoF[currCell] = density[currCell]*scalar[currCell];
            if (d_calcEnthalpy)
              rhoE[currCell] = density[currCell]*enthalpy[currCell];
          }
        }
      }
    }

    filterRho.copy(density, patch->getExtraCellLowIndex(),
        patch->getExtraCellHighIndex());
#ifdef PetscFilter
    d_filter->applyFilter_noPetsc<constCCVariable<double> >(pc, patch, density, filterVolume, cellType, filterRho); 
#endif 

    // making filterRho nonzero 
    sum_vartype den_ref_var;
    if (mmWallID > 0) {
      new_dw->get(den_ref_var, timelabels->ref_density);

      idxLo = patch->getExtraCellLowIndex();
      idxHi = patch->getExtraCellHighIndex();

      for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
        for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
          for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {

            IntVector currCell(colX, colY, colZ);

            if (filterRho[currCell] < 1.0e-15) 
              filterRho[currCell]=den_ref_var;

          }
        }
      }
    }
    if (d_dynScalarModel) {
#ifdef PetscFilter
      if (d_calcScalar)
        d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoF, filterVolume, cellType, filterRhoF);
      if (d_calcEnthalpy)
        d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoE, filterVolume, cellType, filterRhoE);
#endif
    }
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
    constCCVariable<double> scalar;
    constCCVariable<double> enthalpy;
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
    if (d_dynScalarModel) {
      if (d_calcScalar) {
        new_dw->get(scalar,     d_mf_label,   indx, patch, gac, 1);
        new_dw->get(filterRhoF, d_lab->d_filterRhoFLabel, indx, patch, gac, 1);
      }
      if (d_calcEnthalpy) {
        new_dw->get(enthalpy,   d_lab->d_enthalpySPLabel, indx, patch, gac, 1);
        new_dw->get(filterRhoE, d_lab->d_filterRhoELabel, indx, patch, gac, 1);
      }
    }

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();


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
    StencilMatrix<CCVariable<double> > scalarGrad;    //vector
    StencilMatrix<CCVariable<double> > filterScalarGrad;    //vector
    StencilMatrix<CCVariable<double> > enthalpyGrad;    //vector
    StencilMatrix<CCVariable<double> > filterEnthalpyGrad;    //vector
    for (int ii = 0; ii < d_lab->d_vectorMatl->size(); ii++) {
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
        if (d_dynScalarModel) {
          if (d_calcScalar) {
            new_dw->allocateAndPut(scalarGrad[ii],      d_lab->d_scalarGradientCompLabel,       ii, patch);
            new_dw->allocateAndPut(filterScalarGrad[ii],d_lab->d_filterScalarGradientCompLabel, ii, patch);
          }
          if (d_calcEnthalpy) {
            new_dw->allocateAndPut(enthalpyGrad[ii],       d_lab->d_enthalpyGradientCompLabel,       ii, patch);
            new_dw->allocateAndPut(filterEnthalpyGrad[ii], d_lab->d_filterEnthalpyGradientCompLabel, ii, patch);
          }
        }
      }
      else {
        if (d_dynScalarModel) {
          if (d_calcScalar) {
            new_dw->getModifiable(scalarGrad[ii],      d_lab->d_scalarGradientCompLabel,       ii, patch);
            new_dw->getModifiable(filterScalarGrad[ii],d_lab->d_filterScalarGradientCompLabel, ii, patch);
          }
          if (d_calcEnthalpy) {
            new_dw->getModifiable(enthalpyGrad[ii],      d_lab->d_enthalpyGradientCompLabel,      ii, patch);
            new_dw->getModifiable(filterEnthalpyGrad[ii],d_lab->d_filterEnthalpyGradientCompLabel, ii, patch);
          }
        }
      }
      //__________________________________
      if (d_dynScalarModel) {
        if (d_calcScalar) {
          scalarGrad[ii].initialize(0.0);
          filterScalarGrad[ii].initialize(0.0);
        }
        if (d_calcEnthalpy) {
          enthalpyGrad[ii].initialize(0.0);
          filterEnthalpyGrad[ii].initialize(0.0);
        }
      }
    }

    IntVector indexLow = patch->getFortranCellLowIndex();
    IntVector indexHigh = patch->getFortranCellHighIndex();

    for (int colZ =indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          double sewcur = cellinfo->sew[colX];
          double snscur = cellinfo->sns[colY];
          double stbcur = cellinfo->stb[colZ];
          double efaccur = cellinfo->efac[colX];
          double wfaccur = cellinfo->wfac[colX];
          double nfaccur = cellinfo->nfac[colY];
          double sfaccur = cellinfo->sfac[colY];
          double tfaccur = cellinfo->tfac[colZ];
          double bfaccur = cellinfo->bfac[colZ];

          double uep, uwp, unp, usp, utp, ubp;
          double vnp, vsp, vep, vwp, vtp, vbp;
          double wtp, wbp, wep, wwp, wnp, wsp;

          uep = uVel[IntVector(colX+1,colY,colZ)];
          uwp = uVel[currCell];
          // colX,coly,colZ component cancels out when computing derivative,
          // so it has been ommited
          unp = 0.5*VelCC[IntVector(colX,colY+1,colZ)].x();
          usp = 0.5*VelCC[IntVector(colX,colY-1,colZ)].x();
          utp = 0.5*VelCC[IntVector(colX,colY,colZ+1)].x();
          ubp = 0.5*VelCC[IntVector(colX,colY,colZ-1)].x();

          vnp = vVel[IntVector(colX,colY+1,colZ)];
          vsp = vVel[currCell];
          // colX,coly,colZ component cancels out when computing derivative,
          // so it has been ommited
          vep = 0.5*VelCC[IntVector(colX+1,colY,colZ)].y();
          vwp = 0.5*VelCC[IntVector(colX-1,colY,colZ)].y();
          vtp = 0.5*VelCC[IntVector(colX,colY,colZ+1)].y();
          vbp = 0.5*VelCC[IntVector(colX,colY,colZ-1)].y();

          wtp = wVel[IntVector(colX,colY,colZ+1)];
          wbp = wVel[currCell];
          // colX,coly,colZ component cancels out when computing derivative,
          // so it has been ommited
          wep = 0.5*VelCC[IntVector(colX+1,colY,colZ)].z();
          wwp = 0.5*VelCC[IntVector(colX-1,colY,colZ)].z();
          wnp = 0.5*VelCC[IntVector(colX,colY+1,colZ)].z();
          wsp = 0.5*VelCC[IntVector(colX,colY-1,colZ)].z();

          //     calculate the grid strain rate tensor
          (SIJ[0])[currCell] = (uep-uwp)/sewcur;
          (SIJ[1])[currCell] = (vnp-vsp)/snscur;
          (SIJ[2])[currCell] = (wtp-wbp)/stbcur;
          (SIJ[3])[currCell] = 0.5*((unp-usp)/snscur + 
              (vep-vwp)/sewcur);
          (SIJ[4])[currCell] = 0.5*((utp-ubp)/stbcur + 
              (wep-wwp)/sewcur);
          (SIJ[5])[currCell] = 0.5*((vtp-vbp)/stbcur + 
              (wnp-wsp)/snscur);

          double fuep, fuwp, funp, fusp, futp, fubp;
          double fvnp, fvsp, fvep, fvwp, fvtp, fvbp;
          double fwtp, fwbp, fwep, fwwp, fwnp, fwsp;

          fuep = filterRhoU[IntVector(colX+1,colY,colZ)]/
            (0.5*(filterRho[currCell] +
                  filterRho[IntVector(colX+1,colY,colZ)]));
          fuwp = filterRhoU[currCell]/
            (0.5*(filterRho[currCell] +
                  filterRho[IntVector(colX-1,colY,colZ)]));
          // colX,coly,colZ component cancels out when computing derivative,
          // so it has been ommited
          funp = 0.5*(efaccur * filterRhoU[IntVector(colX+1,colY+1,colZ)]/
              (0.5*(filterRho[IntVector(colX,colY+1,colZ)] +
                    filterRho[IntVector(colX+1,colY+1,colZ)])) +
              wfaccur * filterRhoU[IntVector(colX,colY+1,colZ)]/
              (0.5*(filterRho[IntVector(colX,colY+1,colZ)] +
                    filterRho[IntVector(colX-1,colY+1,colZ)])));
          fusp = 0.5*(efaccur * filterRhoU[IntVector(colX+1,colY-1,colZ)]/
              (0.5*(filterRho[IntVector(colX,colY-1,colZ)] +
                    filterRho[IntVector(colX+1,colY-1,colZ)])) +
              wfaccur * filterRhoU[IntVector(colX,colY-1,colZ)]/
              (0.5*(filterRho[IntVector(colX,colY-1,colZ)] +
                    filterRho[IntVector(colX-1,colY-1,colZ)])));
          futp = 0.5*(efaccur * filterRhoU[IntVector(colX+1,colY,colZ+1)]/
              (0.5*(filterRho[IntVector(colX,colY,colZ+1)] +
                    filterRho[IntVector(colX+1,colY,colZ+1)])) +
              wfaccur * filterRhoU[IntVector(colX,colY,colZ+1)]/
              (0.5*(filterRho[IntVector(colX,colY,colZ+1)] +
                    filterRho[IntVector(colX-1,colY,colZ+1)])));
          fubp = 0.5*(efaccur * filterRhoU[IntVector(colX+1,colY,colZ-1)]/
              (0.5*(filterRho[IntVector(colX,colY,colZ-1)] +
                    filterRho[IntVector(colX+1,colY,colZ-1)])) +
              wfaccur * filterRhoU[IntVector(colX,colY,colZ-1)]/
              (0.5*(filterRho[IntVector(colX,colY,colZ-1)] +
                    filterRho[IntVector(colX-1,colY,colZ-1)])));

          fvnp = filterRhoV[IntVector(colX,colY+1,colZ)]/
            (0.5*(filterRho[currCell] +
                  filterRho[IntVector(colX,colY+1,colZ)]));
          fvsp = filterRhoV[currCell]/
            (0.5*(filterRho[currCell] +
                  filterRho[IntVector(colX,colY-1,colZ)]));
          // colX,coly,colZ component cancels out when computing derivative,
          // so it has been ommited
          fvep = 0.5*(nfaccur * filterRhoV[IntVector(colX+1,colY+1,colZ)]/
              (0.5*(filterRho[IntVector(colX+1,colY,colZ)] +
                    filterRho[IntVector(colX+1,colY+1,colZ)])) +
              sfaccur * filterRhoV[IntVector(colX+1,colY,colZ)]/
              (0.5*(filterRho[IntVector(colX+1,colY,colZ)] +
                    filterRho[IntVector(colX+1,colY-1,colZ)])));
          fvwp = 0.5*(nfaccur * filterRhoV[IntVector(colX-1,colY+1,colZ)]/
              (0.5*(filterRho[IntVector(colX-1,colY,colZ)] +
                    filterRho[IntVector(colX-1,colY+1,colZ)])) +
              sfaccur * filterRhoV[IntVector(colX-1,colY,colZ)]/
              (0.5*(filterRho[IntVector(colX-1,colY,colZ)] +
                    filterRho[IntVector(colX-1,colY-1,colZ)])));
          fvtp = 0.5*(nfaccur * filterRhoV[IntVector(colX,colY+1,colZ+1)]/
              (0.5*(filterRho[IntVector(colX,colY,colZ+1)] +
                    filterRho[IntVector(colX,colY+1,colZ+1)])) +
              sfaccur * filterRhoV[IntVector(colX,colY,colZ+1)]/
              (0.5*(filterRho[IntVector(colX,colY,colZ+1)] +
                    filterRho[IntVector(colX,colY-1,colZ+1)])));
          fvbp = 0.5*(nfaccur * filterRhoV[IntVector(colX,colY+1,colZ-1)]/
              (0.5*(filterRho[IntVector(colX,colY,colZ-1)] +
                    filterRho[IntVector(colX,colY+1,colZ-1)])) +
              sfaccur * filterRhoV[IntVector(colX,colY,colZ-1)]/
              (0.5*(filterRho[IntVector(colX,colY,colZ-1)] +
                    filterRho[IntVector(colX,colY-1,colZ-1)])));

          fwtp = filterRhoW[IntVector(colX,colY,colZ+1)]/
            (0.5*(filterRho[currCell] +
                  filterRho[IntVector(colX,colY,colZ+1)]));
          fwbp = filterRhoW[currCell]/
            (0.5*(filterRho[currCell] +
                  filterRho[IntVector(colX,colY,colZ-1)]));
          // colX,coly,colZ component cancels out when computing derivative,
          // so it has been ommited
          fwep = 0.5*(tfaccur * filterRhoW[IntVector(colX+1,colY,colZ+1)]/
              (0.5*(filterRho[IntVector(colX+1,colY,colZ)] +
                    filterRho[IntVector(colX+1,colY,colZ+1)])) +
              bfaccur * filterRhoW[IntVector(colX+1,colY,colZ)]/
              (0.5*(filterRho[IntVector(colX+1,colY,colZ)] +
                    filterRho[IntVector(colX+1,colY,colZ-1)])));
          fwwp = 0.5*(tfaccur * filterRhoW[IntVector(colX-1,colY,colZ+1)]/
              (0.5*(filterRho[IntVector(colX-1,colY,colZ)] +
                    filterRho[IntVector(colX-1,colY,colZ+1)])) +
              bfaccur * filterRhoW[IntVector(colX-1,colY,colZ)]/
              (0.5*(filterRho[IntVector(colX-1,colY,colZ)] +
                    filterRho[IntVector(colX-1,colY,colZ-1)])));
          fwnp = 0.5*(tfaccur * filterRhoW[IntVector(colX,colY+1,colZ+1)]/
              (0.5*(filterRho[IntVector(colX,colY+1,colZ)] +
                    filterRho[IntVector(colX,colY+1,colZ+1)])) +
              bfaccur * filterRhoW[IntVector(colX,colY+1,colZ)]/
              (0.5*(filterRho[IntVector(colX,colY+1,colZ)] +
                    filterRho[IntVector(colX,colY+1,colZ-1)])));
          fwsp = 0.5*(tfaccur * filterRhoW[IntVector(colX,colY-1,colZ+1)]/
              (0.5*(filterRho[IntVector(colX,colY-1,colZ)] +
                    filterRho[IntVector(colX,colY-1,colZ+1)])) +
              bfaccur * filterRhoW[IntVector(colX,colY-1,colZ)]/
              (0.5*(filterRho[IntVector(colX,colY-1,colZ)] +
                    filterRho[IntVector(colX,colY-1,colZ-1)])));

          //     calculate the filtered strain rate tensor
          (filterSIJ[0])[currCell] = (fuep-fuwp)/sewcur;
          (filterSIJ[1])[currCell] = (fvnp-fvsp)/snscur;
          (filterSIJ[2])[currCell] = (fwtp-fwbp)/stbcur;
          (filterSIJ[3])[currCell] = 0.5*((funp-fusp)/snscur + 
              (fvep-fvwp)/sewcur);
          (filterSIJ[4])[currCell] = 0.5*((futp-fubp)/stbcur + 
              (fwep-fwwp)/sewcur);
          (filterSIJ[5])[currCell] = 0.5*((fvtp-fvbp)/stbcur + 
              (fwnp-fwsp)/snscur);

          double scalarxp, scalarxm, scalaryp;
          double scalarym, scalarzp, scalarzm;
          double fscalarxp, fscalarxm, fscalaryp;
          double fscalarym, fscalarzp, fscalarzm;
          double enthalpyxp, enthalpyxm, enthalpyyp;
          double enthalpyym, enthalpyzp, enthalpyzm;
          double fenthalpyxp, fenthalpyxm, fenthalpyyp;
          double fenthalpyym, fenthalpyzp, fenthalpyzm;

          if (d_dynScalarModel) {
            if (d_calcScalar) {

              // colX,coly,colZ component cancels out when computing derivative,
              // so it has been ommited
              scalarxm = 0.5*scalar[IntVector(colX-1,colY,colZ)];
              scalarxp = 0.5*scalar[IntVector(colX+1,colY,colZ)];
              scalarym = 0.5*scalar[IntVector(colX,colY-1,colZ)];
              scalaryp = 0.5*scalar[IntVector(colX,colY+1,colZ)];
              scalarzm = 0.5*scalar[IntVector(colX,colY,colZ-1)];
              scalarzp = 0.5*scalar[IntVector(colX,colY,colZ+1)];

              (scalarGrad[0])[currCell] = (scalarxp-scalarxm)/sewcur;
              (scalarGrad[1])[currCell] = (scalaryp-scalarym)/snscur;
              (scalarGrad[2])[currCell] = (scalarzp-scalarzm)/stbcur;


              // colX,coly,colZ component cancels out when computing derivative,
              // so it has been ommited
              fscalarxm = 0.5*(filterRhoF[IntVector(colX-1,colY,colZ)]/
                  filterRho[IntVector(colX-1,colY,colZ)]);
              fscalarxp = 0.5*(filterRhoF[IntVector(colX+1,colY,colZ)]/
                  filterRho[IntVector(colX+1,colY,colZ)]);
              fscalarym = 0.5*(filterRhoF[IntVector(colX,colY-1,colZ)]/
                  filterRho[IntVector(colX,colY-1,colZ)]);
              fscalaryp = 0.5*(filterRhoF[IntVector(colX,colY+1,colZ)]/
                  filterRho[IntVector(colX,colY+1,colZ)]);
              fscalarzm = 0.5*(filterRhoF[IntVector(colX,colY,colZ-1)]/
                  filterRho[IntVector(colX,colY,colZ-1)]);
              fscalarzp = 0.5*(filterRhoF[IntVector(colX,colY,colZ+1)]/
                  filterRho[IntVector(colX,colY,colZ+1)]);

              (filterScalarGrad[0])[currCell] = (fscalarxp-fscalarxm)/sewcur;
              (filterScalarGrad[1])[currCell] = (fscalaryp-fscalarym)/snscur;
              (filterScalarGrad[2])[currCell] = (fscalarzp-fscalarzm)/stbcur;
            }
            if (d_calcEnthalpy) {

              // colX,coly,colZ component cancels out when computing derivative,
              // so it has been ommited
              enthalpyxm = 0.5*enthalpy[IntVector(colX-1,colY,colZ)];
              enthalpyxp = 0.5*enthalpy[IntVector(colX+1,colY,colZ)];
              enthalpyym = 0.5*enthalpy[IntVector(colX,colY-1,colZ)];
              enthalpyyp = 0.5*enthalpy[IntVector(colX,colY+1,colZ)];
              enthalpyzm = 0.5*enthalpy[IntVector(colX,colY,colZ-1)];
              enthalpyzp = 0.5*enthalpy[IntVector(colX,colY,colZ+1)];

              (enthalpyGrad[0])[currCell] = (enthalpyxp-enthalpyxm)/sewcur;
              (enthalpyGrad[1])[currCell] = (enthalpyyp-enthalpyym)/snscur;
              (enthalpyGrad[2])[currCell] = (enthalpyzp-enthalpyzm)/stbcur;


              // colX,coly,colZ component cancels out when computing derivative,
              // so it has been ommited
              fenthalpyxm = 0.5*(filterRhoE[IntVector(colX-1,colY,colZ)]/
                  filterRho[IntVector(colX-1,colY,colZ)]);
              fenthalpyxp = 0.5*(filterRhoE[IntVector(colX+1,colY,colZ)]/
                  filterRho[IntVector(colX+1,colY,colZ)]);
              fenthalpyym = 0.5*(filterRhoE[IntVector(colX,colY-1,colZ)]/
                  filterRho[IntVector(colX,colY-1,colZ)]);
              fenthalpyyp = 0.5*(filterRhoE[IntVector(colX,colY+1,colZ)]/
                  filterRho[IntVector(colX,colY+1,colZ)]);
              fenthalpyzm = 0.5*(filterRhoE[IntVector(colX,colY,colZ-1)]/
                  filterRho[IntVector(colX,colY,colZ-1)]);
              fenthalpyzp = 0.5*(filterRhoE[IntVector(colX,colY,colZ+1)]/
                  filterRho[IntVector(colX,colY,colZ+1)]);

              (filterEnthalpyGrad[0])[currCell] = (fenthalpyxp-fenthalpyxm)/
                sewcur;
              (filterEnthalpyGrad[1])[currCell] = (fenthalpyyp-fenthalpyym)/
                snscur;
              (filterEnthalpyGrad[2])[currCell] = (fenthalpyzp-fenthalpyzm)/
                stbcur;
            }
          }
        }
      }
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
    TAU_PROFILE_TIMER(compute1, "Compute1", "[reComputeFilterValues::compute1]" , TAU_USER);
    TAU_PROFILE_TIMER(compute2, "Compute2", "[reComputeFilterValues::compute2]" , TAU_USER);
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constCCVariable<Vector> ccVel;
    constCCVariable<double> den;
    constCCVariable<double> scalar;
    constCCVariable<double> enthalpy;
    constCCVariable<double> filterRho;
    constCCVariable<double> filterRhoF;
    constCCVariable<double> filterRhoE;
    constCCVariable<double> filterRhoRF;
    constCCVariable<double> filterVolume; 
    constCCVariable<int> cellType; 


    // Get the velocity and density
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(filterVolume, d_lab->d_filterVolumeLabel, indx, patch, Ghost::None, 0); 
    new_dw->get(cellType,     d_lab->d_cellTypeLabel,     indx, patch, gac, 2);
    new_dw->get(ccVel,     d_lab->d_CCVelocityLabel, indx, patch, gac, 1);
    new_dw->get(den,       d_lab->d_densityCPLabel,      indx, patch, gac, 1);
    new_dw->get(filterRho, d_lab->d_filterRhoLabel,      indx, patch, gac, 1);

    if (d_dynScalarModel) {
      if (d_calcScalar) {
        new_dw->get(scalar,     d_mf_label,   indx, patch, gac, 1);
        new_dw->get(filterRhoF, d_lab->d_filterRhoFLabel, indx, patch, gac, 1);
      }
      if (d_calcEnthalpy) {
        new_dw->get(enthalpy,   d_lab->d_enthalpySPLabel, indx, patch, gac, 1);
        new_dw->get(filterRhoE, d_lab->d_filterRhoELabel, indx, patch, gac, 1);
      }
    }  

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();


    IntVector idxLo = patch->getExtraCellLowIndex(Arches::ONEGHOSTCELL);
    IntVector idxHi = patch->getExtraCellHighIndex(Arches::ONEGHOSTCELL);

    StencilMatrix<constCCVariable<double> > SIJ; //6 point tensor
    StencilMatrix<constCCVariable<double> > SHATIJ; //6 point tensor
    for (int ii = 0; ii < d_lab->d_symTensorMatl->size(); ii++) {
      new_dw->get(SIJ[ii],    d_lab->d_strainTensorCompLabel,      ii, patch,gac, 1);
      new_dw->get(SHATIJ[ii], d_lab->d_filterStrainTensorCompLabel,ii, patch, gac, 1);
    }

    StencilMatrix<constCCVariable<double> > scalarGrad; //vector
    StencilMatrix<constCCVariable<double> > filterScalarGrad; //vector
    StencilMatrix<constCCVariable<double> > enthalpyGrad; //vector
    StencilMatrix<constCCVariable<double> > filterEnthalpyGrad; //vector
    for (int ii = 0; ii < d_lab->d_vectorMatl->size(); ii++) {
      if (d_dynScalarModel) {
        if (d_calcScalar) {
          new_dw->get(scalarGrad[ii],      d_lab->d_scalarGradientCompLabel,      ii, patch, gac, 1);
          new_dw->get(filterScalarGrad[ii],d_lab->d_filterScalarGradientCompLabel,ii, patch, gac, 1);
        }
        if (d_calcEnthalpy) {
          new_dw->get(enthalpyGrad[ii],      d_lab->d_enthalpyGradientCompLabel,      ii, patch, gac, 1);
          new_dw->get(filterEnthalpyGrad[ii],d_lab->d_filterEnthalpyGradientCompLabel,ii, patch, gac, 1);
        }
      }  
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

    StencilMatrix<Array3<double> > scalarBeta;  //vector
    StencilMatrix<Array3<double> > scalarBetaHat; //vector
    StencilMatrix<Array3<double> > enthalpyBeta;  //vector
    StencilMatrix<Array3<double> > enthalpyBetaHat; //vector
    for (int ii = 0; ii < d_lab->d_vectorMatl->size(); ii++) {
      if (d_dynScalarModel) {
        if (d_calcScalar) {
          scalarBeta[ii].resize(idxLo, idxHi);
          scalarBeta[ii].initialize(0.0);
          scalarBetaHat[ii].resize(idxLo, idxHi);
          scalarBetaHat[ii].initialize(0.0);
        }
        if (d_calcEnthalpy) {
          enthalpyBeta[ii].resize(idxLo, idxHi);
          enthalpyBeta[ii].initialize(0.0);
          enthalpyBetaHat[ii].resize(idxLo, idxHi);
          enthalpyBetaHat[ii].initialize(0.0);
        }
      }  
    }

    CCVariable<double> IsImag;
    CCVariable<double> MLI;
    CCVariable<double> MMI;
    CCVariable<double> scalarNum;
    CCVariable<double> scalarDenom;
    CCVariable<double> enthalpyNum;
    CCVariable<double> enthalpyDenom;
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->allocateAndPut(IsImag, d_lab->d_strainMagnitudeLabel,   indx, patch);
      new_dw->allocateAndPut(MLI,    d_lab->d_strainMagnitudeMLLabel, indx, patch);
      new_dw->allocateAndPut(MMI,    d_lab->d_strainMagnitudeMMLabel, indx, patch);
      if (d_dynScalarModel) {
        if (d_calcScalar) {
          new_dw->allocateAndPut(scalarNum,  d_lab->d_scalarNumeratorLabel,   indx, patch);
          new_dw->allocateAndPut(scalarDenom,d_lab->d_scalarDenominatorLabel, indx, patch);
        }
        if (d_calcEnthalpy) {
          new_dw->allocateAndPut(enthalpyNum,  d_lab->d_enthalpyNumeratorLabel, indx, patch);
          new_dw->allocateAndPut(enthalpyDenom,d_lab->d_enthalpyDenominatorLabel, indx, patch);
        }
      }
    }
    else {
      new_dw->getModifiable(IsImag, 
          d_lab->d_strainMagnitudeLabel, indx, patch);
      new_dw->getModifiable(MLI, 
          d_lab->d_strainMagnitudeMLLabel, indx, patch);
      new_dw->getModifiable(MMI, 
          d_lab->d_strainMagnitudeMMLabel, indx, patch);
      if (d_dynScalarModel) {
        if (d_calcScalar) {
          new_dw->getModifiable(scalarNum, 
              d_lab->d_scalarNumeratorLabel, indx, patch);
          new_dw->getModifiable(scalarDenom, 
              d_lab->d_scalarDenominatorLabel, indx, patch);
        }
        if (d_calcEnthalpy) {
          new_dw->getModifiable(enthalpyNum, 
              d_lab->d_enthalpyNumeratorLabel, indx, patch);
          new_dw->getModifiable(enthalpyDenom, 
              d_lab->d_enthalpyDenominatorLabel, indx, patch);
        }
      }  
    }
    IsImag.initialize(0.0);
    MLI.initialize(0.0);
    MMI.initialize(0.0);
    if (d_dynScalarModel) {
      if (d_calcScalar) {
        scalarNum.initialize(0.0);
        scalarDenom.initialize(0.0);
      }
      if (d_calcEnthalpy) {
        enthalpyNum.initialize(0.0);
        enthalpyDenom.initialize(0.0);
      }
    }  


    // compute test filtered velocities, density and product 
    // (den*u*u, den*u*v, den*u*w, den*v*v,
    // den*v*w, den*w*w)
    // using a box filter, generalize it to use other filters such as Gaussian


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
    if (d_dynScalarModel) {
      if (d_calcScalar) {
        rhoFU.resize(idxLo, idxHi);
        rhoFU.initialize(0.0);
        rhoFV.resize(idxLo, idxHi);
        rhoFV.initialize(0.0);
        rhoFW.resize(idxLo, idxHi);
        rhoFW.initialize(0.0);
      }
      if (d_calcEnthalpy) {
        rhoEU.resize(idxLo, idxHi);
        rhoEU.initialize(0.0);
        rhoEV.resize(idxLo, idxHi);
        rhoEV.initialize(0.0);
        rhoEW.resize(idxLo, idxHi);
        rhoEW.initialize(0.0);
      }
    }  
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

    TAU_PROFILE_START(compute1);
    for (int colZ = startZ; colZ < endZ; colZ ++) {
      for (int colY = startY; colY < endY; colY ++) {
        for (int colX = startX; colX < endX; colX ++) {
          IntVector currCell(colX, colY, colZ);
          // calculate absolute value of the grid strain rate
          // computes for the ghost cells too
          double sij0 = (SIJ[0])[currCell];
          double sij1 = (SIJ[1])[currCell];
          double sij2 = (SIJ[2])[currCell];
          double sij3 = (SIJ[3])[currCell];
          double sij4 = (SIJ[4])[currCell];
          double sij5 = (SIJ[5])[currCell];
          double isi_cur = sqrt(2.0*(sij0*sij0 + sij1*sij1 + sij2*sij2 +
                2.0*(sij3*sij3 + sij4*sij4 + sij5*sij5)));
          // trace has been neglected
          //        double trace = (sij0 + sij1 + sij2)/3.0;
          double trace = 0.0;
          double uvel_cur = ccVel[currCell].x();
          double vvel_cur = ccVel[currCell].y();
          double wvel_cur = ccVel[currCell].z();
          double den_cur = den[currCell];

          IsI[currCell] = isi_cur; 

          //    calculate the grid filtered stress tensor, beta

          (betaIJ[0])[currCell] = den_cur*isi_cur*(sij0-trace);
          (betaIJ[1])[currCell] = den_cur*isi_cur*(sij1-trace);
          (betaIJ[2])[currCell] = den_cur*isi_cur*(sij2-trace);
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

          double scalar_cur,enthalpy_cur;
          if (d_dynScalarModel) {
            if (d_calcScalar) {
              scalar_cur = scalar[currCell];
              (scalarBeta[0])[currCell] = den_cur*isi_cur*(scalarGrad[0])[currCell];
              (scalarBeta[1])[currCell] = den_cur*isi_cur*(scalarGrad[1])[currCell];
              (scalarBeta[2])[currCell] = den_cur*isi_cur*(scalarGrad[2])[currCell];
              rhoFU[currCell] = den_cur*scalar_cur*uvel_cur;
              rhoFV[currCell] = den_cur*scalar_cur*vvel_cur;
              rhoFW[currCell] = den_cur*scalar_cur*wvel_cur;
            }
            if (d_calcEnthalpy) {
              enthalpy_cur = enthalpy[currCell];
              (enthalpyBeta[0])[currCell] = den_cur*isi_cur*(enthalpyGrad[0])[currCell];
              (enthalpyBeta[1])[currCell] = den_cur*isi_cur*(enthalpyGrad[1])[currCell];
              (enthalpyBeta[2])[currCell] = den_cur*isi_cur*(enthalpyGrad[2])[currCell];
              rhoEU[currCell] = den_cur*enthalpy_cur*uvel_cur;
              rhoEV[currCell] = den_cur*enthalpy_cur*vvel_cur;
              rhoEW[currCell] = den_cur*enthalpy_cur*wvel_cur;
            }
          }  
        }
      }
    }
    TAU_PROFILE_STOP(compute1);
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
    if (d_dynScalarModel) {
      if (d_calcScalar) {
        filterRhoFU.resize(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
        filterRhoFU.initialize(0.0);
        filterRhoFV.resize(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
        filterRhoFV.initialize(0.0);
        filterRhoFW.resize(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
        filterRhoFW.initialize(0.0);
      }
      if (d_calcEnthalpy) {
        filterRhoEU.resize(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
        filterRhoEU.initialize(0.0);
        filterRhoEV.resize(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
        filterRhoEV.initialize(0.0);
        filterRhoEW.resize(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
        filterRhoEW.initialize(0.0);
      }
    }  

    IntVector indexLow = patch->getFortranCellLowIndex();
    IntVector indexHigh = patch->getFortranCellHighIndex();
    double start_turbTime = Time::currentSeconds();

#ifdef PetscFilter
    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoU,   filterVolume, cellType, filterRhoU);
    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoV,   filterVolume, cellType, filterRhoV);
    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoW,   filterVolume, cellType, filterRhoW);
    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoUU,  filterVolume, cellType,  filterRhoUU);
    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoUV,  filterVolume, cellType,  filterRhoUV);
    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoUW,  filterVolume, cellType,  filterRhoUW);
    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoVV,  filterVolume, cellType,  filterRhoVV);
    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoVW,  filterVolume, cellType,  filterRhoVW);
    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoWW,  filterVolume, cellType,  filterRhoWW);
#endif

    for (int ii = 0; ii < d_lab->d_symTensorMatl->size(); ii++) {
#ifdef PetscFilter
      d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, betaIJ[ii], filterVolume, cellType, betaHATIJ[ii]);
#endif 
    }

    if (d_dynScalarModel) {
      if (d_calcScalar) {
#ifdef PetscFilter
        d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoFU, filterVolume, cellType, filterRhoFU);
        d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoFV, filterVolume, cellType, filterRhoFV);
        d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoFW, filterVolume, cellType, filterRhoFW);
        for (int ii = 0; ii < d_lab->d_vectorMatl->size(); ii++) {
          d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, scalarBeta[ii], filterVolume, cellType, scalarBetaHat[ii]);
        }
#endif
      }
      if (d_calcEnthalpy) {
#ifdef PetscFilter
        d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoEU, filterVolume, cellType, filterRhoEU);
        d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoEV, filterVolume, cellType, filterRhoEV);
        d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoEW, filterVolume, cellType, filterRhoEW);
        for (int ii = 0; ii < d_lab->d_vectorMatl->size(); ii++) {
          d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, enthalpyBeta[ii], filterVolume, cellType, enthalpyBetaHat[ii]);
        }
#endif 
      }
    }  

    if (pc->myrank() == 0)
      cerr << "Time for the Filter operation in Turbulence Model: " << 
        Time::currentSeconds()-start_turbTime << " seconds\n";
    TAU_PROFILE_START(compute2);

    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          double delta = cellinfo->sew[colX]*
            cellinfo->sns[colY]*cellinfo->stb[colZ];
          double filter = pow(delta, 1.0/3.0);

          // test filter width is assumed to be twice that of the basic filter
          // needs following modifications:
          // a) make the test filter work for anisotropic grid
          // b) generalize the filter operation
          double shatij0 = (SHATIJ[0])[currCell];
          double shatij1 = (SHATIJ[1])[currCell];
          double shatij2 = (SHATIJ[2])[currCell];
          double shatij3 = (SHATIJ[3])[currCell];
          double shatij4 = (SHATIJ[4])[currCell];
          double shatij5 = (SHATIJ[5])[currCell];
          double IshatIcur = sqrt(2.0*(shatij0*shatij0 + shatij1*shatij1 +
                shatij2*shatij2 + 2.0*(shatij3*shatij3 + 
                  shatij4*shatij4 + shatij5*shatij5)));
          double filterDencur = filterRho[currCell];
          //        ignoring the trace
          //        double trace = (shatij0 + shatij1 + shatij2)/3.0;
          double trace = 0.0;

          IsImag[currCell] = IsI[currCell]; 

          double MIJ0cur = 2.0*filter*filter*
            ((betaHATIJ[0])[currCell]-
             2.0*2.0*filterDencur*IshatIcur*(shatij0-trace));
          double MIJ1cur = 2.0*filter*filter*
            ((betaHATIJ[1])[currCell]-
             2.0*2.0*filterDencur*IshatIcur*(shatij1-trace));
          double MIJ2cur = 2.0*filter*filter*
            ((betaHATIJ[2])[currCell]-
             2.0*2.0*filterDencur*IshatIcur*(shatij2-trace));
          double MIJ3cur = 2.0*filter*filter*
            ((betaHATIJ[3])[currCell]-
             2.0*2.0*filterDencur*IshatIcur*shatij3);
          double MIJ4cur = 2.0*filter*filter*
            ((betaHATIJ[4])[currCell]-
             2.0*2.0*filterDencur*IshatIcur*shatij4);
          double MIJ5cur =  2.0*filter*filter*
            ((betaHATIJ[5])[currCell]-
             2.0*2.0*filterDencur*IshatIcur*shatij5);


          // compute Leonard stress tensor
          // index 0: L11, 1:L22, 2:L33, 3:L12, 4:L13, 5:L23
          double filterRhoUcur = filterRhoU[currCell];
          double filterRhoVcur = filterRhoV[currCell];
          double filterRhoWcur = filterRhoW[currCell];
          double LIJ0cur = filterRhoUU[currCell] -
            filterRhoUcur*filterRhoUcur/filterDencur;
          double LIJ1cur = filterRhoVV[currCell] -
            filterRhoVcur*filterRhoVcur/filterDencur;
          double LIJ2cur = filterRhoWW[currCell] -
            filterRhoWcur*filterRhoWcur/filterDencur;
          double LIJ3cur = filterRhoUV[currCell] -
            filterRhoUcur*filterRhoVcur/filterDencur;
          double LIJ4cur = filterRhoUW[currCell] -
            filterRhoUcur*filterRhoWcur/filterDencur;
          double LIJ5cur = filterRhoVW[currCell] -
            filterRhoVcur*filterRhoWcur/filterDencur;

          // Explicitly making LIJ traceless here
          // Actually, trace has been ignored          
          //        double LIJtrace = (LIJ0cur + LIJ1cur + LIJ2cur)/3.0;
          double LIJtrace = 0.0;
          LIJ0cur = LIJ0cur - LIJtrace;
          LIJ1cur = LIJ1cur - LIJtrace;
          LIJ2cur = LIJ2cur - LIJtrace;

          // compute the magnitude of ML and MM
          MLI[currCell] = MIJ0cur*LIJ0cur +
            MIJ1cur*LIJ1cur +
            MIJ2cur*LIJ2cur +
            2.0*(MIJ3cur*LIJ3cur +
                MIJ4cur*LIJ4cur +
                MIJ5cur*LIJ5cur );
          // calculate absolute value of the grid strain rate
          MMI[currCell] = MIJ0cur*MIJ0cur +
            MIJ1cur*MIJ1cur +
            MIJ2cur*MIJ2cur +
            2.0*(MIJ3cur*MIJ3cur +
                MIJ4cur*MIJ4cur +
                MIJ5cur*MIJ5cur );

          double filterRhoFcur, scalarLX, scalarLY, scalarLZ;
          double scalarMX, scalarMY, scalarMZ;
          double filterRhoEcur, enthalpyLX, enthalpyLY, enthalpyLZ;
          double enthalpyMX, enthalpyMY, enthalpyMZ;
          if (d_dynScalarModel) {
            if (d_calcScalar) {
              filterRhoFcur = filterRhoF[currCell];
              scalarLX =  filter*filter*
                ((scalarBetaHat[0])[currCell]-
                 2.0*2.0*filterDencur*IshatIcur*
                 (filterScalarGrad[0])[currCell]);
              scalarLY =  filter*filter*
                ((scalarBetaHat[1])[currCell]-
                 2.0*2.0*filterDencur*IshatIcur*
                 (filterScalarGrad[1])[currCell]);
              scalarLZ =  filter*filter*
                ((scalarBetaHat[2])[currCell]-
                 2.0*2.0*filterDencur*IshatIcur*
                 (filterScalarGrad[2])[currCell]);
              scalarMX = filterRhoFU[currCell] -
                filterRhoFcur*filterRhoUcur/filterDencur;
              scalarMY = filterRhoFV[currCell] -
                filterRhoFcur*filterRhoVcur/filterDencur;
              scalarMZ = filterRhoFW[currCell] -
                filterRhoFcur*filterRhoWcur/filterDencur;
              scalarNum[currCell] = scalarLX*scalarLX +
                scalarLY*scalarLY +
                scalarLZ*scalarLZ;
              scalarDenom[currCell] = scalarMX*scalarLX +
                scalarMY*scalarLY +
                scalarMZ*scalarLZ;
            }
            if (d_calcEnthalpy) {
              filterRhoEcur = filterRhoE[currCell];
              enthalpyLX =  filter*filter*
                ((enthalpyBetaHat[0])[currCell]-
                 2.0*2.0*filterDencur*IshatIcur*
                 (filterEnthalpyGrad[0])[currCell]);
              enthalpyLY =  filter*filter*
                ((enthalpyBetaHat[1])[currCell]-
                 2.0*2.0*filterDencur*IshatIcur*
                 (filterEnthalpyGrad[1])[currCell]);
              enthalpyLZ =  filter*filter*
                ((enthalpyBetaHat[2])[currCell]-
                 2.0*2.0*filterDencur*IshatIcur*
                 (filterEnthalpyGrad[2])[currCell]);
              enthalpyMX = filterRhoEU[currCell] -
                filterRhoEcur*filterRhoUcur/filterDencur;
              enthalpyMY = filterRhoEV[currCell] -
                filterRhoEcur*filterRhoVcur/filterDencur;
              enthalpyMZ = filterRhoEW[currCell] -
                filterRhoEcur*filterRhoWcur/filterDencur;
              enthalpyNum[currCell] = enthalpyLX*enthalpyLX +
                enthalpyLY*enthalpyLY +
                enthalpyLZ*enthalpyLZ;
              enthalpyDenom[currCell] = enthalpyMX*enthalpyLX +
                enthalpyMY*enthalpyLY +
                enthalpyMZ*enthalpyLZ;
            }
          }  
        }
      }
    }
    TAU_PROFILE_STOP(compute2);

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
  //  double time = d_lab->d_sharedState->getElapsedTime();
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constCCVariable<double> IsI;
    constCCVariable<double> MLI;
    constCCVariable<double> MMI;
    constCCVariable<double> scalarNum;
    constCCVariable<double> scalarDenom;
    constCCVariable<double> enthalpyNum;
    constCCVariable<double> enthalpyDenom;
    CCVariable<double> Cs; //smag coeff 
    CCVariable<double> ShF; //Shmidt number 
    CCVariable<double> ShE; //Shmidt number 
    CCVariable<double> ShRF; //Shmidt number 
    constCCVariable<double> den;
    constCCVariable<double> voidFraction;
    constCCVariable<int> cellType;
    CCVariable<double> viscosity;
    CCVariable<double> turbViscosity; 
    CCVariable<double> scalardiff;
    CCVariable<double> enthalpydiff;
    constCCVariable<double> filterVolume; 
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->allocateAndPut(Cs, d_lab->d_CsLabel, indx, patch);
      if (d_dynScalarModel) {
        if (d_calcScalar) {
          new_dw->allocateAndPut(ShF, d_lab->d_ShFLabel, indx, patch);
        }
        if (d_calcEnthalpy) {
          new_dw->allocateAndPut(ShE, d_lab->d_ShELabel, indx, patch);
        }
      }      
    }
    else {
      new_dw->getModifiable(Cs, d_lab->d_CsLabel, indx, patch);
      if (d_dynScalarModel) {
        if (d_calcScalar) {
          new_dw->getModifiable(ShF, d_lab->d_ShFLabel, indx, patch);
        }
        if (d_calcEnthalpy) {
          new_dw->getModifiable(ShE, d_lab->d_ShELabel, indx, patch);
        }
      }      
    }
    Cs.initialize(0.0);
    if (d_dynScalarModel) {
      if (d_calcScalar) {
        ShF.initialize(0.0);
      }
      if (d_calcEnthalpy) {
        ShE.initialize(0.0);
      }
    }      

    new_dw->getModifiable(viscosity,         d_lab->d_viscosityCTSLabel,        indx, patch);
    new_dw->getModifiable(turbViscosity,            d_lab->d_turbViscosLabel,              indx, patch);
    if (d_dynScalarModel) {
      if (d_calcScalar) {
        new_dw->getModifiable(scalardiff,    d_lab->d_scalarDiffusivityLabel,   indx, patch);
      }
      if (d_calcEnthalpy) {
        new_dw->getModifiable(enthalpydiff,  d_lab->d_enthalpyDiffusivityLabel, indx, patch);
      }
    } 

    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;

    new_dw->get(IsI, d_lab->d_strainMagnitudeLabel,   indx, patch,   gn, 0);
    // using a box filter of 2*delta...will require more ghost cells if the size of filter is increased
    new_dw->get(MLI, d_lab->d_strainMagnitudeMLLabel, indx, patch, gac, 1);
    new_dw->get(MMI, d_lab->d_strainMagnitudeMMLabel, indx, patch, gac, 1);
    new_dw->get(filterVolume, d_lab->d_filterVolumeLabel, indx, patch, gn, 0); 

    if (d_dynScalarModel) {
      if (d_calcScalar) {
        new_dw->get(scalarNum,   d_lab->d_scalarNumeratorLabel,  indx, patch, gac, 1);
        new_dw->get(scalarDenom, d_lab->d_scalarDenominatorLabel,indx, patch, gac, 1);
      }
      if (d_calcEnthalpy) {
        new_dw->get(enthalpyNum,   d_lab->d_enthalpyNumeratorLabel,  indx, patch, gac, 1);
        new_dw->get(enthalpyDenom, d_lab->d_enthalpyDenominatorLabel,indx, patch, gac, 1);
      }
    }      

    new_dw->get(den, d_lab->d_densityCPLabel, indx, patch,gac, 1);

    if (d_MAlab){
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, indx, patch, gn, 0);
    }

    new_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch, gac, 1);

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    // get physical constants
    double viscos; // molecular viscosity
    viscos = d_physicalConsts->getMolecularViscosity();


    // compute test filtered velocities, density and product 
    // (den*u*u, den*u*v, den*u*w, den*v*v,
    // den*v*w, den*w*w)
    // using a box filter, generalize it to use other filters such as Gaussian
    Array3<double> MLHatI(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex()); // magnitude of strain rate
    MLHatI.initialize(0.0);
    Array3<double> MMHatI(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex()); // magnitude of test filter strain rate
    MLHatI.initialize(0.0);
    Array3<double> scalarNumHat;
    Array3<double> scalarDenomHat;
    Array3<double> enthalpyNumHat;
    Array3<double> enthalpyDenomHat;
    if (d_dynScalarModel) {
      if (d_calcScalar) {
        scalarNumHat.resize(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
        scalarNumHat.initialize(0.0);
        scalarDenomHat.resize(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
        scalarDenomHat.initialize(0.0);
      }
      if (d_calcEnthalpy) {
        enthalpyNumHat.resize(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
        enthalpyNumHat.initialize(0.0);
        enthalpyDenomHat.resize(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
        enthalpyDenomHat.initialize(0.0);
      }
    }      
    IntVector indexLow = patch->getFortranCellLowIndex();
    IntVector indexHigh = patch->getFortranCellHighIndex();

#ifdef PetscFilter
    d_filter->applyFilter_noPetsc<constCCVariable<double> >(pc, patch, MLI, filterVolume, cellType, MLHatI);
    d_filter->applyFilter_noPetsc<constCCVariable<double> >(pc, patch, MMI, filterVolume, cellType, MMHatI);
    if (d_dynScalarModel) {
      if (d_calcScalar) {
        d_filter->applyFilter_noPetsc<constCCVariable<double> >(pc, patch, scalarNum,   filterVolume, cellType, scalarNumHat);
        d_filter->applyFilter_noPetsc<constCCVariable<double> >(pc, patch, scalarDenom, filterVolume, cellType, scalarDenomHat);
      }
      if (d_calcEnthalpy) {
        d_filter->applyFilter_noPetsc<constCCVariable<double> >(pc, patch, enthalpyNum,   filterVolume, cellType, enthalpyNumHat);
        d_filter->applyFilter_noPetsc<constCCVariable<double> >(pc, patch, enthalpyDenom, filterVolume, cellType, enthalpyDenomHat);
      }
    }      
#endif

    CCVariable<double> tempCs;
    tempCs.allocate(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    tempCs.initialize(0.0);
    CCVariable<double> tempShF;
    CCVariable<double> tempShE;
    CCVariable<double> tempShRF;
    if (d_dynScalarModel) {
      if (d_calcScalar) {
        tempShF.allocate(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
        tempShF.initialize(0.0);
      }
      if (d_calcEnthalpy) {
        tempShE.allocate(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
        tempShE.initialize(0.0);
      }
    }      
    //     calculate the local Smagorinsky coefficient
    //     perform "clipping" in case MLij is negative...
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);

          double value;
          if ((MMHatI[currCell] < 1.0e-10)||(MLHatI[currCell] < 1.0e-7))
            value = 0.0;
          else
            value = MLHatI[currCell]/MMHatI[currCell];
          tempCs[currCell] = value;

          // It makes more sence to compute inverse Sh numbers here
          double scalar_value;
          double enthalpy_value;
          if (d_dynScalarModel) {
            if (d_calcScalar) {
              if ((scalarNumHat[currCell] < 1.0e-7)||
                  (scalarDenomHat[currCell] < 1.0e-10))
                scalar_value = 0.0;
              else
                scalar_value = scalarDenomHat[currCell]/scalarNumHat[currCell];
              tempShF[currCell] = scalar_value;
            }
            if (d_calcEnthalpy) {
              if ((enthalpyNumHat[currCell] < 1.0e-7/scalarNumHat[currCell]*enthalpyNumHat[currCell])||
                  (enthalpyDenomHat[currCell] < 1.0e-10))
                enthalpy_value = 0.0;
              else
                enthalpy_value = enthalpyDenomHat[currCell]/
                  enthalpyNumHat[currCell];
              tempShE[currCell] = enthalpy_value;
            }
          }      
        }
      }
    }

    if ((d_filter_cs_squared)&&(!(d_3d_periodic))) {
      // filtering for periodic case is not implemented 
      // if it needs to be then tempCs will require 1 layer of boundary cells to be computed
#ifdef PetscFilter
      d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, tempCs, filterVolume, cellType, Cs);
      if (d_dynScalarModel) {
        if (d_calcScalar) {
          d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, tempShF, filterVolume, cellType, ShF);
        }
        if (d_calcEnthalpy) {
          d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, tempShE, filterVolume, cellType,  ShE);
        }
      }      
#endif 
    }
    else
      Cs.copy(tempCs, tempCs.getLowIndex(),
          tempCs.getHighIndex());
    if (d_dynScalarModel) {
      if (d_calcScalar) {
        ShF.copy(tempShF, tempShF.getLowIndex(),
            tempShF.getHighIndex());
      }
      if (d_calcEnthalpy) {
        ShE.copy(tempShE, tempShE.getLowIndex(),
            tempShE.getHighIndex());
      }
    }      

    double factor = 1.0;
#if 0
    if (time < 2.0)
      factor = (time+0.000001)*0.5;
#endif

    // Laminar Pr number is taken to be 0.7, shouldn't make much difference
    double laminarPrNo = 0.7;
    if (d_MAlab) {

      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
            IntVector currCell(colX, colY, colZ);
            double delta = cellinfo->sew[colX]*
              cellinfo->sns[colY]*cellinfo->stb[colZ];
            double filter = pow(delta, 1.0/3.0);

            Cs[currCell] = Min(Cs[currCell],10.0);
            Cs[currCell] = factor * sqrt(Cs[currCell]);

            viscosity[currCell] =  Cs[currCell] * Cs[currCell] *
              filter * filter *
              IsI[currCell] * den[currCell] +
              viscos*voidFraction[currCell];

            turbViscosity[currCell] = viscosity[currCell] - viscos*voidFraction[currCell]; 

            if (d_dynScalarModel) {
              if (d_calcScalar) {
                ShF[currCell] = Min(ShF[currCell],10.0);
                scalardiff[currCell] = filter * filter *
                  IsI[currCell] * den[currCell] *
                  ShF[currCell] + viscos*
                  voidFraction[currCell]/laminarPrNo;
              }
              if (d_calcEnthalpy) {
                ShE[currCell] = Min(ShE[currCell],10.0);
                enthalpydiff[currCell] = filter * filter *
                  IsI[currCell] * den[currCell] *
                  ShE[currCell] + viscos*
                  voidFraction[currCell]/laminarPrNo;
              }
            }      
          }
        }
      }
    }
    else {
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
            IntVector currCell(colX, colY, colZ);
            double delta = cellinfo->sew[colX]*
              cellinfo->sns[colY]*cellinfo->stb[colZ];
            double filter = pow(delta, 1.0/3.0);

            Cs[currCell] = Min(Cs[currCell],10.0);
            Cs[currCell] = factor * sqrt(Cs[currCell]);
            viscosity[currCell] =  Cs[currCell] * Cs[currCell] *
              filter * filter *
              IsI[currCell] * den[currCell] + viscos;

            turbViscosity[currCell] = viscosity[currCell] - viscos; 

            if (d_dynScalarModel) {
              if (d_calcScalar) {
                ShF[currCell] = Min(ShF[currCell],10.0);
                scalardiff[currCell] = filter * filter *
                  IsI[currCell] * den[currCell] *
                  ShF[currCell] + viscos/laminarPrNo;
              }
              if (d_calcEnthalpy) {
                ShE[currCell] = Min(ShE[currCell],10.0);
                enthalpydiff[currCell] = filter * filter *
                  IsI[currCell] * den[currCell] *
                  ShE[currCell] + viscos/laminarPrNo;
              }
            }      
          }
        }
      }
    }

    // boundary conditions...make a separate function apply Boundary
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    int wall_celltypeval = BoundaryCondition::WALL; //d_boundaryCondition->wallCellType();
    if (xminus) {
      int colX = indexLow.x();
      for (int colZ = indexLow.z(); colZ <=  indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <=  indexHigh.y(); colY ++) {
          IntVector currCell(colX-1, colY, colZ);
          if (cellType[currCell] != wall_celltypeval) {
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
            //          viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
            //                             *den[currCell]/den[IntVector(colX,colY,colZ)];
            if (d_dynScalarModel) {
              if (d_calcScalar) {
                scalardiff[currCell] = scalardiff[IntVector(colX,colY,colZ)];
              }
              if (d_calcEnthalpy) {
                enthalpydiff[currCell] = enthalpydiff[IntVector(colX,colY,colZ)];
              }
            }
          }          
        }
      }
    }
    if (xplus) {
      int colX =  indexHigh.x();
      for (int colZ = indexLow.z(); colZ <=  indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <=  indexHigh.y(); colY ++) {
          IntVector currCell(colX+1, colY, colZ);
          if (cellType[currCell] != wall_celltypeval) {
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
            //          viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
            //                    *den[currCell]/den[IntVector(colX,colY,colZ)];
            if (d_dynScalarModel) {
              if (d_calcScalar) {
                scalardiff[currCell] = scalardiff[IntVector(colX,colY,colZ)];
              }
              if (d_calcEnthalpy) {
                enthalpydiff[currCell] = enthalpydiff[IntVector(colX,colY,colZ)];
              }
            }
          }          
        }
      }
    }
    if (yminus) {
      int colY = indexLow.y();
      for (int colZ = indexLow.z(); colZ <=  indexHigh.z(); colZ ++) {
        for (int colX = indexLow.x(); colX <=  indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY-1, colZ);
          if (cellType[currCell] != wall_celltypeval) {
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
            //          viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
            //                    *den[currCell]/den[IntVector(colX,colY,colZ)];
            if (d_dynScalarModel) {
              if (d_calcScalar) {
                scalardiff[currCell] = scalardiff[IntVector(colX,colY,colZ)];
              }
              if (d_calcEnthalpy) {
                enthalpydiff[currCell] = enthalpydiff[IntVector(colX,colY,colZ)];
              }
            }
          }          
        }
      }
    }
    if (yplus) {
      int colY =  indexHigh.y();
      for (int colZ = indexLow.z(); colZ <=  indexHigh.z(); colZ ++) {
        for (int colX = indexLow.x(); colX <=  indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY+1, colZ);
          if (cellType[currCell] != wall_celltypeval) {
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
            //          viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
            //                    *den[currCell]/den[IntVector(colX,colY,colZ)];
            if (d_dynScalarModel) {
              if (d_calcScalar) {
                scalardiff[currCell] = scalardiff[IntVector(colX,colY,colZ)];
              }
              if (d_calcEnthalpy) {
                enthalpydiff[currCell] = enthalpydiff[IntVector(colX,colY,colZ)];
              }
            }
          }          
        }
      }
    }
    if (zminus) {
      int colZ = indexLow.z();
      for (int colY = indexLow.y(); colY <=  indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <=  indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY, colZ-1);
          if (cellType[currCell] != wall_celltypeval) {
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
            //          viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
            //                    *den[currCell]/den[IntVector(colX,colY,colZ)];
            if (d_dynScalarModel) {
              if (d_calcScalar) {
                scalardiff[currCell] = scalardiff[IntVector(colX,colY,colZ)];
              }
              if (d_calcEnthalpy) {
                enthalpydiff[currCell] = enthalpydiff[IntVector(colX,colY,colZ)];
              }
            }
          }          
        }
      }
    }
    if (zplus) {
      int colZ =  indexHigh.z();
      for (int colY = indexLow.y(); colY <=  indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <=  indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY, colZ+1);
          if (cellType[currCell] != wall_celltypeval) {
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
            //          viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
            //                    *den[currCell]/den[IntVector(colX,colY,colZ)];
            if (d_dynScalarModel) {
              if (d_calcScalar) {
                scalardiff[currCell] = scalardiff[IntVector(colX,colY,colZ)];
              }
              if (d_calcEnthalpy) {
                enthalpydiff[currCell] = enthalpydiff[IntVector(colX,colY,colZ)];
              }
            }
          }          
        }
      }
    }

  }
}

//______________________________________________________________________
  void 
CompDynamicProcedure::sched_computeScalarVariance(SchedulerP& sched, 
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls,
                                                  const TimeIntegratorLabel* timelabels)
{
  string taskname =  "CompDynamicProcedure::computeScalarVaraince" +
    timelabels->integrator_step_name;
    
  Task* tsk = scinew Task(taskname, this,
      &CompDynamicProcedure::computeScalarVariance,
      timelabels);


  // Requires, only the scalar corresponding to matlindex = 0 is
  //           required. For multiple scalars this will be put in a loop

  Ghost::GhostType  gac = Ghost::AroundCells;

  tsk->requires(Task::NewDW, d_mf_label,              gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,  gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_filterVolumeLabel, Ghost::None, 0); 

  int mmWallID = d_boundaryCondition->getMMWallId();
  if (mmWallID > 0)
    tsk->requires(Task::OldDW, d_lab->d_refDensity_label);

  // Computes
  if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::First ) {
    tsk->computes(d_lab->d_scalarVarSPLabel);
    tsk->computes(d_lab->d_normalizedScalarVarLabel);
  }
  else {
    tsk->modifies(d_lab->d_scalarVarSPLabel);
    tsk->modifies(d_lab->d_normalizedScalarVarLabel);
  }

  sched->addTask(tsk, patches, matls);
}

//______________________________________________________________________
//
  void 
CompDynamicProcedure::computeScalarVariance(const ProcessorGroup* pc,
                                            const PatchSubset* patches,
                                            const MaterialSubset*,
                                            DataWarehouse* old_dw,
                                            DataWarehouse* new_dw,
                                            const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constCCVariable<double> scalar;
    constCCVariable<double> density;
    CCVariable<double> scalarVar;
    CCVariable<double> normalizedScalarVar;
    constCCVariable<double> filterVolume;
    new_dw->get(filterVolume, d_lab->d_filterVolumeLabel, indx, patch, Ghost::None, 0); 
    // Get the velocity, density and viscosity from the old data warehouse

    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(scalar, d_mf_label, indx,  patch, gac, 1);

    new_dw->get(density, d_lab->d_densityCPLabel, indx, patch, gac, 1);

    if ((timelabels->integrator_step_number == TimeIntegratorStepNumber::First)) {
      new_dw->allocateAndPut(scalarVar,           d_lab->d_scalarVarSPLabel,         indx, patch);
      new_dw->allocateAndPut(normalizedScalarVar, d_lab->d_normalizedScalarVarLabel, indx,patch);
    }
    else {
      new_dw->getModifiable(scalarVar,           d_lab->d_scalarVarSPLabel,         indx,patch);
      new_dw->getModifiable(normalizedScalarVar, d_lab->d_normalizedScalarVarLabel, indx,patch);
    }
    scalarVar.initialize(0.0);
    normalizedScalarVar.initialize(0.0);

    constCCVariable<int> cellType;
    new_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch, gac, 1);


    IntVector idxLo = patch->getExtraCellLowIndex(Arches::ONEGHOSTCELL);
    IntVector idxHi = patch->getExtraCellHighIndex(Arches::ONEGHOSTCELL);
    Array3<double> rhoPhi(idxLo, idxHi);
    Array3<double> rhoPhiSqr(idxLo, idxHi);
    rhoPhi.initialize(0.0);
    rhoPhiSqr.initialize(0.0);

    for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          rhoPhi[currCell] = density[currCell]*scalar[currCell];
          rhoPhiSqr[currCell] = density[currCell]*
            scalar[currCell]*scalar[currCell];

        }
      }
    }

    Array3<double> filterRho(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    Array3<double> filterRhoPhi(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    Array3<double> filterRhoPhiSqr(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRho.initialize(0.0);
    filterRhoPhi.initialize(0.0);
    filterRhoPhiSqr.initialize(0.0);

    IntVector indexLow = patch->getFortranCellLowIndex();
    IntVector indexHigh = patch->getFortranCellHighIndex();

#ifdef PetscFilter
    d_filter->applyFilter_noPetsc<constCCVariable<double> >(pc, patch, density, filterVolume, cellType, filterRho);
    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoPhi, filterVolume, cellType, filterRhoPhi);
    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoPhiSqr, filterVolume, cellType, filterRhoPhiSqr);
#endif

    // making filterRho nonzero 
    sum_vartype den_ref_var;
    int mmWallID = d_boundaryCondition->getMMWallId();
    if (mmWallID > 0) {
      old_dw->get(den_ref_var, d_lab->d_refDensity_label);
    }

    double small = 1.0e-10;
    double var_limit = 0.0;
    double filterPhi = 0.0;
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);

          if ((mmWallID > 0)&&(filterRho[currCell] < 1.0e-15)) {
            filterRho[currCell]=den_ref_var;
          }

          // compute scalar variance
          filterPhi = filterRhoPhi[currCell]/filterRho[currCell];
          scalarVar[currCell] = d_CFVar*
            (filterRhoPhiSqr[currCell]/filterRho[currCell]-
             filterPhi*filterPhi);

          // now, check variance bounds and normalize
          if (d_filter_var_limit_scalar)
            var_limit = filterPhi * (1.0 - filterPhi);
          else
            var_limit = scalar[currCell] * (1.0 - scalar[currCell]);

          if(scalarVar[currCell] < small)
            scalarVar[currCell] = 0.0;
          if(scalarVar[currCell] > var_limit)
            scalarVar[currCell] = var_limit;

          normalizedScalarVar[currCell] = scalarVar[currCell]/(var_limit+small);
        }
      }
    }

    // boundary conditions
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    int outlet_celltypeval = d_boundaryCondition->outletCellType();
    int pressure_celltypeval = d_boundaryCondition->pressureCellType();
    if (xminus) {
      int colX = indexLow.x();
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          IntVector currCell(colX-1, colY, colZ);
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval))
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)]) {
              scalarVar[currCell] = scalarVar[IntVector(colX,colY,colZ)];
              normalizedScalarVar[currCell] = 
                normalizedScalarVar[IntVector(colX,colY,colZ)];

            }
        }
      }
    }

    if (xplus) {
      int colX = indexHigh.x();
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          IntVector currCell(colX+1, colY, colZ);
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval))
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)]) {
              scalarVar[currCell] = scalarVar[IntVector(colX,colY,colZ)];
              normalizedScalarVar[currCell] = 
                normalizedScalarVar[IntVector(colX,colY,colZ)];
            }
        }
      }
    }
    if (yminus) {
      int colY = indexLow.y();
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY-1, colZ);
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval))
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)]) {
              scalarVar[currCell] = scalarVar[IntVector(colX,colY,colZ)];
              normalizedScalarVar[currCell] = 
                normalizedScalarVar[IntVector(colX,colY,colZ)];
            }
        }
      }
    }
    if (yplus) {
      int colY = indexHigh.y();
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY+1, colZ);
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval))
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)]) {
              scalarVar[currCell] = scalarVar[IntVector(colX,colY,colZ)];
              normalizedScalarVar[currCell] = 
                normalizedScalarVar[IntVector(colX,colY,colZ)];
            }
        }
      }
    }
    if (zminus) {
      int colZ = indexLow.z();
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY, colZ-1);
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval))
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)]) {
              scalarVar[currCell] = scalarVar[IntVector(colX,colY,colZ)];
              normalizedScalarVar[currCell] = 
                normalizedScalarVar[IntVector(colX,colY,colZ)];
            }
        }
      }
    }
    if (zplus) {
      int colZ = indexHigh.z();
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY, colZ+1);
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval))
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)]) {
              scalarVar[currCell] = scalarVar[IntVector(colX,colY,colZ)];
              normalizedScalarVar[currCell] = 
                normalizedScalarVar[IntVector(colX,colY,colZ)];
            }
        }
      }
    }

  }
}
//****************************************************************************
// Schedule recomputation of the turbulence sub model 
//****************************************************************************
  void 
CompDynamicProcedure::sched_computeScalarDissipation(SchedulerP& sched, 
                                                     const PatchSet* patches,
                                                     const MaterialSet* matls,
                                                     const TimeIntegratorLabel* timelabels)
{
  string taskname =  "CompDynamicProcedure::computeScalarDissipation" +
    timelabels->integrator_step_name;
    
  Task* tsk = scinew Task(taskname, this,
                         &CompDynamicProcedure::computeScalarDissipation,
                         timelabels);


  // Requires, only the scalar corresponding to matlindex = 0 is
  //           required. For multiple scalars this will be put in a loop
  // assuming scalar dissipation is computed before turbulent viscosity calculation 
  Ghost::GhostType  gac = Ghost::AroundCells;
  tsk->requires(Task::NewDW, d_mf_label,  gac, 1);

  if (d_dynScalarModel){
    tsk->requires(Task::NewDW, d_lab->d_scalarDiffusivityLabel, gac, 1);
  }else{
    tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,      gac, 1);
  }
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,            gac, 1);

  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, Ghost::None);

  // Computes
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
    tsk->computes(d_lab->d_scalarDissSPLabel);
  }else{
    tsk->modifies(d_lab->d_scalarDissSPLabel);
  }
  sched->addTask(tsk, patches, matls);
}

//______________________________________________________________________
//
  void 
CompDynamicProcedure::computeScalarDissipation(const ProcessorGroup*,
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
    constCCVariable<double> viscosity;
    constCCVariable<double> scalar;
    CCVariable<double> scalarDiss;  // dissipation..chi

    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(scalar, d_mf_label, indx, patch, gac, 1);

    if (d_dynScalarModel){
      new_dw->get(viscosity, d_lab->d_scalarDiffusivityLabel, indx, patch, gac, 1);
    }else{
      new_dw->get(viscosity, d_lab->d_viscosityCTSLabel,      indx, patch, gac, 1);
    }

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      new_dw->allocateAndPut(scalarDiss, d_lab->d_scalarDissSPLabel, indx, patch);
    }else{
      new_dw->getModifiable(scalarDiss, d_lab->d_scalarDissSPLabel,  indx, patch);
    }
    scalarDiss.initialize(0.0);

    constCCVariable<int> cellType;
    new_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch, gac, 1);

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    // compatible with fortran index
    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          double scale = 0.5*(scalar[currCell]+
              scalar[IntVector(colX+1,colY,colZ)]);
          double scalw = 0.5*(scalar[currCell]+
              scalar[IntVector(colX-1,colY,colZ)]);
          double scaln = 0.5*(scalar[currCell]+
              scalar[IntVector(colX,colY+1,colZ)]);
          double scals = 0.5*(scalar[currCell]+
              scalar[IntVector(colX,colY-1,colZ)]);
          double scalt = 0.5*(scalar[currCell]+
              scalar[IntVector(colX,colY,colZ+1)]);
          double scalb = 0.5*(scalar[currCell]+
              scalar[IntVector(colX,colY,colZ-1)]);
          double dfdx = (scale-scalw)/cellinfo->sew[colX];
          double dfdy = (scaln-scals)/cellinfo->sns[colY];
          double dfdz = (scalt-scalb)/cellinfo->stb[colZ];
          scalarDiss[currCell] = viscosity[currCell]*
            (dfdx*dfdx + dfdy*dfdy + dfdz*dfdz)/
            d_turbPrNo; 
        }
      }
    }


    // boundary conditions
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    int outlet_celltypeval   = d_boundaryCondition->outletCellType();
    int pressure_celltypeval = d_boundaryCondition->pressureCellType();

    if (xminus) {
      int colX = idxLo.x();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
          IntVector currCell(colX-1, colY, colZ);
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval))
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)])
              scalarDiss[currCell] = scalarDiss[IntVector(colX,colY,colZ)];
        }
      }
    }
    if (xplus) {
      int colX = idxHi.x();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
          IntVector currCell(colX+1, colY, colZ);
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval))
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)])
              scalarDiss[currCell] = scalarDiss[IntVector(colX,colY,colZ)];
        }
      }
    }
    if (yminus) {
      int colY = idxLo.y();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY-1, colZ);
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval))
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)])
              scalarDiss[currCell] = scalarDiss[IntVector(colX,colY,colZ)];
        }
      }
    }
    if (yplus) {
      int colY = idxHi.y();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY+1, colZ);
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval))
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)])
              scalarDiss[currCell] = scalarDiss[IntVector(colX,colY,colZ)];
        }
      }
    }
    if (zminus) {
      int colZ = idxLo.z();
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY, colZ-1);
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval))
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)])
              scalarDiss[currCell] = scalarDiss[IntVector(colX,colY,colZ)];
        }
      }
    }
    if (zplus) {
      int colZ = idxHi.z();
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY, colZ+1);
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval))
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)])
              scalarDiss[currCell] = scalarDiss[IntVector(colX,colY,colZ)];
        }
      }
    }

  }
}
