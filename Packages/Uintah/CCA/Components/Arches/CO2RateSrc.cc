//----- CO2RateSrc.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/CO2RateSrc.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>

using namespace std;
using namespace Uintah;

//****************************************************************************
// Interface constructor for CO2RateSrc
//****************************************************************************
CO2RateSrc::CO2RateSrc(const ArchesLabel* label, 
		       const MPMArchesLabel* MAlb,
                       const VarLabel* d_src_label):
                       ExtraScalarSrc(label, MAlb, d_src_label)
{
}

//****************************************************************************
// Destructor
//****************************************************************************
CO2RateSrc::~CO2RateSrc()
{
}

//****************************************************************************
// Problem Setup 
//****************************************************************************
void 
CO2RateSrc::problemSetup(const ProblemSpecP& params)
{
}
//****************************************************************************
// Schedule source computation
//****************************************************************************
void
CO2RateSrc::sched_addExtraScalarSrc(SchedulerP& sched, 
                                   const PatchSet* patches,
				   const MaterialSet* matls,
				   const TimeIntegratorLabel* timelabels)
{
  
  string taskname =  "CO2RateSrc::addExtraScalarSrc" +
      	              timelabels->integrator_step_name+
                      d_scalar_nonlin_src_label->getName();
  //cout << taskname << endl;
  Task* tsk = scinew Task(taskname, this,
      		    &CO2RateSrc::addExtraScalarSrc,
      		    timelabels);
  tsk->modifies(d_scalar_nonlin_src_label);
  tsk->requires(Task::NewDW, d_lab->d_co2RateLabel, 
  		Ghost::None, Arches::ZEROGHOSTCELLS);

  sched->addTask(tsk, patches, matls);
  
}
//****************************************************************************
// Actual source computation 
//****************************************************************************
void 
CO2RateSrc::addExtraScalarSrc(const ProcessorGroup* pc,
				      const PatchSubset* patches,
				      const MaterialSubset*,
				      DataWarehouse*,
				      DataWarehouse* new_dw,
				      const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> scalarNonlinSrc;
    new_dw->getModifiable(scalarNonlinSrc, d_scalar_nonlin_src_label,
                          matlIndex, patch);

    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex();

    constCCVariable<double> CO2rate;
    new_dw->get(CO2rate, d_lab->d_co2RateLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    for (int colZ =indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  
          scalarNonlinSrc[currCell] += CO2rate[currCell];
        }
      }
    }
  }
}
