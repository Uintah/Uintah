//----- ZeroExtraScalarSrc.cc ----------------------------------------------

#include <CCA/Components/Arches/ZeroExtraScalarSrc.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <CCA/Ports/Scheduler.h>

using namespace std;
using namespace Uintah;

//****************************************************************************
// Interface constructor for ZeroExtraScalarSrc
//****************************************************************************
ZeroExtraScalarSrc::ZeroExtraScalarSrc(const ArchesLabel* label, 
			       const MPMArchesLabel* MAlb,
                               const VarLabel* d_src_label):
                               ExtraScalarSrc(label, MAlb, d_src_label)
{
}

//****************************************************************************
// Destructor
//****************************************************************************
ZeroExtraScalarSrc::~ZeroExtraScalarSrc()
{
}

//****************************************************************************
// Problem Setup 
//****************************************************************************
void 
ZeroExtraScalarSrc::problemSetup(const ProblemSpecP& params)
{
}
//****************************************************************************
// Schedule source computation
//****************************************************************************
void
ZeroExtraScalarSrc::sched_addExtraScalarSrc(SchedulerP& sched, 
                                   const PatchSet* patches,
				   const MaterialSet* matls,
				   const TimeIntegratorLabel* timelabels)
{
  
  string taskname =  "ZeroExtraScalarSrc::addExtraScalarSrc" +
      	              timelabels->integrator_step_name+
                      d_scalar_nonlin_src_label->getName();
  //cout << taskname << endl;
  Task* tsk = scinew Task(taskname, this,
      		    &ZeroExtraScalarSrc::addExtraScalarSrc,
      		    timelabels);

  tsk->modifies(d_scalar_nonlin_src_label);

  sched->addTask(tsk, patches, matls);
  
}
//****************************************************************************
// Actual source computation 
//****************************************************************************
void 
ZeroExtraScalarSrc::addExtraScalarSrc(const ProcessorGroup* pc,
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

    //cout << "adding source for " << d_scalar_nonlin_src_label->getName() << endl;
    for (int colZ =indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);

          scalarNonlinSrc[currCell] += 0.0;
        }
      }
    }
  }
}
