/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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


//----- ZeroExtraScalarSrc.cc ----------------------------------------------

#include <CCA/Components/Arches/ZeroExtraScalarSrc.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CCVariable.h>
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
  tsk->modifies(d_lab->d_zerosrcVarLabel);

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
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> scalarNonlinSrc;
    CCVariable<double> zerosrcVar;
    
    new_dw->getModifiable(scalarNonlinSrc, d_scalar_nonlin_src_label,indx, patch);
    new_dw->getModifiable(zerosrcVar,     d_lab->d_zerosrcVarLabel,  indx, patch);

    //cout << "adding source for " << d_scalar_nonlin_src_label->getName() << endl;
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      scalarNonlinSrc[*iter] += 0.0;
      zerosrcVar[*iter] += 0.0;
    }
  }
}
