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


//----- CO2RateSrc.cc ----------------------------------------------

#include <CCA/Components/Arches/CO2RateSrc.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Ports/Scheduler.h>

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
  ProblemSpecP db = params;
  //Get the name of what to look for in the table
  // we "require" this because this source is specifically 
  // designed for a table-read source term.
  db->require("tableName", d_tableName);
  db->getWithDefault("scaleFactor", d_scaleFactor, 1.0);

  //Initialize
  setTableIndex(-1);
      
  //__________________________________
  //  bulletproofing
  bool test;  
  ProblemSpecP root      = db->getRootNode();
  ProblemSpecP cfd_ps    = root->findBlock("CFD");
  ProblemSpecP arches_ps = cfd_ps->findBlock("ARCHES");
  ProblemSpecP BC_ps     = arches_ps->findBlock("BoundaryConditions");
  BC_ps->getWithDefault("carbon_balance_es", test, false);  
  
  if (test == false){
    throw ProblemSetupException("The CO2Rate Source term requires that carbon_balance_es be set to true! \n",__FILE__, __LINE__);
  }
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
  
  Task* tsk = scinew Task(taskname, this,
                          &CO2RateSrc::addExtraScalarSrc,
                          timelabels);

  //variables needed:
  tsk->modifies(d_scalar_nonlin_src_label);
  tsk->requires(Task::NewDW, d_lab->d_co2RateLabel, Ghost::None, 0);

  //add the task:
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
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> scalarNonlinSrc;
    new_dw->getModifiable(scalarNonlinSrc, d_scalar_nonlin_src_label, indx, patch);

    //going to estimate volume for all cells.
    //this will need to be fixed when going to stretched meshes
    Vector dx = patch->dCell();
    double vol = dx.x()*dx.y()*dx.z();

    constCCVariable<double> CO2rate;
    new_dw->get(CO2rate, d_lab->d_co2RateLabel, indx, patch, Ghost::None, 0);

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      scalarNonlinSrc[*iter] += d_scaleFactor*CO2rate[*iter]*vol*44000; //44000 = conversion from mol/cm^3/s to kg/m^3/s
    }
  }
}
