/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#include <CCA/Components/OnTheFlyAnalysis/radiometer.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Parallel/ProcessorGroup.h>

#include <Core/Util/DebugStream.h>
#include <iostream>
#include <cstdio>


using namespace Uintah;
using namespace std;
/*______________________________________________________________________
          TO DO
    - Generalize how the varLabels are registered for each component

    - Add bulletproofing in problem setup to catch if the user doesn't
      save the fluxes.

    - add radiometerFreq variable
______________________________________________________________________*/
static DebugStream cout_doing("RADIOMETER_DOING_COUT", false);

OnTheFly_radiometer::OnTheFly_radiometer(ProblemSpecP& module_spec,
                                         SimulationStateP& sharedState,
                                         Output* dataArchiver)
  : AnalysisModule(module_spec, sharedState, dataArchiver)
{

  d_sharedState = sharedState;
  d_RMCRT  = scinew Radiometer();
}

//__________________________________
OnTheFly_radiometer::~OnTheFly_radiometer()
{
  cout_doing << " Doing: destorying OntheFly_radiometer " << endl;

  delete d_RMCRT;
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
//______________________________________________________________________
void OnTheFly_radiometer::problemSetup(const ProblemSpecP& prob_spec,
                                       const ProblemSpecP& ,
                                       GridP& grid,
                                       SimulationStateP& sharedState)
{
  cout_doing << "Doing problemSetup \t\t\t\t OnTheFly_radiometer" << endl;
#if 0
  //__________________________________
  // find the material .  Default is matl 0.
  // The user can use either
  //  <material>   atmosphere </material>
  //  <materialIndex> 1 </materialIndex>
  const Material* matl;
  if( prob_spec->findBlock("material") ){
    matl = d_sharedState->parseAndLookupMaterial(d_prob_spec, "material");
  } else if ( prob_spec->findBlock("materialIndex") ){
    int indx;
    prob_spec->get("materialIndex", indx);
    matl = d_sharedState->getMaterial(indx);
  } else {
    matl = d_sharedState->getMaterial(0);
  }

  int matl_index = matl->getDWIndex();



  d_RMCRT->registerVarLabels(_matl_index,
                          _prop_calculator->get_abskg_label(),
                          _absorpLabel,
                          _tempLabel,
                          _cellTypeLabel,
                          _src_label);
#endif

  d_RMCRT->problemSetup(prob_spec, prob_spec, sharedState);
}

//______________________________________________________________________
//
//______________________________________________________________________
void OnTheFly_radiometer::scheduleInitialize(SchedulerP& sched,
                                             const LevelP& level)
{
  return;  // do nothing
}

void OnTheFly_radiometer::initialize(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset*,
                                     DataWarehouse*,
                                     DataWarehouse* new_dw)
{
}

void OnTheFly_radiometer::restartInitialize()
{
}

//______________________________________________________________________
//
//______________________________________________________________________
void OnTheFly_radiometer::scheduleDoAnalysis(SchedulerP& sched,
                                             const LevelP& level)
{
  d_radiometerCalc_freq =1;

  Task::WhichDW temp_dw     = Task::NewDW;
  Task::WhichDW abskg_dw    = Task::NewDW;
  Task::WhichDW sigmaT4_dw  = Task::NewDW;
  Task::WhichDW celltype_dw = Task::NewDW;

  d_RMCRT->sched_radiometer( level, sched, abskg_dw, sigmaT4_dw, celltype_dw, d_radiometerCalc_freq );
}
