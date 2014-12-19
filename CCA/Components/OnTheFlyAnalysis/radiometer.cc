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
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Parallel/ProcessorGroup.h>

#include <Core/Util/DebugStream.h>
#include <iostream>
#include <cstdio>


using namespace Uintah;
using namespace std;

//#define USE_RADIOMETER
/*______________________________________________________________________
          TO DO
      - Clean up the hardwiring in the problem setup
______________________________________________________________________*/
static DebugStream cout_doing("radiometer", false);

OnTheFly_radiometer::OnTheFly_radiometer(ProblemSpecP& module_spec,
                                         SimulationStateP& sharedState,
                                         Output* dataArchiver)
  : AnalysisModule(module_spec, sharedState, dataArchiver)
{
  d_sharedState = sharedState;

#ifdef USE_RADIOMETER
  d_radiometer  = scinew Radiometer( TypeDescription::double_type );          // HARDWIRED: double;
#endif
  d_module_ps   = module_spec;
  d_dataArchiver = dataArchiver;
}

//__________________________________
OnTheFly_radiometer::~OnTheFly_radiometer()
{
  cout_doing << " Doing: destorying OntheFly_radiometer " << endl;
#ifdef USE_RADIOMETER
  delete d_radiometer;
#endif
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
//______________________________________________________________________
void OnTheFly_radiometer::problemSetup(const ProblemSpecP& ,
                                       const ProblemSpecP& ,
                                       GridP& grid,
                                       SimulationStateP& sharedState)
{

#ifdef USE_RADIOMETER
  cout_doing << "Doing problemSetup \t\t\t\t OnTheFly_radiometer" << endl;

  //__________________________________
  // find the material .  Default is matl 0.
  // The user can use either
  //  <material>   atmosphere </material>
  //  <materialIndex> 1 </materialIndex>
 
  Material* matl;
 
  if( d_module_ps->findBlock("material") ){
    matl = d_sharedState->parseAndLookupMaterial( d_module_ps, "material" );
  } else if ( d_module_ps->findBlock("materialIndex") ){
    int indx;
    d_module_ps->get("materialIndex", indx);
    matl = d_sharedState->getMaterial(indx);
  } else {
    matl = d_sharedState->getMaterial(0);
  }

  int matl_index = matl->getDWIndex();
  
  ProblemSpecP rad_ps = d_module_ps->findBlock("Radiometer");
  if (!rad_ps){
    throw ProblemSetupException("ERROR Radiometer: Couldn't find <Radiometer> xml node", __FILE__, __LINE__);    
  }
  
  //__________________________________
  //  read in the VarLabel names
  string temp     = "NULL";
  string cellType = "NULL";
  string abskg    = "NULL";
  
  if ( rad_ps->findBlock( "temperature" ) ){ 
    rad_ps->findBlock( "temperature" )->getAttribute( "label",temp ); 
  } 
  if ( rad_ps->findBlock( "cellType" ) ){ 
    rad_ps->findBlock( "cellType" )->getAttribute( "label",cellType ); 
  }
  if ( rad_ps->findBlock( "abskg" ) ){ 
    rad_ps->findBlock( "abskg" )->getAttribute( "label",abskg ); 
  }
  
  
  //__________________________________
  //  bulletproofing
  const VarLabel* tempLabel      = VarLabel::find( temp );
  const VarLabel* cellTypeLabel  = VarLabel::find( cellType );
  const VarLabel* abskgLabel     = VarLabel::find( abskg );
  const VarLabel* notUsed = 0;
  
  if( tempLabel == NULL || cellTypeLabel == NULL || abskgLabel == NULL ){
    ostringstream warn;
    warn << "ERROR OnTheFly_radiometer One of the VarLabels need to do the analysis does not exist\n"
         << "    temperature address: " << tempLabel << "\n"
         << "    celltype:             " << cellTypeLabel << "\n"
         << "    abskg:                " << abskgLabel << "\n";
    throw InternalError(warn.str(), __FILE__, __LINE__);
  }


  //__________________________________
  // register the component VarLabels the RMCRT:Radiometer
  d_radiometer->registerVarLabels( matl_index,
                                   abskgLabel,
                                   tempLabel,
                                   cellTypeLabel,
                                   notUsed);

  d_module_ps->getWithDefault( "radiometerCalc_freq", d_radiometerCalc_freq, 1 );
  bool getExtraInputs = true;
  d_radiometer->problemSetup(rad_ps, rad_ps, grid, d_sharedState, getExtraInputs);
  
  if(!d_dataArchiver->isLabelSaved( "VRFlux" ) ){
    throw ProblemSetupException("\nERROR:  You've activated the radiometer but your not saving the variable (VRFlux)\n",__FILE__, __LINE__);
  }
#endif  
}

//______________________________________________________________________
//
//______________________________________________________________________
void OnTheFly_radiometer::scheduleInitialize(SchedulerP& sched,
                                             const LevelP& level)
{
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
#ifdef USE_RADIOMETER 
  printSchedule(level, cout_doing, "OnTheFly_radiometer::scheduleDoAnalysis");

  Task::WhichDW temp_dw     = Task::NewDW;
  Task::WhichDW abskg_dw    = Task::NewDW;
  Task::WhichDW sigmaT4_dw  = Task::NewDW;
  Task::WhichDW celltype_dw = Task::NewDW;
  bool includeEC = true;
 
  d_radiometer->sched_initializeRadVars( level, sched, d_radiometerCalc_freq );
  
  d_radiometer->sched_sigmaT4( level, sched, temp_dw, d_radiometerCalc_freq, includeEC );

  d_radiometer->sched_radiometer( level, sched, abskg_dw, sigmaT4_dw, celltype_dw, d_radiometerCalc_freq );
#endif
}
