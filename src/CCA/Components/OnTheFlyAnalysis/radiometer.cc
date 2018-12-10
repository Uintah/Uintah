/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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
#include <Core/Grid/MaterialManager.h>
#include <Core/Parallel/ProcessorGroup.h>

#include <Core/Util/DebugStream.h>
#include <iostream>
#include <cstdio>


using namespace Uintah;
using namespace std;

//#define USE_RADIOMETER   // Circular dependency on OSX.  Disable so we can compile on bigmac
/*______________________________________________________________________
          TO DO
      - Clean up the hardwiring in the problem setup
______________________________________________________________________*/
static DebugStream cout_doing("radiometer", false);

OnTheFly_radiometer::OnTheFly_radiometer(const ProcessorGroup* myworld,
                                         const MaterialManagerP materialManager,
                                         const ProblemSpecP& module_spec)
  : AnalysisModule(myworld, materialManager, module_spec)
{
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
                                       std::vector<std::vector<const VarLabel* > > &PState,
                                       std::vector<std::vector<const VarLabel* > > &PState_preReloc)
{

#ifdef USE_RADIOMETER
  cout_doing << "Doing problemSetup \t\t\t\t OnTheFly_radiometer" << endl;

  //__________________________________
  // find the material .  Default is matl 0.
  // The user can use either
  //  <material>   atmosphere </material>
  //  <materialIndex> 1 </materialIndex>
 
  Material* matl;
 
  if( m_module_spec->findBlock("material") ){
    matl = d_materialManager->parseAndLookupMaterial( m_module_spec, "material" );
  } else if ( m_module_spec->findBlock("materialIndex") ){
    int indx;
    m_module_spec->get("materialIndex", indx);
    matl = d_materialManager->getMaterial(indx);
  } else {
    matl = d_materialManager->getMaterial(0);
  }

  int matl_index = matl->getDWIndex();
  
  ProblemSpecP rad_ps = m_module_spec->findBlock("Radiometer");
  if (!rad_ps){
    throw ProblemSetupException("ERROR Radiometer: Couldn't find <Radiometer> xml node", __FILE__, __LINE__);    
  }
  
  //__________________________________
  //  read in the VarLabel names
  string temp     = "nullptr";
  string cellType = "nullptr";
  string abskg    = "nullptr";
  
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
  
  if( tempLabel == nullptr || cellTypeLabel == nullptr || abskgLabel == nullptr ){
    ostringstream warn;
    warn << "ERROR OnTheFly_radiometer One of the VarLabels need to do the analysis does not exist\n"
         << "    temperature address: " << tempLabel << "\n"
         << "    celltype:             " << cellTypeLabel << "\n"
         << "    abskg:                " << abskgLabel << "\n";
    throw InternalError(warn.str(), __FILE__, __LINE__);
  }

   //__________________________________
   // using float or doubles for all-to-all variables
   map<string,string> type;
   rad_ps->getAttributes(type);

   string isFloat = type["type"];

   if( isFloat == "float" ){
     d_radiometer = scinew Radiometer( TypeDescription::float_type );
   } else {
     d_radiometer = scinew Radiometer( TypeDescription::double_type );
   }

  //__________________________________
  // register the component VarLabels the RMCRT:Radiometer
  d_radiometer->registerVarLabels( matl_index,
                                   abskgLabel,
                                   tempLabel,
                                   cellTypeLabel,
                                   notUsed);

  bool getExtraInputs = true;
  d_radiometer->problemSetup(rad_ps, rad_ps, grid, d_materialManager, getExtraInputs);
  
  if(!d_output->isLabelSaved( "VRFlux" ) ){
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

  // carry forward if it is time
  d_radiometer->sched_CarryForward_Var ( level, sched, d_radiometer->d_sigmaT4Label, RMCRTCommon::TG_CARRY_FORWARD);
 
  d_radiometer->sched_initializeRadVars( level, sched );
  
  // convert abskg:dbl -> abskg:flt if needed
  d_radiometer->sched_DoubleToFloat( level, sched, abskg_dw );
  
  d_radiometer->sched_sigmaT4( level, sched, temp_dw, includeEC );

  d_radiometer->sched_radiometer( level, sched, abskg_dw, sigmaT4_dw, celltype_dw );
#endif
}
