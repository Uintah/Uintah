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

#include <CCA/Components/Parent/ApplicationFactory.h>
#include <CCA/Components/Parent/Switcher.h>
#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <CCA/Components/Solvers/SolverFactory.h>
#include <CCA/Components/SwitchingCriteria/None.h>
#include <CCA/Components/SwitchingCriteria/SwitchingCriteriaFactory.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/Regridder.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SolverInterface.h>
#include <CCA/Ports/SwitchingCriteria.h>

#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/OS/Dir.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/StringUtil.h>

#include <sci_defs/uintah_defs.h>

using namespace Uintah;

//______________________________________________________________________
// Initialize class static variables:

// export SCI_DEBUG="SWITCHER:+"
DebugStream Switcher::switcher_dbg("SWITCHER", "Switcher", "Switcher debug stream", false);
//______________________________________________________________________
//

#define ALL_LEVELS  99

// ToDo:
// - test carry over and init vars
// - test in parallel
// - test restarting capability
// - fix so each subcomponent filebase name is used for uda.
// - test different components (mpmice, impm) for subcomponents


//__________________________________
// In the constructor read the master ups file
// For each subcomponent in the ups file:
//     - 
Switcher::Switcher( const ProcessorGroup * myworld,
                    const MaterialManagerP materialManager,
                          ProblemSpecP   & d_master_ups,
                    const std::string    & uda )
  : ApplicationCommon(myworld, materialManager)
{
  proc0cout << "-----------------------------Switcher::Switcher top"<< std::endl;

  d_switch_label =
    VarLabel::create("switchFlag", max_vartype::getTypeDescription());

  int num_components = 0;
  d_componentIndex   = 0;
  d_switchState      = idle;
  d_restarting       = false;

  std::set<std::string>       simComponents;

  ProblemSpecP sim_block = d_master_ups->findBlock("SimulationComponent");
  ProblemSpecP child     = sim_block->findBlock("subcomponent");

  //__________________________________
  //  loop over the subcomponents
  for(; child != nullptr; child = child->findNextBlock("subcomponent")) {

    //__________________________________
    //  Read in subcomponent ups file and store the filename
    std::string input_file("");
    if (!child->get("input_file",input_file)) {
      throw ProblemSetupException("Need 'input_file' for subcomponent", __FILE__, __LINE__);
    }
    
    proc0cout << "Input file:\t\t" << input_file << std::endl;
    
    d_in_file.push_back(input_file);
    ProblemSpecP subCompUps = ProblemSpecReader().readInputFile(input_file);

    // get the component name from the input file, and the uda arg is
    // not needed for normal simulations...
    std::string sim_comp;
    ProblemSpecP sim_ps = subCompUps->findBlock("SimulationComponent");
    sim_ps->getAttribute( "type", sim_comp );
    simComponents.insert(sim_comp);

    //__________________________________
    // create simulation port and attach it switcher component    
    UintahParallelComponent* comp =
      ApplicationFactory::create(subCompUps, myworld, m_materialManager, "");

    ApplicationInterface* app = dynamic_cast<ApplicationInterface*>(comp);
    attachPort( "application", app );

    // Create solver port and attach it to the switcher component.
    SolverInterface * solver = SolverFactory::create( subCompUps, myworld );    
    attachPort( "sub_solver", solver );

    comp->attachPort( "solver", solver );

    //__________________________________
    // create switching criteria port and attach it switcher component
    SwitchingCriteria * switch_criteria =
      SwitchingCriteriaFactory::create( child, myworld );

    if( switch_criteria ) {
      switch_criteria->setSwitchLabel( d_switch_label );

      attachPort(      "switch_criteria", switch_criteria);
      comp->attachPort("switch_criteria", switch_criteria);
    }

    //__________________________________
    // Get the variables that will need to be initialized by this subcomponent
    initVars* initVar = scinew initVars;
    for( ProblemSpecP var = child->findBlock("init"); var != nullptr; var = var->findNextBlock("init") ) {

      std::map<std::string, std::string> attributes;
      var->getAttributes(attributes);
      
      // matlsetNames
      std::string matls = attributes["matls"];
      initVar->matlSetNames.push_back(matls);
      
      // levels
      std::stringstream s_level(attributes["levels"]);
      int levels = ALL_LEVELS;
      s_level >> levels;
      initVar->levels.push_back(levels);
     
      // variable name
      std::string varName =  attributes["var"];
      initVar->varNames.push_back(varName);
    }
    
    d_initVars[num_components] =initVar;
    
    num_components++;
    
    proc0cout << "\n";
  }  // loop over subcomponents

  //__________________________________
  // Bulletproofing:
  if ( simComponents.count("mpm") && simComponents.count("rmpmice") ){
    throw ProblemSetupException("Switcher: The simulation subComponents rmpmice and mpm cannot be used together", __FILE__, __LINE__);
  }

  //__________________________________
  // Bulletproofing:
  // Make sure that a switching criteria was specified.  For n subcomponents,
  // there should be n-1 switching critiera specified.
  int num_switch_criteria = 0;
  for (int i = 0; i < num_components; i++) {
    UintahParallelComponent* comp =
      dynamic_cast<UintahParallelComponent*>(getPort("application",i));

    SwitchingCriteria* sw =
      dynamic_cast<SwitchingCriteria*>(comp->getPort("switch_criteria"));

    if (sw) {
      num_switch_criteria++;
    }
  }
  
  if (num_switch_criteria != num_components-1) {
    throw  ProblemSetupException( "Do not have enough switching criteria specified for the number of components.", __FILE__, __LINE__ );
  }
  
  //__________________________________  
  // Add the "None" SwitchCriteria to the last component, so the
  // switchFlag label is computed in the last stage.
  UintahParallelComponent* last_comp =
    dynamic_cast<UintahParallelComponent*>(getPort("application",num_components-1));

  SwitchingCriteria* none_switch_criteria = scinew None();
  
  // Attaching to switcher so that the switcher can delete it
  none_switch_criteria->setSwitchLabel( d_switch_label );
  
  attachPort(           "switch_criteria",none_switch_criteria);
  last_comp->attachPort("switch_criteria",none_switch_criteria);

  //__________________________________
  // Get the vars that will need to be carried over 
  for (ProblemSpecP var = sim_block->findBlock("carry_over"); var != nullptr; var = var->findNextBlock("carry_over")) {
    std::map<std::string, std::string> attributes;
    var->getAttributes(attributes);
    std::string name  = attributes["var"];
    std::string matls = attributes["matls"];
    std::string level = attributes["level"];

    if (name != "") {
      d_carryOverVars.push_back(name);
    }

    MaterialSubset* carry_over_matls = 0;
    if (matls != "") {
      carry_over_matls = scinew MaterialSubset;
      ConsecutiveRangeSet crs = matls;
      ConsecutiveRangeSet::iterator iter = crs.begin();

      for (; iter != crs.end(); iter++) {
        carry_over_matls->add(*iter);
      }
      carry_over_matls->addReference();
    }

    d_carryOverVarMatls.push_back(carry_over_matls);
    if (level == "finest") {
      d_carryOverFinestLevelOnly.push_back(true);
    }
    else {
      d_carryOverFinestLevelOnly.push_back(false);
    }
  }  // loop over 

  d_numComponents = num_components;
  d_computedVars.clear();

  proc0cout << "Number of components " << d_numComponents << std::endl;
  proc0cout << "-----------------------------Switcher::Switcher bottom" << std::endl;
}

//______________________________________________________________________
//
Switcher::~Switcher()
{
  switcher_dbg << d_myworld->myRank() << " Switcher::~Switcher" << std::endl;

  VarLabel::destroy(d_switch_label);

  for (unsigned i = 0; i < d_carryOverVarMatls.size(); i++) {
    if (d_carryOverVarMatls[i] && d_carryOverVarMatls[i]->removeReference()) {
      delete d_carryOverVarMatls[i];
    }
  }
  d_carryOverVarMatls.clear();
}

//______________________________________________________________________
// 
void
Switcher::problemSetup( const ProblemSpecP     & params,
                        const ProblemSpecP     & restart_prob_spec,
                              GridP            & grid )
{  
  switcher_dbg << "Doing ProblemSetup \t\t\t\tSwitcher"<< std::endl;

  if (restart_prob_spec){
    readSwitcherState(restart_prob_spec, m_materialManager);
  }

  switchApplication( restart_prob_spec, grid );

  proc0cout << "__________________________________\n\n";
  
  //__________________________________
  // init Variables:
  //   - determine the label from the string names
  //   - determine the MaterialSet from the string matlSetName
  //   - store this info to be used later
  std::map<int, initVars*>::iterator it;
  for (it = d_initVars.begin(); it != d_initVars.end(); it++)
  {
    int comp = it->first;
    initVars* vars = it->second;

    switcher_dbg << " init Variables:  component: " << comp << std::endl;
    
    // Find the varLabel   
    std::vector<std::string>& varNames = vars->varNames;
    std::vector<VarLabel*> varLabels = vars->varLabels;

    for (unsigned j = 0; j < varNames.size(); j++) {

      std::string varName = varNames[j];
      VarLabel* label = VarLabel::find(varName);

      if (!label) {
        std::string error =
          "ERROR: Switcher: Cannot find init VarLabel" + varName;
        throw ProblemSetupException(error, __FILE__, __LINE__);
      }

      varLabels.push_back(label);
      // so the variable is not scrubbed from the data warehouse
      m_scheduler->overrideVariableBehavior(varName, false, false, true, false, false);
    }

    d_initVars[comp]->varLabels = varLabels;
  }
  
  // Carry over labels
  for (unsigned i = 0; i < d_carryOverVars.size(); i++)
  {
    VarLabel* label = VarLabel::find(d_carryOverVars[i]);

    if (label) {
      d_carryOverVarLabels.push_back(label);

      // So variables are not scrubbed from the data warehouse.
      m_scheduler->overrideVariableBehavior(d_carryOverVars[i], false, false, true, false, false);
    }
    else {
      std::string error =
        "ERROR: Switcher: Cannot find carry_over VarLabel" + d_carryOverVars[i];
      throw ProblemSetupException(error, __FILE__, __LINE__);
    }
  }
}

//______________________________________________________________________
// 
void Switcher::scheduleInitialize(const LevelP     & level,
                                        SchedulerP & sched)
{
  printSchedule(level,switcher_dbg,"Switcher::scheduleInitialize");
  d_app->scheduleInitialize(level,sched);
}

//______________________________________________________________________
// 
void Switcher::scheduleRestartInitialize(const LevelP     & level,
                                               SchedulerP & sched)
{
  printSchedule(level,switcher_dbg,"Switcher::scheduleRestartInitialize");
  d_app->scheduleRestartInitialize(level,sched);
}
//______________________________________________________________________
//
void Switcher::scheduleComputeStableTimeStep(const LevelP     & level,
                                                   SchedulerP & sched)
{
  printSchedule(level,switcher_dbg,"Switcher::scheduleComputeStableTimeStep");
  d_app->scheduleComputeStableTimeStep(level,sched);
}

//______________________________________________________________________
//
void
Switcher::scheduleTimeAdvance(const LevelP     & level,
                                    SchedulerP & sched)
{
  printSchedule(level,switcher_dbg,"Switcher::scheduleTimeAdvance");
  d_app->scheduleTimeAdvance(level,sched);
}

//______________________________________________________________________
//
void
Switcher::scheduleFinalizeTimestep( const LevelP     & level,
                                          SchedulerP & sched)
{
  printSchedule(level,switcher_dbg,"Switcher::scheduleFinalizeTimestep");
  
  d_app->scheduleFinalizeTimestep(level, sched); 
  
  scheduleSwitchTest(level,sched);

  // compute variables that are required from the old_dw for the next subcomponent
  scheduleInitNewVars(level,sched);

  scheduleSwitchInitialization(level,sched);

  // carry over vars that will be needed by a future component
  scheduleCarryOverVars(level,sched);
 
}

//______________________________________________________________________
//
void Switcher::scheduleSwitchInitialization(const LevelP     & level,
                                                  SchedulerP & sched)
{
  if (d_doSwitching[level->getIndex()]) {
    printSchedule(level,switcher_dbg,"Switcher::scheduleSwitchInitialization");
    d_app->scheduleSwitchInitialization(level, sched);
  }
}

//______________________________________________________________________
//
void Switcher::scheduleSwitchTest(const LevelP     & level,
                                        SchedulerP & sched)
{
  printSchedule(level,switcher_dbg,"Switcher::scheduleSwitchTest");
  
  d_app->scheduleSwitchTest(level,sched); // generates switch test data;

  Task* t = scinew Task("Switcher::switchTest", this, & Switcher::switchTest);

  t->setType(Task::OncePerProc);
  
  // the component is responsible for determining when it is to switch.
  t->requires(Task::NewDW, d_switch_label);
  sched->addTask(t, m_loadBalancer->getPerProcessorPatchSet(level),m_materialManager->allMaterials());
}

//______________________________________________________________________
//
void Switcher::scheduleInitNewVars(const LevelP     & level,
                                         SchedulerP & sched)
{
  unsigned int nextComp_indx = d_componentIndex+1;
  
  if( nextComp_indx >= d_numComponents ) {
    return;
  }
  
  printSchedule(level,switcher_dbg,"Switcher::scheduleInitNewVars");
  
  Task* t = scinew Task("Switcher::initNewVars",this, & Switcher::initNewVars);
  
  initVars* initVar  = d_initVars.find(nextComp_indx)->second;
  
  std::vector<const MaterialSet*> matlSet;
 
  for (unsigned i = 0; i < initVar->varLabels.size(); i++) {
    
    VarLabel* label = initVar->varLabels[i]; 
     
    // Find the MaterialSet for this variable
    // and put that set in the global structure
    const MaterialSet* matls;

    std::string nextComp_matls = initVar->matlSetNames[i];

    if (nextComp_matls == "all_matls") {
      matls = m_materialManager->allMaterials();
    }
#ifndef NO_ICE
    else if (nextComp_matls == "ice_matls" ) {
      matls = m_materialManager->allMaterials( "ICE" );
    }
#endif
#ifndef NO_MPM
    else if (nextComp_matls == "mpm_matls" ) {
      matls = m_materialManager->allMaterials( "MPM" );
    }
#endif
    else {
      throw ProblemSetupException("Bad material set", __FILE__, __LINE__);
    }
    
    matlSet.push_back(matls);

    switcher_dbg << "init Variable  " << initVar->varNames[i] << " \t matls: " 
                 << nextComp_matls << " levels " << initVar->levels[i]
                 << std::endl;
    
    const MaterialSubset* matl_ss = matls->getUnion();
    
    t->computes(label, matl_ss);
  }

  d_initVars[nextComp_indx]->matls = matlSet;

  t->requires(Task::NewDW, d_switch_label);
  sched->addTask(t,level->eachPatch(),m_materialManager->allMaterials());
}

//______________________________________________________________________
//
//
void Switcher::scheduleCarryOverVars(const LevelP     & level,
                                           SchedulerP & sched)
{
  printSchedule(level,switcher_dbg,"Switcher::scheduleCarryOverVars");
  int L_indx = level->getIndex();
  
  if (d_computedVars.size() == 0) {
    // get the set of computed vars like this, because by scheduling a carry-over var, we add to the compute list
    d_computedVars = sched->getComputedVars();
  }

  if ( d_doSwitching[L_indx] || d_restarting ) {
    // clear and reset carry-over db
    if ( L_indx >= (int) d_doCarryOverVarPerLevel.size() ) {
      d_doCarryOverVarPerLevel.resize( L_indx+1 );
    }
    d_doCarryOverVarPerLevel[L_indx].clear();

    // rebuild carry-over database

    // mark each var as carry over if it's not in the computed list
    for (unsigned i = 0; i < d_carryOverVarLabels.size(); i++) {
      
      bool do_on_this_level = !d_carryOverFinestLevelOnly[i] || L_indx == level->getGrid()->numLevels()-1;
      
      bool no_computes      = d_computedVars.find( d_carryOverVarLabels[i] ) == d_computedVars.end();
      
      bool trueFalse = ( do_on_this_level && no_computes );
      d_doCarryOverVarPerLevel[L_indx].push_back( trueFalse );
    }
  }
  
  //__________________________________
  //
  Task* t = scinew Task("Switcher::carryOverVars",this, 
                       & Switcher::carryOverVars);
                        
  // schedule the vars to be carried over (if this happens before a switch, don't do it)
  if ( L_indx < (int) d_doCarryOverVarPerLevel.size() ) {
    
    for (unsigned int i = 0; i < d_carryOverVarLabels.size(); i++) { 
    
      if ( d_doCarryOverVarPerLevel[L_indx][i] ) {
      
        VarLabel* var         = d_carryOverVarLabels[i];
        MaterialSubset* matls = d_carryOverVarMatls[i];
      
        t->requires(Task::OldDW, var, matls, Ghost::None, 0);
        t->computes(var, matls);
     
        if(d_myworld->myRank() == 0) {
          if (matls) {
            std::cout << d_myworld->myRank() << "  Carry over " << *var << "\t\tmatls: " << *matls << " on level " << L_indx << std::endl;
          }
          else {
            std::cout << d_myworld->myRank() << "  Carry over " << *var << "\t\tAll matls on level " << L_indx << "\n";
          }
        }
      }
    }  
  }
  sched->addTask(t,level->eachPatch(),m_materialManager->allOriginalMaterials());
}
//______________________________________________________________________
//  Set the flag if switch criteria has been satisfied.
void Switcher::switchTest(const ProcessorGroup *,
                          const PatchSubset    * patches,
                          const MaterialSubset * matls,
                                DataWarehouse  * old_dw,
                                DataWarehouse  * new_dw)
{
  max_vartype switch_condition;
  new_dw->get(switch_condition, d_switch_label, 0);

  if (switch_condition) {
    // actually PERFORM the switch during the next needRecompile; set back to idle then
    d_switchState = switching;
  }
  else {
    d_switchState = idle;
  }
}

//______________________________________________________________________
//  This only get executed if a switching components has been called for.
void Switcher::initNewVars(const ProcessorGroup *,
                           const PatchSubset    * patches,
                           const MaterialSubset * matls,
                                 DataWarehouse  * old_dw,
                                 DataWarehouse  * new_dw)
{
  max_vartype switch_condition;
  new_dw->get(switch_condition, d_switch_label, 0);

  if (!switch_condition)
    return; 
    
  switcher_dbg << "__________________________________" << std::endl
               << "initNewVars \t\t\t\tSwitcher" << std::endl;
  //__________________________________
  // loop over the init vars, initialize them and put them in the new_dw
  initVars* initVar  = d_initVars.find(d_componentIndex+1)->second;
  
  for (unsigned i = 0; i < initVar->varLabels.size(); i++) {
    
    VarLabel* l = initVar->varLabels[i];
    const MaterialSubset* matls = initVar->matls[i]->getUnion();

    //__________________________________
    //initialize a variable on this level?
    const Level* level  = getLevel(patches);
    int numLevels       = level->getGrid()->numLevels();
    int L_indx          = getLevel(patches)->getIndex();
    int relative_indx   = L_indx - numLevels;
    int init_Levels     = initVar->levels[i];
    
    switcher_dbg << "    varName: " << l->getName()
                 << " \t\t matls " << initVar->matlSetNames[i]
                 << " level " << init_Levels << std::endl;
    
    bool onThisLevel = false;

    if (init_Levels == L_indx     ||     // user can specify: a level,
        init_Levels == ALL_LEVELS  ||    // all levels,
        init_Levels == relative_indx) {  // or a relative indx, -1, -2
      onThisLevel = true;
    }
  
    if (onThisLevel == false) {
      continue;
    }
    
    // Bulletproofing
    if (l->typeDescription()->getType() == TypeDescription::ParticleVariable && relative_indx != -1) {
      std::ostringstream warn;
      warn << " \nERROR: switcher: subcomponent: init var: (" << l->getName()
           << ") \n particle variables can only be initialized on the finest level \n"
           << " of a multilevel grid.  Add levels=\"-1\" to that variable" << std::endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    
    //__________________________________
    // initialization section
    for (int m = 0; m < matls->size(); m++) {
      const int indx = matls->get(m);
         
      for (int p = 0; p < patches->size(); p++) {
        const Patch* patch = patches->get(p);
        
        switcher_dbg << "    indx: " << indx << " patch " << *patch << " "
                     << l->getName() << std::endl;
        
        switch (l->typeDescription()->getType()) {

          //__________________________________
          //
          case TypeDescription::CCVariable :
            switch (l->typeDescription()->getSubType()->getType()) {
              case TypeDescription::double_type : {
                CCVariable<double> q;
                new_dw->allocateAndPut(q, l, indx, patch);
                q.initialize(0);
                break;
              }
              case TypeDescription::Vector : {
                CCVariable<Vector> q;
                new_dw->allocateAndPut(q, l, indx, patch);
                q.initialize(Vector(0, 0, 0));
                break;
              }
              default :
                throw InternalError("ERROR:Switcher::initNewVars Unknown CCVariable type", __FILE__, __LINE__);
            }
            break;
            //__________________________________
            //
          case TypeDescription::NCVariable :
            switch (l->typeDescription()->getSubType()->getType()) {
              case TypeDescription::double_type : {
                NCVariable<double> q;
                new_dw->allocateAndPut(q, l, indx, patch);
                q.initialize(0);
                break;
              }
              case TypeDescription::Vector : {
                NCVariable<Vector> q;
                new_dw->allocateAndPut(q, l, indx, patch);
                q.initialize(Vector(0, 0, 0));
                break;
              }
              default :
                throw InternalError("ERROR:Switcher::initNewVars Unknown NCVariable type", __FILE__, __LINE__);
            }
            break;
            //__________________________________
            //
          case TypeDescription::ParticleVariable : {

            ParticleSubset* pset = old_dw->getParticleSubset(indx, patch);
            switch (l->typeDescription()->getSubType()->getType()) {
              case TypeDescription::int_type : {
                ParticleVariable<int> q;
                new_dw->allocateAndPut(q, l, pset);

                for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
                  q[*iter] = 0;
                }

                break;
              }
              case TypeDescription::double_type : {
                ParticleVariable<double> q;
                new_dw->allocateAndPut(q, l, pset);

                for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
                  q[*iter] = 0;
                }
                break;
              }
              case TypeDescription::Vector : {
                ParticleVariable<Vector> q;
                new_dw->allocateAndPut(q, l, pset);

                for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
                  q[*iter] = Vector(0, 0, 0);
                }
                break;
              }
              case TypeDescription::Matrix3 : {
                ParticleVariable<Matrix3> q;
                new_dw->allocateAndPut(q, l, pset);
                for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
                  q[*iter].Identity();
                }
                break;
              }
              default :
                throw InternalError("ERROR:Switcher::initNewVars Unknown particle type", __FILE__, __LINE__);
            }
            break;
          }
          default :
            throw InternalError("ERROR:Switcher::initNewVars Unknown Variable type", __FILE__, __LINE__);
        }
      }  // patch loop
    }  // matl loop
  }  // varlabel loop

  switcher_dbg << "__________________________________" << std::endl;
}
//______________________________________________________________________
//
void Switcher::carryOverVars(const ProcessorGroup *,
                             const PatchSubset    * patches,
                             const MaterialSubset * matls,
                                   DataWarehouse  * old_dw, 
                                   DataWarehouse  * new_dw)
{
  const Level* level = getLevel(patches);
  int L_indx = level->getIndex();

  if (L_indx < (int)d_doCarryOverVarPerLevel.size()) {

    for (unsigned int i = 0; i < d_carryOverVarLabels.size(); i++) {

      if (d_doCarryOverVarPerLevel[L_indx][i]) {

        const VarLabel* label = d_carryOverVarLabels[i];
        const MaterialSubset* xfer_matls = d_carryOverVarMatls[i] == 0 ? matls : d_carryOverVarMatls[i];

        //__________________________________
        //  reduction variables
        if (label->typeDescription()->isReductionVariable()) {

          switch (label->typeDescription()->getSubType()->getType()) {
            case Uintah::TypeDescription::double_type : {
              ReductionVariable<double, Reductions::Max<double> > var_d;
              old_dw->get(var_d, label);
              new_dw->put(var_d, label);
            }
              break;
            default :
              throw InternalError("ERROR:Switcher::carryOverVars - Unknown reduction variable type", __FILE__, __LINE__);
          }
        }
        else {  // all grid variables
          new_dw->transferFrom(old_dw, label, patches, xfer_matls);
        }
      }
    }
  }
}
  
//______________________________________________________________________
// 
void Switcher::switchApplication( const ProblemSpecP     & restart_prob_spec,
                                                          const GridP            & grid )
{
  // Get the initial simulation component and initialize the need components
  proc0cout << "\n------------ Switching to application (" << d_componentIndex <<") \n";
  proc0cout << "  Reading input file: " << d_in_file[d_componentIndex] << "\n";

  // Read the ups file for the first subcomponent.
  ProblemSpecP subCompUps = ProblemSpecReader().readInputFile(d_in_file[d_componentIndex]);

  UintahParallelComponent* appComp = dynamic_cast<UintahParallelComponent*>( getPort("application", d_componentIndex) );
  
  d_app = dynamic_cast<ApplicationInterface*>( appComp );
  d_app->setComponents( this );

  // Send the subcomponent's UPS file to it's sim interface.
  d_app->problemSetup(subCompUps, restart_prob_spec, const_cast<GridP&>(grid) );  

  // Send the subcomponent's UPS file to the data archiver to get the
  // output and checkpointing parameters.
  m_output->problemSetup( subCompUps, restart_prob_spec, m_materialManager );
  
  // Read in the grid adaptivity flag from the subcomponent's UPS file.
  if (m_regridder) {
    m_regridder->switchInitialize( subCompUps );
  }
  
  // Send the subcomponent's UPS file to the switcher's simulation
  // time.  Note this goes into the switcher not the subcomponent.
  proc0cout << "  Reading the <Time> block from: "
            << Uintah::basename(subCompUps->getFile()) << "\n";

  ProblemSpecP time_ps = subCompUps->findBlock("Time");

  if ( !time_ps ) {
    throw ProblemSetupException("ERROR SimulationTime \n"
                                "Can not find the <Time> block.",
                                __FILE__, __LINE__);
  }
  
  time_ps->require( "delt_min", m_delTMin );
  time_ps->require( "delt_max", m_delTMax );
  time_ps->require( "timestep_multiplier", m_delTMultiplier );

  if( !time_ps->get("delt_init", m_delTInitialMax) &&
      !time_ps->get("max_initial_delt", m_delTInitialMax) ) {
    m_delTInitialMax = 0;
  }

  if( !time_ps->get("initial_delt_range", m_delTInitialRange) ) {
    m_delTInitialRange = 0;
  }

  if( !time_ps->get("max_delt_increase", m_delTMaxIncrease) ) {
    m_delTMaxIncrease = 0;
  }
  
  if( !time_ps->get( "override_restart_delt", m_delTOverrideRestart) ) {
    m_delTOverrideRestart = 0;
  }

  // Set flags for checking reduction vars - done after the
  // subcomponent problem spec is read because the values may be based
  // on the solver being requested in the problem setup.
  setReductionVariables( appComp );
}

//______________________________________________________________________
//  This is where the actual component switching takes place.
bool
Switcher::needRecompile( const GridP & grid )
{
  switcher_dbg << "  Doing Switcher::needRecompile " << std::endl;
  
  d_restarting = true;
  d_doSwitching.resize(grid->numLevels());
  
  for (int i = 0; i < grid->numLevels(); i++) {
    d_doSwitching[i] = ( d_switchState == switching );
  }

  if (d_switchState == switching) {
    d_switchState = idle;
    d_computedVars.clear();
    d_componentIndex++;

    d_app->setupForSwitching();
    m_materialManager->clearMaterials();

    // Reseting the GeometryPieceFactory only (I believe) will ever
    // need to be done by the Switcher component...
    GeometryPieceFactory::resetFactory();

    switchApplication( nullptr, grid );
    
    // Each application has their own maximum initial delta T
    // specified.  On a switch from one application to the next, delT
    // needs to be adjusted to the value specified in the input file.
    proc0cout << "Switching the next delT from " << m_delT
              << " to " << m_delTInitialMax
              << std::endl;
    
    setDelT( m_delTInitialMax );

    // This is needed to get the "ICE surrounding matl"
    d_app->restartInitialize();
    m_materialManager->finalizeMaterials();

    proc0cout << "__________________________________\n\n";
    
    m_output->setSwitchState(true);
    
    return true;
  } 
  else
  {
    m_output->setSwitchState(false);
    
    return false;
  }
}
//______________________________________________________________________
//
void
Switcher::outputProblemSpec(ProblemSpecP& ps)
{
  ps->appendElement( "switcherComponentIndex", (int) d_componentIndex );
  ps->appendElement( "switcherState",          (int) d_switchState );
  ps->appendElement( "switcherCarryOverMatls", m_materialManager->allOriginalMaterials()->getUnion()->size());
  d_app->outputProblemSpec( ps );
}

//______________________________________________________________________
//

void
Switcher::readSwitcherState( const ProblemSpecP     & spec,
                                   MaterialManagerP & materialManager )
{
  ProblemSpecP ps = (ProblemSpecP)spec;

  int tmp;
  ps->get("switcherComponentIndex", tmp);
  d_componentIndex = tmp;

  ps->get("switcherState", tmp);
  d_switchState = (switchState)tmp;

  int numMatls = 0;
  ps->get("switcherCarryOverMatls", numMatls);

  if (numMatls != 0) {
    MaterialSet* new_matls = scinew MaterialSet;
    new_matls->addReference();
    new_matls->createEmptySubsets(1);

    for (int i = 0; i < numMatls; i++) {
      new_matls->getSubset(0)->add(i);
    }

    materialManager->setOriginalMatlsFromRestart(new_matls);
  }

  proc0cout << "  Switcher RESTART: component index = " << d_componentIndex << std::endl;
}

//______________________________________________________________________
//
void Switcher::restartInitialize()
{
  d_restarting = true;
  d_app->restartInitialize();
}

//______________________________________________________________________
//
double Switcher::recomputeDelT(const double delT)
{
  return d_app->recomputeDelT( delT );
}

//______________________________________________________________________
//     AMR
void Switcher::scheduleRefineInterface(const LevelP     & fineLevel,
                                             SchedulerP & sched,
                                             bool         needCoarseOld,
                                             bool         needCoarseNew)
{
  d_app->scheduleRefineInterface(fineLevel,sched, needCoarseOld, needCoarseNew);
}

//______________________________________________________________________
//                                    
void Switcher::scheduleRefine (const PatchSet   * patches,
                                     SchedulerP & sched){
  d_app->scheduleRefine(patches, sched);
}

//______________________________________________________________________
//
void Switcher::scheduleCoarsen(const LevelP     & coarseLevel,
                                     SchedulerP & sched)
{
  d_app->scheduleCoarsen(coarseLevel, sched);
}

//______________________________________________________________________
//
void Switcher::scheduleInitialErrorEstimate(const LevelP     & coarseLevel,
                                                  SchedulerP & sched)
{
  d_app->scheduleInitialErrorEstimate(coarseLevel,sched);
}

//______________________________________________________________________
//                                          
void Switcher::scheduleErrorEstimate(const LevelP     & coarseLevel,
                                           SchedulerP & sched)
{
  d_app->scheduleErrorEstimate(coarseLevel,sched);
}
