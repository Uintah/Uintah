/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#include <CCA/Components/Parent/ComponentFactory.h>
#include <CCA/Components/Parent/MultiScaleSwitcher.h>
#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <CCA/Components/Solvers/SolverFactory.h>
#include <CCA/Components/SwitchingCriteria/None.h>
#include <CCA/Components/SwitchingCriteria/SwitchingCriteriaFactory.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/ModelMaker.h>
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
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/OS/Dir.h>
#include <Core/Parallel/Parallel.h>


#include <iomanip>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>

using namespace Uintah;

static DebugStream dbg("MULTISCALE_SWITCHER", false);

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
MultiScaleSwitcher::MultiScaleSwitcher( const ProcessorGroup * myworld,
                                              ProblemSpecP   & d_master_ups,
                                              bool             doAMR,
                                        const std::string    & uda )
  : UintahParallelComponent(myworld)
{
  proc0cout << "-----------------------------MultiScaleSwitcher::MultiScaleSwitcher top"<< std::endl;

  int num_components = 0;
  d_componentIndex   = 0;
  d_switchState      = idle;
  d_restarting       = false;

  std::set<std::string>       simComponents;

  ProblemSpecP sim_block = d_master_ups->findBlock("SimulationComponent");
  ProblemSpecP child     = sim_block->findBlock("subcomponent");

  //__________________________________
  //  loop over the subcomponents
  for (; child != 0; child = child->findNextBlock("subcomponent")) {
    
    //__________________________________
    //  Read in subcomponent ups file and store the filename
    std::string input_file("");
    if (!child->get("input_file", input_file)) {
      throw ProblemSetupException("Need 'input_file' for subcomponent", __FILE__, __LINE__);
    }
    
    proc0cout << "Input file:\t\t" << input_file << std::endl;
    
    d_in_file.push_back(input_file);
    ProblemSpecP subCompUps = ProblemSpecReader().readInputFile(input_file);


    // get the component name from the input file, and the uda arg is not needed for normal simulations...  
    std::string sim_comp;
    ProblemSpecP sim_ps = subCompUps->findBlock("SimulationComponent");
    sim_ps->getAttribute("type", sim_comp);
    simComponents.insert(sim_comp);
    d_componentNameIndexMap.insert( std::pair<std::string, int>(sim_comp, num_components));

    //__________________________________
    // create simulation port and attach it switcher component    
    UintahParallelComponent* comp = ComponentFactory::create(subCompUps, myworld, doAMR, "");
    SimulationInterface* sim = dynamic_cast<SimulationInterface*>(comp);
    attachPort("sim", sim);

    //__________________________________
    // Create solver port and attach it to the switcher component.
    SolverInterface * solver = SolverFactory::create(subCompUps, myworld);

    attachPort("sub_solver", solver);
    comp->attachPort("solver", solver);

    //__________________________________
    // create switching criteria port and attach it switcher component
    SwitchingCriteria * switch_criteria = SwitchingCriteriaFactory::create(child, myworld);

    if (switch_criteria) {
      attachPort("switch_criteria", switch_criteria);
      comp->attachPort("switch_criteria", switch_criteria);
    }

    //__________________________________
    // Get the variables that will need to be initialized by this subcomponent
    initVars* initVar = scinew initVars;
    for (ProblemSpecP var = child->findBlock("init"); var != 0; var = var->findNextBlock("init")) {

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
      std::string varName = attributes["var"];
      initVar->varNames.push_back(varName);
    }

    d_initVars[num_components] = initVar;

    num_components++;


    proc0cout << "\n";
  }  // loop over subcomponents
  
  
  //__________________________________
  // Bulletproofing:
  if (simComponents.count("mpm") && simComponents.count("rmpmice")) {
    throw ProblemSetupException("MultiScaleSwitcher: The simulation subComponents rmpmice and mpm cannot be used together", __FILE__,
                                __LINE__);
  }
  
  
  //__________________________________
  // Bulletproofing:
  // Make sure that a switching criteria was specified.  For n subcomponents,
  // there should be n-1 switching critiera specified.
  int num_switch_criteria = 0;
  for (int i = 0; i < num_components; i++) {
    UintahParallelComponent* comp = dynamic_cast<UintahParallelComponent*>(getPort("sim", i));
    SwitchingCriteria* sw = dynamic_cast<SwitchingCriteria*>(comp->getPort("switch_criteria"));
    if (sw) {
      num_switch_criteria++;
    }
  }

  if (num_switch_criteria != num_components - 1) {
    throw ProblemSetupException("Do not have enough switching criteria specified for the number of components.", __FILE__, __LINE__);
  }
  
  //__________________________________
  // Add the "None" SwitchCriteria to the last component, so the switchFlag label
  // is computed in the last stage.

  UintahParallelComponent* last_comp = dynamic_cast<UintahParallelComponent*>(getPort("sim", num_components - 1));

  SwitchingCriteria* none_switch_criteria = scinew None();

  // Attaching to switcher so that the switcher can delete it
  attachPort("switch_criteria", none_switch_criteria);
  last_comp->attachPort("switch_criteria", none_switch_criteria);
  
  
  
  //__________________________________
  // Get the vars that will need to be carried over 
  for (ProblemSpecP var = sim_block->findBlock("carry_over"); var != 0; var = var->findNextBlock("carry_over")) {
    std::map<std::string, std::string> attributes;
    var->getAttributes(attributes);
    std::string name = attributes["var"];
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
  }  // loop over carry-over variables

  d_numComponents = num_components;
  d_computedVars.clear();

  proc0cout << "Number of components " << d_numComponents << std::endl;
  proc0cout << "-----------------------------MultiScaleSwitcher::MultiScaleSwitcher bottom" << std::endl;
}
//______________________________________________________________________
//
MultiScaleSwitcher::~MultiScaleSwitcher()
{

  if (dbg.active()) {
    dbg << d_myworld->myrank() << " Switcher::~Switcher" << std::endl;
  }

  for (unsigned i = 0; i < d_carryOverVarMatls.size(); i++) {
    if (d_carryOverVarMatls[i] && d_carryOverVarMatls[i]->removeReference()) {
      delete d_carryOverVarMatls[i];
    }
  }
  d_carryOverVarMatls.clear();
}
//_____________________________________________________________________
// Do pre-grid setup related stuff
void
MultiScaleSwitcher::preGridProblemSetup(const ProblemSpecP&     params,
                                              GridP&            grid,
                                              SimulationStateP& state)
{
  // Do nothing (for now)
}

//______________________________________________________________________
// Setup the first component 
void
MultiScaleSwitcher::problemSetup( const ProblemSpecP     & /*params*/,
                                  const ProblemSpecP     & restart_prob_spec,
                                        GridP            & grid,
                                        SimulationStateP & sharedState )
{  
  if (dbg.active()) {
    dbg << "Doing ProblemSetup \t\t\t\tSwitcher" << std::endl;
  }

  if (restart_prob_spec) {
    readSwitcherState(restart_prob_spec, sharedState);
  }

  d_sharedState = sharedState;
  d_sim                         = dynamic_cast<SimulationInterface*>(     getPort("sim",d_componentIndex) );
  UintahParallelComponent* comp = dynamic_cast<UintahParallelComponent*>( getPort("sim",d_componentIndex) );
  Scheduler* sched              = dynamic_cast<Scheduler*>(               getPort("scheduler") );
  Output* dataArchiver          = dynamic_cast<Output*>(                  getPort("output") );
  ModelMaker* modelmaker        = dynamic_cast<ModelMaker*>(              getPort("modelmaker") );
  comp->attachPort("scheduler", sched);
  comp->attachPort("output",    dataArchiver);
  comp->attachPort("modelmaker",modelmaker);

  //__________________________________
  //Read the ups file for the first subcomponent
  ProblemSpecP subCompUps = ProblemSpecReader().readInputFile(d_in_file[d_componentIndex]);

  dataArchiver->problemSetup( subCompUps, d_sharedState.get_rep() );

  d_sim->problemSetup(subCompUps, restart_prob_spec, grid, sharedState );

  // read in the grid adaptivity flag from the ups file
  Regridder* regridder = dynamic_cast<Regridder*>(getPort("regridder"));
  if (regridder) {
    regridder->switchInitialize(subCompUps);
  }

  // read in <Time> block from ups file
  d_sharedState->d_simTime->problemSetup( subCompUps );

  //__________________________________
  // init Variables:
  //   - determine the label from the string names
  //   - determine the MaterialSet from the string matlSetName
  //   - store this info to be used later
  proc0cout << "\n-----------------------------------\n";
  std::map<int, initVars*>::iterator it;
  for (it = d_initVars.begin(); it != d_initVars.end(); it++) {

    int comp = it->first;
    proc0cout << "  init Variables:  component: " << comp << std::endl;
    initVars* tmp = it->second;

    // Find the varLabel   
    std::vector<std::string>& varNames = tmp->varNames;
    std::vector<VarLabel*> varLabels = tmp->varLabels;

    for (unsigned j = 0; j < varNames.size(); j++) {

      std::string varName = varNames[j];
      VarLabel* label = VarLabel::find(varName);

      if (!label) {
        std::string error = "ERROR: Switcher: Cannot find init VarLabel" + varName;
        throw ProblemSetupException(error, __FILE__, __LINE__);
      }

      varLabels.push_back(label);
      // so the variable is not scrubbed from the data warehouse
      sched->overrideVariableBehavior(varName, false, false, true, false, false);
    }

    d_initVars[comp]->varLabels = varLabels;
  }

  //__________________________________
  // Carry over labels
  for (unsigned i = 0; i < d_carryOverVars.size(); i++) {
    VarLabel* label = VarLabel::find(d_carryOverVars[i]);
    if (label) {
      d_carryOverVarLabels.push_back(label);

      // so variable is not scrubbed from the data warehouse
      sched->overrideVariableBehavior(d_carryOverVars[i], false, false, true, false, false);
    }
    else {
      std::string error = "ERROR: Switcher: Cannot find carry_over VarLabel" + d_carryOverVars[i];
      throw ProblemSetupException(error, __FILE__, __LINE__);
    }
  }
  proc0cout << "-----------------------------------\n" << std::endl;
}

SimulationInterface* MultiScaleSwitcher::matchComponentToLevelset(const LevelP&     level)
{
  GridP grid = level->getGrid();
  std::string levelSetComponent = grid->getSubsetComponentName(level->getIndex());
  int         componentIndex = d_componentNameIndexMap.find(levelSetComponent)->second;

  proc0cout << "Matching: " << levelSetComponent << " level: " << level->getIndex()
            << " to level set: " << componentIndex << std::endl;
  SimulationInterface* component = dynamic_cast<SimulationInterface*> (getPort("sim", componentIndex));
  return component;
}

//______________________________________________________________________
// 
void MultiScaleSwitcher::scheduleInitialize( const LevelP & level,
                                             SchedulerP   & sched )
{
  printSchedule(level, dbg, "MultiScaleSwitcher::scheduleInitialize");
  SimulationInterface* component = matchComponentToLevelset(level);
  component->scheduleInitialize(level,sched);
}

//______________________________________________________________________
// 
void MultiScaleSwitcher::scheduleRestartInitialize( const LevelP     & level,
                                                          SchedulerP & sched )
{
  printSchedule(level, dbg, "MultiScaleSwitcher::scheduleRestartInitialize");
  SimulationInterface* component = matchComponentToLevelset(level);
  component->scheduleRestartInitialize(level,sched);
}
//______________________________________________________________________
//
void MultiScaleSwitcher::scheduleComputeStableTimestep( const LevelP     & level,
                                                              SchedulerP & sched )
{
  printSchedule(level, dbg, "MultiScaleSwitcher::scheduleComputeStableTimestep");
  SimulationInterface* component = matchComponentToLevelset(level);
  component->scheduleComputeStableTimestep(level,sched);
}

//______________________________________________________________________
//
void
MultiScaleSwitcher::scheduleTimeAdvance( const LevelP     & level,
                                               SchedulerP & sched )
{
  printSchedule(level, dbg, "MultiScaleSwitcher::scheduleTimeAdvance");
  SimulationInterface* component = matchComponentToLevelset(level);
  proc0cout << " Scheduling Time Advance on Level: " << level->getIndex()
            << " Component: " << level->getGrid()->getSubsetComponentName(level->getIndex())
            << std::endl;
  component->scheduleTimeAdvance(level,sched);
}

//______________________________________________________________________
//
void
MultiScaleSwitcher::scheduleFinalizeTimestep( const LevelP     & level,
                                                    SchedulerP & sched )
{
  printSchedule(level, dbg, "MultiScaleSwitcher::scheduleFinalizeTimestep");
  SimulationInterface* component = matchComponentToLevelset(level);
  component->scheduleFinalizeTimestep(level,sched);

  scheduleSwitchTest(level, sched);

  // compute variables that are required from the old_dw for the next subcomponent
  scheduleInitNewVars(level, sched);

  scheduleSwitchInitialization(level, sched);

  // carry over vars that will be needed by a future component
  scheduleCarryOverVars(level, sched);
 
}

//______________________________________________________________________
//
void MultiScaleSwitcher::scheduleSwitchInitialization( const LevelP     & level,
                                                             SchedulerP & sched )
{
  if (d_doSwitching[level->getIndex()]) {
    printSchedule(level, dbg, "MultiScaleSwitcher::scheduleSwitchInitialization");
    SimulationInterface* component = matchComponentToLevelset(level);
    component->switchInitialize(level,sched);
//    d_sim->switchInitialize(level, sched);
  }
}

//______________________________________________________________________
//
void MultiScaleSwitcher::scheduleSwitchTest( const LevelP     & level,
                                                   SchedulerP & sched )
{
  printSchedule(level, dbg, "MultiScaleSwitcher::scheduleSwitchTest");
  SimulationInterface* component = matchComponentToLevelset(level);
  component->scheduleSwitchTest(level, sched);  // generates switch test data;

  Task* t = scinew Task("MultiScaleSwitcher::switchTest", this, &MultiScaleSwitcher::switchTest);

  t->setType(Task::OncePerProc);

  // the component is responsible for determining when it is to switch.
  t->requires(Task::NewDW, d_sharedState->get_switch_label());
  sched->addTask(t, sched->getLoadBalancer()->getPerProcessorPatchSet(level->getSubsetIndex()), d_sharedState->allMaterials());
}

//______________________________________________________________________
//
void MultiScaleSwitcher::scheduleInitNewVars( const LevelP     & level,
                                                    SchedulerP & sched )
{
  unsigned int nextComp_indx = d_componentIndex + 1;

  if (nextComp_indx >= d_numComponents) {
    return;
  }

  printSchedule(level, dbg, "MultiScaleSwitcher::scheduleInitNewVars");

  Task* t = scinew Task("MultiScaleSwitcher::initNewVars", this, &MultiScaleSwitcher::initNewVars);

  initVars* initVar = d_initVars.find(nextComp_indx)->second;

  std::vector<const MaterialSet*> matlSet;

  for (unsigned i = 0; i < initVar->varLabels.size(); i++) {

    VarLabel* label = initVar->varLabels[i];

    // Find the MaterialSet for this variable and put that set in the global structure
    const MaterialSet* matls;

    std::string nextComp_matls = initVar->matlSetNames[i];
    if (nextComp_matls == "ice_matls") {
      matls = d_sharedState->allICEMaterials();
    }
    else if (nextComp_matls == "mpm_matls") {
      matls = d_sharedState->allMPMMaterials();
    }
    else if (nextComp_matls == "md_matls") {
      matls = d_sharedState->allMDMaterials();
    }
    else if (nextComp_matls == "all_matls") {
      matls = d_sharedState->allMaterials();
    }
    else {
      throw ProblemSetupException("Bad material set", __FILE__, __LINE__);
    }

    matlSet.push_back(matls);

    proc0cout << "init Variable  " << initVar->varNames[i] << " \t matls: " << nextComp_matls << " levels " << initVar->levels[i]
              << std::endl;

    const MaterialSubset* matl_ss = matls->getUnion();

    if (label->typeDescription()->getType() == TypeDescription::ReductionVariable) {
      t->computes(label);
    }
    else {
      t->computes(label, matl_ss);
    }
  }

  d_initVars[nextComp_indx]->matls = matlSet;

  t->requires(Task::NewDW, d_sharedState->get_switch_label());
  sched->addTask(t, level->eachPatch(), d_sharedState->allMaterials());
}

//______________________________________________________________________
//
//
void MultiScaleSwitcher::scheduleCarryOverVars( const LevelP     & level,
                                                      SchedulerP & sched )
{
  printSchedule(level, dbg, "MultiScaleSwitcher::scheduleCarryOverVars");

  int L_indx = level->getIndex();

  if (d_computedVars.size() == 0) {
    // get the set of computed vars like this, because by scheduling a carry-over var, we add to the compute list
    d_computedVars = sched->getComputedVars();
  }

  if (d_doSwitching[L_indx] || d_restarting) {
    // clear and reset carry-over db
    if (L_indx >= (int)d_doCarryOverVarPerLevel.size()) {
      d_doCarryOverVarPerLevel.resize(L_indx + 1);
    }
    d_doCarryOverVarPerLevel[L_indx].clear();

    // rebuild carry-over database - mark each var as carry over if it's not in the computed list
    for (unsigned i = 0; i < d_carryOverVarLabels.size(); i++) {

      bool do_on_this_level = !d_carryOverFinestLevelOnly[i] || L_indx == level->getGrid()->numLevels() - 1;
      bool no_computes = d_computedVars.find(d_carryOverVarLabels[i]) == d_computedVars.end();
      bool trueFalse = (do_on_this_level && no_computes);

      d_doCarryOverVarPerLevel[L_indx].push_back(trueFalse);
    }
  }
  
  //__________________________________
  //
  Task* t = scinew Task("MultiScaleSwitcher::carryOverVars", this, &MultiScaleSwitcher::carryOverVars);

  // schedule the vars to be carried over (if this happens before a switch, don't do it)
  if (L_indx < (int)d_doCarryOverVarPerLevel.size()) {

    for (unsigned int i = 0; i < d_carryOverVarLabels.size(); i++) {

      if (d_doCarryOverVarPerLevel[L_indx][i]) {

        VarLabel* var = d_carryOverVarLabels[i];
        MaterialSubset* matls = d_carryOverVarMatls[i];

        t->requires(Task::OldDW, var, matls, Ghost::None, 0);
        t->computes(var, matls);

        if (UintahParallelComponent::d_myworld->myrank() == 0) {
          if (matls) {
            std::cout << d_myworld->myrank() << "  Carry over " << *var << "\t\tmatls: " << *matls << " on level " << L_indx
                      << std::endl;
          }
          else {
            std::cout << d_myworld->myrank() << "  Carry over " << *var << "\t\tAll matls on level " << L_indx << "\n";
          }
        }
      }
    }
  }
  sched->addTask(t, level->eachPatch(), d_sharedState->originalAllMaterials());
}
//______________________________________________________________________
//  Set the flag if switch criteria has been satisfied.
void MultiScaleSwitcher::switchTest( const ProcessorGroup * /*pg*/,
                                     const PatchSubset    * patches,
                                     const MaterialSubset * matls,
                                           DataWarehouse  * old_dw,
                                           DataWarehouse  * new_dw )
{
  max_vartype switch_condition;
  new_dw->get(switch_condition, d_sharedState->get_switch_label(), 0);

  // ParticleSubset debugging output --------------------------------------------
  std::cout << std::endl << " Old Data Warehouse: " << std::endl;
  OnDemandDataWarehouse* odOldDW = static_cast<OnDemandDataWarehouse*> (old_dw);
  odOldDW->printParticleSubsets();
  std::cout << std::endl << " New Data Warehouse: " << std::endl;
  OnDemandDataWarehouse* odNewDW = static_cast<OnDemandDataWarehouse*> (new_dw);
  odNewDW->printParticleSubsets();
  std::cout << std::endl;
  // ParticleSubset debugging output --------------------------------------------

  if (switch_condition) {
    // actually PERFORM the switch during the next needRecompile; set back to idle then
    d_switchState = switching;
  }
  else {
    d_switchState = idle;
  }
}

//______________________________________________________________________
//  This only gets executed if a switching component has been called for.
void MultiScaleSwitcher::initNewVars( const ProcessorGroup * /*pg*/,
                                      const PatchSubset    * patches,
                                      const MaterialSubset * matls,
                                            DataWarehouse  * old_dw,
                                            DataWarehouse  * new_dw)
{
  max_vartype switch_condition;
  new_dw->get(switch_condition, d_sharedState->get_switch_label(), 0);

  if (!switch_condition) {
    return;
  }

  proc0cout << "\n-----------------------------------\n";
  proc0cout << "initNewVars \t\t\t\tSwitcher" << std::endl;
  //__________________________________
  // loop over the init vars, initialize them and put them in the new_dw
  initVars* initVar = d_initVars.find(d_componentIndex + 1)->second;

  for (unsigned i = 0; i < initVar->varLabels.size(); i++) {

    VarLabel* l = initVar->varLabels[i];
    const MaterialSubset* matls = initVar->matls[i]->getUnion();

    //__________________________________
    //initialize a variable on this level?
    const Level* level = getLevel(patches);
    int numLevels = level->getGrid()->numLevels();
    int L_indx = getLevel(patches)->getIndex();
    int relative_indx = L_indx - numLevels;
    int init_Levels = initVar->levels[i];

    proc0cout << "    varName: " << l->getName() << " \t\t matls " << initVar->matlSetNames[i] << " level " << init_Levels
              << std::endl;

    bool onThisLevel = false;

    if (init_Levels == L_indx ||    // user can specify: a level,
        init_Levels == ALL_LEVELS ||    // nothing,
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

        proc0cout << "    indx: " << indx << " patch " << *patch << " " << l->getName() << std::endl;

        switch ( l->typeDescription()->getType() ) {

          //__________________________________
          //
          case TypeDescription::CCVariable :
            switch ( l->typeDescription()->getSubType()->getType() ) {
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
            switch ( l->typeDescription()->getSubType()->getType() ) {
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
            switch ( l->typeDescription()->getSubType()->getType() ) {
              case TypeDescription::int_type : {
                ParticleVariable<int> q;
                new_dw->allocateAndPut(q, l, pset);

                for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
                  q[*iter] = 0;
                }

                break;
              }
              case TypeDescription::long64_type : {
                ParticleVariable<long64> q;
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
          case TypeDescription::ReductionVariable : {
            switch ( l->typeDescription()->getSubType()->getType() ) {
              case Uintah::TypeDescription::double_type : {
                ReductionVariable<double, Reductions::Sum<double> > var_d;
                new_dw->put(var_d, l);
                break;
              }
              default :
                throw InternalError("ERROR:Switcher::initNewVars Unknown ReductionVariable type", __FILE__, __LINE__);
            }
            break;
          }
          default :
            throw InternalError("ERROR:Switcher::initNewVars Unknown Variable type", __FILE__, __LINE__);
        }
      }  // patch loop
    }  // matl loop
  }  // varlabel loop
  proc0cout << "\n-----------------------------------" << std::endl;
}

//______________________________________________________________________
//
void MultiScaleSwitcher::carryOverVars( const ProcessorGroup * /*pg*/,
                                        const PatchSubset    * patches,
                                        const MaterialSubset * matls,
                                              DataWarehouse  * old_dw,
                                              DataWarehouse  * new_dw )
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

          switch ( label->typeDescription()->getSubType()->getType() ) {
            case Uintah::TypeDescription::double_type : {
              ReductionVariable<double, Reductions::Sum<double> > var_d;
              new_dw->put(var_d, label);
              break;
            }
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
//  This is where the actual component switching takes place.
bool
MultiScaleSwitcher::needRecompile(       double   time,
                                         double   delt,
                                   const GridP  & grid )
{
  if (dbg.active()) {
    dbg << "  Doing Switcher::needRecompile " << std::endl;
  }

  bool retval = false;
  d_restarting = true;

  d_doSwitching.resize(grid->numLevels());
  for (int i = 0; i < grid->numLevels(); i++) {
    d_doSwitching[i] = (d_switchState == switching);
  }

  if (d_switchState == switching) {

    d_switchState = idle;
    d_computedVars.clear();
    d_componentIndex++;
    d_componentIndex %= d_numComponents;

    // For multi-scale, we don't want to clear materials, need to keep them around
//    d_sharedState->clearMaterials();
    d_sharedState->d_switchState = true;

    // Reseting the GeometryPieceFactory only should ever need to be done by the Switcher component...
//    GeometryPieceFactory::resetFactory();

    //__________________________________
    // get the next simulation component
    // and initialize the scheduler and dataArchiver
    d_sim                                  = dynamic_cast<SimulationInterface*>( getPort("sim",d_componentIndex) );
    UintahParallelComponent * comp         = dynamic_cast<UintahParallelComponent*>( getPort("sim",d_componentIndex) );
    Scheduler               * sched        = dynamic_cast<Scheduler*>( getPort("scheduler") );
    Output                  * dataArchiver = dynamic_cast<Output*>( getPort("output") );
    ModelMaker              * modelmaker   = dynamic_cast<ModelMaker*>( getPort("modelmaker") );
    
    comp->attachPort("scheduler", sched);
    comp->attachPort("output",    dataArchiver);
    comp->attachPort("modelmaker",modelmaker);

//    // clean up old models
//    if (modelmaker) {
//      modelmaker->clearModels();
//    }

    proc0cout << "\n__________________________________ Switching to component (" << d_componentIndex << ") \n";
    proc0cout << "  Reading input file: " << d_in_file[d_componentIndex] << "\n";

    // read in the problemSpec on next subcomponent
    ProblemSpecP prob_spec = 0;
    ProblemSpecP subCompUps = ProblemSpecReader().readInputFile(d_in_file[d_componentIndex]);
    SimulationStateP subcomp_sharedstate = scinew SimulationState(subCompUps);

    // TODO APH JBH - We'll probably want to create a new SimulationTime for MD instances and read in from teh md input file
    // For now - assign original SimulationTime object to new shared state
    subcomp_sharedstate->d_simTime = d_sharedState->d_simTime;

    // execute the subcomponent ProblemSetup
    d_sim->problemSetup(subCompUps, prob_spec, const_cast<GridP&>(grid), subcomp_sharedstate);

    // read in <DataArchiver> section
    dataArchiver->problemSetup(subCompUps, subcomp_sharedstate.get_rep());

//    // we need this to get the "ICE surrounding matl"
//    d_sim->restartInitialize();
    subcomp_sharedstate->finalizeMaterials();

    // read in the grid adaptivity flag from the ups file
    Regridder* regridder = dynamic_cast<Regridder*>(getPort("regridder"));
    if (regridder) {
      regridder->switchInitialize(subCompUps);
    }

    LevelSet running_level_set;
    running_level_set.addAll(grid->getLevelSubset(d_componentIndex)->getVector());

    int numSubsets = running_level_set.size();
    for (int subsetIndex = 0; subsetIndex < numSubsets; ++subsetIndex) {
      LevelSubset* currSubset = running_level_set.getSubset(subsetIndex);
      int numInSubset = currSubset->size();
      for (int indexInSubset = 0; indexInSubset < numInSubset; ++indexInSubset) {
        LevelP levelHandle = grid->getLevel(currSubset->get(indexInSubset)->getIndex());
        levelHandle->setSubsetIndex(indexInSubset);
      }
    }

    LoadBalancer* lb = sched->getLoadBalancer();
    lb->possiblyDynamicallyReallocate(running_level_set, LoadBalancer::init);

    // create subscheduler for MD component
    d_subScheduler = sched->createSubScheduler();
    d_subScheduler->initialize(1,1);
    d_subScheduler->clearMappings();
    d_subScheduler->mapDataWarehouse(Task::ParentOldDW, 0);
    d_subScheduler->mapDataWarehouse(Task::ParentNewDW, 1);
    d_subScheduler->mapDataWarehouse(Task::OldDW, 0);
    d_subScheduler->mapDataWarehouse(Task::NewDW, 1);

    // Initialize the per-levelset data
    const LevelSubset* level_subset = grid->getLevelSubset(d_componentIndex);
    int num_levels = level_subset->size();
    for (int i =  0; i < num_levels; ++i) {
      LevelP level_handle = grid->getLevel(level_subset->get(i)->getIndex());
      d_sim->scheduleInitialize(level_handle, d_subScheduler);
      d_sim->scheduleComputeStableTimestep(level_handle, d_subScheduler);
    }

    d_subScheduler->advanceDataWarehouse(grid);

    d_subScheduler->compile(&running_level_set);

//    d_subScheduler->get_dw(0)->setScrubbing(DataWarehouse::ScrubNone);
    d_subScheduler->execute();

    d_subScheduler->initialize(1,1);
    d_subScheduler->clearMappings();
    d_subScheduler->mapDataWarehouse(Task::ParentOldDW, 0);
    d_subScheduler->mapDataWarehouse(Task::ParentNewDW, 1);
    d_subScheduler->mapDataWarehouse(Task::OldDW, 0);
    d_subScheduler->mapDataWarehouse(Task::NewDW, 1);


    for (int i =  0; i < num_levels; ++i) {
      LevelP level_handle = grid->getLevel(level_subset->get(i)->getIndex());
      d_sim->scheduleTimeAdvance(level_handle, d_subScheduler);
      d_sim->scheduleComputeStableTimestep(level_handle, d_subScheduler);
    }

    d_subScheduler->compile(&running_level_set);
    d_subScheduler->advanceDataWarehouse(grid);
//    d_subScheduler->get_dw(0)->setScrubbing(DataWarehouse::ScrubNone);

    d_subScheduler->execute();

    retval = false;
    proc0cout << "\n-----------------------------------" << std::endl;

    d_componentIndex++;
    d_componentIndex %= d_numComponents;

    d_sharedState->d_switchState = false;
  } 
  else {
    d_sharedState->d_switchState = false;
  }

  retval |= d_sim->needRecompile(time, delt, grid);

  return retval;
}

//______________________________________________________________________
//
void
MultiScaleSwitcher::outputProblemSpec( ProblemSpecP& ps )
{
  ps->appendElement( "switcherComponentIndex", (int) d_componentIndex );
  ps->appendElement( "switcherState",          (int) d_switchState );
  ps->appendElement( "switcherCarryOverMatls", d_sharedState->originalAllMaterials()->getUnion()->size());
  d_sim->outputProblemSpec( ps );
}

//______________________________________________________________________
//
void
MultiScaleSwitcher::outputPS( Dir& dir )
{
#if 0
  // TURN THIS OFF.  It appears to be working without it.
  // We can problem remove outputPS from the source tree
  for (unsigned i = 0; i < d_numComponents; i++) {

    std::stringstream stream;
    stream << i;
    std::string inputname = dir.getName() + "/input.xml." + stream.str();
    std::cout << "switcher:outputing file " << inputname << std::endl;
    d_master_ups->output(inputname.c_str());
  }

  std::string inputname = dir.getName() + "/input.xml";
  ProblemSpecP inputDoc = ProblemSpecReader().readInputFile(inputname);

  int count = 0;
  ProblemSpecP sim_block = inputDoc->findBlock("SimulationComponent");
  for (ProblemSpecP child = sim_block->findBlock("subcomponent"); child != 0; child = child->findNextBlock("subcomponent")) {

    ProblemSpecP in_file = child->findBlock("input_file");
    std::string nodeName = in_file->getNodeName();
    std::cout << "nodeName = " << nodeName << std::endl;

    if (nodeName == "input_file") {
      std::stringstream stream;
      stream << count++;
      std::string inputname = "input.xml." + stream.str();
      std::cout << "inputname = " << inputname << std::endl;
      child->appendElement("input_file", inputname);
    }
    child->removeChild(in_file);

  }
  inputDoc->output(inputname.c_str());
#endif
}

//______________________________________________________________________
//
void
MultiScaleSwitcher::readSwitcherState( const ProblemSpecP     & spec,
                                             SimulationStateP & state )
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

    state->setOriginalMatlsFromRestart(new_matls);
  }

  proc0cout << "  Switcher RESTART: component index = " << d_componentIndex << std::endl;
}

//______________________________________________________________________
//
void MultiScaleSwitcher::restartInitialize()
{
  d_restarting = true;
  d_sim->restartInitialize();
}

//______________________________________________________________________
//
bool MultiScaleSwitcher::restartableTimesteps()
{
  return d_sim->restartableTimesteps();
}

//______________________________________________________________________
//
double MultiScaleSwitcher::recomputeTimestep( double dt )
{
  return d_sim->recomputeTimestep(dt);
}
