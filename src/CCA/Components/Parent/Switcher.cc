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

#include <CCA/Components/Parent/ComponentFactory.h>
#include <CCA/Components/Parent/Switcher.h>
#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <CCA/Components/Solvers/SolverFactory.h>
#include <CCA/Components/SwitchingCriteria/SwitchingCriteriaFactory.h>
#include <CCA/Components/SwitchingCriteria/None.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/ModelMaker.h>
#include <CCA/Ports/ProblemSpecInterface.h>
#include <CCA/Ports/Regridder.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SolverInterface.h>
#include <CCA/Ports/SwitchingCriteria.h>
#include <CCA/Ports/Output.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/OS/Dir.h>
#include <Core/Util/FileUtils.h>

#include <sstream>

using namespace std;
using namespace Uintah;

#define ALL_LEVELS  99

Switcher::Switcher( const ProcessorGroup* myworld, 
                    ProblemSpecP& ups, 
                    bool doAMR,
                    const string & uda ) : 
  UintahParallelComponent(myworld)
{
  int num_components = 0;
  d_componentIndex   = 0;
  d_switchState      = idle;
  d_restarting       = false;
  d_problemSpec      = ups;

  ProblemSpecP sim_block = ups->findBlock("SimulationComponent");
  ProblemSpecP child = sim_block->findBlock("subcomponent");

  for(; child != 0; child = child->findNextBlock("subcomponent")) {

    vector<string> init_vars;
    vector<string> init_matls;
    vector<int> init_levels;
    
    string input_file("");
    if (!child->get("input_file",input_file)) {
      throw ProblemSetupException("Need 'input_file' for subcomponent", __FILE__, __LINE__);
    }

    if( uda != "" ) {
      throw ProblemSetupException( "Uda != ''", __FILE__, __LINE__);
    }

    // This will get the component name from the input file, and the uda arg is not needed for normal simulations...
    UintahParallelComponent* comp = ComponentFactory::create(child, myworld, doAMR, "");

    SimulationInterface* sim = dynamic_cast<SimulationInterface*>(comp);

    attachPort( "sim", sim );

    string            no_solver_specified("");
    SolverInterface * solver = SolverFactory::create( child, myworld, no_solver_specified );
    
    // Attaching to switcher so that the switcher can delete it
    attachPort("sub_solver", solver);
    comp->attachPort("solver", solver);

    SwitchingCriteria * switch_criteria = SwitchingCriteriaFactory::create( child,myworld );

    if( switch_criteria ) {
      // Attaching to switcher so that the switcher can delete it
      attachPort("switch_criteria",switch_criteria);
      comp->attachPort("switch_criteria",switch_criteria);
    }

    // Get the vars that will need to be initialized by this component
    for( ProblemSpecP var = child->findBlock("init"); var != 0; var = var->findNextBlock("init") ) {
      map<string,string> attributes;
      var->getAttributes(attributes);
      string name   = attributes["var"];
      string matls  = attributes["matls"];
      
      stringstream s_level(attributes["levels"]);
      int levels = ALL_LEVELS;
      s_level >> levels;
      
      if (name != ""){ 
        init_vars.push_back(name);
      }else{
        continue;
      }
      
      init_levels.push_back(levels);
      init_matls.push_back(matls);
    }
    d_initVars.push_back(init_vars);
    d_initMatls.push_back(init_matls);
    d_initLevels.push_back(init_levels);
    num_components++;
  }

  // Make sure that a switching criteria was specified.  For n subcomponents,
  // there should be n-1 switching critiera specified.

  int num_switch_criteria = 0;
  for (int i = 0; i < num_components; i++) {
    UintahParallelComponent* comp = 
      dynamic_cast<UintahParallelComponent*>(getPort("sim",i));
    SwitchingCriteria* sw = 
      dynamic_cast<SwitchingCriteria*>(comp->getPort("switch_criteria"));
    if (sw) {
      num_switch_criteria++;
    }
  }
  
  // Add the None SwitchCriteria to the last component, so the switchFlag label
  // is computed in the last stage.

  UintahParallelComponent* last_comp =
    dynamic_cast<UintahParallelComponent*>(getPort("sim",num_components-1));

  SwitchingCriteria* none_switch_criteria = scinew None();
  
  // Attaching to switcher so that the switcher can delete it
  attachPort("switch_criteria",none_switch_criteria);
  last_comp->attachPort("switch_criteria",none_switch_criteria);
  
  if (num_switch_criteria != num_components-1) {
    throw  ProblemSetupException( "Do not have enough switching criteria specified for the number of components.",
                                  __FILE__, __LINE__ );
  }
  
  // Get the vars that will need to be initialized by this component
  for( ProblemSpecP var = sim_block->findBlock("carry_over"); var != 0; var = var->findNextBlock("carry_over") ) {
    map<string,string> attributes;
    var->getAttributes(attributes);
    string name = attributes["var"];
    string matls = attributes["matls"];
    string level = attributes["level"];
    if (name != "") {
      d_carryOverVars.push_back(name);
    }
    MaterialSubset* carry_over_matls = 0;
    if (matls != "") {
      carry_over_matls = scinew MaterialSubset;
      ConsecutiveRangeSet crs = matls;
      ConsecutiveRangeSet::iterator iter = crs.begin();
      for (; iter != crs.end(); iter++)
        carry_over_matls->add(*iter);
      carry_over_matls->addReference();
    }
    d_carryOverVarMatls.push_back(carry_over_matls);
    if (level == "finest") {
      d_carryOverFinestLevelOnly.push_back(true);
    }
    else {
      d_carryOverFinestLevelOnly.push_back(false);
    }
  }
  d_numComponents = num_components;
  d_computedVars.clear();
  //d_switchLabel = VarLabel::create("postSwitchTest", max_vartype::getTypeDescription());
}

Switcher::~Switcher()
{
  for (unsigned i = 0; i < d_carryOverVarMatls.size(); i++)
    if (d_carryOverVarMatls[i] && d_carryOverVarMatls[i]->removeReference())
      delete d_carryOverVarMatls[i];
  d_carryOverVarMatls.clear();

  for (unsigned i = 0; i < numConnections("sim"); i++)
    delete getPort("sim",i);
  
  for (unsigned i = 0; i < numConnections("switch_criteria"); i++)
    delete getPort("switch_criteria",i);
  
  for (unsigned i = 0; i < numConnections("sub_solver"); i++)
    delete getPort("sub_solver",i);
  
  for (unsigned i = 0; i < numConnections("problem spec"); i++)
    delete getPort("problem spec",i);

  //VarLabel::destroy(d_switchLabel);
}

void
Switcher::problemSetup( const ProblemSpecP& params, 
                        const ProblemSpecP& restart_prob_spec, GridP& grid,
                        SimulationStateP& sharedState )
{
  if( params.get_rep() != d_problemSpec.get_rep() ) {
    throw InternalError( "Switcher problemSetup ProblemSpec is different from initialization ProblemSpec ", __FILE__, __LINE__);    
  }

  d_sim = dynamic_cast<SimulationInterface*>(getPort("sim",d_componentIndex));

  // Some components need the output port attached to each individual component
  // At the time of Switcher constructor, the data archiver is not available.
  Output* output = dynamic_cast<Output*>(getPort("output"));
  Scheduler* sched = dynamic_cast<Scheduler*>(getPort("scheduler"));
  ModelMaker* modelmaker = dynamic_cast<ModelMaker*>(getPort("modelmaker"));
  
  for (unsigned i = 0; i < d_numComponents; i++) {
    UintahParallelComponent* comp = dynamic_cast<UintahParallelComponent*>(getPort("sim",i));

    comp->attachPort("output",output);
    comp->attachPort("scheduler",sched);
    comp->attachPort("modelmaker",modelmaker);
    // Do the material maker stuff
  }

  for (unsigned i = 0; i < d_componentIndex; i++) {
    // qwerty: this doesn't seem to call problemSetup() for the first component...?

    SimulationInterface * sim = dynamic_cast<SimulationInterface*> (getPort("sim",i));	 
    sim->problemSetup( params, restart_prob_spec, grid, sharedState );
    sharedState->clearMaterials();
  }

  // get the varLabels for carryOver and init Vars from the strings we found above
  for (unsigned i = 0; i < d_initVars.size(); i++) {
    vector<string>& names = d_initVars[i];
    vector<VarLabel*> labels;
    for (unsigned j = 0; j < names.size(); j++) {
      VarLabel* label = VarLabel::find(names[j]);
      if (label) {
        labels.push_back(label);
        sched->overrideVariableBehavior(names[j], false, false, true);
      }
      else {
        throw ProblemSetupException("Cannot find VarLabel", __FILE__, __LINE__);
      }
    }
    d_initVarLabels.push_back(labels);
  }

  for (unsigned i = 0; i < d_carryOverVars.size(); i++) {
    VarLabel* label = VarLabel::find(d_carryOverVars[i]);
    if (label) {
      d_carryOverVarLabels.push_back(label);
      sched->overrideVariableBehavior(d_carryOverVars[i], false, false, true);
    }
    else {
      string error = "Cannot find VarLabel = " + d_carryOverVars[i];
      throw ProblemSetupException(error, __FILE__, __LINE__);
    }
  }

  d_sharedState = sharedState;

  // re-initialize the DataArchiver to output according the the new component's specs
  dynamic_cast<Output*>(getPort("output"))->problemSetup( params, d_sharedState.get_rep() );

  // do this again, in case of a restart
  Regridder* regridder = dynamic_cast<Regridder*>(getPort("regridder"));
  if (regridder) {
    regridder->switchInitialize( params );
  }

  // re-initialize the time info
  d_sharedState->d_simTime->problemSetup( params );
}
 
void Switcher::scheduleInitialize(const LevelP& level,
                                  SchedulerP& sched)
{
  d_sim->scheduleInitialize(level,sched);
}
 
void Switcher::scheduleComputeStableTimestep(const LevelP& level,
                                             SchedulerP& sched)
{
  d_sim->scheduleComputeStableTimestep(level,sched);
}

void
Switcher::scheduleTimeAdvance(const LevelP& level, SchedulerP& sched)
{
  d_sim->scheduleTimeAdvance(level,sched);
}

void
Switcher::scheduleFinalizeTimestep( const LevelP& level, SchedulerP& sched)
{
  d_sim->scheduleFinalizeTimestep(level, sched);

  scheduleSwitchTest(level,sched);

  // compute vars for the next component that may not have been computed by the current
  scheduleInitNewVars(level,sched);

  scheduleSwitchInitialization(level,sched);

  // carry over vars that will be needed by a future component
  scheduleCarryOverVars(level,sched);
}

void Switcher::scheduleSwitchInitialization(const LevelP& level, 
                                            SchedulerP& sched)
{
  if (d_doSwitching[level->getIndex()]) {
    d_sim->switchInitialize(level,sched);
  }
}

void Switcher::scheduleSwitchTest(const LevelP& level, SchedulerP& sched)
{
  d_sim->scheduleSwitchTest(level,sched); // generates switch test data;

  Task* t = scinew Task("Switcher::switchTest",
                        this, & Switcher::switchTest);

  t->setType(Task::OncePerProc);
  
  // the component is responsible to determine when it is to switch.
  t->requires(Task::NewDW,d_sharedState->get_switch_label());
  //t->computes(d_switchLabel, level.get_rep(), d_sharedState->refineFlagMaterials());
  sched->addTask(t,sched->getLoadBalancer()->getPerProcessorPatchSet(level),d_sharedState->allMaterials());
}

void Switcher::scheduleInitNewVars(const LevelP& level, SchedulerP& sched)
{
  Task* t = scinew Task("Switcher::initNewVars",
                        this, & Switcher::initNewVars);
  t->requires(Task::NewDW,d_sharedState->get_switch_label());
  sched->addTask(t,level->eachPatch(),d_sharedState->allMaterials());
}

// note - won't work if number of levels changes
void Switcher::scheduleCarryOverVars(const LevelP& level, SchedulerP& sched)
{
  if (d_computedVars.size() == 0) {
    // get the set of computed vars like this, because by scheduling a carry-over
    // var, we add to the compute list
    d_computedVars = sched->getComputedVars();
  }

  if (d_doSwitching[level->getIndex()] || d_restarting) {
    // clear and reset carry-over db
    if (level->getIndex() >= (int) d_doCarryOverVarPerLevel.size()) {
      d_doCarryOverVarPerLevel.resize(level->getIndex()+1);
    }
    d_doCarryOverVarPerLevel[level->getIndex()].clear();

    // rebuild carry-over db

    // mark each var as carry over if it's not in the computed list
    for (unsigned i = 0; i < d_carryOverVarLabels.size(); i++) {
      bool do_on_this_level = !d_carryOverFinestLevelOnly[i] || level->getIndex() == level->getGrid()->numLevels()-1;
      bool no_computes = d_computedVars.find(d_carryOverVarLabels[i]) == d_computedVars.end();
      d_doCarryOverVarPerLevel[level->getIndex()].push_back(do_on_this_level && no_computes);
    }
  }

  Task* t = scinew Task("Switcher::carryOverVars",
                        this, & Switcher::carryOverVars);
  // schedule the vars for carrying over (if this happens before a switch, don't do it)
  if (level->getIndex() < (int) d_doCarryOverVarPerLevel.size()) {
    for (unsigned int i = 0; i < d_carryOverVarLabels.size(); i++) { 
      if (d_doCarryOverVarPerLevel[level->getIndex()][i]) {
        VarLabel* var = d_carryOverVarLabels[i];
        MaterialSubset* matls = d_carryOverVarMatls[i];
        t->requires(Task::OldDW, var, matls, Ghost::None, 0);
        t->computes(var, matls);
     
        if(UintahParallelComponent::d_myworld->myrank() == 0){
          if (matls)
            cout << d_myworld->myrank() << "  Carry over " << *var << "\t\tmatls: " << *matls << " on level " << level->getIndex() << endl;
          else
            cout << d_myworld->myrank() << "  Carry over " << *var << "\t\tAll matls on level " << level->getIndex() << "\n";
        }
      }
    }  
  }
  sched->addTask(t,level->eachPatch(),d_sharedState->originalAllMaterials());
  
}

void Switcher::switchTest(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  max_vartype switch_condition;
  new_dw->get(switch_condition,d_sharedState->get_switch_label(),0);

  if (switch_condition) {
    // actually PERFORM the switch during the next needRecompile; set back to idle then
    d_switchState = switching;
  } else {
    d_switchState = idle;
  }
}

void Switcher::initNewVars(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  max_vartype switch_condition;
  new_dw->get(switch_condition,d_sharedState->get_switch_label(),0);

  if (!switch_condition)
    return;

  for (unsigned i = 0; i < d_initVarLabels[d_componentIndex+1].size(); i++) {
    VarLabel* l = d_initVarLabels[d_componentIndex+1][i];
    const MaterialSubset* matls;
    if (d_initMatls[d_componentIndex+1][i] == "ice_matls")
      matls = d_sharedState->allICEMaterials()->getSubset(0);
    else if (d_initMatls[d_componentIndex+1][i] == "mpm_matls")
      matls = d_sharedState->allMPMMaterials()->getSubset(0);
    else if (d_initMatls[d_componentIndex+1][i] == "all_matls")
      matls = d_sharedState->allMaterials()->getSubset(0);
    else 
      throw ProblemSetupException("Bad material set", __FILE__, __LINE__);
    //__________________________________
    //initialize a variable on this level?
    const Level* level = getLevel(patches);
    int numLevels = level->getGrid()->numLevels();
    int L_indx = getLevel(patches)->getIndex();
    int relative_indx = L_indx - numLevels;
    int init_Levels = d_initLevels[d_componentIndex+1][i];
    
    bool onThisLevel = false;

    if( init_Levels == L_indx      ||   // user can specify: a level,
        init_Levels == ALL_LEVELS  ||   // nothing,
        init_Levels == relative_indx){  // or a relative indx, -1, -2
      onThisLevel = true;
    }
  
    if(onThisLevel == false){
      continue;
    }
    
    // Bulletproofing
    if(l->typeDescription()->getType() == TypeDescription::ParticleVariable &&
       relative_indx != -1){
      ostringstream warn;
      warn << " \nERROR: switcher: subcomponent: init var: (" << l->getName() 
           << ") \n particle variables can only be initialized on the finest level \n"
           << " of a multilevel grid.  Add levels=\"-1\" to that variable" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    
    
    for (int m = 0; m < matls->size(); m++) {
      const int indx = matls->get(m);
        
      for (int p = 0; p < patches->size(); p++) {
        const Patch* patch = patches->get(p);
        // loop over certain vars and init them into the DW
        switch(l->typeDescription()->getType()) {
        case TypeDescription::CCVariable:
          switch(l->typeDescription()->getSubType()->getType()) {
          case TypeDescription::double_type:
            {
            CCVariable<double> q;
            new_dw->allocateAndPut(q, l, indx,patch);
            q.initialize(0);
            break;
            }
          case TypeDescription::Vector:
            {
            CCVariable<Vector> q;
            new_dw->allocateAndPut(q, l, indx,patch);
            q.initialize(Vector(0,0,0));
            break;
            }
          default:
            throw ProblemSetupException("Unknown type", __FILE__, __LINE__);
          }
          break;
        case TypeDescription::NCVariable:
          switch(l->typeDescription()->getSubType()->getType()) {
          case TypeDescription::double_type:
            {
            NCVariable<double> q;
            new_dw->allocateAndPut(q, l, indx,patch);
            q.initialize(0);
            break;
            }
          case TypeDescription::Vector:
            {
            NCVariable<Vector> q;
            new_dw->allocateAndPut(q, l, indx,patch);
            q.initialize(Vector(0,0,0));
            break;
            }
          default:
            throw ProblemSetupException("Unknown type", __FILE__, __LINE__);
          }
          break;
        case TypeDescription::ParticleVariable:
          {
          ParticleSubset* pset = new_dw->getParticleSubset(indx, patch);
          switch(l->typeDescription()->getSubType()->getType()) {
          case TypeDescription::int_type:
            {
            ParticleVariable<int> q;
            constParticleVariable<int> qcopy;
            new_dw->allocateAndPut(q, l, pset);
            for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++)
              q[*iter] = 0;
            break;
            }
          case TypeDescription::double_type:
            {
            ParticleVariable<double> q;
            constParticleVariable<double> qcopy;
            new_dw->allocateAndPut(q, l, pset);
            for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++)
              q[*iter] = 0;
            break;
            }
          case TypeDescription::Vector:
            {
            ParticleVariable<Vector> q;
            new_dw->allocateAndPut(q, l, pset);
            for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++)
              q[*iter] = Vector(0,0,0);
            break;
            }
          case TypeDescription::Matrix3:
            {
            ParticleVariable<Matrix3> q;
            new_dw->allocateAndPut(q, l, pset);
            for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++)
              q[*iter].Identity();
            break;
            }
          default:
            throw ProblemSetupException("Unknown type", __FILE__, __LINE__);
          }          
          break;
          }
        default:
          throw ProblemSetupException("Unknown type", __FILE__, __LINE__);
        }
      }
    }
  }
}

void Switcher::carryOverVars(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  if (level->getIndex() < (int) d_doCarryOverVarPerLevel.size()) {
    for (unsigned int i = 0; i < d_carryOverVarLabels.size(); i++) { 
      if (d_doCarryOverVarPerLevel[level->getIndex()][i]) {
        VarLabel* var = d_carryOverVarLabels[i];
        const MaterialSubset* xfer_matls = d_carryOverVarMatls[i] == 0 ? matls : d_carryOverVarMatls[i];
        new_dw->transferFrom(old_dw, var, patches, xfer_matls);
      }
    }  
  }
}

bool
Switcher::needRecompile( double time, double delt, const GridP& grid )
{
  bool retval = false;
  d_restarting = true;
  d_doSwitching.resize(grid->numLevels());
  for (int i = 0; i < grid->numLevels(); i++) {
    d_doSwitching[i] = ( d_switchState == switching );
  }

  if (d_switchState == switching) {
    d_switchState = idle;
    d_computedVars.clear();
    if (d_myworld->myrank() == 0) {
      cout << "------------ Switching components: ";
    }

    d_componentIndex++;
    d_sharedState->clearMaterials();
    d_sharedState->d_switchState = true;
    d_sim = 
      dynamic_cast<SimulationInterface*>(getPort("sim",d_componentIndex)); 
#if 0
    ProblemSpecInterface* psi = 
      dynamic_cast<ProblemSpecInterface*>(getPort("problem spec",
                                                  d_componentIndex));
    ProblemSpecP ups,restart_prob_spec=0;
    if (psi) {
      ups = psi->readInputFile();
      d_sim->problemSetup(ups,restart_prob_spec,const_cast<GridP&>(grid),
                          d_sharedState);
        
      if (d_myworld->myrank() == 0){
        string filename = psi->getInputFile();
        cout << " Reading input file " << filename << "\n";;
      }                                 
    }
#endif
    // we need this to get the "ICE surrounding matl"
    d_sim->restartInitialize();
    d_sharedState->finalizeMaterials();

    // re-initialize the DataArchiver to output according the the new component's specs
//  dynamic_cast<Output*>(getPort("output"))->problemSetup( ups, d_sharedState.get_rep() );
    dynamic_cast<Output*>(getPort("output"))->problemSetup( d_problemSpec, d_sharedState.get_rep() );
    // re-initialize some Regridder Parameters
    Regridder* regridder = dynamic_cast<Regridder*>(getPort("regridder"));
    if (regridder) {
//    regridder->switchInitialize(ups);
      regridder->switchInitialize( d_problemSpec );
    }

//  d_sharedState->d_simTime->problemSetup(ups);
    d_sharedState->d_simTime->problemSetup( d_problemSpec );

    retval = true;
  } 
  else {
    d_sharedState->d_switchState = false;
  }
  retval |= d_sim->needRecompile(time, delt, grid);
  return retval;
}

void
Switcher::outputProblemSpec(ProblemSpecP& ps)
{
  d_sim->outputProblemSpec( ps );
}

void
Switcher::outputPS( Dir & dir )
{
  for (unsigned i = 0; i < d_numComponents; i++) {
    
#if 0
    ProblemSpecInterface* psi = 
      dynamic_cast<ProblemSpecInterface*>(getPort("problem spec",i));
    ProblemSpecP ups = psi->readInputFile();
#endif
    std::stringstream stream;
    stream << i;
    string inputname = dir.getName() + "/input.xml." + stream.str();
    cout << "outputing file " << inputname << endl;
//  ups->output(inputname.c_str());
    d_problemSpec->output(inputname.c_str());
  }
  
  string inputname = dir.getName()+"/input.xml";
  ProblemSpecP inputDoc = ProblemSpecReader().readInputFile( inputname );

  int count = 0;
  ProblemSpecP sim_block = inputDoc->findBlock("SimulationComponent");
  for (ProblemSpecP child = sim_block->findBlock("subcomponent"); child != 0; 
       child = child->findNextBlock("subcomponent")) {
    ProblemSpecP in_file = child->findBlock("input_file");
    string nodeName = in_file->getNodeName();
    cout << "nodeName = " << nodeName << endl;
    if (nodeName == "input_file") {
      std::stringstream stream;
      stream << count++;
//    string inputname = dir.getName() + "/input.xml." + stream.str();
      string inputname = "input.xml." + stream.str();
      cout << "inputname = " << inputname << endl;
      child->appendElement("input_file",inputname);
    }
    child->removeChild(in_file);
    
  }
  inputDoc->output( inputname.c_str() );
//inputDoc->releaseDocument();
}

void
Switcher::addToTimestepXML(ProblemSpecP& spec)
{
  spec->appendElement( "switcherComponentIndex", (int) d_componentIndex );
  spec->appendElement( "switcherState", (int) d_switchState );
  spec->appendElement( "switcherCarryOverMatls", d_sharedState->originalAllMaterials()->getUnion()->size());
}

void
Switcher::readFromTimestepXML(const ProblemSpecP& spec,SimulationStateP& state)
{
  // problemSpec doesn't handle unsigned
  ProblemSpecP ps = (ProblemSpecP) spec;
  int tmp;
  ps->get("switcherComponentIndex", tmp);
  d_componentIndex = tmp; 
  ps->get("switcherState", tmp);
  d_switchState = (switchState) tmp;
  
  int numMatls = 0;
  ps->get("switcherCarryOverMatls",numMatls);

  if (numMatls != 0) {
    MaterialSet* new_matls = scinew MaterialSet;
    new_matls->addReference();
    new_matls->createEmptySubsets(1);
    for (int i = 0; i < numMatls; i++)
      new_matls->getSubset(0)->add(i);
    state->setOriginalMatlsFromRestart(new_matls);
  }
  
  if (d_myworld->myrank() == 0) {
    cout << "  Switcher RESTART: component index = " << d_componentIndex << endl;
  }
}

void Switcher::addMaterial(const ProblemSpecP& params, GridP& grid,
                           SimulationStateP& state)
{
  d_sim->addMaterial(params, grid, state);
}

void Switcher::scheduleInitializeAddedMaterial(const LevelP& level,
                                               SchedulerP& sched)
{
  d_sim->scheduleInitializeAddedMaterial(level, sched);
}

void Switcher::restartInitialize() {
  d_restarting = true;
  d_sim->restartInitialize();
}

bool Switcher::restartableTimesteps() {
  return d_sim->restartableTimesteps();
}


double Switcher::recomputeTimestep(double dt) {
  return d_sim->recomputeTimestep(dt);
}

//______________________________________________________________________
//     AMR
void Switcher::scheduleRefineInterface(const LevelP& fineLevel,
                                       SchedulerP& sched,
                                       bool needCoarseOld, 
                                       bool needCoarseNew)
{
  d_sim->scheduleRefineInterface(fineLevel,sched, needCoarseOld, needCoarseNew);
}
                                    
void Switcher::scheduleRefine (const PatchSet* patches, 
                               SchedulerP& sched){
  d_sim->scheduleRefine(patches, sched);
}

void Switcher::scheduleCoarsen(const LevelP& coarseLevel, 
                               SchedulerP& sched){
  d_sim->scheduleCoarsen(coarseLevel, sched);
}


void Switcher::scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                            SchedulerP& sched){
  d_sim->scheduleInitialErrorEstimate(coarseLevel,sched);
}
                                          
void Switcher::scheduleErrorEstimate(const LevelP& coarseLevel,
                                     SchedulerP& sched){
  d_sim->scheduleErrorEstimate(coarseLevel,sched);
}
