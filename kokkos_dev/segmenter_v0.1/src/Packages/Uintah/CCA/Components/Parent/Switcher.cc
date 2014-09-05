#include <Packages/Uintah/CCA/Components/Parent/Switcher.h>
#include <Packages/Uintah/CCA/Components/Parent/ComponentFactory.h>
#include <Packages/Uintah/CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/ProblemSpecInterface.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/CCA/Ports/ModelMaker.h>
#include <Packages/Uintah/CCA/Ports/SwitchingCriteria.h>
#include <Packages/Uintah/CCA/Components/Solvers/SolverFactory.h>
#include <Packages/Uintah/CCA/Components/SwitchingCriteria/SwitchingCriteriaFactory.h>
#include <Packages/Uintah/CCA/Components/SwitchingCriteria/None.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/Core/Grid/Variables/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/SimpleMaterial.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>



using namespace Uintah;

Switcher::Switcher(const ProcessorGroup* myworld, ProblemSpecP& ups, 
                   bool doAMR) : UintahParallelComponent(myworld)
{
  int num_components = 0;
  d_componentIndex = 0;
  d_switchState = idle;

  ProblemSpecP sim_block = ups->findBlock("SimulationComponent");
  for (ProblemSpecP child = sim_block->findBlock("subcomponent"); child != 0; 
       child = child->findNextBlock("subcomponent")) {

    vector<string> init_vars;
    vector<string> init_matls;
    string in("");
    if (!child->get("input_file",in))
      throw ProblemSetupException("Need input file for subcomponent", __FILE__, __LINE__);

    // it will get the component name from the input file, and the uda arg is not needed for normal sims
    UintahParallelComponent* comp = ComponentFactory::create(child, myworld, doAMR, "", "");
    SimulationInterface* sim = dynamic_cast<SimulationInterface*>(comp);
    attachPort("sim", sim);
    attachPort("problem spec", scinew ProblemSpecReader(in));

    string no_solver_specified("");
    SolverInterface* solver = SolverFactory::create(child,myworld,
                                                    no_solver_specified);

    comp->attachPort("solver", solver);

    SwitchingCriteria* switch_criteria = 
      SwitchingCriteriaFactory::create(child,myworld);

    if (switch_criteria)
      comp->attachPort("switch_criteria",switch_criteria);
                                                                         

    // get the vars that will need to be initialized by this component
    for (ProblemSpecP var=child->findBlock("init"); var != 0; var = var->findNextBlock("init")) {
      map<string,string> attributes;
      var->getAttributes(attributes);
      string name = attributes["var"];
      string matls = attributes["matls"];
      if (name != "") 
        init_vars.push_back(name);
      else
        continue;
      init_matls.push_back(matls);
    }
    d_initVars.push_back(init_vars);
    d_initMatls.push_back(init_matls);
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
    if (sw)
      num_switch_criteria++;
  }
  
  // Add the None SwitchCriteria to the last component, so the switchFlag label
  // is computed in the last stage.

  UintahParallelComponent* last_comp =
    dynamic_cast<UintahParallelComponent*>(getPort("sim",num_components-1));

  SwitchingCriteria* none_switch_criteria = new None();
  last_comp->attachPort("switch_criteria",none_switch_criteria);
  
  if (num_switch_criteria != num_components-1) {
    throw  ProblemSetupException("Do not have enough switching criteria specified for the number of components.",
                                 __FILE__, __LINE__);
  }
      
  
  // get the vars that will need to be initialized by this component
  for (ProblemSpecP var=sim_block->findBlock("carry_over"); var != 0; var = var->findNextBlock("carry_over")) {
    map<string,string> attributes;
    var->getAttributes(attributes);
    string name = attributes["var"];
    if (name != "") {
      d_carryOverVars.push_back(name);
    }
  }
  d_numComponents = num_components;

  //d_switchLabel = VarLabel::create("postSwitchTest", max_vartype::getTypeDescription());
}

Switcher::~Switcher()
{
  for (unsigned i = 0; i < d_carryOverVarMatls.size(); i++)
    if (d_carryOverVarMatls[i] && d_carryOverVarMatls[i]->removeReference())
      delete d_carryOverVarMatls[i];
  d_carryOverVarMatls.clear();
  //VarLabel::destroy(d_switchLabel);
}

void Switcher::problemSetup(const ProblemSpecP& params, 
                            const ProblemSpecP& materials_ps, GridP& grid,
                            SimulationStateP& sharedState)
{
  d_sim = dynamic_cast<SimulationInterface*>(getPort("sim",d_componentIndex));

  // Some components need the output port attached to each individual component
  // At the time of Switcher constructor, the data archiver is not available.
  Output* output = dynamic_cast<Output*>(getPort("output"));
  ModelMaker* modelmaker = dynamic_cast<ModelMaker*>(getPort("modelmaker"));
  
  for (unsigned i = 0; i < d_numComponents; i++) {
    UintahParallelComponent* comp =
      dynamic_cast<UintahParallelComponent*>(getPort("sim",i));

    comp->attachPort("output",output);
    comp->attachPort("modelmaker",modelmaker);
    // Do the materialmaker stuff

  }

  for (unsigned i = 0; i < d_componentIndex; i++) {
    SimulationInterface* sim = dynamic_cast<SimulationInterface*> (getPort("sim",i));	 
    ProblemSpecInterface* psi = 
      dynamic_cast<ProblemSpecInterface*>(getPort("problem spec",i));
    ProblemSpecP ups = psi->readInputFile();
    sim->problemSetup(ups,materials_ps,grid,sharedState);
    sharedState->clearMaterials();
  }
  
  // clear it out and do the first one again
  //sharedState->clearMaterials();
  ProblemSpecInterface* psi = 
    dynamic_cast<ProblemSpecInterface*>(getPort("problem spec",d_componentIndex));
  ProblemSpecP ups;
  if (psi) {
    ups = psi->readInputFile();
    d_sim->problemSetup(ups,materials_ps,grid,sharedState);
  } else {
    throw InternalError("psi dynamic_cast failed", __FILE__, __LINE__);
  }

  // get the varLabels for carryOver and init Vars from the strings we found above
  for (unsigned i = 0; i < d_initVars.size(); i++) {
    vector<string>& names = d_initVars[i];
    vector<VarLabel*> labels;
    for (unsigned j = 0; j < names.size(); j++) {
      VarLabel* label = VarLabel::find(names[j]);
      if (label) {
        labels.push_back(label);
      }
      else
        throw ProblemSetupException("Cannot find VarLabel", __FILE__, __LINE__);
    }
    d_initVarLabels.push_back(labels);
  }

  for (unsigned i = 0; i < d_carryOverVars.size(); i++) {
    VarLabel* label = VarLabel::find(d_carryOverVars[i]);
    if (label) {
      d_carryOverVarLabels.push_back(label);
    }
    else {
      string error = "Cannot find VarLabel = " + d_carryOverVars[i];
      throw ProblemSetupException(error, __FILE__, __LINE__);
    }
  }

  d_sharedState = sharedState;

  // re-initialize the DataArchiver to output according the the new component's specs
  dynamic_cast<Output*>(getPort("output"))->problemSetup(ups, d_sharedState.get_rep());

  // re-initialize the time info
  d_sharedState->d_simTime->problemSetup(ups);
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
Switcher::scheduleTimeAdvance(const LevelP& level, SchedulerP& sched,
                              int a, int b )
{
  d_sim->scheduleTimeAdvance(level,sched,a,b);
  scheduleSwitchTest(level,sched);

  // compute vars for the next component that may not have been computed by the current
  scheduleInitNewVars(level,sched);

  // carry over vars that will be needed by a future component
  scheduleCarryOverVars(level,sched);
}

void Switcher::scheduleSwitchTest(const LevelP& level, SchedulerP& sched)
{
  d_sim->scheduleSwitchTest(level,sched); // generates switch test data;

  Task* t = scinew Task("Switcher::switchTest",
                        this, & Switcher::switchTest);

  // the component is responsible to determine when it is to switch.
  t->requires(Task::NewDW,d_sharedState->get_switch_label(), level.get_rep());
  //t->computes(d_switchLabel, level.get_rep(), d_sharedState->refineFlagMaterials());
  sched->addTask(t,level->eachPatch(),d_sharedState->allMaterials());
}

void Switcher::scheduleInitNewVars(const LevelP& level, SchedulerP& sched)
{
  Task* t = scinew Task("Switcher::initNewVars",
                        this, & Switcher::initNewVars);
  t->requires(Task::NewDW,d_sharedState->get_switch_label(), level.get_rep());
  sched->addTask(t,level->eachPatch(),d_sharedState->allMaterials());
}

// note - won't work if number of levels changes
void Switcher::scheduleCarryOverVars(const LevelP& level, SchedulerP& sched)
{

  static bool first = true;
  Task* t = scinew Task("Switcher::carryOverVars",
                        this, & Switcher::carryOverVars);
  if (d_switchState == switching || first) {
    first = false;
    // clear carry-over db
    if (level->getIndex() >= (int) d_matlVarsDB.size()) {
      d_matlVarsDB.resize(level->getIndex()+1);
    }
    for (matlVarsType::iterator iter = d_matlVarsDB[level->getIndex()].begin(); iter != d_matlVarsDB[level->getIndex()].end(); iter++) {
      if (iter->second && iter->second->removeReference()) {
        delete iter->second;
      }
    }
    d_matlVarsDB[level->getIndex()].clear();
    const PatchSet* procset = sched->getLoadBalancer()->createPerProcessorPatchSet(level);
    // rebuild carry-over db
    for (unsigned i = 0; i < d_carryOverVarLabels.size(); i++) {
      if (!d_carryOverVarLabels[i])
        continue;
      bool found = false;
      for (int j = 0; j < sched->getNumTasks(); j++) {
        for(Task::Dependency* dep = sched->getTask(j)->getComputes(); dep != 0; dep=dep->next){
          //if(!sched->isOldDW(dep->mapDataWarehouse()))
          //continue;
          if (dep->var == d_carryOverVarLabels[i]) {
            found = true;
            break;
          }
        }
        // it's already being required by somebody else, so ignore it
        if (found)
          break;
      }
      if (found) {
        //cout << "  Not carrying over " << *d_carryOverVarLabels[i] << endl;
        continue;
      }
      MaterialSubset* matls = scinew MaterialSubset;
      matls->addReference();
      for (int j = 0; j < d_sharedState->getMaxMatlIndex(); j++) {
        // if it exists in the old DW for this level (iff it is 
        // on the level, it will be in the procSet's patches)
        if (sched->get_dw(0)->exists(d_carryOverVarLabels[i], j, procset->getSubset(d_myworld->myrank())->get(0))) {
          matls->add(j);
        }
      }
      d_matlVarsDB[level->getIndex()][d_carryOverVarLabels[i]] = matls;
    }
  }

  // schedule the vars for carrying over (if this happens before a switch, don't do it)
  if (level->getIndex() < (int) d_matlVarsDB.size()) {
    for (matlVarsType::iterator iter = d_matlVarsDB[level->getIndex()].begin(); 
         iter != d_matlVarsDB[level->getIndex()].end();
         iter++) {
      t->requires(Task::OldDW, iter->first, iter->second, Task::OutOfDomain, Ghost::None, 0);
      t->computes(iter->first, iter->second, Task::OutOfDomain);
      //cout << d_myworld->myrank() << "  Carry over " << *iter->first << " matl " << *iter->second << endl;
    }  
  }
  sched->addTask(t,level->eachPatch(),d_sharedState->allMaterials());
  
  d_switchState = idle;
}

void Switcher::switchTest(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  max_vartype switch_condition;
  new_dw->get(switch_condition,d_sharedState->get_switch_label(),getLevel(patches));

  if (switch_condition) {
    // we'll set it back to idle at the bottom of the next timestep's scheduleCarryOverVars
    // actually PERFORM the switch during the needRecompile
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
  new_dw->get(switch_condition,d_sharedState->get_switch_label(),getLevel(patches));

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
        case TypeDescription::ParticleVariable:
          {
          ParticleSubset* pset = new_dw->getParticleSubset(indx, patch);
          switch(l->typeDescription()->getSubType()->getType()) {
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
  if (level->getIndex() < (int) d_matlVarsDB.size()) {
    for (matlVarsType::iterator iter = d_matlVarsDB[level->getIndex()].begin(); 
         iter != d_matlVarsDB[level->getIndex()].end();
         iter++) {
      new_dw->transferFrom(old_dw, iter->first, patches, iter->second);
    }  
  }
}
bool Switcher::needRecompile(double time, double delt, const GridP& grid)
{
  bool retval = false;
  if (d_switchState == switching) {
    if (d_myworld->myrank() == 0)
      cout << "   Switching components!\n";
    d_componentIndex++;
    d_sharedState->clearMaterials();
    d_sharedState->d_switchState = true;
    d_sim = 
      dynamic_cast<SimulationInterface*>(getPort("sim",d_componentIndex)); 
    ProblemSpecInterface* psi = 
      dynamic_cast<ProblemSpecInterface*>(getPort("problem spec",
                                                  d_componentIndex));

    ProblemSpecP ups,materials_ps=0;
    if (psi) {
      ups = psi->readInputFile();
      d_sim->problemSetup(ups,materials_ps,const_cast<GridP&>(grid),
                          d_sharedState);
    }
    // we need this to get the "ICE surrounding matl"
    d_sim->restartInitialize();
    d_sharedState->finalizeMaterials();

    // re-initialize the DataArchiver to output according the the new component's specs
    dynamic_cast<Output*>(getPort("output"))->problemSetup(ups, d_sharedState.get_rep());

    d_sharedState->d_simTime->problemSetup(ups);

    retval = true;
  } else
    d_sharedState->d_switchState = false;
  retval |= d_sim->needRecompile(time, delt, grid);
  return retval;
}

void Switcher::addToTimestepXML(ProblemSpecP& spec)
{
  spec->appendElement( "switcherComponentIndex", (int) d_componentIndex );
  spec->appendElement( "switcherState", (int) d_switchState );
}

void Switcher::readFromTimestepXML(const ProblemSpecP& spec)
{
  // problemSpec doesn't handle unsigned
  ProblemSpecP ps = (ProblemSpecP) spec;
  int tmp;
  ps->get("switcherComponentIndex", tmp);
  d_componentIndex = tmp; 
  ps->get("switcherState", tmp);
  d_switchState = (switchState) tmp;
  if (d_myworld->myrank() == 0)
    cout << "  Switcher RESTART: component index = " << d_componentIndex << endl;
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
  d_sim->restartInitialize();
}

bool Switcher::restartableTimesteps() {
  return d_sim->restartableTimesteps();
}


double Switcher::recomputeTimestep(double dt) {
  return d_sim->recomputeTimestep(dt);
}
