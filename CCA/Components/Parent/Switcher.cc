#include <Packages/Uintah/CCA/Components/Parent/Switcher.h>
#include <Packages/Uintah/CCA/Components/Parent/ComponentFactory.h>
#include <Packages/Uintah/CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/ProblemSpecInterface.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/CCA/Ports/ModelMaker.h>
#include <Packages/Uintah/CCA/Components/Solvers/SolverFactory.h>
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

    UintahParallelComponent* comp = ComponentFactory::create(child, myworld, doAMR);
    SimulationInterface* sim = dynamic_cast<SimulationInterface*>(comp);
    attachPort("sim", sim);
    attachPort("problem spec", scinew ProblemSpecReader(in));
    string no_solver_specified("");
    SolverInterface* solver = SolverFactory::create(child,myworld,
                                                    no_solver_specified);
    comp->attachPort("solver", solver);


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
      if (matls != "")
        init_matls.push_back(matls);
              
    }
    d_initVars.push_back(init_vars);
    d_initMatls.push_back(init_matls);
    num_components++;
  }
  
  // get the vars that will need to be initialized by this component
  for (ProblemSpecP var=sim_block->findBlock("carry_over"); var != 0; var = var->findNextBlock("init")) {
    map<string,string> attributes;
    var->getAttributes(attributes);
    string name = attributes["var"];
    if (name != "") 
      d_carryOverVars.push_back(name);
  }
  d_numComponents = num_components;

  switchLabel = VarLabel::create("switch.bool",
                                 SoleVariable<bool>::getTypeDescription());
}

Switcher::~Switcher()
{
  VarLabel::destroy(switchLabel);
}

void Switcher::problemSetup(const ProblemSpecP& params, GridP& grid,
                            SimulationStateP& sharedState)
{
  d_sim = dynamic_cast<SimulationInterface*>(getPort("sim",d_componentIndex));
  
  // Some components need the output port attached to each individual component
  // At the time of Switcher constructor, the data archiver is not available.
  Output* output = dynamic_cast<Output*>(getPort("output"));

  ModelMaker* modelmaker = dynamic_cast<ModelMaker*>(getPort("modelmaker"));

  for (unsigned int i = 0; i < d_numComponents; i++) {
    UintahParallelComponent* comp = 
      dynamic_cast<UintahParallelComponent*>(getPort("sim",i));
    comp->attachPort("output",output);
    comp->attachPort("modelmaker",modelmaker);
  }

  // maybe not the best way to do this right now, but we need to do this to get 
  // all the VarLabels created, so we don't have to do intermediate timesteps.
  // also, this will enable us to automatically init_vars and carry_over_vars.

  for (unsigned i = 0; i < d_numComponents; i++) {
    ProblemSpecInterface* psi = 
      dynamic_cast<ProblemSpecInterface*>(getPort("problem spec",i));
    if (psi) {
      ProblemSpecP ups = psi->readInputFile();
      dynamic_cast<SimulationInterface*>(getPort("sim",i))->problemSetup(ups,grid,sharedState);
    } else {
      throw InternalError("psi dynamic_cast failed", __FILE__, __LINE__);
    }
  }

  // clear it out and do the first one again
  sharedState->clearMaterials();
  ProblemSpecInterface* psi = 
    dynamic_cast<ProblemSpecInterface*>(getPort("problem spec",d_componentIndex));
  if (psi) {
    ProblemSpecP ups = psi->readInputFile();
    d_sim->problemSetup(ups,grid,sharedState);
  } else {
    throw InternalError("psi dynamic_cast failed", __FILE__, __LINE__);
  }

  // get the varLabels from the strings we found above
  for (unsigned i = 0; i < d_initVars.size(); i++) {
    vector<string>& names = d_initVars[i];
    vector<VarLabel*> labels;
    for (unsigned j = 0; j < names.size(); j++) {
      VarLabel* label = VarLabel::find(names[j]);
      if (label)
        labels.push_back(label);
      else
        throw ProblemSetupException("Cannot find VarLabel", __FILE__, __LINE__);
    }
    d_initVarLabels.push_back(labels);
  }

  for (unsigned i = 0; i < d_carryOverVars.size(); i++) {
    VarLabel* label = VarLabel::find(d_carryOverVars[i]);
    if (label)
      d_carryOverVarLabels.push_back(label);
    else
      throw ProblemSetupException("Cannot find VarLabel", __FILE__, __LINE__);
  }

  d_sharedState = sharedState;
}
 
void Switcher::scheduleInitialize(const LevelP& level,
                                  SchedulerP& sched)
{
  d_sim->scheduleInitialize(level,sched);
}
 
void Switcher::scheduleComputeStableTimestep(const LevelP& level,
                                             SchedulerP& sched)
{
  cout << "Switcher::scheduleComputeStableTimestep" << endl;
  d_sim->scheduleComputeStableTimestep(level,sched);
}

void
Switcher::scheduleTimeAdvance(const LevelP& level, SchedulerP& sched,
                              int a, int b )
{
  cout << "d_sim = " << d_sim << endl;
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

  // requires switch_test data;
  t->requires(Task::NewDW,switchLabel);
  sched->addTask(t,level->eachPatch(),d_sharedState->allMaterials());
  if (d_switchState == switching) {
    cout << "RUNNING computeStableTimestep" << endl;
    //d_sim->scheduleComputeStableTimestep(level,sched);
  }
}

void Switcher::scheduleInitNewVars(const LevelP& level, SchedulerP& sched)
{
  Task* t = scinew Task("Switcher::initNewVars",
                        this, & Switcher::initNewVars);
  sched->addTask(t,level->eachPatch(),d_sharedState->allMaterials());
  t->requires(Task::NewDW, switchLabel);
}

void Switcher::scheduleCarryOverVars(const LevelP& level, SchedulerP& sched)
{
  Task* t = scinew Task("Switcher::carryOverVars",
                        this, & Switcher::carryOverVars);
  sched->addTask(t,level->eachPatch(),d_sharedState->allMaterials());
}

void Switcher::switchTest(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw, DataWarehouse* new_dw)
{

  SoleVariable<bool> switch_condition;
  new_dw->get(switch_condition,switchLabel,getLevel(patches));
  cout << "switch_condition = " << switch_condition << endl;

  if (switch_condition) {
    d_switchState = switching;
  } else
    d_switchState = idle;

}

void Switcher::initNewVars(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  if (d_switchState != switching)
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

}
bool Switcher::needRecompile(double time, double delt, const GridP& grid)
{
  cout << "In needRecompile, returning " << (d_switchState == switching) << endl;
  bool retval = false;
  if (d_switchState == switching) {
    d_componentIndex++;
    d_sharedState->clearMaterials();
    d_sharedState->d_switchState = true;
    d_sim = 
      dynamic_cast<SimulationInterface*>(getPort("sim",d_componentIndex)); 
    ProblemSpecInterface* psi = 
      dynamic_cast<ProblemSpecInterface*>(getPort("problem spec",
                                                  d_componentIndex));
    if (psi) {
      ProblemSpecP ups = psi->readInputFile();
      d_sim->problemSetup(ups,const_cast<GridP&>(grid),d_sharedState);
    }
    d_sharedState->finalizeMaterials();
    retval = true;
  } else
    d_sharedState->d_switchState = false;
  retval |= d_sim->needRecompile(time, delt, grid);
  return retval;
}

void Switcher::addToTimestepXML(ProblemSpecP& spec)
{
  spec->appendElement("switcherComponentIndex", (int) d_componentIndex, true);
  spec->appendElement("switcherState", (int) d_switchState, true);
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
  cout << "  RESTART index = " << d_componentIndex << " STATE = " << d_switchState << endl;
}
