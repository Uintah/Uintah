#include <CCA/Components/Arches/ChemMixV2/ChemMixFactory.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
//Specific property evaluators:
#include <CCA/Components/Arches/ChemMixV2/ConstantStateProperties.h>
#include <CCA/Components/Arches/ChemMixV2/ColdFlowProperties.h>

using namespace Uintah;

ChemMixFactory::ChemMixFactory( const ApplicationCommon* arches ) : TaskFactoryBase(arches)
{
  _factory_name = "ChemMixFactory";
}

ChemMixFactory::~ChemMixFactory()
{}

void
ChemMixFactory::register_all_tasks( ProblemSpecP& db )
{

  if ( db->findBlock("StateProperties") ){

    ProblemSpecP db_sp = db->findBlock("StateProperties");

    for ( ProblemSpecP db_p = db_sp->findBlock("model");
          db_p.get_rep() != nullptr;
          db_p = db_p->findNextBlock("model")){

      std::string label;
      std::string type;

      db_p->getAttribute("label", label);
      db_p->getAttribute("type", type);

      if ( type == "constant" ){
        TaskInterface::TaskBuilder* tsk = scinew ConstantStateProperties::Builder( label, 0 );
        register_task( label, tsk );
      } else if ( type == "coldflow" ){
        TaskInterface::TaskBuilder* tsk = scinew ColdFlowProperties::Builder( label, 0 );
        register_task( label, tsk );
      } else {
        throw InvalidValue("Error: Unknown state property evaluator type: "+type, __FILE__, __LINE__);
      }
    }
  }

}

void
ChemMixFactory::build_all_tasks( ProblemSpecP& db )
{

  if ( db->findBlock("StateProperties") ){

    ProblemSpecP db_sp = db->findBlock("StateProperties");

    for ( ProblemSpecP db_p = db_sp->findBlock("model");
	  db_p.get_rep() != nullptr;
          db_p = db_p->findNextBlock("model")){

      std::string label;
      std::string type;

      db_p->getAttribute("label", label);
      db_p->getAttribute("type", type);

      TaskInterface* tsk = retrieve_task( label );
      tsk->problemSetup( db_p );
      tsk->create_local_labels();

      m_task_order.push_back( label );

    }
  }
}

void
ChemMixFactory::add_task( ProblemSpecP& db )
{
}

//--------------------------------------------------------------------------------------------------
void ChemMixFactory::schedule_initialization( const LevelP& level,
                                              SchedulerP& sched,
                                              const MaterialSet* matls,
                                              bool doing_restart ){

  const bool pack_tasks = false;
  schedule_task_group( "Ordered_ChemMixV2_initialization", m_task_init_order, TaskInterface::INITIALIZE,
                       pack_tasks, level, sched, matls );

}

//--------------------------------------------------------------------------------------------------
void ChemMixFactory::schedule_applyBCs( const LevelP& level,
                                        SchedulerP& sched,
                                        const MaterialSet* matls,
                                        const int time_substep ){

  const bool pack_tasks = false;
  schedule_task_group( "Ordered_ChemMixV2_bc", m_task_init_order, TaskInterface::BC,
                       pack_tasks, level, sched, matls );
}
