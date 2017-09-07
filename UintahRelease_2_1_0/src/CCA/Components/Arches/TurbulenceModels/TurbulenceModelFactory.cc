#include <CCA/Components/Arches/TurbulenceModels/TurbulenceModelFactory.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>

//Specific models:
#include <CCA/Components/Arches/TurbulenceModels/SGSsigma.h>
#include <CCA/Components/Arches/TurbulenceModels/Smagorinsky.h>
#include <CCA/Components/Arches/TurbulenceModels/WALE.h>

using namespace Uintah;

//--------------------------------------------------------------------------------------------------
TurbulenceModelFactory::TurbulenceModelFactory( )
{}

//--------------------------------------------------------------------------------------------------
TurbulenceModelFactory::~TurbulenceModelFactory()
{}

//--------------------------------------------------------------------------------------------------
void
TurbulenceModelFactory::register_all_tasks( ProblemSpecP& db )
{

  /*
   * <Arches>
   *  <TurbulenceModels>
   *    <model label="my_turb_model" type="the_type">
   *      .....
   *    </model>
   *    ...
   *  </TurbulenceModels>
   *  ...
   * </Arches>
   */

  if ( db->findBlock("TurbulenceModels")){

    ProblemSpecP db_m = db->findBlock("TurbulenceModels");

    for ( ProblemSpecP db_model = db_m->findBlock("model"); db_model != nullptr;
          db_model=db_model->findNextBlock("model")){

      std::string name;
      std::string type;
      db_model->getAttribute("label", name);
      db_model->getAttribute("type", type);

      if ( type == "sigma" ){

        TaskInterface::TaskBuilder* tsk_builder = scinew SGSsigma::Builder( name, 0 );
        register_task( name, tsk_builder );
        m_momentum_closure_tasks.push_back(name);

      } else if (type == "constant_smagorinsky"){

        TaskInterface::TaskBuilder* tsk_builder = scinew Smagorinsky::Builder( name, 0 );
        register_task( name, tsk_builder );
        m_momentum_closure_tasks.push_back(name);
        
      } else if (type == "wale"){

        TaskInterface::TaskBuilder* tsk_builder = scinew WALE::Builder( name, 0 );
        register_task( name, tsk_builder );
        m_momentum_closure_tasks.push_back(name);

      } else {

        throw InvalidValue(
          "Error: Turbulence model not recognized: "+type+", "+name, __FILE__, __LINE__ );

      }

      assign_task_to_type_storage(name, type);

    }
  }
}

//--------------------------------------------------------------------------------------------------
void
TurbulenceModelFactory::build_all_tasks( ProblemSpecP& db )
{

  if ( db->findBlock("TurbulenceModels")){
    ProblemSpecP db_m = db->findBlock("TurbulenceModels");

    for ( ProblemSpecP db_model = db_m->findBlock("model"); db_model != nullptr;
          db_model=db_model->findNextBlock("model")){

      std::string name;
      std::string type;
      db_model->getAttribute("label", name);
      db_model->getAttribute("type", type);

      TaskInterface* tsk = retrieve_task(name);
      tsk->problemSetup(db_model);
      tsk->create_local_labels();

    }
  }

}

//--------------------------------------------------------------------------------------------------
void TurbulenceModelFactory::schedule_initialization( const LevelP& level,
                                                      SchedulerP& sched,
                                                      const MaterialSet* matls,
                                                      bool doing_restart ){

  for ( auto i = m_task_init_order.begin(); i != m_task_init_order.end(); i++ ){
    TaskInterface* tsk = retrieve_task( *i );
    tsk->schedule_init( level, sched, matls, doing_restart );
  }

}
