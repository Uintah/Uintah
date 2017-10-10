#include <CCA/Components/Arches/TurbulenceModels/TurbulenceModelFactory.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Task/TaskController.h>

//Specific models:
#include <CCA/Components/Arches/TurbulenceModels/SGSsigma.h>
#include <CCA/Components/Arches/TurbulenceModels/Smagorinsky.h>
#include <CCA/Components/Arches/TurbulenceModels/WALE.h>
#include <CCA/Components/Arches/TurbulenceModels/DSFT.h>
#include <CCA/Components/Arches/TurbulenceModels/DSmaCs.h>
#include <CCA/Components/Arches/TurbulenceModels/DSmaMMML.h>

using namespace Uintah;

//--------------------------------------------------------------------------------------------------
TurbulenceModelFactory::TurbulenceModelFactory(  )
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

   using namespace ArchesCore;

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

      } else if (type == "dynamic_smagorinsky"){

        TaskController& tsk_controller = TaskController::self();
        const TaskController::Packing& packed_tasks = tsk_controller.get_packing_info();

        if ( packed_tasks.turbulence ){
          name = "DS_task1";
          TaskInterface::TaskBuilder* tsk_builder = scinew DSFT::Builder( name, 0 );
          register_task( name, tsk_builder );
          m_momentum_closure_tasks.push_back(name);

          name = "DS_task2";
          TaskInterface::TaskBuilder* tsk_builder2 = scinew DSmaMMML< CCVariable<double> >::Builder( name, 0 );
          register_task( name, tsk_builder2 );
          m_momentum_closure_tasks.push_back(name);

          name = "DS_task3";
          TaskInterface::TaskBuilder* tsk_builder3 = scinew DSmaCs< CCVariable<double> >::Builder( name, 0 );
          register_task( name, tsk_builder3 );
          m_momentum_closure_tasks.push_back(name);
        } else {
          name = "DS_task1";
          TaskInterface::TaskBuilder* tsk_builder = scinew DSFT::Builder( name, 0 );
          register_task( name, tsk_builder );
          m_momentum_closure_tasks.push_back(name);

          name = "DS_task2";
          TaskInterface::TaskBuilder* tsk_builder2 = scinew DSmaMMML< constCCVariable<double> >::Builder( name, 0 );
          register_task( name, tsk_builder2 );
          m_momentum_closure_tasks.push_back(name);

          name = "DS_task3";
          TaskInterface::TaskBuilder* tsk_builder3 = scinew DSmaCs< constCCVariable<double> >::Builder( name, 0 );
          register_task( name, tsk_builder3 );
          m_momentum_closure_tasks.push_back(name);
        }

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

      if (type == "dynamic_smagorinsky" ) {

        name = "DS_task1"; 
        TaskInterface* tsk = retrieve_task(name);
        tsk->problemSetup(db_model);
        tsk->create_local_labels();

        name = "DS_task2"; 
        TaskInterface* tsk2 = retrieve_task(name);
        tsk2->problemSetup(db_model);
        tsk2->create_local_labels();

        name = "DS_task3"; 
        TaskInterface* tsk3 = retrieve_task(name);
        tsk3->problemSetup(db_model);
        tsk3->create_local_labels();

      } else {

        TaskInterface* tsk = retrieve_task(name);
        tsk->problemSetup(db_model);
        tsk->create_local_labels();
      }
    }
  }
}

//--------------------------------------------------------------------------------------------------
void TurbulenceModelFactory::schedule_initialization( const LevelP& level,
                                                      SchedulerP& sched,
                                                      const MaterialSet* matls,
                                                      bool doing_restart ){

  for ( auto i = _active_tasks.begin(); i != _active_tasks.end(); i++ ){
    TaskInterface* tsk = retrieve_task( *i );
    tsk->schedule_init( level, sched, matls, doing_restart );
  }

}
