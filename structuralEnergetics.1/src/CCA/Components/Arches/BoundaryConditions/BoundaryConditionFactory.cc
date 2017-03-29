#include <CCA/Components/Arches/BoundaryConditions/BoundaryConditionFactory.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
//Specific models:
#include <CCA/Components/Arches/BoundaryConditions/HandOff.h>

using namespace Uintah;

BoundaryConditionFactory::BoundaryConditionFactory( )
{
  _factory_name = "BoundaryConditionFactory"; 
}

BoundaryConditionFactory::~BoundaryConditionFactory()
{}

void
BoundaryConditionFactory::register_all_tasks( ProblemSpecP & db )
{

  if ( db->findBlock( "BoundaryConditions" ) ) {

    ProblemSpecP db_m = db->findBlock( "BoundaryConditions" );

    for( ProblemSpecP db_bc = db_m->findBlock("bc"); db_bc != nullptr; db_bc = db_bc->findNextBlock("bc") ) {

      std::string name;
      std::string type;
      db_bc->getAttribute("label", name);
      db_bc->getAttribute("type", type);

      if ( type == "handoff" ){

        TaskInterface::TaskBuilder* tsk = scinew HandOff<CCVariable<double> >::Builder( name, 0 );
        register_task( name, tsk );

      }
      else {
        throw InvalidValue("Error: Property model not recognized: "+type,__FILE__,__LINE__);
      }

      assign_task_to_type_storage(name, type);
    }
  }
}

void
BoundaryConditionFactory::build_all_tasks( ProblemSpecP& db )
{

  if ( db->findBlock("BoundaryConditions")){

    ProblemSpecP db_m = db->findBlock("BoundaryConditions");

    for ( ProblemSpecP db_bc = db_m->findBlock("bc"); db_bc != nullptr; db_bc=db_bc->findNextBlock("bc") ) {

      std::string name;
      std::string type;
      db_bc->getAttribute("label", name);
      db_bc->getAttribute("type", type);

      print_task_setup_info( name, type );
      TaskInterface* tsk = retrieve_task(name);
      tsk->problemSetup(db_bc);
      tsk->create_local_labels();
    }
  }
}

void
BoundaryConditionFactory::add_task( ProblemSpecP& db )
{

  if ( db->findBlock( "BoundaryConditions" ) ) {

    ProblemSpecP db_m = db->findBlock( "BoundaryConditions" );

    for ( ProblemSpecP db_bc = db_m->findBlock("bc"); db_bc != nullptr; db_bc=db_bc->findNextBlock("bc") ) {

      std::string name;
      std::string type;
      db_bc->getAttribute("label", name);
      db_bc->getAttribute("type", type);

      if ( type == "handoff" ){

        std::string grid_type;
        db_bc->findBlock("grid")->getAttribute("type", grid_type);

        TaskInterface::TaskBuilder* tsk;
        if ( grid_type == "CC" ){
          tsk = scinew HandOff<CCVariable<double> >::Builder( name, 0 );
        }
        else if ( grid_type == "FX" ){
          tsk = scinew HandOff<SFCXVariable<double> >::Builder( name, 0 );
        }
        else if ( grid_type == "FY" ){
          tsk = scinew HandOff<SFCYVariable<double> >::Builder( name, 0 );
        }
        else if ( grid_type == "FZ" ){
          tsk = scinew HandOff<SFCZVariable<double> >::Builder( name, 0 );
        }
        else {
          throw InvalidValue("Error: Grid type not recognized for bc labeled: "+name, __FILE__, __LINE__ );
        }

        register_task( name, tsk );

      }
      else {
        throw InvalidValue("Error: Property model not recognized.",__FILE__,__LINE__);
      }

      assign_task_to_type_storage(name, type);

      //also build the task here
      TaskInterface* tsk = retrieve_task(name);
      tsk->problemSetup(db_bc);
      tsk->create_local_labels();
    }
  }
}

//--------------------------------------------------------------------------------------------------
void BoundaryConditionFactory::schedule_initialization( const LevelP& level,
                                                        SchedulerP& sched,
                                                        const MaterialSet* matls,
                                                        bool doing_restart ){

  for ( auto i = _tasks.begin(); i != _tasks.end(); i++ ){

    TaskInterface* tsk = retrieve_task( i->first );
    tsk->schedule_init( level, sched, matls, doing_restart );

  }
}
