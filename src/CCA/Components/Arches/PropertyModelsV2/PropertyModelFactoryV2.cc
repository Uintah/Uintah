#include <CCA/Components/Arches/PropertyModelsV2/PropertyModelFactoryV2.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
//Specific models:
#include <CCA/Components/Arches/PropertyModelsV2/WallHFVariable.h>
#include <CCA/Components/Arches/PropertyModelsV2/VariableStats.h>
#include <CCA/Components/Arches/PropertyModelsV2/DensityPredictor.h>
#include <CCA/Components/Arches/PropertyModelsV2/OneDWallHT.h>
#include <CCA/Components/Arches/PropertyModelsV2/ConstantProperty.h>
#include <CCA/Components/Arches/PropertyModelsV2/FaceVelocities.h>
#include <CCA/Components/Arches/PropertyModelsV2/UFromRhoU.h>
#include <CCA/Components/Arches/PropertyModelsV2/BurnsChriston.h>
#include <CCA/Components/Arches/PropertyModelsV2/cloudBenchmark.h>
#include <CCA/Components/Arches/PropertyModelsV2/CO.h>

using namespace Uintah;

PropertyModelFactoryV2::PropertyModelFactoryV2( )
{}

PropertyModelFactoryV2::~PropertyModelFactoryV2()
{}

void
PropertyModelFactoryV2::register_all_tasks( ProblemSpecP& db )
{

  /*

  <PropertyModelsV2>
    <model name="A_MODEL" type="A_TYPE">
      <stuff....>
    </model>
  </PropertyModelsV2>

  */

  //Force the face velocity property model to be created:

  // going to look for a <VarID>. If not found, use the standary velocity names.
  m_vel_name = "face_velocities";
  TaskInterface::TaskBuilder* vel_tsk = scinew FaceVelocities::Builder( m_vel_name, 0 );
  register_task(m_vel_name, vel_tsk);
  _pre_update_property_tasks.push_back(m_vel_name);

  if ( db->findBlock("KMomentum") ){
    TaskInterface::TaskBuilder* u_from_rho_u_tsk = scinew UFromRhoU::Builder( "u_from_rho_u", 0);
    register_task("u_from_rho_u", u_from_rho_u_tsk);
  }

  if ( db->findBlock("PropertyModelsV2")){

    ProblemSpecP db_m = db->findBlock("PropertyModelsV2");

    for ( ProblemSpecP db_model = db_m->findBlock("model"); db_model != 0;
          db_model=db_model->findNextBlock("model")){

      std::string name;
      std::string type;
      db_model->getAttribute("label", name);
      db_model->getAttribute("type", type);

      if ( type == "wall_heatflux_variable" ){

        TaskInterface::TaskBuilder* tsk = scinew WallHFVariable::Builder( name, 0, _shared_state );
        register_task( name, tsk );
        _pre_update_property_tasks.push_back( name );

      } else if ( type == "variable_stats" ){

        TaskInterface::TaskBuilder* tsk = scinew VariableStats::Builder( name, 0 );
        register_task( name, tsk );
        _var_stats_tasks.push_back( name );

      } else if ( type == "density_predictor" ) {

        TaskInterface::TaskBuilder* tsk = scinew DensityPredictor::Builder( name, 0 );
        register_task( name, tsk );

      } else if ( type == "one_d_wallht" ) {

        TaskInterface::TaskBuilder* tsk = scinew OneDWallHT::Builder( name, 0 );
        register_task( name, tsk );
        _pre_update_property_tasks.push_back( name );


      } else if ( type == "CO" ) {

        TaskInterface::TaskBuilder* tsk = scinew CO::Builder( name, 0 );
        register_task( name, tsk );
        _finalize_property_tasks.push_back( name );

      } else if ( type == "burns_christon" ) {

        TaskInterface::TaskBuilder* tsk = scinew BurnsChriston::Builder( name, 0 );
        register_task( name, tsk );

      } else if ( type == "cloudBenchmark" ) {

        TaskInterface::TaskBuilder* tsk = scinew cloudBenchmark::Builder( name, 0 );
        register_task( name, tsk );


      } else if ( type == "constant_property"){

        std::string var_type = "NA";
        db_model->findBlock("grid")->getAttribute("type",var_type);

        TaskInterface::TaskBuilder* tsk;
        if ( var_type == "CC" ){
          tsk = scinew ConstantProperty<CCVariable<double> >::Builder(name, 0);
          _pre_update_property_tasks.push_back(name);
        } else if ( var_type == "FX" ){
          tsk = scinew ConstantProperty<SFCXVariable<double> >::Builder(name, 0);
          _pre_update_property_tasks.push_back(name);
        } else if ( var_type == "FY" ){
          tsk = scinew ConstantProperty<SFCYVariable<double> >::Builder(name, 0);
          _pre_update_property_tasks.push_back(name);
        } else if ( var_type == "FZ" ){
          tsk = scinew ConstantProperty<SFCZVariable<double> >::Builder(name, 0);
          _pre_update_property_tasks.push_back(name);
        } else {
          throw InvalidValue("Error: Property grid type not recognized for model: "+name,__FILE__,__LINE__);
        }
        register_task( name, tsk );
        _pre_update_property_tasks.push_back(name);

      } else {

        throw InvalidValue("Error: Property model not recognized: "+type,__FILE__,__LINE__);

      }

      assign_task_to_type_storage(name, type);

    }
  }
}

void
PropertyModelFactoryV2::build_all_tasks( ProblemSpecP& db )
{

  TaskInterface* vel_tsk = retrieve_task(m_vel_name);
  vel_tsk->problemSetup(db);
  vel_tsk->create_local_labels();

  _task_order.push_back( m_vel_name );


  if ( db->findBlock("KMomentum") ){

    TaskInterface* u_from_rho_u_tsk = retrieve_task( "u_from_rho_u");
    u_from_rho_u_tsk->problemSetup(db);
    u_from_rho_u_tsk->create_local_labels();

  }


  if ( db->findBlock("PropertyModelsV2")){

    ProblemSpecP db_m = db->findBlock("PropertyModelsV2");

    for ( ProblemSpecP db_model = db_m->findBlock("model"); db_model != 0; db_model=db_model->findNextBlock("model")){
      std::string name;
      db_model->getAttribute("label", name);
      TaskInterface* tsk = retrieve_task(name);
      tsk->problemSetup(db_model);
      tsk->create_local_labels();

      //Assuming that everything here is independent:
      _task_order.push_back(name);

    }
  }

  if ( db->findBlock("KMomentum") ){
    //Requires density so putting it last
    _task_order.push_back( "u_from_rho_u");
  }

}

void
PropertyModelFactoryV2::add_task( ProblemSpecP& db )
{
  if ( db->findBlock("PropertyModelsV2")){

    ProblemSpecP db_m = db->findBlock("PropertyModelsV2");

    for ( ProblemSpecP db_model = db_m->findBlock("model"); db_model != 0;
          db_model=db_model->findNextBlock("model")){

      std::string name;
      std::string type;
      db_model->getAttribute("label", name);
      db_model->getAttribute("type", type);

      if ( type == "wall_heatflux_variable" ){

        TaskInterface::TaskBuilder* tsk = scinew WallHFVariable::Builder( name, 0, _shared_state );
        register_task( name, tsk );
        _pre_update_property_tasks.push_back( name );

      } else if ( type == "variable_stats" ){

        TaskInterface::TaskBuilder* tsk = scinew VariableStats::Builder( name, 0 );
        register_task( name, tsk );
        _finalize_property_tasks.push_back( name );

      } else if ( type == "density_predictor" ) {

        TaskInterface::TaskBuilder* tsk = scinew DensityPredictor::Builder( name, 0 );
        register_task( name, tsk );

      } else if ( type == "one_d_wallht" ) {

        TaskInterface::TaskBuilder* tsk = scinew OneDWallHT::Builder( name, 0 );
        register_task( name, tsk );
        _pre_update_property_tasks.push_back( name );

      } else if ( type == "CO" ) {

        TaskInterface::TaskBuilder* tsk = scinew CO::Builder( name, 0 );
        register_task( name, tsk );
        _finalize_property_tasks.push_back( name );

      } else {

        throw InvalidValue("Error: Property model not recognized.",__FILE__,__LINE__);

      }

      assign_task_to_type_storage(name, type);
      print_task_setup_info( name, type );

      //also build the task here
      TaskInterface* tsk = retrieve_task(name);
      tsk->problemSetup(db_model);
      tsk->create_local_labels();

    }
  }
}

//--------------------------------------------------------------------------------------------------
void PropertyModelFactoryV2::schedule_initialization( const LevelP& level,
                                                      SchedulerP& sched,
                                                      const MaterialSet* matls,
                                                      bool doing_restart ){

  for ( auto i = _task_order.begin(); i != _task_order.end(); i++ ){

    TaskInterface* tsk = retrieve_task( *i );
    tsk->schedule_init( level, sched, matls, doing_restart );

  }

}
