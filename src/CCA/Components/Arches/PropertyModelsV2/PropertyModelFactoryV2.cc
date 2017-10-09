#include <CCA/Components/Arches/PropertyModelsV2/PropertyModelFactoryV2.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
//Specific models:
#include <CCA/Components/Arches/Utility/TaskAlgebra.h>
#include <CCA/Components/Arches/PropertyModelsV2/WallHFVariable.h>
#include <CCA/Components/Arches/PropertyModelsV2/VariableStats.h>
#include <CCA/Components/Arches/PropertyModelsV2/DensityPredictor.h>
#include <CCA/Components/Arches/PropertyModelsV2/OneDWallHT.h>
#include <CCA/Components/Arches/PropertyModelsV2/ConstantProperty.h>
#include <CCA/Components/Arches/PropertyModelsV2/FaceVelocities.h>
#include <CCA/Components/Arches/PropertyModelsV2/VarInterpolation.h>
#include <CCA/Components/Arches/PropertyModelsV2/UFromRhoU.h>
#include <CCA/Components/Arches/PropertyModelsV2/CCVel.h>
#include <CCA/Components/Arches/PropertyModelsV2/BurnsChriston.h>
#include <CCA/Components/Arches/PropertyModelsV2/cloudBenchmark.h>
#include <CCA/Components/Arches/PropertyModelsV2/sumRadiation.h>
#include <CCA/Components/Arches/PropertyModelsV2/gasRadProperties.h>
#include <CCA/Components/Arches/PropertyModelsV2/spectralProperties.h>
#include <CCA/Components/Arches/PropertyModelsV2/partRadProperties.h>
#include <CCA/Components/Arches/PropertyModelsV2/sootVolumeFrac.h>
#include <CCA/Components/Arches/PropertyModelsV2/CO.h>
#include <libxml/parser.h>
#include <libxml/tree.h>

using namespace Uintah;

PropertyModelFactoryV2::PropertyModelFactoryV2( )
{
  _factory_name = "PropertyModelFactory";
}

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

  bool check_for_radiation=false;
  if ( db->findBlock("PropertyModelsV2")){

    ProblemSpecP db_m = db->findBlock("PropertyModelsV2");

    for ( ProblemSpecP db_model = db_m->findBlock("model"); db_model != nullptr; db_model=db_model->findNextBlock("model") ){

      std::string name;
      std::string type;
      db_model->getAttribute("label", name);
      db_model->getAttribute("type", type);

      if ( type == "wall_heatflux_variable" ){

        TaskInterface::TaskBuilder* tsk = scinew WallHFVariable::Builder( name, 0, _shared_state );
        register_task( name, tsk );

      } else if ( type == "Interpolation_var" ){

        std::string grid_type="NA";
        std::string grid_type2="NA";
        db_model->findBlock("variable")->getAttribute("type", grid_type);
        db_model->findBlock("new_variable")->getAttribute("type", grid_type2);

        TaskInterface::TaskBuilder* tsk = nullptr;

        if ( grid_type == "CC" ){

          typedef typename ArchesCore::VariableHelper<CCVariable<double> >::Type T;
          typedef typename ArchesCore::VariableHelper<T>::ConstType CT;

          if (grid_type2 == "FX"){
            typedef typename ArchesCore::VariableHelper<T>::XFaceType IT;
            tsk = scinew VarInterpolation<CT, IT>::Builder(name, 0);
          } else if (grid_type2 == "FY") {
            typedef typename ArchesCore::VariableHelper<T>::YFaceType IT;
            tsk = scinew VarInterpolation<CT, IT>::Builder(name, 0);
          } else if (grid_type2 == "FZ") {
            typedef typename ArchesCore::VariableHelper<T>::ZFaceType IT;
            tsk = scinew VarInterpolation<CT, IT>::Builder(name, 0);
          }

        } else if ( grid_type == "FX" || grid_type == "FY" || grid_type == "FZ" ){

          if ( grid_type2 != "CC" ){
            std::stringstream msg;
            msg << "Error: When using the variable_interpolator model, only F(X,Y,Z) -> CC allowed." << std::endl;
            throw InvalidValue(msg.str(), __FILE__, __LINE__);
          }

          if ( grid_type == "FX" ){
            typedef typename ArchesCore::VariableHelper<SFCXVariable<double> >::Type T;
            typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
            tsk = scinew VarInterpolation<CT, CCVariable<double> >::Builder(name, 0);
          } else if ( grid_type == "FY" ){
            typedef typename ArchesCore::VariableHelper<SFCYVariable<double> >::Type T;
            typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
            tsk = scinew VarInterpolation<CT, CCVariable<double> >::Builder(name, 0);
          } else if ( grid_type == "FZ" ){
            typedef typename ArchesCore::VariableHelper<SFCZVariable<double> >::Type T;
            typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
            tsk = scinew VarInterpolation<CT, CCVariable<double> >::Builder(name, 0);
          }

        } else {

          throw InvalidValue("Error: grid_type not recognized.",__FILE__,__LINE__);

        }

        register_task( name, tsk );
        _pre_update_property_tasks.push_back( name );

      } else if ( type == "wall_thermal_resistance" ){

        TaskInterface::TaskBuilder* tsk = scinew TaskAlgebra<CCVariable<double> >::Builder(name, 0);
        register_task(name , tsk);
        _pre_update_property_tasks.push_back(name);

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
      } else if ( type == "sootVolumeFrac" ){

        TaskInterface::TaskBuilder* tsk = scinew sootVolumeFrac::Builder( name, 0 );
        register_task( name, tsk );
        _pre_table_post_iv_update.push_back(name);

      } else if ( type == "gasRadProperties" ){

        TaskInterface::TaskBuilder* tsk = scinew gasRadProperties::Builder( name, 0 );
        register_task( name, tsk );
        _pre_table_post_iv_update.push_back(name);
        //_finalize_property_tasks.push_back( name );

        check_for_radiation=true;
      } else if ( type == "spectralProperties" ){

        TaskInterface::TaskBuilder* tsk = scinew spectralProperties::Builder( name, 0 );
        register_task( name, tsk );
        _pre_table_post_iv_update.push_back(name);

        check_for_radiation=true;
      } else if ( type == "partRadProperties" ){

        std::string calculator_type = "NA";
        db_model->require("subModel", calculator_type);
        TaskInterface::TaskBuilder* tsk;

        tsk = scinew partRadProperties::Builder( name, 0 );
        register_task( name, tsk );
        _pre_table_post_iv_update.push_back(name);
        //_finalize_property_tasks.push_back( name );
        check_for_radiation=true;


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

  // MOVE THESE TO THE MODEL SPECIFIC LOCATIONS AND USE ADD_TASK FUNCTION INSTEAD:
  //----Need to generate Property for Rad if present------//
  if(check_for_radiation){
    if(db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns") != nullptr){
      if(db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources") != nullptr){
        ProblemSpecP db_source = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources");
        for ( ProblemSpecP db_src = db_source->findBlock("src"); db_src != nullptr; db_src = db_src->findNextBlock("src")){
          std::string radiation_model;
          db_src->getAttribute("type", radiation_model);
          if (radiation_model == "do_radiation" || radiation_model== "rmcrt_radiation"){
            std::string task_name="sumRadiation::abskt";
            TaskInterface::TaskBuilder* tsk = scinew sumRadiation::Builder( task_name, 0 );
            register_task( task_name, tsk );
            _pre_table_post_iv_update.push_back(task_name);
            //_finalize_property_tasks.push_back( task_name );
            assign_task_to_type_storage(task_name,"sumRadiation");
            break;
          }
        }
      }
    }
  }

  m_vel_name = "face_velocities";
  TaskInterface::TaskBuilder* vel_tsk = scinew FaceVelocities::Builder( m_vel_name, 0 );
  register_task(m_vel_name, vel_tsk);
  _pre_update_property_tasks.push_back(m_vel_name);

  if ( db->findBlock("KMomentum") ){

    TaskInterface::TaskBuilder* u_from_rho_u_tsk = scinew UFromRhoU::Builder( "u_from_rho_u", 0);
    register_task("u_from_rho_u", u_from_rho_u_tsk);

    TaskInterface::TaskBuilder* cc_u = scinew CCVel::Builder("compute_cc_velocities", 0 );
    register_task("compute_cc_velocities", cc_u );

  }

}

void
PropertyModelFactoryV2::build_all_tasks( ProblemSpecP& db )
{

// Temporarily commenting out Derek's commits
// To aid in a messy merge.
ProblemSpecP db_m=db;
if ( db->findBlock("PropertyModelsV2") != nullptr){
  db_m = db->findBlock("PropertyModelsV2");
}//
//for (unsigned int ix=0; ix< m_task_init_order.size(); ix++){
//  TaskInterface* tsk = retrieve_task(m_task_init_order[ix]); // builds task
//  ProblemSpecP db_model = matchNametoSpec(db_m,m_task_init_order[ix]);

  if ( db->findBlock("PropertyModelsV2")){

    ProblemSpecP db_m = db->findBlock("PropertyModelsV2");

    for ( ProblemSpecP db_model = db_m->findBlock("model");
          db_model != nullptr; db_model=db_model->findNextBlock("model")){

      std::string name;
      std::string type;

      db_model->getAttribute("label", name);
      db_model->getAttribute("type", type);

      TaskInterface* tsk = retrieve_task(name);

      ProblemSpecP db_passed;

      if ( type == "wall_thermal_resistance" ){
        db_passed = create_taskAlegebra_spec( db_model, name );
      } else {
        db_passed = db_model;
      }
      tsk->problemSetup(db_passed);
      tsk->create_local_labels();

      //Assuming that everything here is independent:
      _task_order.push_back(name);

    }
  }

  TaskInterface* tsk = retrieve_task("face_velocities");
  tsk->problemSetup(db);
  tsk->create_local_labels();

  if ( db->findBlock("KMomentum")){
    tsk = retrieve_task("u_from_rho_u");
    tsk->problemSetup(db);
    tsk->create_local_labels();

    tsk = retrieve_task("compute_cc_velocities");
    tsk->problemSetup(db);
    tsk->create_local_labels();
  }

  for ( auto i = m_task_init_order.begin(); i != m_task_init_order.end(); i++ ){
    if( *i == "sumRadiation::abskt"){
      TaskInterface* tsk = retrieve_task(*i);
      tsk->problemSetup( db_m );
      tsk->create_local_labels();
    }
  }

}

void
PropertyModelFactoryV2::add_task( ProblemSpecP& db )
{
  if ( db->findBlock("PropertyModelsV2")){

    ProblemSpecP db_m = db->findBlock("PropertyModelsV2");

    for ( ProblemSpecP db_model = db_m->findBlock("model"); db_model != nullptr; db_model=db_model->findNextBlock("model")){

      std::string name;
      std::string type;
      db_model->getAttribute("label", name);
      db_model->getAttribute("type", type);

      if ( type == "wall_heatflux_variable" ){

        TaskInterface::TaskBuilder* tsk = scinew WallHFVariable::Builder( name, 0, _shared_state );
        register_task( name, tsk );

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

  for ( auto i = m_task_init_order.begin(); i != m_task_init_order.end(); i++ ){

    TaskInterface* tsk = retrieve_task( *i );
    tsk->schedule_init( level, sched, matls, doing_restart );
  }

}

//--------------------------------------------------------------------------------------------------
ProblemSpecP
PropertyModelFactoryV2::matchNametoSpec( ProblemSpecP& db_m, std::string name){

  ProblemSpecP db = db_m;
  for ( ProblemSpecP db_model = db->findBlock("model"); db_model != nullptr; db_model=db_model->findNextBlock("model")){
    std::string tname;
    db_model->getAttribute("label", tname);
    if (name==tname){
      return db_model;
    }
  }
  return db;
}

//--------------------------------------------------------------------------------------------------
ProblemSpecP PropertyModelFactoryV2::create_taskAlegebra_spec( ProblemSpecP db_model,
                                                               const std::string name ){
  /*
  <model label="R_total" type="wall_resistance">
    <layer R_eff="some_label"/> <!-- where R_eff = k/dx -->
    <layer R_eff="some_other_label"/>
    ....
  </model>
  */

  xmlDocPtr doc = NULL;       /* document pointer */
  xmlNodePtr root_node = NULL; /* root node */
  doc = xmlNewDoc(BAD_CAST "1.0");
  root_node = xmlNewNode(NULL, BAD_CAST "root");
  xmlDocSetRootElement(doc, root_node);

  ProblemSpec* temp_db = scinew ProblemSpec( root_node ); /* create a temp db */
  ProblemSpecP temp_model_db = temp_db->appendChild("model");
  temp_model_db->setAttribute("type", "task_math");
  temp_model_db->setAttribute("label", name);

  std::vector<std::string> dx_vec;
  std::vector<std::string> k_vec;

  for ( ProblemSpecP db_layer = db_model->findBlock("layer"); db_layer != nullptr;
        db_layer =db_layer->findNextBlock("layer") ){

    std::string dx;
    std::string k;
    db_layer->getAttribute("dx", dx);
    db_layer->getAttribute("k", k);

    dx_vec.push_back(dx);
    k_vec.push_back(k);

  }

  const int N = dx_vec.size();

  if ( N > 1 ){

    for ( int i = 0; i < N; i++ ){

      ProblemSpecP db_op = temp_model_db->appendChild("op");
      std::stringstream string_i;
      string_i << i;

      db_op->setAttribute("type","DIVIDE");
      db_op->setAttribute("label","op_"+string_i.str());

      db_op->appendElement("dep","temporary_variable");
      db_op->appendChild("dep_is_temp");
      db_op->appendChild("sum_into_dep");

      db_op->appendElement("ind1", dx_vec[i]);
      db_op->appendElement("ind2", k_vec[i]);

    }

    //add temp into dw variable
    ProblemSpecP db_op = temp_model_db->appendChild("op");
    std::stringstream string_i;
    string_i << N-1;

    db_op->setAttribute("type","EQUALS");
    db_op->setAttribute("label","final_op");
    db_op->appendElement("dep", name);
    db_op->appendElement("ind1", "temporary_variable" );
    db_op->appendChild("new_variable");
    db_op->appendChild("ind1_is_temp");

    ProblemSpecP exe_db = temp_model_db->appendChild("exe_order");

    for ( int i = 0; i < N; i++ ){

      std::stringstream istr;
      istr << i;
      ProblemSpecP op_db = exe_db->appendChild("op");
      op_db->setAttribute("label", "op_"+istr.str() );
    }

    ProblemSpecP op_db = exe_db->appendChild("op");
    op_db->setAttribute("label", "final_op");

  } else {

    throw ProblemSetupException("Error: You don\'t have enough operations for R_eff", __FILE__, __LINE__);

  }

  //temp_model_db->output("inputFrag.xml");

  return temp_model_db;

}
