#include <CCA/Components/Arches/SourceTermsV2/SourceTermFactoryV2.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Utility/GridInfo.h>
//Specific source terms:
#include <CCA/Components/Arches/SourceTermsV2/MMS_mom_csmag.h>
#include <CCA/Components/Arches/SourceTermsV2/MMS_mom.h>
#include <CCA/Components/Arches/SourceTermsV2/MMS_Shunn.h>
#include <CCA/Components/Arches/SourceTermsV2/MMS_scalar.h>

using namespace Uintah;

//--------------------------------------------------------------------------------------------------
SourceTermFactoryV2::SourceTermFactoryV2()
{
  _factory_name = "SourceTermFactoryV2";
}

SourceTermFactoryV2::~SourceTermFactoryV2()
{}

//--------------------------------------------------------------------------------------------------
void
SourceTermFactoryV2::register_all_tasks( ProblemSpecP& db )
{

  /*

      <SourceV2>
        <src label= "Oscar_test" type = "MMS" >

        </src>
      </SourceV2>

  */

  if ( db->findBlock("SourceV2") ){

    ProblemSpecP db_init = db->findBlock("SourceV2");

    for (ProblemSpecP db_src = db_init->findBlock("src"); db_src != nullptr; db_src = db_src->findNextBlock("src")){

      std::string name;
      std::string type;
      db_src->getAttribute("label",name );
      db_src->getAttribute("type", type );

      if ( type == "MMS_mom" ) {

        std::string var_type;
        db_src->findBlock("variable")->getAttribute("type", var_type);

        TaskInterface::TaskBuilder* tsk;

        if ( var_type == "CC" ){
          tsk = scinew MMS_mom<CCVariable<double> >::Builder( name, 0, _shared_state );
        } else if ( var_type == "FX" ){
          tsk = scinew MMS_mom<SFCXVariable<double> >::Builder( name, 0, _shared_state );
        } else if ( var_type == "FY" ){
          tsk = scinew MMS_mom<SFCYVariable<double> >::Builder( name, 0, _shared_state );
        } else {
          tsk = scinew MMS_mom<SFCZVariable<double> >::Builder( name, 0, _shared_state );
        }

        register_task( name, tsk );
        _pre_update_source_tasks.push_back( name );

      } else if ( type == "MMS_Shunn" ) {

        std::string var_type;
        db_src->findBlock("variable")->getAttribute("type", var_type);

        TaskInterface::TaskBuilder* tsk;

        if ( var_type == "CC" ){
          tsk = scinew MMS_Shunn<CCVariable<double> >::Builder( name, 0, _shared_state );
        } else if ( var_type == "FX" ){
          tsk = scinew MMS_Shunn<SFCXVariable<double> >::Builder( name, 0, _shared_state );
        } else if ( var_type == "FY" ){
          tsk = scinew MMS_Shunn<SFCYVariable<double> >::Builder( name, 0, _shared_state );
        } else {
          tsk = scinew MMS_Shunn<SFCZVariable<double> >::Builder( name, 0, _shared_state );
        }

        register_task( name, tsk );
        _pre_update_source_tasks.push_back( name );

      } else if ( type == "MMS_mom_csmag" ) {

        std::string var_type;
        db_src->findBlock("variable")->getAttribute("type", var_type);

        TaskInterface::TaskBuilder* tsk;

        if ( var_type == "CC" ){
          tsk = scinew MMS_mom_csmag<CCVariable<double> >::Builder( name, 0, _shared_state );
        } else if ( var_type == "FX" ){
          tsk = scinew MMS_mom_csmag<SFCXVariable<double> >::Builder( name, 0, _shared_state );
        } else if ( var_type == "FY" ){
          tsk = scinew MMS_mom_csmag<SFCYVariable<double> >::Builder( name, 0, _shared_state );
        } else {
          tsk = scinew MMS_mom_csmag<SFCZVariable<double> >::Builder( name, 0, _shared_state );
        }

        register_task( name, tsk );
        _pre_update_source_tasks.push_back( name );


      } else if ( type == "MMS_scalar" ) {

        TaskInterface::TaskBuilder* tsk = scinew MMS_scalar::Builder( name, 0 , _shared_state );
        register_task( name, tsk );
        _pre_update_source_tasks.push_back( name );

      } else {

        throw InvalidValue("Error: Source term not recognized: "+type,__FILE__,__LINE__);

      }

      assign_task_to_type_storage(name, type);
      }
    }
  }

//--------------------------------------------------------------------------------------------------
void
SourceTermFactoryV2::build_all_tasks( ProblemSpecP& db )
{

  if ( db->findBlock("SourceV2") ){

    ProblemSpecP db_init = db->findBlock("SourceV2");

    for (ProblemSpecP db_src = db_init->findBlock("src"); db_src != nullptr; db_src = db_src->findNextBlock("src")){

      std::string name;
      std::string type;
      db_src->getAttribute("label",name );
      db_src->getAttribute("type", type );

      TaskInterface* tsk = retrieve_task(name);
      tsk->problemSetup( db_src );
      tsk->create_local_labels();

      //Assuming that everything here is independent:
      //_task_order.push_back(name);

    }
  }
}

//--------------------------------------------------------------------------------------------------

void
SourceTermFactoryV2::add_task( ProblemSpecP& db )
{

  if ( db->findBlock("SourceV2") ){

    ProblemSpecP db_init = db->findBlock("SourceV2");

    for (ProblemSpecP db_src = db_init->findBlock("src"); db_src != nullptr; db_src = db_src->findNextBlock("src")){

      std::string name;
      std::string type;
      db_src->getAttribute("label",name );
      db_src->getAttribute("type", type );

      if ( type == "MMS_scalar" ) {

        TaskInterface::TaskBuilder* tsk = scinew MMS_scalar::Builder( name, 0 , _shared_state );
        register_task( name, tsk );
        _pre_update_source_tasks.push_back( name );

      } else if ( type == "MMS_mom" ) {

        std::string var_type;
        db_src->findBlock("variable")->getAttribute("type", var_type);

        TaskInterface::TaskBuilder* tsk;

        if ( var_type == "CC" ){
          tsk = scinew MMS_mom<CCVariable<double> >::Builder( name, 0, _shared_state );
        } else if ( var_type == "FX" ){
          tsk = scinew MMS_mom<SFCXVariable<double> >::Builder( name, 0, _shared_state );
        } else if ( var_type == "FY" ){
          tsk = scinew MMS_mom<SFCYVariable<double> >::Builder( name, 0, _shared_state );
        } else {
          tsk = scinew MMS_mom<SFCZVariable<double> >::Builder( name, 0, _shared_state );
        }
        register_task( name, tsk );
        _pre_update_source_tasks.push_back( name );

      } else if ( type == "MMS_Shunn" ) {

        std::string var_type;
        db_src->findBlock("variable")->getAttribute("type", var_type);

        TaskInterface::TaskBuilder* tsk;

        if ( var_type == "CC" ){
          tsk = scinew MMS_Shunn<CCVariable<double> >::Builder( name, 0, _shared_state );
        } else if ( var_type == "FX" ){
          tsk = scinew MMS_Shunn<SFCXVariable<double> >::Builder( name, 0, _shared_state );
        } else if ( var_type == "FY" ){
          tsk = scinew MMS_Shunn<SFCYVariable<double> >::Builder( name, 0, _shared_state );
        } else {
          tsk = scinew MMS_Shunn<SFCZVariable<double> >::Builder( name, 0, _shared_state );
        }
        register_task( name, tsk );
        _pre_update_source_tasks.push_back( name );
      } else if ( type == "MMS_mom_csmag" ) {

        std::string var_type;
        db_src->findBlock("variable")->getAttribute("type", var_type);

        TaskInterface::TaskBuilder* tsk;

        if ( var_type == "CC" ){
          tsk = scinew MMS_mom_csmag<CCVariable<double> >::Builder( name, 0, _shared_state );
        } else if ( var_type == "FX" ){
          tsk = scinew MMS_mom_csmag<SFCXVariable<double> >::Builder( name, 0, _shared_state );
        } else if ( var_type == "FY" ){
          tsk = scinew MMS_mom_csmag<SFCYVariable<double> >::Builder( name, 0, _shared_state );
        } else {
          tsk = scinew MMS_mom_csmag<SFCZVariable<double> >::Builder( name, 0, _shared_state );
        }

        register_task( name, tsk );
        _pre_update_source_tasks.push_back( name );

      } else {

        throw InvalidValue("Error: Source term not recognized: "+type,__FILE__,__LINE__);

      }

      assign_task_to_type_storage(name, type);
      print_task_setup_info( name, type );

      //also build the task here
      TaskInterface* tsk = retrieve_task(name);
      tsk->problemSetup(db_src);
      tsk->create_local_labels();

    }
  }
}
//--------------------------------------------------------------------------------------------------

void SourceTermFactoryV2::schedule_initialization( const LevelP& level,
                                                 SchedulerP& sched,
                                                 const MaterialSet* matls,
                                                 bool doing_restart ){

  for ( auto i = _tasks.begin(); i != _tasks.end(); i++ ){

    TaskInterface* tsk = retrieve_task( i->first );
    tsk->schedule_init( level, sched, matls, doing_restart );

  }
}
