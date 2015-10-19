#include <CCA/Components/Arches/Transport/TransportFactory.h>
#include <CCA/Components/Arches/Transport/ScalarRHS.h>
#include <CCA/Components/Arches/Transport/KScalarRHS.h>
#include <CCA/Components/Arches/Transport/URHS.h>
#include <CCA/Components/Arches/Transport/FEUpdate.h>
#include <CCA/Components/Arches/Transport/KFEUpdate.h>
#include <CCA/Components/Arches/Transport/SSPInt.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>

using namespace Uintah;

TransportFactory::TransportFactory()
{}

TransportFactory::~TransportFactory()
{}

void
TransportFactory::register_all_tasks( ProblemSpecP& db )
{

  /*

     <ScalarTransport>

      <eqn label="my_eqn">
        <diffusion/>
        <convection type="super_bee"/>
        <stage value="1"/> ???
        <clip high="1.0" low="0.0"/>
        <initialization  type="constant">...</intialization>
      </eqn>

      <eqn label=....>
      .... and so on....

     </ScalarTransport>

     <Momentum>
      <convection type="..."/>
     </Momentum>


  */

  if ( db->findBlock("ScalarTransport") ){

    ProblemSpecP db_st = db->findBlock("ScalarTransport");

    for (ProblemSpecP eqn_db = db_st->findBlock("eqn"); eqn_db != 0; eqn_db = eqn_db->findNextBlock("eqn")){

      std::string eqn_name = "null";
      eqn_db->getAttribute("label", eqn_name);
      TaskInterface::TaskBuilder* tsk = scinew ScalarRHS::Builder(eqn_name,0);
      register_task( eqn_name, tsk );

      _scalar_builders.push_back(eqn_name);

    }

    typedef SpatialOps::SVolField SVol;
    std::string update_task_name = "scalar_fe_update";
    FEUpdate<SVol>::Builder* tsk = scinew FEUpdate<SVol>::Builder( update_task_name, 0, _scalar_builders );
    register_task( update_task_name, tsk );

    typedef SpatialOps::SVolField SVol;
    std::string ssp_task_name = "scalar_ssp_update";
    SSPInt<SVol>::Builder* tsk2 = scinew SSPInt<SVol>::Builder( ssp_task_name, 0, _scalar_builders );
    register_task( ssp_task_name, tsk2 );

    _scalar_update.push_back( update_task_name );
    _scalar_ssp.push_back( ssp_task_name );

  }

  if ( db->findBlock("KScalarTransport") ){

    ProblemSpecP db_st = db->findBlock("KScalarTransport");

    for (ProblemSpecP eqn_db = db_st->findBlock("eqn"); eqn_db != 0; eqn_db = eqn_db->findNextBlock("eqn")){

      std::string eqn_name = "null";
      std::string type = "null";
      eqn_db->getAttribute("label", eqn_name);
      eqn_db->getAttribute("type", type );

      TaskInterface::TaskBuilder* tsk;
      if ( type == "CC" ){
        tsk = scinew KScalarRHS<CCVariable<double> >::Builder(eqn_name, 0);
        _scalar_builders.push_back(eqn_name);
      } else if ( type == "SX" ){
        tsk = scinew KScalarRHS<SFCXVariable<double> >::Builder(eqn_name, 0);
        _momentum_builders.push_back(eqn_name);
      } else if ( type == "SY" ){
        tsk = scinew KScalarRHS<SFCYVariable<double> >::Builder(eqn_name, 0);
        _momentum_builders.push_back(eqn_name);
      } else if ( type == "SZ" ){
        tsk = scinew KScalarRHS<SFCZVariable<double> >::Builder(eqn_name, 0);
        _momentum_builders.push_back(eqn_name);
      } else {
        throw InvalidValue("Error: Eqn type not recognized",__FILE__,__LINE__);
      }
      register_task( eqn_name, tsk );

    }

    std::string update_task_name = "scalar_fe_update";
    KFEUpdate<CCVariable<double> >::Builder* tsk = scinew KFEUpdate<CCVariable<double> >::Builder( update_task_name, 0, _scalar_builders );
    register_task( update_task_name, tsk );

    // std::string ssp_task_name = "scalar_ssp_update";
    // SSPInt::Builder* tsk2 = scinew SSPInt::Builder( ssp_task_name, 0, _scalar_builders );
    // register_task( ssp_task_name, tsk2 );

    _scalar_update.push_back( update_task_name );
    // _scalar_ssp.push_back( ssp_task_name );

  }

  //Momentum:
  if ( db->findBlock("MomentumTransport")){

    typedef SpatialOps::XVolField XVol;
    typedef SpatialOps::YVolField YVol;
    typedef SpatialOps::ZVolField ZVol;

    //---u
    std::string u_name = "uvelocity";
    std::string v_name = "vvelocity";
    std::string w_name = "wvelocity";

    std::string task_name = "umom";

    TaskInterface::TaskBuilder* umom_tsk = scinew URHS<XVol>::Builder( task_name, 0, u_name, v_name, w_name );
    register_task( task_name, umom_tsk );

    _momentum_builders.push_back(task_name);

    //---v
    task_name = "vmom";
    TaskInterface::TaskBuilder* vmom_tsk = scinew URHS<YVol>::Builder( task_name, 0, v_name, w_name, u_name );
    register_task( task_name, vmom_tsk );

    _momentum_builders.push_back(task_name);

    //---w
    task_name = "wmom";
    TaskInterface::TaskBuilder* wmom_tsk = scinew URHS<ZVol>::Builder( task_name, 0, w_name, u_name, v_name );
    register_task( task_name, wmom_tsk );

    _momentum_builders.push_back(task_name);

    typedef SpatialOps::XVolField XVol;
    typedef SpatialOps::YVolField YVol;
    typedef SpatialOps::ZVolField ZVol;
    std::string update_task_name = "umom_fe_update";
    std::vector<std::string> u_up_tsk;
    u_up_tsk.push_back("umom");
    FEUpdate<XVol>::Builder* utsk = scinew FEUpdate<XVol>::Builder( update_task_name, 0, u_up_tsk );
    register_task( update_task_name, utsk );
    _momentum_update.push_back( update_task_name );

    update_task_name = "vmom_fe_update";
    std::vector<std::string> v_up_tsk;
    v_up_tsk.push_back("vmom");
    FEUpdate<YVol>::Builder* vtsk = scinew FEUpdate<YVol>::Builder( update_task_name, 0, v_up_tsk );
    register_task( update_task_name, vtsk );
    _momentum_update.push_back( update_task_name );

    update_task_name = "wmom_fe_update";
    std::vector<std::string> w_up_tsk;
    w_up_tsk.push_back("wmom");
    FEUpdate<YVol>::Builder* wtsk = scinew FEUpdate<YVol>::Builder( update_task_name, 0, w_up_tsk );
    register_task( update_task_name, wtsk );
    //_momentum_update.push_back( update_task_name );

    //std::string ssp_task_name = "umom_ssp_update";
    //SSPInt<XVol>::Builder* ssp_utsk = scinew SSPInt<XVol>::Builder( ssp_task_name, 0, u_up_tsk );
    //register_task( ssp_task_name, ssp_utsk );

    //ssp_task_name = "vmom_ssp_update";
    //SSPInt<YVol>::Builder* ssp_vtsk = scinew SSPInt<YVol>::Builder( ssp_task_name, 0, v_up_tsk );
    //register_task( ssp_task_name, ssp_vtsk );

    //ssp_task_name = "wmom_ssp_update";
    //SSPInt<ZVol>::Builder* ssp_wtsk = scinew SSPInt<ZVol>::Builder( ssp_task_name, 0, w_up_tsk );
    //register_task( ssp_task_name, ssp_wtsk );

  }
}

void
TransportFactory::build_all_tasks( ProblemSpecP& db )
{

  if ( db->findBlock("ScalarTransport") ){

    ProblemSpecP db_st = db->findBlock("ScalarTransport");

    for (ProblemSpecP eqn_db = db_st->findBlock("eqn"); eqn_db != 0; eqn_db = eqn_db->findNextBlock("eqn")){

      std::string eqn_name = "null";
      eqn_db->getAttribute("label", eqn_name);
      TaskInterface* tsk = retrieve_task(eqn_name);
      tsk->problemSetup( eqn_db );

      tsk->create_local_labels();

    }

    TaskInterface* tsk = retrieve_task("scalar_fe_update");
    tsk->problemSetup( db );

    tsk->create_local_labels();

    tsk = retrieve_task("scalar_ssp_update");
    tsk->problemSetup( db );

    tsk->create_local_labels();

  }

  if ( db->findBlock("KScalarTransport") ){

    ProblemSpecP db_st = db->findBlock("KScalarTransport");

    for (ProblemSpecP eqn_db = db_st->findBlock("eqn"); eqn_db != 0; eqn_db = eqn_db->findNextBlock("eqn")){

      std::string eqn_name = "null";
      eqn_db->getAttribute("label", eqn_name);
      TaskInterface* tsk = retrieve_task(eqn_name);
      tsk->problemSetup( eqn_db );

      tsk->create_local_labels();

    }

    TaskInterface* tsk = retrieve_task("scalar_fe_update");
    tsk->problemSetup( db );

    tsk->create_local_labels();

    // tsk = retrieve_task("scalar_ssp_update");
    // tsk->problemSetup( db );
    //
    // tsk->create_local_labels();

  }

  if ( db->findBlock("MomentumTransport")){

    ProblemSpecP db_mt = db->findBlock("MomentumTransport");

    TaskInterface* utsk = retrieve_task("umom");
    utsk->problemSetup(db_mt);
    utsk->create_local_labels();

    TaskInterface* vtsk = retrieve_task("vmom");
    vtsk->problemSetup(db_mt);
    vtsk->create_local_labels();

    TaskInterface* wtsk = retrieve_task("wmom");
    wtsk->problemSetup(db_mt);
    wtsk->create_local_labels();

    utsk = retrieve_task("umom_fe_update");
    utsk->problemSetup(db_mt);
    utsk->create_local_labels();

    vtsk = retrieve_task("vmom_fe_update");
    vtsk->problemSetup(db_mt);
    vtsk->create_local_labels();

    wtsk = retrieve_task("wmom_fe_update");
    wtsk->problemSetup(db_mt);
    wtsk->create_local_labels();
  }
}
