#include <CCA/Components/Arches/Transport/TransportFactory.h>
#include <CCA/Components/Arches/Transport/ScalarRHS.h>
#include <CCA/Components/Arches/Transport/KScalarRHS.h>
#include <CCA/Components/Arches/Transport/ComputePsi.h>
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
      TaskInterface::TaskBuilder* tsk = new ScalarRHS::Builder(eqn_name,0);
      register_task( eqn_name, tsk );

      _scalar_builders.push_back(eqn_name);

    }

    typedef SpatialOps::SVolField SVol;
    std::string update_task_name = "scalar_fe_update";
    FEUpdate<SVol>::Builder* tsk = new FEUpdate<SVol>::Builder( update_task_name, 0, _scalar_builders );
    register_task( update_task_name, tsk );

    typedef SpatialOps::SVolField SVol;
    std::string ssp_task_name = "scalar_ssp_update";
    SSPInt<SVol>::Builder* tsk2 = new SSPInt<SVol>::Builder( ssp_task_name, 0, _scalar_builders );
    register_task( ssp_task_name, tsk2 );

    _scalar_update.push_back( update_task_name );
    _scalar_ssp.push_back( ssp_task_name );

  }

  if ( db->findBlock("KScalarTransport") ){

    ProblemSpecP db_st = db->findBlock("KScalarTransport");

    for (ProblemSpecP eqn_db = db_st->findBlock("eqn_group"); eqn_db != 0;
         eqn_db = eqn_db->findNextBlock("eqn_group")){

      std::string group_name = "null";
      std::string type = "null";
      eqn_db->getAttribute("label", group_name);
      eqn_db->getAttribute("type", type );

      TaskInterface::TaskBuilder* tsk;
      if ( type == "CC" ){
        tsk = new KScalarRHS<CCVariable<double> >::Builder(group_name, 0);
        _scalar_builders.push_back(group_name);
      } else if ( type == "FX" ){
        tsk = new KScalarRHS<SFCXVariable<double> >::Builder(group_name, 0);
        _momentum_builders.push_back(group_name);
      } else if ( type == "FY" ){
        tsk = new KScalarRHS<SFCYVariable<double> >::Builder(group_name, 0);
        _momentum_builders.push_back(group_name);
      } else if ( type == "FZ" ){
        tsk = new KScalarRHS<SFCZVariable<double> >::Builder(group_name, 0);
        _momentum_builders.push_back(group_name);
      } else {
        throw InvalidValue("Error: Eqn type for group not recognized named: "+group_name+" with type: "+type,__FILE__,__LINE__);
      }
      register_task( group_name, tsk );

      //Generate a psi function for each scalar and fe updates:
      if ( type == "CC" ){

        std::string compute_psi_name = "compute_scalar_psi_"+group_name;
        TaskInterface::TaskBuilder* compute_psi_tsk =
        new ComputePsi<CCVariable<double> >::Builder( compute_psi_name, 0 );
        _scalar_compute_psi.push_back(compute_psi_name);
        register_task( compute_psi_name, compute_psi_tsk );

        std::string update_task_name = "scalar_fe_update_"+group_name;
        KFEUpdate<CCVariable<double> >::Builder* update_tsk =
        new KFEUpdate<CCVariable<double> >::Builder( update_task_name, 0 );
        register_task( update_task_name, update_tsk );
        _scalar_update.push_back( update_task_name );

      } else if ( type == "FX" ){

        std::string compute_psi_name = "compute_momentum_psi_"+group_name;
        TaskInterface::TaskBuilder* compute_psi_tsk =
        new ComputePsi<SFCXVariable<double> >::Builder( compute_psi_name, 0 );
        _momentum_compute_psi.push_back(compute_psi_name);
        register_task( compute_psi_name, compute_psi_tsk );

        std::string update_task_name = "momentum_fe_update_"+group_name;
        KFEUpdate<SFCXVariable<double> >::Builder* update_tsk =
        new KFEUpdate<SFCXVariable<double> >::Builder( update_task_name, 0 );
        register_task( update_task_name, update_tsk );
        _momentum_update.push_back( update_task_name );

      } else if ( type == "FY" ){

        std::string compute_psi_name = "compute_momentum_psi_"+group_name;
        TaskInterface::TaskBuilder* compute_psi_tsk =
        new ComputePsi<SFCYVariable<double> >::Builder( compute_psi_name, 0 );
        _momentum_compute_psi.push_back(compute_psi_name);
        register_task( compute_psi_name, compute_psi_tsk );

        std::string update_task_name = "momentum_fe_update_"+group_name;
        KFEUpdate<SFCYVariable<double> >::Builder* update_tsk =
        new KFEUpdate<SFCYVariable<double> >::Builder( update_task_name, 0 );
        register_task( update_task_name, update_tsk );
        _momentum_update.push_back( update_task_name );

      } else if ( type == "FZ" ){

        std::string compute_psi_name = "compute_momentum_psi_"+group_name;
        TaskInterface::TaskBuilder* compute_psi_tsk =
        new ComputePsi<SFCZVariable<double> >::Builder( compute_psi_name, 0 );
        _momentum_compute_psi.push_back(compute_psi_name);
        register_task( compute_psi_name, compute_psi_tsk );

        std::string update_task_name = "momentum_fe_update_"+group_name;
        KFEUpdate<SFCZVariable<double> >::Builder* update_tsk =
        new KFEUpdate<SFCZVariable<double> >::Builder( update_task_name, 0 );
        register_task( update_task_name, update_tsk );
        _momentum_update.push_back( update_task_name );

      }

      // std::string ssp_task_name = "scalar_ssp_update";
      // SSPInt::Builder* tsk2 = new SSPInt::Builder( ssp_task_name, 0, _scalar_builders );
      // register_task( ssp_task_name, tsk2 );

      // _scalar_ssp.push_back( ssp_task_name );

    }

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

    TaskInterface::TaskBuilder* umom_tsk = new URHS<XVol>::Builder( task_name, 0, u_name, v_name, w_name );
    register_task( task_name, umom_tsk );

    _momentum_builders.push_back(task_name);

    //---v
    task_name = "vmom";
    TaskInterface::TaskBuilder* vmom_tsk = new URHS<YVol>::Builder( task_name, 0, v_name, w_name, u_name );
    register_task( task_name, vmom_tsk );

    _momentum_builders.push_back(task_name);

    //---w
    task_name = "wmom";
    TaskInterface::TaskBuilder* wmom_tsk = new URHS<ZVol>::Builder( task_name, 0, w_name, u_name, v_name );
    register_task( task_name, wmom_tsk );

    _momentum_builders.push_back(task_name);

    typedef SpatialOps::XVolField XVol;
    typedef SpatialOps::YVolField YVol;
    typedef SpatialOps::ZVolField ZVol;
    std::string update_task_name = "umom_fe_update";
    std::vector<std::string> u_up_tsk;
    u_up_tsk.push_back("umom");
    FEUpdate<XVol>::Builder* utsk = new FEUpdate<XVol>::Builder( update_task_name, 0, u_up_tsk );
    register_task( update_task_name, utsk );
    _momentum_update.push_back( update_task_name );

    update_task_name = "vmom_fe_update";
    std::vector<std::string> v_up_tsk;
    v_up_tsk.push_back("vmom");
    FEUpdate<YVol>::Builder* vtsk = new FEUpdate<YVol>::Builder( update_task_name, 0, v_up_tsk );
    register_task( update_task_name, vtsk );
    _momentum_update.push_back( update_task_name );

    update_task_name = "wmom_fe_update";
    std::vector<std::string> w_up_tsk;
    w_up_tsk.push_back("wmom");
    FEUpdate<YVol>::Builder* wtsk = new FEUpdate<YVol>::Builder( update_task_name, 0, w_up_tsk );
    register_task( update_task_name, wtsk );
    //_momentum_update.push_back( update_task_name );

    //std::string ssp_task_name = "umom_ssp_update";
    //SSPInt<XVol>::Builder* ssp_utsk = new SSPInt<XVol>::Builder( ssp_task_name, 0, u_up_tsk );
    //register_task( ssp_task_name, ssp_utsk );

    //ssp_task_name = "vmom_ssp_update";
    //SSPInt<YVol>::Builder* ssp_vtsk = new SSPInt<YVol>::Builder( ssp_task_name, 0, v_up_tsk );
    //register_task( ssp_task_name, ssp_vtsk );

    //ssp_task_name = "wmom_ssp_update";
    //SSPInt<ZVol>::Builder* ssp_wtsk = new SSPInt<ZVol>::Builder( ssp_task_name, 0, w_up_tsk );
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

    for (ProblemSpecP group_db = db_st->findBlock("eqn_group"); group_db != 0;
         group_db = group_db->findNextBlock("eqn_group")){

      std::string group_name = "null";
      std::string type = "null";
      group_db->getAttribute("label", group_name);
      group_db->getAttribute("type", type );

      //RHS builders
      TaskInterface* tsk = retrieve_task(group_name);
      tsk->problemSetup(group_db);
      tsk->create_local_labels();

      //PSI generation and FE update:
      if ( type == "CC"){

        TaskInterface* psi_tsk = retrieve_task("compute_scalar_psi_"+group_name);
        psi_tsk->problemSetup( group_db );
        psi_tsk->create_local_labels();

        TaskInterface* fe_tsk = retrieve_task("scalar_fe_update_"+group_name);
        fe_tsk->problemSetup( group_db );
        fe_tsk->create_local_labels();

      } else {

        TaskInterface* psi_tsk = retrieve_task("compute_momentum_psi_"+group_name);
        psi_tsk->problemSetup( group_db );
        psi_tsk->create_local_labels();

        TaskInterface* fe_tsk = retrieve_task("momentum_fe_update_"+group_name);
        fe_tsk->problemSetup( group_db );
        fe_tsk->create_local_labels();

      }

      // tsk = retrieve_task("scalar_ssp_update_"+group_name);
      // tsk->problemSetup( group_db );
      //
      // tsk->create_local_labels();

    }
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
