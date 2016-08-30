#include <CCA/Components/Arches/Transport/TransportFactory.h>
#include <CCA/Components/Arches/Transport/KScalarRHS.h>
#include <CCA/Components/Arches/Transport/ComputePsi.h>
#include <CCA/Components/Arches/Transport/KFEUpdate.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

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
        tsk = scinew KScalarRHS<CCVariable<double> >::Builder(group_name, 0);
        _scalar_builders.push_back(group_name);
      } else if ( type == "FX" ){
        tsk = scinew KScalarRHS<SFCXVariable<double> >::Builder(group_name, 0);
        _momentum_builders.push_back(group_name);
      } else if ( type == "FY" ){
        tsk = scinew KScalarRHS<SFCYVariable<double> >::Builder(group_name, 0);
        _momentum_builders.push_back(group_name);
      } else if ( type == "FZ" ){
        tsk = scinew KScalarRHS<SFCZVariable<double> >::Builder(group_name, 0);
        _momentum_builders.push_back(group_name);
      } else {
        throw InvalidValue("Error: Eqn type for group not recognized named: "+group_name+" with type: "+type,__FILE__,__LINE__);
      }
      register_task( group_name, tsk );

      //Generate a psi function for each scalar and fe updates:
      if ( type == "CC" ){

        std::string compute_psi_name = "compute_scalar_psi_"+group_name;
        TaskInterface::TaskBuilder* compute_psi_tsk =
        scinew ComputePsi<CCVariable<double> >::Builder( compute_psi_name, 0 );
        _scalar_compute_psi.push_back(compute_psi_name);
        register_task( compute_psi_name, compute_psi_tsk );

        std::string update_task_name = "scalar_fe_update_"+group_name;
        KFEUpdate<CCVariable<double> >::Builder* update_tsk =
        scinew KFEUpdate<CCVariable<double> >::Builder( update_task_name, 0 );
        register_task( update_task_name, update_tsk );
        _scalar_update.push_back( update_task_name );

      } else if ( type == "FX" ){

        std::string compute_psi_name = "compute_momentum_psi_"+group_name;
        TaskInterface::TaskBuilder* compute_psi_tsk =
        scinew ComputePsi<SFCXVariable<double> >::Builder( compute_psi_name, 0 );
        _momentum_compute_psi.push_back(compute_psi_name);
        register_task( compute_psi_name, compute_psi_tsk );

        std::string update_task_name = "momentum_fe_update_"+group_name;
        KFEUpdate<SFCXVariable<double> >::Builder* update_tsk =
        scinew KFEUpdate<SFCXVariable<double> >::Builder( update_task_name, 0 );
        register_task( update_task_name, update_tsk );
        _momentum_update.push_back( update_task_name );

      } else if ( type == "FY" ){

        std::string compute_psi_name = "compute_momentum_psi_"+group_name;
        TaskInterface::TaskBuilder* compute_psi_tsk =
        scinew ComputePsi<SFCYVariable<double> >::Builder( compute_psi_name, 0 );
        _momentum_compute_psi.push_back(compute_psi_name);
        register_task( compute_psi_name, compute_psi_tsk );

        std::string update_task_name = "momentum_fe_update_"+group_name;
        KFEUpdate<SFCYVariable<double> >::Builder* update_tsk =
        scinew KFEUpdate<SFCYVariable<double> >::Builder( update_task_name, 0 );
        register_task( update_task_name, update_tsk );
        _momentum_update.push_back( update_task_name );

      } else if ( type == "FZ" ){

        std::string compute_psi_name = "compute_momentum_psi_"+group_name;
        TaskInterface::TaskBuilder* compute_psi_tsk =
        scinew ComputePsi<SFCZVariable<double> >::Builder( compute_psi_name, 0 );
        _momentum_compute_psi.push_back(compute_psi_name);
        register_task( compute_psi_name, compute_psi_tsk );

        std::string update_task_name = "momentum_fe_update_"+group_name;
        KFEUpdate<SFCZVariable<double> >::Builder* update_tsk =
        scinew KFEUpdate<SFCZVariable<double> >::Builder( update_task_name, 0 );
        register_task( update_task_name, update_tsk );
        _momentum_update.push_back( update_task_name );

      }

      // std::string ssp_task_name = "scalar_ssp_update";
      // SSPInt::Builder* tsk2 = scinew SSPInt::Builder( ssp_task_name, 0, _scalar_builders );
      // register_task( ssp_task_name, tsk2 );

      // _scalar_ssp.push_back( ssp_task_name );

    }

  }

}

void
TransportFactory::build_all_tasks( ProblemSpecP& db )
{

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

  // if ( db->findBlock("MomentumTransport")){
  //
  //   ProblemSpecP db_mt = db->findBlock("MomentumTransport");
  //
  //   TaskInterface* utsk = retrieve_task("umom");
  //   utsk->problemSetup(db_mt);
  //   utsk->create_local_labels();
  //
  //   TaskInterface* vtsk = retrieve_task("vmom");
  //   vtsk->problemSetup(db_mt);
  //   vtsk->create_local_labels();
  //
  //   TaskInterface* wtsk = retrieve_task("wmom");
  //   wtsk->problemSetup(db_mt);
  //   wtsk->create_local_labels();
  //
  //   utsk = retrieve_task("umom_fe_update");
  //   utsk->problemSetup(db_mt);
  //   utsk->create_local_labels();
  //
  //   vtsk = retrieve_task("vmom_fe_update");
  //   vtsk->problemSetup(db_mt);
  //   vtsk->create_local_labels();
  //
  //   wtsk = retrieve_task("wmom_fe_update");
  //   wtsk->problemSetup(db_mt);
  //   wtsk->create_local_labels();
  // }
}
} //namespace Uintah
