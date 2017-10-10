#include <CCA/Components/Arches/Transport/TransportFactory.h>
#include <CCA/Components/Arches/Transport/KScalarRHS.h>
#include <CCA/Components/Arches/Transport/KMomentum.h>
#include <CCA/Components/Arches/Transport/ComputePsi.h>
#include <CCA/Components/Arches/Transport/KFEUpdate.h>
#include <CCA/Components/Arches/Transport/PressureEqn.h>
#include <CCA/Components/Arches/Transport/VelRhoHatBC.h>
#include <CCA/Components/Arches/Transport/AddPressGradient.h>
#include <CCA/Components/Arches/Transport/PressureBC.h>
#include <CCA/Components/Arches/Transport/StressTensor.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Task/AtomicTaskInterface.h>

namespace Uintah{

TransportFactory::TransportFactory()
{
  _factory_name = "TransportFactory";
}

TransportFactory::~TransportFactory()
{}

void
TransportFactory::register_all_tasks( ProblemSpecP& db )
{

  if ( db->findBlock("KScalarTransport") ){

    ProblemSpecP db_st = db->findBlock("KScalarTransport");

    if ( db_st->findBlock("pack_transport_construction") ) m_pack_transport_construction_tasks = true;

    for (ProblemSpecP eqn_db = db_st->findBlock("eqn_group"); eqn_db != nullptr; eqn_db = eqn_db->findNextBlock("eqn_group")){

      std::string group_name = "null";
      std::string type = "null";
      eqn_db->getAttribute("label", group_name);
      eqn_db->getAttribute("type", type );

      TaskInterface::TaskBuilder* tsk;
      if ( type == "CC" ){
        typedef typename ArchesCore::VariableHelper<CCVariable<double> >::Type C;
        typedef typename ArchesCore::VariableHelper<CCVariable<double> >::XFaceType FXT;
        typedef typename ArchesCore::VariableHelper<CCVariable<double> >::YFaceType FYT;
        typedef typename ArchesCore::VariableHelper<CCVariable<double> >::ZFaceType FZT;
        typedef typename ArchesCore::VariableHelper<CCVariable<double> >::ConstXFaceType CFXT;
        typedef typename ArchesCore::VariableHelper<CCVariable<double> >::ConstYFaceType CFYT;
        typedef typename ArchesCore::VariableHelper<CCVariable<double> >::ConstZFaceType CFZT;
        if ( m_pack_transport_construction_tasks ){
          tsk = scinew KScalarRHS<C, FXT, FYT, FZT >::Builder(group_name, 0);
        } else {
          tsk = scinew KScalarRHS<C, CFXT, CFYT, CFZT >::Builder(group_name, 0);
        }
      } else if ( type == "FX" ){
        typedef typename ArchesCore::VariableHelper<SFCXVariable<double> >::Type C;
        typedef typename ArchesCore::VariableHelper<SFCXVariable<double> >::XFaceType FXT;
        typedef typename ArchesCore::VariableHelper<SFCXVariable<double> >::YFaceType FYT;
        typedef typename ArchesCore::VariableHelper<SFCXVariable<double> >::ZFaceType FZT;
        typedef typename ArchesCore::VariableHelper<SFCXVariable<double> >::ConstXFaceType CFXT;
        typedef typename ArchesCore::VariableHelper<SFCXVariable<double> >::ConstYFaceType CFYT;
        typedef typename ArchesCore::VariableHelper<SFCXVariable<double> >::ConstZFaceType CFZT;
        if ( m_pack_transport_construction_tasks ){
          tsk = scinew KScalarRHS<C, FXT, FYT, FZT >::Builder(group_name, 0);
        } else {
          tsk = scinew KScalarRHS<C, CFXT, CFYT, CFZT >::Builder(group_name, 0);
        }
      } else if ( type == "FY" ){
        typedef typename ArchesCore::VariableHelper<SFCYVariable<double> >::Type C;
        typedef typename ArchesCore::VariableHelper<SFCYVariable<double> >::XFaceType FXT;
        typedef typename ArchesCore::VariableHelper<SFCYVariable<double> >::YFaceType FYT;
        typedef typename ArchesCore::VariableHelper<SFCYVariable<double> >::ZFaceType FZT;
        typedef typename ArchesCore::VariableHelper<SFCYVariable<double> >::ConstXFaceType CFXT;
        typedef typename ArchesCore::VariableHelper<SFCYVariable<double> >::ConstYFaceType CFYT;
        typedef typename ArchesCore::VariableHelper<SFCYVariable<double> >::ConstZFaceType CFZT;
        if ( m_pack_transport_construction_tasks ){
          tsk = scinew KScalarRHS<C, FXT, FYT, FZT >::Builder(group_name, 0);
        } else {
          tsk = scinew KScalarRHS<C, CFXT, CFYT, CFZT >::Builder(group_name, 0);
        }
      } else if ( type == "FZ" ){
        typedef typename ArchesCore::VariableHelper<SFCZVariable<double> >::Type C;
        typedef typename ArchesCore::VariableHelper<SFCZVariable<double> >::XFaceType FXT;
        typedef typename ArchesCore::VariableHelper<SFCZVariable<double> >::YFaceType FYT;
        typedef typename ArchesCore::VariableHelper<SFCZVariable<double> >::ZFaceType FZT;
        typedef typename ArchesCore::VariableHelper<SFCZVariable<double> >::ConstXFaceType CFXT;
        typedef typename ArchesCore::VariableHelper<SFCZVariable<double> >::ConstYFaceType CFYT;
        typedef typename ArchesCore::VariableHelper<SFCZVariable<double> >::ConstZFaceType CFZT;
        if ( m_pack_transport_construction_tasks ){
          tsk = scinew KScalarRHS<C, FXT, FYT, FZT >::Builder(group_name, 0);
        } else {
          tsk = scinew KScalarRHS<C, CFXT, CFYT, CFZT >::Builder(group_name, 0);
        }
      } else {
        throw InvalidValue("Error: Eqn type for group not recognized named: "+group_name+" with type: "+type,__FILE__,__LINE__);
      }
      _scalar_builders.push_back(group_name);
      register_task( group_name, tsk );

      //Generate a psi function for each scalar and fe updates:
      std::string compute_psi_name = "compute_scalar_psi_"+group_name;
      std::string update_task_name = "scalar_fe_update_"+group_name;
      if ( type == "CC" ){
        TaskInterface::TaskBuilder* compute_psi_tsk =
        scinew ComputePsi<CCVariable<double> >::Builder( compute_psi_name, 0 );
        register_task( compute_psi_name, compute_psi_tsk );
        KFEUpdate<CCVariable<double> >::Builder* update_tsk =
        scinew KFEUpdate<CCVariable<double> >::Builder( update_task_name, 0 );
        register_task( update_task_name, update_tsk );
      } else if ( type == "FX" ){
        TaskInterface::TaskBuilder* compute_psi_tsk =
        scinew ComputePsi<SFCXVariable<double> >::Builder( compute_psi_name, 0 );
        register_task( compute_psi_name, compute_psi_tsk );
        KFEUpdate<SFCXVariable<double> >::Builder* update_tsk =
        scinew KFEUpdate<SFCXVariable<double> >::Builder( update_task_name, 0 );
        register_task( update_task_name, update_tsk );
      } else if ( type == "FY" ){
        TaskInterface::TaskBuilder* compute_psi_tsk =
        scinew ComputePsi<SFCYVariable<double> >::Builder( compute_psi_name, 0 );
        register_task( compute_psi_name, compute_psi_tsk );
        KFEUpdate<SFCYVariable<double> >::Builder* update_tsk =
        scinew KFEUpdate<SFCYVariable<double> >::Builder( update_task_name, 0 );
        register_task( update_task_name, update_tsk );
      } else if ( type == "FZ" ){
        TaskInterface::TaskBuilder* compute_psi_tsk =
        scinew ComputePsi<SFCZVariable<double> >::Builder( compute_psi_name, 0 );
        register_task( compute_psi_name, compute_psi_tsk );
        KFEUpdate<SFCZVariable<double> >::Builder* update_tsk =
        scinew KFEUpdate<SFCZVariable<double> >::Builder( update_task_name, 0 );
        register_task( update_task_name, update_tsk );
      }
      _scalar_compute_psi.push_back(compute_psi_name);
      _scalar_update.push_back( update_task_name );

    }
  }

  if ( db->findBlock("KMomentum") ){

    // stress tensor
    std::string stress_name = "stress_tensor";
    TaskInterface::TaskBuilder* stress_tsk = scinew StressTensor::Builder( stress_name, 0 );
    register_task(stress_name, stress_tsk);
    _momentum_builders.push_back( stress_name );
    _momentum_solve.push_back( stress_name );

    // X-mom
    std::string compute_psi_name = "x-mom-psi";
    std::string update_task_name = "x-mom-update";
    std::string mom_task_name = "x-mom";
    TaskInterface::TaskBuilder* x_tsk = scinew KMomentum<SFCXVariable<double> >::Builder(mom_task_name, 0);
    register_task( mom_task_name, x_tsk );
    TaskInterface::TaskBuilder* x_compute_psi_tsk =
    scinew ComputePsi<SFCXVariable<double> >::Builder( compute_psi_name, 0 );
    register_task( compute_psi_name, x_compute_psi_tsk );
    KFEUpdate<SFCXVariable<double> >::Builder* x_update_tsk =
    scinew KFEUpdate<SFCXVariable<double> >::Builder( update_task_name, 0 );
    register_task( update_task_name, x_update_tsk );

    _momentum_builders.push_back(mom_task_name);
    _momentum_compute_psi.push_back(compute_psi_name);
    _momentum_update.push_back(update_task_name);

    _momentum_solve.push_back( compute_psi_name );
    _momentum_solve.push_back( mom_task_name );

    // Y-mom
    compute_psi_name = "y-mom-psi";
    update_task_name = "y-mom-update";
    mom_task_name = "y-mom";
    TaskInterface::TaskBuilder* y_tsk = scinew KMomentum<SFCYVariable<double> >::Builder(mom_task_name, 0);
    register_task( mom_task_name, y_tsk );
    TaskInterface::TaskBuilder* y_compute_psi_tsk =
    scinew ComputePsi<SFCYVariable<double> >::Builder( compute_psi_name, 0 );
    register_task( compute_psi_name, y_compute_psi_tsk );
    TaskInterface::TaskBuilder* y_update_tsk =
    scinew KFEUpdate<SFCYVariable<double> >::Builder( update_task_name, 0 );
    register_task( update_task_name, y_update_tsk );

    _momentum_builders.push_back(mom_task_name);
    _momentum_compute_psi.push_back(compute_psi_name);
    _momentum_update.push_back(update_task_name);

    _momentum_solve.push_back( compute_psi_name );
    _momentum_solve.push_back( mom_task_name );

    // Z-mom
    compute_psi_name = "z-mom-psi";
    update_task_name = "z-mom-update";
    mom_task_name = "z-mom";
    TaskInterface::TaskBuilder* z_tsk = scinew KMomentum<SFCZVariable<double> >::Builder(mom_task_name, 0);
    register_task( mom_task_name, z_tsk );
    TaskInterface::TaskBuilder* z_compute_psi_tsk =
    scinew ComputePsi<SFCZVariable<double> >::Builder( compute_psi_name, 0 );
    register_task( compute_psi_name, z_compute_psi_tsk );
    TaskInterface::TaskBuilder* z_update_tsk =
    scinew KFEUpdate<SFCZVariable<double> >::Builder( update_task_name, 0 );
    register_task( update_task_name, z_update_tsk );

    _momentum_builders.push_back(mom_task_name);
    _momentum_compute_psi.push_back(compute_psi_name);
    _momentum_update.push_back(update_task_name);

    _momentum_solve.push_back( compute_psi_name );
    _momentum_solve.push_back( mom_task_name );

    //Pressure eqn
    TaskInterface::TaskBuilder* press_tsk = scinew PressureEqn::Builder("build_pressure_system", 0, _shared_state);
    register_task( "build_pressure_system", press_tsk );
    _pressure_eqn.push_back("build_pressure_system");

    //BC for velrhohat
    AtomicTaskInterface::AtomicTaskBuilder* velrhohatbc_tsk = scinew VelRhoHatBC::Builder("vel_rho_hat_bc", 0);
    register_atomic_task( "vel_rho_hat_bc", velrhohatbc_tsk);

    //pressure bcs
    AtomicTaskInterface::AtomicTaskBuilder* press_bc_tsk = scinew ArchesCore::PressureBC::Builder("pressure_bcs", 0);
    register_atomic_task( "pressure_bcs", press_bc_tsk );

    //pressure Gradient
    AtomicTaskInterface::AtomicTaskBuilder* gradP_tsk = scinew AddPressGradient::Builder("pressure_correction", 0);
    register_atomic_task( "pressure_correction", gradP_tsk);

  }
}

void
TransportFactory::build_all_tasks( ProblemSpecP& db )
{

  if ( db->findBlock("KScalarTransport") ){

    ProblemSpecP db_st = db->findBlock("KScalarTransport");

    for (ProblemSpecP group_db = db_st->findBlock("eqn_group"); group_db != nullptr; group_db = group_db->findNextBlock("eqn_group")){

      std::string group_name = "null";
      std::string type = "null";
      group_db->getAttribute("label", group_name);
      group_db->getAttribute("type", type );

      //RHS builders
      TaskInterface* tsk = retrieve_task(group_name);
      proc0cout << "       Task: " << group_name << "  Type: " << "compute_RHS" << std::endl;
      tsk->problemSetup(group_db);
      tsk->create_local_labels();

      TaskInterface* psi_tsk = retrieve_task("compute_scalar_psi_"+group_name);
      proc0cout << "       Task: " << group_name << "  Type: " << "compute_psi" << std::endl;
      psi_tsk->problemSetup( group_db );
      psi_tsk->create_local_labels();

      TaskInterface* fe_tsk = retrieve_task("scalar_fe_update_"+group_name);
      proc0cout << "       Task: " << group_name << "  Type: " << "scalar_fe_update" << std::endl;
      fe_tsk->problemSetup( group_db );
      fe_tsk->create_local_labels();

      // tsk = retrieve_task("scalar_ssp_update_"+group_name);
      // tsk->problemSetup( group_db );
      //
      // tsk->create_local_labels();

    }
  }

  ProblemSpecP db_mom = db->findBlock("KMomentum");

  if ( db_mom != nullptr ){

    TaskInterface* stress_tsk = retrieve_task("stress_tensor");
    print_task_setup_info( "stress_tensor", "compute stress tensor");
    stress_tsk->problemSetup( db_mom );
    stress_tsk->create_local_labels();

    TaskInterface* tsk = retrieve_task( "x-mom" );
    print_task_setup_info( "x-mom-compute-rhs", "compute rhs");
    tsk->problemSetup( db_mom );
    tsk->create_local_labels();

    TaskInterface* psi_tsk = retrieve_task("x-mom-psi");
    print_task_setup_info( "x-mom-compute-psi", "compute psi");
    psi_tsk->problemSetup( db_mom );
    psi_tsk->create_local_labels();

    TaskInterface* fe_tsk = retrieve_task("x-mom-update");
    print_task_setup_info( "x-mom-update", "fe update");
    fe_tsk->problemSetup( db_mom );
    fe_tsk->create_local_labels();

    tsk = retrieve_task( "y-mom" );
    print_task_setup_info( "y-mom-compute-rhs", "compute rhs");
    tsk->problemSetup( db_mom );
    tsk->create_local_labels();

    psi_tsk = retrieve_task("y-mom-psi");
    print_task_setup_info( "y-mom-compute-psi", "compute psi");
    psi_tsk->problemSetup( db_mom );
    psi_tsk->create_local_labels();

    fe_tsk = retrieve_task("y-mom-update");
    print_task_setup_info( "y-mom-update", "fe update");
    fe_tsk->problemSetup( db_mom );
    fe_tsk->create_local_labels();

    tsk = retrieve_task( "z-mom" );
    print_task_setup_info( "z-mom-compute-rhs", "compute rhs");
    tsk->problemSetup( db_mom );
    tsk->create_local_labels();

    psi_tsk = retrieve_task("z-mom-psi");
    print_task_setup_info( "z-mom-compute-psi", "compute psi");
    psi_tsk->problemSetup( db_mom );
    psi_tsk->create_local_labels();

    fe_tsk = retrieve_task("z-mom-update");
    print_task_setup_info( "z-mom-update", "fe update");
    fe_tsk->problemSetup( db_mom );
    fe_tsk->create_local_labels();

    TaskInterface* press_tsk = retrieve_task("build_pressure_system");
    print_task_setup_info("build_pressure_system", "building pressure terms, A, b");
    press_tsk->problemSetup( db_mom );
    press_tsk->create_local_labels();

    AtomicTaskInterface* rhouhatbc_tsk = retrieve_atomic_task("vel_rho_hat_bc");
    print_task_setup_info("vel_rho_hat_bc", "applies bc on rhouhat");
    rhouhatbc_tsk->problemSetup(db_mom);
    rhouhatbc_tsk->create_local_labels();

    AtomicTaskInterface* gradP_tsk = retrieve_atomic_task("pressure_correction");
    print_task_setup_info("pressure_correction", "correction for vels");
    gradP_tsk->problemSetup(db_mom);
    gradP_tsk->create_local_labels();

    AtomicTaskInterface* press_bc_tsk = retrieve_atomic_task("pressure_bcs");
    print_task_setup_info("pressure_bcs", "apply bcs to the solution of the linear pressure system");
    press_bc_tsk->problemSetup(db_mom);
    press_bc_tsk->create_local_labels();


  }
}

//--------------------------------------------------------------------------------------------------
void TransportFactory::schedule_initialization( const LevelP& level,
                                                SchedulerP& sched,
                                                const MaterialSet* matls,
                                                bool doing_restart ){

  for ( auto i = _tasks.begin(); i != _tasks.end(); i++ ){

    if ( i->first != "build_pressure_system" ){
      TaskInterface* tsk = retrieve_task( i->first );
      tsk->schedule_init( level, sched, matls, doing_restart );
    }

  }

  //because this relies on momentum solvers.
  if ( has_task( "build_pressure_system" ) ){
    TaskInterface* tsk = retrieve_task( "build_pressure_system" );
    tsk->schedule_init( level, sched, matls, doing_restart );
  }

}

} //namespace Uintah
