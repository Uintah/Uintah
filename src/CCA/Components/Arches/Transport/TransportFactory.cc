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
#include <CCA/Components/Arches/Transport/Diffusion.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Task/AtomicTaskInterface.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <libxml/parser.h>
#include <libxml/tree.h>

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

        typedef CCVariable<double> C;
        typedef typename ArchesCore::VariableHelper<CCVariable<double> >::ConstType CT;

        if ( m_pack_transport_construction_tasks ){
          tsk = scinew KScalarRHS<C, C >::Builder(group_name, 0);
        } else {
          tsk = scinew KScalarRHS<C, CT >::Builder(group_name, 0);
        }

      } else if ( type == "FX" ){

        typedef SFCXVariable<double> C;
        typedef typename ArchesCore::VariableHelper<SFCXVariable<double> >::ConstType CT;

        if ( m_pack_transport_construction_tasks ){
          tsk = scinew KScalarRHS<C, C >::Builder(group_name, 0);
        } else {
          tsk = scinew KScalarRHS<C, CT >::Builder(group_name, 0);
        }

      } else if ( type == "FY" ){

        typedef SFCYVariable<double> C;
        typedef typename ArchesCore::VariableHelper<SFCYVariable<double> >::ConstType CT;

        if ( m_pack_transport_construction_tasks ){
          tsk = scinew KScalarRHS<C, C >::Builder(group_name, 0);
        } else {
          tsk = scinew KScalarRHS<C, CT >::Builder(group_name, 0);
        }

      } else if ( type == "FZ" ){

        typedef SFCZVariable<double> C;
        typedef typename ArchesCore::VariableHelper<SFCZVariable<double> >::ConstType CT;

        if ( m_pack_transport_construction_tasks ){
          tsk = scinew KScalarRHS<C, C >::Builder(group_name, 0);
        } else {
          tsk = scinew KScalarRHS<C, CT >::Builder(group_name, 0);
        }

      } else {
        throw InvalidValue("Error: Eqn type for group not recognized named: "+group_name+" with type: "+type,__FILE__,__LINE__);
      }

      _scalar_builders.push_back(group_name);
      register_task( group_name, tsk );

      //Generate a psi function for each scalar and fe updates:
      std::string compute_psi_name = "compute_scalar_psi_"+group_name;
      std::string update_task_name = "scalar_fe_update_"+group_name;
      std::string diffusion_task_name = "diffusion_"+group_name;
      if ( type == "CC" ){
        TaskInterface::TaskBuilder* compute_psi_tsk =
        scinew ComputePsi<CCVariable<double> >::Builder( compute_psi_name, 0 );
        register_task( compute_psi_name, compute_psi_tsk );
        KFEUpdate<CCVariable<double> >::Builder* update_tsk =
        scinew KFEUpdate<CCVariable<double> >::Builder( update_task_name, 0 );
        register_task( update_task_name, update_tsk );

        //diffusion term:
        TaskInterface::TaskBuilder* diff_tsk =
          scinew Diffusion<CCVariable<double> >::Builder( diffusion_task_name, 0 );
        register_task( diffusion_task_name, diff_tsk );

      } else if ( type == "FX" ){
        TaskInterface::TaskBuilder* compute_psi_tsk =
        scinew ComputePsi<SFCXVariable<double> >::Builder( compute_psi_name, 0 );
        register_task( compute_psi_name, compute_psi_tsk );
        KFEUpdate<SFCXVariable<double> >::Builder* update_tsk =
        scinew KFEUpdate<SFCXVariable<double> >::Builder( update_task_name, 0 );
        register_task( update_task_name, update_tsk );

        //diffusion term:
        TaskInterface::TaskBuilder* diff_tsk =
          scinew Diffusion<SFCXVariable<double> >::Builder( diffusion_task_name, 0 );
        register_task( diffusion_task_name, diff_tsk );

      } else if ( type == "FY" ){
        TaskInterface::TaskBuilder* compute_psi_tsk =
        scinew ComputePsi<SFCYVariable<double> >::Builder( compute_psi_name, 0 );
        register_task( compute_psi_name, compute_psi_tsk );
        KFEUpdate<SFCYVariable<double> >::Builder* update_tsk =
        scinew KFEUpdate<SFCYVariable<double> >::Builder( update_task_name, 0 );
        register_task( update_task_name, update_tsk );

        //diffusion term:
        TaskInterface::TaskBuilder* diff_tsk =
          scinew Diffusion<SFCYVariable<double> >::Builder( diffusion_task_name, 0 );
        register_task( diffusion_task_name, diff_tsk );

      } else if ( type == "FZ" ){
        TaskInterface::TaskBuilder* compute_psi_tsk =
        scinew ComputePsi<SFCZVariable<double> >::Builder( compute_psi_name, 0 );
        register_task( compute_psi_name, compute_psi_tsk );
        KFEUpdate<SFCZVariable<double> >::Builder* update_tsk =
        scinew KFEUpdate<SFCZVariable<double> >::Builder( update_task_name, 0 );
        register_task( update_task_name, update_tsk );

        //diffusion term:
        TaskInterface::TaskBuilder* diff_tsk =
          scinew Diffusion<SFCZVariable<double> >::Builder( diffusion_task_name, 0 );
        register_task( diffusion_task_name, diff_tsk );

      }

      _scalar_compute_psi.push_back(compute_psi_name);
      _scalar_update.push_back( update_task_name );
      _scalar_diffusion.push_back( diffusion_task_name );

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

    if ( db ->findBlock("KMomentum")->findBlock("PressureSolver")){
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

  if ( db->findBlock("DQMOM") ){

    if ( db->findBlock("DQMOM")->findBlock("kokkos_translate") ) {
      register_DQMOM(db->findBlock("DQMOM"));
    }

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
      print_task_setup_info(group_name, "Compute RHS.");
      tsk->problemSetup(group_db);
      tsk->create_local_labels();

      TaskInterface* psi_tsk = retrieve_task("compute_scalar_psi_"+group_name);
      print_task_setup_info(group_name, "Compute PSI function.");
      psi_tsk->problemSetup( group_db );
      psi_tsk->create_local_labels();

      std::string diffusion_task_name = "diffusion_"+group_name;
      print_task_setup_info(diffusion_task_name, "Compute diffusive fluxes.");
      TaskInterface* diff_tsk = retrieve_task(diffusion_task_name);
      diff_tsk->problemSetup( group_db );
      diff_tsk->create_local_labels();

      TaskInterface* fe_tsk = retrieve_task("scalar_fe_update_"+group_name);
      print_task_setup_info(group_name, "FE update.");
      fe_tsk->problemSetup( group_db );
      fe_tsk->create_local_labels();

      // tsk = retrieve_task("scalar_ssp_update_"+group_name);
      // tsk->problemSetup( group_db );
      // tsk->create_local_labels();

    }
  }

  if ( db->findBlock("DQMOM")){
    if ( db->findBlock("DQMOM")->findBlock("kokkos_translate") ){
      build_DQMOM( db );
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

    if ( db_mom ->findBlock("PressureSolver")){
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
   } else {
     throw ProblemSetupException("Error: Please update UPS file to include a <PressureSolver> tag since momentum was detected.", __FILE__, __LINE__);
   }

  }
}

//--------------------------------------------------------------------------------------------------
void TransportFactory::schedule_initialization( const LevelP& level,
                                                SchedulerP& sched,
                                                const MaterialSet* matls,
                                                bool doing_restart ){

  for ( auto i = _tasks.begin(); i != _tasks.end(); i++ ){

    if ( i->first != "build_pressure_system" ){
      schedule_task( i->first, TaskInterface::INITIALIZE, level, sched, matls );
    }

  }

  //because this relies on momentum solvers.
  if ( has_task( "build_pressure_system" ) ){
    schedule_task( "build_pressure_system", TaskInterface::INITIALIZE, level, sched, matls ); 
  }

}

//--------------------------------------------------------------------------------------------------
void TransportFactory::register_DQMOM( ProblemSpecP db_dqmom ){

  unsigned int nQn = ArchesCore::get_num_env( db_dqmom, ArchesCore::DQMOM_METHOD );

  for ( int i = 0; i < int(nQn); i++ ){

    std::stringstream dqmom_eqn_grp_env;
    dqmom_eqn_grp_env << m_dqmom_grp_name << "_" << i;

    std::string group_name = dqmom_eqn_grp_env.str();

    TaskInterface::TaskBuilder* tsk;
    typedef CCVariable<double> C;
    typedef typename ArchesCore::VariableHelper<CCVariable<double> >::ConstType CT;

    if ( m_pack_transport_construction_tasks ){
      tsk = scinew KScalarRHS<C, C >::Builder(group_name, 0);
    } else {
      tsk = scinew KScalarRHS<C, CT >::Builder(group_name, 0);
    }

    _dqmom_eqns.push_back(group_name);
    register_task( group_name, tsk );

    std::string compute_psi_name = "dqmom_psi_builders_"+group_name;
    std::string update_task_name = "dqmom_fe_update_"+group_name;

    TaskInterface::TaskBuilder* compute_psi_tsk =
    scinew ComputePsi<CCVariable<double> >::Builder( compute_psi_name, 0 );
    register_task( compute_psi_name, compute_psi_tsk );

    KFEUpdate<CCVariable<double> >::Builder* update_tsk =
    scinew KFEUpdate<CCVariable<double> >::Builder( update_task_name, 0 );
    register_task( update_task_name, update_tsk );

    _dqmom_fe_update.push_back( update_task_name );
    _dqmom_compute_psi.push_back( compute_psi_name );

    if ( db_dqmom->findBlock("diffusion") ) {
      std::string diffusion_task_name = "dqmom_diffusion_"+group_name;
      TaskInterface::TaskBuilder* diff_tsk =
        scinew Diffusion<CCVariable<double> >::Builder( diffusion_task_name, 0);
      register_task( diffusion_task_name, diff_tsk );
      _dqmom_compute_diff.push_back( diffusion_task_name );
    }

  }
}

//--------------------------------------------------------------------------------------------------
void TransportFactory::build_DQMOM( ProblemSpecP db ){

  ProblemSpecP db_dqmom = db->findBlock("DQMOM");

  std::string output_xml_name;
  bool print_ups_with_dqmom = false;
  if ( db_dqmom->findBlock("write_input_with_dqmom_eqns") ){
    db_dqmom->getWithDefault("write_input_with_dqmom_eqns", output_xml_name, "your_input_with_dqmom_eqns.xml");
    print_ups_with_dqmom = true;
  }

  ProblemSpecP db_transport;

  if ( db->findBlock("KScalarTransport") ){
    db_transport = db->findBlock("KScalarTransport");
  } else {
    db_transport = db->appendChild("KScalarTransport");
  }

  unsigned int nQn = ArchesCore::get_num_env( db_dqmom, ArchesCore::DQMOM_METHOD );

  for ( int i = 0; i < int(nQn); i++ ){

    std::stringstream dqmom_eqn_grp_env;
    dqmom_eqn_grp_env << m_dqmom_grp_name << "_" << i;
    std::string grp_name = dqmom_eqn_grp_env.str();

  //Create weights
    ProblemSpecP db_eqn_group = db_transport->appendChild("eqn_group");
    db_eqn_group->setAttribute("label", dqmom_eqn_grp_env.str());
    db_eqn_group->setAttribute("type", "CC");
    db_eqn_group->setAttribute("class", "dqmom");

    std::string conv_scheme;
    bool do_convection = false;
    if ( db_dqmom->findBlock("convection") ) {
      do_convection = true;
      db_dqmom->findBlock("convection")->getAttribute("scheme", conv_scheme);
    }

    std::string diff_scheme;
    std::string D_label;
    bool do_diffusion = false;
    if ( db_dqmom->findBlock("diffusion") ) {
      do_diffusion = true;
      db_dqmom->findBlock("diffusion")->getAttribute("scheme", diff_scheme);
      db_dqmom->findBlock("diffusion")->getAttribute("D_label", D_label);
      db_eqn_group->appendChild("diffusion_coef")->setAttribute("label", D_label);
    }

    ProblemSpecP db_weight = db_dqmom->findBlock("Weights");
    if ( !db_weight ){
      throw ProblemSetupException("Error: No <Weights> spec found in <DQMOM>.",__FILE__,__LINE__);
    }

    ProblemSpecP eqn_db = db_eqn_group->appendChild("eqn");
    std::stringstream this_qn;
    this_qn << i;

    if ( db_dqmom->findBlock("velocity") ){
      //This sets one velocity for the distribution
      std::string xvel_label;
      std::string yvel_label;
      std::string zvel_label;
      db_dqmom->findBlock("velocity")->getAttribute("xlabel", xvel_label);
      db_dqmom->findBlock("velocity")->getAttribute("ylabel", yvel_label);
      db_dqmom->findBlock("velocity")->getAttribute("zlabel", zvel_label);
      ProblemSpecP conv_vel = db_eqn_group->appendChild("velocity");
      conv_vel->setAttribute("xlabel", xvel_label);
      conv_vel->setAttribute("ylabel", yvel_label);
      conv_vel->setAttribute("zlabel", zvel_label);
    } else {
      //This uses the internal coordinates of velocity for the distribution
      ProblemSpecP conv_vel = db_eqn_group->appendChild("velocity");
      conv_vel->setAttribute("xlabel", "pvel_face_x_" + this_qn.str());
      conv_vel->setAttribute("ylabel", "pvel_face_y_" + this_qn.str());
      conv_vel->setAttribute("zlabel", "pvel_fave_z_" + this_qn.str());
    }


    eqn_db->setAttribute("label", "w_qn"+this_qn.str());
    if ( do_convection ){
      ProblemSpecP conv_db = eqn_db->appendChild("convection");
      conv_db->setAttribute("scheme", conv_scheme );
    }
    if ( do_diffusion ){
      ProblemSpecP diff_db = eqn_db->appendChild("diffusion");
      diff_db->setAttribute("scheme", diff_scheme );
    }

    //link the models with the Weight:
    //if ( db_weight->findBlock("model") ){
      //for ( ProblemSpecP db_model = db_weight->findBlock("model"); db_model != nullptr;
            //db_model = db_model->findNextBlock("model") ){

        //std::string label;
        //db_model->getAttribute("label", label);

        //ProblemSpecP src_db = eqn_db->appendChild("src");
        //src_db->setAttribute( "label", label );

      //}
    //}

    ProblemSpecP src_db = eqn_db->appendChild("src");
    src_db->setAttribute("label", "w_qn"+this_qn.str()+"_src");

    ProblemSpecP db_init = db_weight->findBlock("initialization");
    if ( db_init != nullptr ){

      std::string type;
      db_init->getAttribute("type", type);
      if ( type != "env_constant" ){
        throw ProblemSetupException("Error: Only env_constant is allowed for DQMOM. Use the <Initialize> tag instead.", __FILE__, __LINE__);
      }

      for ( ProblemSpecP db_env = db_init->findBlock("env_constant"); db_env != nullptr;
            db_env = db_env->findNextBlock("env_constant") ){

        int this_qn;
        db_env->getAttribute("qn", this_qn );

        if ( this_qn == i ){
          std::string value;
          db_env->getAttribute("value", value );
          ProblemSpecP db_new_init = eqn_db->appendChild("initialize");
          db_new_init->setAttribute("value", value);
        }
      }
    }

    ProblemSpecP db_scal = db_weight->findBlock("scaling_const");
    if ( db_scal ){
      std::vector<std::string> scaling_constants;
      db_weight->require("scaling_const", scaling_constants);

      if ( scaling_constants.size() != nQn ){
        throw ProblemSetupException("Error: number of scaling constants != number quadrature nodes.", __FILE__, __LINE__);
      }

      eqn_db->appendChild("scaling")->setAttribute("value", scaling_constants[i]);

    }

    for ( ProblemSpecP db_ic = db_dqmom->findBlock("Ic"); db_ic != nullptr;
          db_ic =db_ic->findNextBlock("Ic") ){

      std::string ic_label;
      db_ic->getAttribute("label", ic_label);

      ProblemSpecP eqn_db = db_eqn_group->appendChild("eqn");
      std::stringstream this_qn;
      this_qn << i;
      eqn_db->setAttribute("label", ic_label+"_qn"+this_qn.str());

      //transport:
      if ( do_convection ){
        ProblemSpecP conv_db = eqn_db->appendChild("convection");
        conv_db->setAttribute("scheme", conv_scheme );
      }
      if ( do_diffusion ){
        ProblemSpecP diff_db = eqn_db->appendChild("diffusion");
        diff_db->setAttribute("scheme", diff_scheme );
      }

        //link the models with the Ic:
        //if ( db_ic->findBlock("model") ){
          //for ( ProblemSpecP db_model = db_ic->findBlock("model"); db_model != nullptr;
                //db_model = db_model->findNextBlock("model") ){

            //std::string label;
            //db_model->getAttribute("label", label);

            //ProblemSpecP src_db = eqn_db->appendChild("src");
            //src_db->setAttribute( "label", label+"_"+this_qn.str() );

          //}
        //}
      ProblemSpecP src_db = eqn_db->appendChild("src");
      src_db->setAttribute("label", ic_label+"_qn"+this_qn.str()+"_src");

      ProblemSpecP db_init = db_ic->findBlock("initialization");
      if ( db_init != nullptr ){

        std::string type;
        db_init->getAttribute("type", type);
        if ( type != "env_constant" ){
          throw ProblemSetupException("Error: Only env_constant is allowed for DQMOM. Use the <Initialize> tag instead.", __FILE__, __LINE__);
        }

        for ( ProblemSpecP db_env = db_init->findBlock("env_constant"); db_env != nullptr;
              db_env = db_env->findNextBlock("env_constant") ){

          int this_qn;
          db_env->getAttribute("qn", this_qn );

          if ( this_qn == i ){
            std::string value;
            db_env->getAttribute("value", value );
            ProblemSpecP db_new_init = eqn_db->appendChild("initialize");
            db_new_init->setAttribute("value", value);
          }
        }
      }
    }

    // RHSs
    std::string group_name = dqmom_eqn_grp_env.str();
    TaskInterface* tsk = retrieve_task(group_name);
    print_task_setup_info( group_name, "DQMOM rhs construction.");
    tsk->problemSetup(db_eqn_group);
    tsk->create_local_labels();

    // Psi
    TaskInterface* psi_tsk = retrieve_task("dqmom_psi_builders_"+group_name);
    print_task_setup_info( "dqmom_psi_builders_"+group_name, "DQMOM compute psi.");
    psi_tsk->problemSetup( db_eqn_group );
    psi_tsk->create_local_labels();

    // Diffusion
    if ( do_diffusion ){
      TaskInterface* diff_tsk = retrieve_task("dqmom_diffusion_"+group_name);
      diff_tsk->problemSetup( db_eqn_group );
      diff_tsk->create_local_labels();
    }

    // FE update
    TaskInterface* fe_tsk = retrieve_task("dqmom_fe_update_"+group_name);
    print_task_setup_info( "dqmom_fe_update_"+group_name, "DQMOM FE update.");
    fe_tsk->problemSetup( db_eqn_group );
    fe_tsk->create_local_labels();

  }

  // Print to a temp file if user requests
  if ( print_ups_with_dqmom ){
    db_transport->output(output_xml_name.c_str());
  }

  for ( int i = 0; i < int(nQn); i++ ){

    std::stringstream dqmom_eqn_grp_env;
    dqmom_eqn_grp_env << m_dqmom_grp_name << "_" << i;
    std::string grp_name = dqmom_eqn_grp_env.str();
    //Going to remove the input that I just created so that restarts work.
    //Otherwise, the input isn't parsed properly for restart.
    for ( ProblemSpecP db_grp = db_transport->findBlock("eqn_group"); db_grp != nullptr;
              db_grp = db_grp->findNextBlock("eqn_group") ){
      std::string grp_name;
      db_grp->getAttribute("label", grp_name);
      if ( grp_name == dqmom_eqn_grp_env.str() ){
        db_transport->removeChild(db_grp);
        break;
      }
    }
  }
}

} //namespace Uintah
