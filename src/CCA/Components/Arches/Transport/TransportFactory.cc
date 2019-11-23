#include <CCA/Components/Arches/Transport/TransportFactory.h>
#include <CCA/Components/Arches/Transport/KScalarRHS.h>
#include <CCA/Components/Arches/Transport/KMomentum.h>
#include <CCA/Components/Arches/Transport/ComputePsi.h>
#include <CCA/Components/Arches/Transport/KFEUpdate.h>
#include <CCA/Components/Arches/Transport/TimeAve.h>
#include <CCA/Components/Arches/Transport/PressureEqn.h>
#include <CCA/Components/Arches/Transport/VelRhoHatBC.h>
#include <CCA/Components/Arches/Transport/AddPressGradient.h>
#include <CCA/Components/Arches/Transport/PressureBC.h>
#include <CCA/Components/Arches/Transport/StressTensor.h>
#include <CCA/Components/Arches/Transport/Diffusion.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Task/AtomicTaskInterface.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/PropertyModelsV2/UnweightVariable.h>
#include <libxml/parser.h>
#include <libxml/tree.h>

namespace Uintah{

TransportFactory::TransportFactory( const ApplicationCommon* arches ) : TaskFactoryBase( arches )
{
  _factory_name = "TransportFactory";
}

TransportFactory::~TransportFactory()
{}

void
TransportFactory::register_all_tasks( ProblemSpecP& db )
{

  using namespace ArchesCore;

  if ( db->findBlock("KScalarTransport") ){

    ProblemSpecP db_st = db->findBlock("KScalarTransport");

    if ( db_st->findBlock("pack_transport_construction") ) m_pack_transport_construction_tasks = true;

    for (ProblemSpecP eqn_db = db_st->findBlock("eqn_group"); eqn_db != nullptr; eqn_db = eqn_db->findNextBlock("eqn_group")){

      std::string group_name = "null";
      std::string type = "null";
      std::string grp_class = "density_weighted"; // ie, default is rho*phi

      eqn_db->getAttribute("label", group_name);
      eqn_db->getAttribute("type", type );
      eqn_db->getAttribute("class", grp_class );
      ArchesCore::EQUATION_CLASS enum_grp_class = ArchesCore::assign_eqn_class_enum( grp_class );

      bool prop_scalar = false;
      if ( eqn_db->findBlock("determines_properties") != nullptr ){
        prop_scalar = true;
      }

      std::string rhs_task_name = "[KScalarRHS]"+group_name;

      TaskInterface::TaskBuilder* tsk_builder;
      if ( type == "CC" ){

        typedef CCVariable<double> C;
        typedef typename ArchesCore::VariableHelper<CCVariable<double> >::ConstType CT;

        if ( m_pack_transport_construction_tasks ){
          tsk_builder = scinew KScalarRHS<C, C >::Builder(rhs_task_name, eqn_db, 0);
        } else {
          tsk_builder = scinew KScalarRHS<C, CT >::Builder(rhs_task_name, eqn_db, 0);
        }

        if ( enum_grp_class == DENSITY_WEIGHTED ){
          std::string unweight_task_name = "[UnweightVariable]"+group_name;
          TaskInterface::TaskBuilder* unweight_var_tsk =
          scinew UnweightVariable<C>::Builder( "NULL", unweight_task_name,
                                               enum_grp_class, 0 );
          register_task( unweight_task_name, unweight_var_tsk, eqn_db );
          if ( prop_scalar ){
            _prop_phi_from_rho_phi.push_back(unweight_task_name);
          } else {
            _phi_from_rho_phi.push_back(unweight_task_name);
          }
        }

      } else if ( type == "FX" ){

        typedef SFCXVariable<double> C;
        typedef typename ArchesCore::VariableHelper<SFCXVariable<double> >::ConstType CT;

        if ( m_pack_transport_construction_tasks ){
          tsk_builder = scinew KScalarRHS<C, C >::Builder(rhs_task_name, eqn_db, 0);
        } else {
          tsk_builder = scinew KScalarRHS<C, CT >::Builder(rhs_task_name, eqn_db, 0);
        }

        if ( enum_grp_class == DENSITY_WEIGHTED ){
          std::string unweight_task_name = "[UnweightVariable]"+group_name;
          TaskInterface::TaskBuilder* unweight_var_tsk =
          scinew UnweightVariable<C>::Builder( "NULL", unweight_task_name,
                                               enum_grp_class, 0 );
          register_task( unweight_task_name, unweight_var_tsk, eqn_db );
          _phi_from_rho_phi.push_back(unweight_task_name);
        }

      } else if ( type == "FY" ){

        typedef SFCYVariable<double> C;
        typedef typename ArchesCore::VariableHelper<SFCYVariable<double> >::ConstType CT;

        if ( m_pack_transport_construction_tasks ){
          tsk_builder = scinew KScalarRHS<C, C >::Builder(rhs_task_name, eqn_db, 0);
        } else {
          tsk_builder = scinew KScalarRHS<C, CT >::Builder(rhs_task_name, eqn_db, 0);
        }

        if ( enum_grp_class == DENSITY_WEIGHTED ){
          std::string unweight_task_name = "[UnweightVariable]"+group_name;
          TaskInterface::TaskBuilder* unweight_var_tsk =
          scinew UnweightVariable<C>::Builder( "NULL", unweight_task_name,
                                               enum_grp_class, 0 );
          register_task( unweight_task_name, unweight_var_tsk, eqn_db );
          _phi_from_rho_phi.push_back(unweight_task_name);
        }

      } else if ( type == "FZ" ){

        typedef SFCZVariable<double> C;
        typedef typename ArchesCore::VariableHelper<SFCZVariable<double> >::ConstType CT;

        if ( m_pack_transport_construction_tasks ){
          tsk_builder = scinew KScalarRHS<C, C >::Builder(rhs_task_name, eqn_db, 0);
        } else {
          tsk_builder = scinew KScalarRHS<C, CT >::Builder(rhs_task_name, eqn_db, 0);
        }

        if ( enum_grp_class == DENSITY_WEIGHTED ){
          std::string unweight_task_name = "[UnweightVariable]"+group_name;
          TaskInterface::TaskBuilder* unweight_var_tsk =
          scinew UnweightVariable<C>::Builder( "NULL", unweight_task_name,
                                               enum_grp_class, 0 );
          register_task( unweight_task_name, unweight_var_tsk, eqn_db );
          _phi_from_rho_phi.push_back(unweight_task_name);
        }

      } else {

        throw InvalidValue("Error: Eqn type for group not recognized named: "+group_name+" with type: "+type,__FILE__,__LINE__);

      }

      if ( enum_grp_class == DENSITY_WEIGHTED || enum_grp_class == VOLUMETRIC ){

        if ( prop_scalar ){
          _prop_scalar_builders.push_back(rhs_task_name);
        } else {
          _scalar_builders.push_back(rhs_task_name);
        }
        register_task( rhs_task_name, tsk_builder, eqn_db );

      } else if ( enum_grp_class != DQMOM ){

        throw ProblemSetupException("Error: Unable to classify eqn group: "+group_name, __FILE__, __LINE__);

      } else {

        delete tsk_builder;  //this task doesnt get registered here...

      }

      std::string diffusion_task_name = "[Diffusion]" + group_name;
      std::string rk_time_ave_task_name = "[TimeAve]" + group_name;
      std::string scalar_update_task_name = "[KFEUpdate]" + group_name;

      if ( type == "CC" ){

        if ( enum_grp_class == DENSITY_WEIGHTED || enum_grp_class == VOLUMETRIC ){
          //diffusion term:
          TaskInterface::TaskBuilder* diff_tsk =
          scinew Diffusion<CCVariable<double> >::Builder( diffusion_task_name, 0 );
          register_task( diffusion_task_name, diff_tsk, eqn_db );

          //split KFEUpdate and time ave in two task for scalar
          // This is needed for the algorithm
          TaskInterface::TaskBuilder* rk_ta_tsk =
          scinew TimeAve<CCVariable<double> >::Builder( rk_time_ave_task_name, 0 );
          register_task( rk_time_ave_task_name, rk_ta_tsk, eqn_db );

          TaskInterface::TaskBuilder* sup_tsk =
          scinew KFEUpdate<CCVariable<double> >::Builder( scalar_update_task_name, 0, false );
          register_task( scalar_update_task_name, sup_tsk, eqn_db );
        }

      } else if ( type == "FX" ){
        //diffusion term:
        TaskInterface::TaskBuilder* diff_tsk =
        scinew Diffusion<SFCXVariable<double> >::Builder( diffusion_task_name, 0 );
        register_task( diffusion_task_name, diff_tsk, eqn_db );

        TaskInterface::TaskBuilder* rk_ta_tsk =
        scinew TimeAve<SFCXVariable<double> >::Builder( rk_time_ave_task_name, 0 );
        register_task( rk_time_ave_task_name, rk_ta_tsk, eqn_db );

        TaskInterface::TaskBuilder* sup_tsk =
        scinew KFEUpdate<SFCXVariable<double> >::Builder( scalar_update_task_name, 0, false );
        register_task( scalar_update_task_name, sup_tsk, eqn_db );

      } else if ( type == "FY" ){
        //diffusion term:
        TaskInterface::TaskBuilder* diff_tsk =
        scinew Diffusion<SFCYVariable<double> >::Builder( diffusion_task_name, 0 );
        register_task( diffusion_task_name, diff_tsk, eqn_db );

        TaskInterface::TaskBuilder* rk_ta_tsk =
        scinew TimeAve<SFCYVariable<double> >::Builder( rk_time_ave_task_name, 0 );
        register_task( rk_time_ave_task_name, rk_ta_tsk, eqn_db );

        TaskInterface::TaskBuilder* sup_tsk =
        scinew KFEUpdate<SFCYVariable<double> >::Builder( scalar_update_task_name, 0, false );
        register_task( scalar_update_task_name, sup_tsk, eqn_db );

      } else if ( type == "FZ" ){
        //diffusion term:
        TaskInterface::TaskBuilder* diff_tsk =
        scinew Diffusion<SFCZVariable<double> >::Builder( diffusion_task_name, 0 );
        register_task( diffusion_task_name, diff_tsk, eqn_db );

        TaskInterface::TaskBuilder* rk_ta_tsk =
        scinew TimeAve<SFCZVariable<double> >::Builder( rk_time_ave_task_name, 0 );
        register_task( rk_time_ave_task_name, rk_ta_tsk, eqn_db );

        TaskInterface::TaskBuilder* sup_tsk =
        scinew KFEUpdate<SFCZVariable<double> >::Builder( scalar_update_task_name, 0, false );
        register_task( scalar_update_task_name, sup_tsk, eqn_db );

      }

      if ( enum_grp_class == DENSITY_WEIGHTED || enum_grp_class == VOLUMETRIC ){
        if ( prop_scalar ){
          _prop_scalar_diffusion.push_back( diffusion_task_name );
          _prop_scalar_update.push_back( scalar_update_task_name );
          _prop_rk_time_ave.push_back( rk_time_ave_task_name );
        } else {
          _scalar_diffusion.push_back( diffusion_task_name );
          _scalar_update.push_back( scalar_update_task_name );
          _rk_time_ave.push_back( rk_time_ave_task_name );
        }
      }

    }
  }

  if ( db->findBlock("KMomentum") ){

    using namespace ArchesCore;

    ProblemSpecP db_mom = db->findBlock("KMomentum");

    m_u_vel_name = parse_ups_for_role( UVELOCITY_ROLE, db, ArchesCore::default_uVel_name );
    m_v_vel_name = parse_ups_for_role( VVELOCITY_ROLE, db, ArchesCore::default_vVel_name );
    m_w_vel_name = parse_ups_for_role( WVELOCITY_ROLE, db, ArchesCore::default_wVel_name );
    m_density_name = parse_ups_for_role( DENSITY_ROLE, db, "density" );

    if ( !db_mom->findBlock("inviscid")) {
      std::string stress_name = "[StressTensor]";
      TaskInterface::TaskBuilder* stress_tsk = scinew StressTensor::Builder( stress_name, 0 );
      register_task(stress_name, stress_tsk, db_mom);
      _momentum_builders.push_back( stress_name );
      _momentum_stress_tensor.push_back( stress_name );
    }

    // X-mom
    std::string update_task_name = "[KFEUpdate]x-mom-update";
    std::string mom_task_name = "[KMomentum]"+ArchesCore::default_uMom_name;
    TaskInterface::TaskBuilder* x_tsk = scinew KMomentum<SFCXVariable<double> >::Builder(mom_task_name, 0);
    register_task( mom_task_name, x_tsk, db_mom );
    KFEUpdate<SFCXVariable<double> >::Builder* x_update_tsk =
      scinew KFEUpdate<SFCXVariable<double> >::Builder( update_task_name, 0, true );
    register_task( update_task_name, x_update_tsk, db_mom );
    // compute u from rhou
    std::string unweight_xmom_name = ArchesCore::default_uMom_name;
    TaskInterface::TaskBuilder* unw_x_tsk =
      scinew UnweightVariable<SFCXVariable<double>>::Builder( unweight_xmom_name, m_u_vel_name,
                                                            ArchesCore::MOMENTUM, 0 );
    register_task( unweight_xmom_name, unw_x_tsk, db_mom );

    _u_from_rho_u.push_back( unweight_xmom_name );
    _momentum_builders.push_back(mom_task_name);
    _momentum_update.push_back(update_task_name);
    _momentum_conv.push_back( mom_task_name );

    // Y-mom
    update_task_name = "[KFEUpdate]y-mom-update";
    mom_task_name = "[KMomentum]"+ArchesCore::default_vMom_name;
    std::string unweight_ymom_name = ArchesCore::default_vMom_name;
    TaskInterface::TaskBuilder* y_tsk = scinew KMomentum<SFCYVariable<double> >::Builder(mom_task_name, 0);
    register_task( mom_task_name, y_tsk, db_mom );
    TaskInterface::TaskBuilder* y_update_tsk =
      scinew KFEUpdate<SFCYVariable<double> >::Builder( update_task_name, 0, true );
    register_task( update_task_name, y_update_tsk, db_mom );
    // compute v from rhov
    TaskInterface::TaskBuilder* unw_y_tsk =
      scinew UnweightVariable<SFCYVariable<double>>::Builder( unweight_ymom_name, m_v_vel_name,
                                                            ArchesCore::MOMENTUM, 0 );
    register_task( unweight_ymom_name, unw_y_tsk, db_mom );

    _u_from_rho_u.push_back( unweight_ymom_name );
    _momentum_builders.push_back(mom_task_name);
    _momentum_update.push_back(update_task_name);
    _momentum_conv.push_back( mom_task_name );

    // Z-mom
    update_task_name = "[KFEUpdate]z-mom-update";
    mom_task_name = "[KMomentum]"+ArchesCore::default_wMom_name;
    std::string unweight_zmom_name = ArchesCore::default_wMom_name;
    TaskInterface::TaskBuilder* z_tsk = scinew KMomentum<SFCZVariable<double> >::Builder(mom_task_name, 0);
    register_task( mom_task_name, z_tsk, db_mom );
    TaskInterface::TaskBuilder* z_update_tsk =
      scinew KFEUpdate<SFCZVariable<double> >::Builder( update_task_name, 0, true );
    register_task( update_task_name, z_update_tsk, db_mom );
    // compute w from rhow
    TaskInterface::TaskBuilder* unw_z_tsk =
      scinew UnweightVariable<SFCZVariable<double>>::Builder( unweight_zmom_name, m_w_vel_name,
                                                            ArchesCore::MOMENTUM, 0 );
    register_task( unweight_zmom_name, unw_z_tsk, db_mom );

    _u_from_rho_u.push_back( unweight_zmom_name );
    _momentum_builders.push_back(mom_task_name);
    _momentum_update.push_back(update_task_name);
    _momentum_conv.push_back( mom_task_name );

    if ( db->findBlock("KMomentum")->findBlock("PressureSolver")){
      //Pressure eqn
      ProblemSpecP db_press = db ->findBlock("KMomentum")->findBlock("PressureSolver");
      std::string tsk_name = "[PressureEqn]";
      TaskInterface::TaskBuilder* press_tsk = scinew PressureEqn::Builder(tsk_name, 0, _materialManager);
      register_task( tsk_name, press_tsk, db_press);
      _pressure_eqn.push_back(tsk_name);

      //BC for velrhohat
      tsk_name = "[VelRhoHatBC]";
      AtomicTaskInterface::AtomicTaskBuilder* velrhohatbc_tsk = scinew VelRhoHatBC::Builder(tsk_name, 0);
      register_atomic_task( tsk_name, velrhohatbc_tsk, db);

      //pressure bcs
      tsk_name = "[PressureBC]";
      AtomicTaskInterface::AtomicTaskBuilder* press_bc_tsk = scinew ArchesCore::PressureBC::Builder(tsk_name, 0);
      register_atomic_task( tsk_name, press_bc_tsk, db );

      //pressure Gradient
      tsk_name = "[AddPressGradient]";
      AtomicTaskInterface::AtomicTaskBuilder* gradP_tsk = scinew AddPressGradient::Builder(tsk_name, 0);
      register_atomic_task( tsk_name, gradP_tsk, db);
    }

  }

  if ( db->findBlock("DQMOM") ){

    if ( db->findBlock("DQMOM")->findBlock("kokkos_translate") ) {
      register_DQMOM(db);
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

void TransportFactory::register_DQMOM( ProblemSpecP db ){

  ProblemSpecP db_transport;
  ProblemSpecP db_dqmom = db->findBlock("DQMOM");

  bool add_to_ups = false; //only add elements to the ups file if this is a fresh startup
  if ( ! db_dqmom->findBlock("DQMOM_built")){
    db_dqmom->appendChild("DQMOM_built");
    add_to_ups = true;
  }

  if ( db->findBlock("KScalarTransport") ){
    db_transport = db->findBlock("KScalarTransport");
  } else {
    db_transport = db->appendChild("KScalarTransport");
  }

  unsigned int nQn = ArchesCore::get_num_env( db_dqmom, ArchesCore::DQMOM_METHOD );
  std::vector<std::string> group_dqmom_names;

  if ( add_to_ups ){

    // This section inserts the transport and source information into the
    // UPS file based on the DQMOM input and creates the builders for the
    // new transport elements.
    for ( int i = 0; i < int(nQn); i++ ){

      std::stringstream dqmom_eqn_grp_env;
      std::stringstream dqmom_eqn_grp_env_noclass;
      dqmom_eqn_grp_env << "[KScalarRHS]" << m_dqmom_grp_name << "_" << i;
      dqmom_eqn_grp_env_noclass << m_dqmom_grp_name << "_" << i;

      std::string grp_name_noclass = dqmom_eqn_grp_env_noclass.str();

      //Create weights
      ProblemSpecP db_eqn_group = db_transport->appendChild("eqn_group");
      db_eqn_group->setAttribute("label", grp_name_noclass);
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

      std::stringstream this_qn;
      this_qn << i;

      ProblemSpecP db_weight = db_dqmom->findBlock("Weights");

      if ( !db_weight ){
        throw ProblemSetupException("Error: No <Weights> spec found in <DQMOM>.",__FILE__,__LINE__);
      }

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
        conv_vel->setAttribute("xlabel", "face_pvel_x_" + this_qn.str());
        conv_vel->setAttribute("ylabel", "face_pvel_y_" + this_qn.str());
        conv_vel->setAttribute("zlabel", "face_pvel_z_" + this_qn.str());

      }

      ProblemSpecP denominator = db_eqn_group->appendChild("weight_factor");
      denominator->setAttribute("label", "w_qn" + this_qn.str());

      ProblemSpecP env_num = db_eqn_group->appendChild("env_number");
      env_num->setAttribute("number", this_qn.str());

      // Here creating the weight eqn
      ProblemSpecP eqn_db = db_eqn_group->appendChild("eqn");
      eqn_db->setAttribute("label", "w_qn"+this_qn.str());
      eqn_db->appendChild("no_weight_factor");

      if ( do_convection ){
        ProblemSpecP conv_db = eqn_db->appendChild("convection");
        conv_db->setAttribute("scheme", conv_scheme );
      }
      if ( do_diffusion ){
        ProblemSpecP diff_db = eqn_db->appendChild("diffusion");
        diff_db->setAttribute("scheme", diff_scheme );
      }

      // The sources are summed and added automatically in DQMOMNoInversion
      // ProblemSpecP src_db = eqn_db->appendChild("src");
      // src_db->setAttribute("label", "w_qn"+this_qn.str()+"_src");

      ProblemSpecP db_init = db_weight->findBlock("initialization");

      if ( db_init != nullptr ){

        std::string type;
        db_init->getAttribute("type", type);
        if ( type != "env_constant" ){
          throw ProblemSetupException("Error: Only env_constant is allowed for DQMOM. Use <ARCHES><Initialize> instead.", __FILE__, __LINE__);
        }

        ProblemSpecP db_scal = db_weight->findBlock("scaling_const");
        std::vector<double> scaling_constants_dbl;
        std::vector<std::string> scaling_constants_str;

        if ( db_scal ){

          db_weight->require("scaling_const", scaling_constants_str);
          db_weight->require("scaling_const", scaling_constants_dbl);

          if ( scaling_constants_str.size() != nQn ){
            throw ProblemSetupException("Error: number of scaling constants != number quadrature nodes.", __FILE__, __LINE__);
          }

          eqn_db->appendChild("scaling")->setAttribute("value", scaling_constants_str[i]);

        } else {

          //assume scaling constants of 1.
          for ( unsigned int ii = 0; ii < nQn; ii++ ){
            scaling_constants_dbl.push_back(1.0);
            scaling_constants_str.push_back("1.0");
          }

          eqn_db->appendChild("scaling")->setAttribute("value", scaling_constants_str[i]);

        }

        if ( db_init->findBlock("env_constant") ){
          for ( ProblemSpecP db_env = db_init->findBlock("env_constant"); db_env != nullptr;
                  db_env = db_env->findNextBlock("env_constant") ){

            int this_qn;
            db_env->getAttribute("qn", this_qn );

            if ( this_qn == i ){
              //std::string value;
              double value;
              //ups file: intial value for unscaled variable
              db_env->getAttribute("value", value );
              ProblemSpecP db_new_init = eqn_db->appendChild("initialize");
              value /= scaling_constants_dbl[i];
              std::stringstream value_s;
              value_s << value;
              db_new_init->setAttribute("value", value_s.str());
            }
          }
        }

      } else {

        std::string type = "env_constant";

        ProblemSpecP db_scal = db_weight->findBlock("scaling_const");
        std::vector<double> scaling_constants_dbl;
        std::vector<std::string> scaling_constants_str;

        if ( db_scal ){

          db_weight->require("scaling_const", scaling_constants_str);
          db_weight->require("scaling_const", scaling_constants_dbl);

          if ( scaling_constants_str.size() != nQn ){
            throw ProblemSetupException("Error: number of scaling constants != number quadrature nodes.", __FILE__, __LINE__);
          }

          eqn_db->appendChild("scaling")->setAttribute("value", scaling_constants_str[i]);

        } else {

          //assume scaling constants of 1.
          for ( unsigned int ii = 0; ii < nQn; ii++ ){
            scaling_constants_str.push_back("1.0");
            scaling_constants_dbl.push_back(1.0);
          }

          eqn_db->appendChild("scaling")->setAttribute("value", scaling_constants_str[i]);

        }

        for ( unsigned int ii = 0; ii < nQn; ii++ ){

          unsigned int this_qn = i;
          if ( this_qn == ii ){
            double value = 0.0;
            ProblemSpecP db_new_init = eqn_db->appendChild("initialize");
            value /= scaling_constants_dbl[i];
            std::stringstream value_s;
            value_s << value;
            db_new_init->setAttribute("value", value_s.str());
          }
        }
      }

      // --+-- Now create the IC transport eqns:
      for ( ProblemSpecP db_ic = db_dqmom->findBlock("Ic"); db_ic != nullptr;
            db_ic =db_ic->findNextBlock("Ic") ){

        std::string ic_label;
        db_ic->getAttribute("label", ic_label);

        eqn_db = db_eqn_group->appendChild("eqn");

        eqn_db->setAttribute("label", ic_label+"_"+this_qn.str());

        //transport:
        if ( do_convection ){
          ProblemSpecP conv_db = eqn_db->appendChild("convection");
          conv_db->setAttribute("scheme", conv_scheme );
        }
        if ( do_diffusion ){
          ProblemSpecP diff_db = eqn_db->appendChild("diffusion");
          diff_db->setAttribute("scheme", diff_scheme );
        }

        // The sources are summed and added in DQMOMNoInversion
        // ProblemSpecP src_db = eqn_db->appendChild("src");
        // src_db->setAttribute("label", ic_label+"_qn"+this_qn.str()+"_src");

        ProblemSpecP db_scal = db_ic->findBlock("scaling_const");
        std::vector<double> scaling_constants_dbl;
        if ( db_scal ){
          std::vector<std::string> scaling_constants_str;
          db_ic->require("scaling_const", scaling_constants_str);
          db_ic->require("scaling_const", scaling_constants_dbl);

          if ( scaling_constants_str.size() != nQn ){
            throw ProblemSetupException("Error: number of scaling constants != number quadrature nodes.", __FILE__, __LINE__);
          }

          eqn_db->appendChild("scaling")->setAttribute("value", scaling_constants_str[i]);

        }

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
              //double value;
              db_env->getAttribute("value", value );
              ProblemSpecP db_new_init = eqn_db->appendChild("initialize");
              db_new_init->setAttribute("value", value);
            }
          }
        }
      }

      //Now create the builders for the DQMOM Transport
      //-----------------------------------------------
      TaskInterface::TaskBuilder* tsk;
      typedef CCVariable<double> C;
      typedef typename ArchesCore::VariableHelper<CCVariable<double> >::ConstType CT;

      if ( m_pack_transport_construction_tasks ){
        tsk = scinew KScalarRHS<C, C >::Builder(dqmom_eqn_grp_env.str(), db_eqn_group, 0);
      } else {
        tsk = scinew KScalarRHS<C, CT >::Builder(dqmom_eqn_grp_env.str(), db_eqn_group, 0);
      }
      register_task( dqmom_eqn_grp_env.str(), tsk, db_eqn_group );
      _dqmom_builders.push_back(dqmom_eqn_grp_env.str());

      std::stringstream update_task_name;
      update_task_name << "[KFEUpdate]" << m_dqmom_grp_name << "_" << i;

      KFEUpdate<CCVariable<double> >::Builder* update_tsk =
      scinew KFEUpdate<CCVariable<double> >::Builder( update_task_name.str(), 0, true );
      register_task( update_task_name.str(), update_tsk, db_eqn_group );
      _dqmom_fe_update.push_back( update_task_name.str() );

      std::stringstream unweight_task_name;
      unweight_task_name << "[UnweightVariable]" << m_dqmom_grp_name << "_" << i;

      TaskInterface::TaskBuilder* weight_var_tsk =
      scinew UnweightVariable<CCVariable<double>>::Builder( "NULL", unweight_task_name.str(),
                                                            ArchesCore::DQMOM, 0 );
      register_task( unweight_task_name.str(), weight_var_tsk, db_eqn_group );
      _ic_from_w_ic.push_back( unweight_task_name.str() );

      if ( db_dqmom->findBlock("diffusion") ) {
        std::stringstream diffusion_task_name;
        diffusion_task_name << "[Diffusion]" << m_dqmom_grp_name << "_" << i;
        TaskInterface::TaskBuilder* diff_tsk =
          scinew Diffusion<CCVariable<double> >::Builder( diffusion_task_name.str(), 0);
        register_task( diffusion_task_name.str(), diff_tsk, db_eqn_group );
        _dqmom_compute_diff.push_back( diffusion_task_name.str() );
      }
    }
  } else {

      //UPS fragments were added previously do just parse the exisiting INFORMATION
      for (ProblemSpecP db_eqn_group = db_transport->findBlock("eqn_group");
        db_eqn_group != nullptr; db_eqn_group = db_eqn_group->findNextBlock("eqn_group")){

          std::string group_name = "null";
          std::string type = "null";
          std::string grp_class = "density_weighted"; // ie, default is rho*phi

          db_eqn_group->getAttribute("label", group_name);
          db_eqn_group->getAttribute("type", type );
          db_eqn_group->getAttribute("class", grp_class );
          ArchesCore::EQUATION_CLASS enum_grp_class = ArchesCore::assign_eqn_class_enum( grp_class );

          if ( enum_grp_class == ArchesCore::DQMOM ){
            //Only create the tasks for DQMOM eqn_groups.
            TaskInterface::TaskBuilder* tsk;
            typedef CCVariable<double> C;
            typedef typename ArchesCore::VariableHelper<CCVariable<double> >::ConstType CT;

            int env;
            db_eqn_group->findBlock("env_number")->getAttribute("number",env);

            std::stringstream dqmom_eqn_grp_env;
            std::stringstream dqmom_eqn_grp_env_noclass;
            dqmom_eqn_grp_env << "[KScalarRHS]" << m_dqmom_grp_name << "_" << env;
            dqmom_eqn_grp_env_noclass << m_dqmom_grp_name << "_" << env;

            if ( m_pack_transport_construction_tasks ){
              tsk = scinew KScalarRHS<C, C >::Builder(dqmom_eqn_grp_env.str(), db_eqn_group, 0);
            } else {
              tsk = scinew KScalarRHS<C, CT >::Builder(dqmom_eqn_grp_env.str(), db_eqn_group, 0);
            }
            register_task( dqmom_eqn_grp_env.str(), tsk, db_eqn_group );
            _dqmom_builders.push_back(dqmom_eqn_grp_env.str());

            std::stringstream update_task_name;
            update_task_name << "[KFEUpdate]" << m_dqmom_grp_name << "_" << env;

            KFEUpdate<CCVariable<double> >::Builder* update_tsk =
            scinew KFEUpdate<CCVariable<double> >::Builder( update_task_name.str(), 0, true );
            register_task( update_task_name.str(), update_tsk, db_eqn_group );
            _dqmom_fe_update.push_back( update_task_name.str() );

            std::stringstream unweight_task_name;
            unweight_task_name << m_dqmom_grp_name << "_" << env;

            TaskInterface::TaskBuilder* weight_var_tsk =
            scinew UnweightVariable<CCVariable<double>>::Builder( "NULL", unweight_task_name.str(),
                                                                  ArchesCore::DQMOM, 0 );
            register_task( unweight_task_name.str(), weight_var_tsk, db_eqn_group );
            _ic_from_w_ic.push_back( unweight_task_name.str() );

            if ( db_dqmom->findBlock("diffusion") ) {
              std::stringstream diffusion_task_name;
              diffusion_task_name << "[Diffusion]" << m_dqmom_grp_name << "_" << env;
              TaskInterface::TaskBuilder* diff_tsk =
                scinew Diffusion<CCVariable<double> >::Builder( diffusion_task_name.str(), 0);
              register_task( diffusion_task_name.str(), diff_tsk, db_eqn_group );
              _dqmom_compute_diff.push_back( diffusion_task_name.str() );
            }
          }

      }

    }
  }
} //namespace Uintah
