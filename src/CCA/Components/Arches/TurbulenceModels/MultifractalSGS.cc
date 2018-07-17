#include <CCA/Components/Arches/TurbulenceModels/MultifractalSGS.h>
#include <CCA/Components/Arches/GridTools.h>

namespace Uintah{

  typedef ArchesFieldContainer AFC;

  //---------------------------------------------------------------------------------
  MultifractalSGS::MultifractalSGS( std::string task_name, int matl_index ) :
    TaskInterface( task_name, matl_index )
  {
    U_ctr_name = "uVelocitySPBC";
    V_ctr_name = "vVelocitySPBC";
    W_ctr_name = "wVelocitySPBC";

    //face_VelCell-3*3 component:
    Ux_face_name = "ucell_xvel";
    Uy_face_name = "ucell_yvel";
    Uz_face_name = "ucell_zvel";
    Vx_face_name = "vcell_xvel";
    Vy_face_name = "vcell_yvel";
    Vz_face_name = "vcell_zvel";
    Wx_face_name = "wcell_xvel";
    Wy_face_name = "wcell_yvel";
    Wz_face_name = "wcell_zvel";
    // Create UD scale velocities
    m_VelDelta_names.resize(15);
    m_VelDelta_names[0]  = "uD_ctr";
    m_VelDelta_names[1]  = "ucell_XvelD";
    m_VelDelta_names[2]  = "ucell_YvelD";
    m_VelDelta_names[3]  = "ucell_ZvelD";
    m_VelDelta_names[4]  = "vD_ctr";
    m_VelDelta_names[5]  = "vcell_XvelD";
    m_VelDelta_names[6]  = "vcell_YvelD";
    m_VelDelta_names[7]  = "vcell_ZvelD";
    m_VelDelta_names[8]  = "wD_ctr";
    m_VelDelta_names[9]  = "wcell_XvelD";
    m_VelDelta_names[10] = "wcell_YvelD";
    m_VelDelta_names[11] = "wcell_ZvelD";
    //create U2D velocity
    m_VelDelta_names[12] = "u2D_ctr";
    m_VelDelta_names[13] = "v2D_ctr";
    m_VelDelta_names[14] = "w2D_ctr";

    // Create SGS stress
    //m_StrainRateUD_names.resize(6);
    //m_StrainRateUD_names[0] = "uuStrainUD";
    //m_StrainRateUD_names[1] = "uvStrainUD";
    //m_StrainRateUD_names[2] = "vvStrainUD";
    //m_StrainRateUD_names[3] = "vwStrainUD";
    //m_StrainRateUD_names[4] = "wwStrainUD";
    //m_StrainRateUD_names[5] = "wuStrainUD";

    //m_StrainRateU2D_names.resize(6);
    //m_StrainRateU2D_names[0] = "uuStrainU2D";
    //m_StrainRateU2D_names[1] = "uvStrainU2D";
    //m_StrainRateU2D_names[2] = "vvStrainU2D";
    //m_StrainRateU2D_names[3] = "vwStrainU2D";
    //m_StrainRateU2D_names[4] = "wwStrainU2D";
    //m_StrainRateU2D_names[5] = "wuStrainU2D";

    //// Create new label
    //// Create SGS stress
    m_SgsStress_names.resize(9);
    m_SgsStress_names[0] = "ucell_xSgsStress";
    m_SgsStress_names[1] = "ucell_ySgsStress";
    m_SgsStress_names[2] = "ucell_zSgsStress";
    m_SgsStress_names[3] = "vcell_xSgsStress";
    m_SgsStress_names[4] = "vcell_ySgsStress";
    m_SgsStress_names[5] = "vcell_zSgsStress";
    m_SgsStress_names[6] = "wcell_xSgsStress";
    m_SgsStress_names[7] = "wcell_ySgsStress";
    m_SgsStress_names[8] = "wcell_zSgsStress";
  }

  //---------------------------------------------------------------------------------
  MultifractalSGS::~MultifractalSGS()
  {}

  //---------------------------------------------------------------------------------
  void
    MultifractalSGS::problemSetup( ProblemSpecP& db ){

      using namespace Uintah::ArchesCore;

      Nghost_cells = 1;

      m_u_vel_name = parse_ups_for_role( UVELOCITY, db, "uVelocitySPBC" );
      m_v_vel_name = parse_ups_for_role( VVELOCITY, db, "vVelocitySPBC" );
      m_w_vel_name = parse_ups_for_role( WVELOCITY, db, "wVelocitySPBC" );

      m_cc_u_vel_name = m_u_vel_name + "_cc";
      m_cc_v_vel_name = m_v_vel_name + "_cc";
      m_cc_w_vel_name = m_w_vel_name + "_cc";

      db->getWithDefault("Cs", m_Cs, .5);

      if (db->findBlock("use_my_name_viscosity")){
        db->findBlock("use_my_name_viscosity")->getAttribute("label",m_t_vis_name);
      } else{
        m_t_vis_name = parse_ups_for_role( TOTAL_VISCOSITY, db );
      }

      const ProblemSpecP params_root = db->getRootNode();
      if (params_root->findBlock("PhysicalConstants")) {
        params_root->findBlock("PhysicalConstants")->require("viscosity",
            m_molecular_visc);
        if( m_molecular_visc == 0 ) {
          std::stringstream msg;
          msg << "ERROR: Constant MultifractalSGS: problemSetup(): Zero viscosity specified \n"
            << "       in <PhysicalConstants> section of input file." << std::endl;
          throw InvalidValue(msg.str(),__FILE__,__LINE__);
        }
      } else {
        std::stringstream msg;
        msg << "ERROR: Constant MultifractalSGS: problemSetup(): Missing <PhysicalConstants> \n"
          << "       section in input file!" << std::endl;
        throw InvalidValue(msg.str(),__FILE__,__LINE__);
      }

    }

  //---------------------------------------------------------------------------------
  void
    MultifractalSGS::create_local_labels(){
      // create subgrid stress
      //U-CELL LABELS:
      register_new_variable<SFCXVariable<double> >("ucell_xSgsStress");
      register_new_variable<SFCXVariable<double> >("ucell_ySgsStress");
      register_new_variable<SFCXVariable<double> >("ucell_zSgsStress");
      //V-CELL LABELS:
      register_new_variable<SFCYVariable<double> >("vcell_xSgsStress");
      register_new_variable<SFCYVariable<double> >("vcell_ySgsStress");
      register_new_variable<SFCYVariable<double> >("vcell_zSgsStress");
      //W-CELL LABELS:
      register_new_variable<SFCZVariable<double> >("wcell_xSgsStress");
      register_new_variable<SFCZVariable<double> >("wcell_ySgsStress");
      register_new_variable<SFCZVariable<double> >("wcell_zSgsStress");

      //std::vector names{32};
      //names[0] = "var1";
      ////..
      //names[31] = "var32";

      //for ( auto i = names.begin(); i != names.end(); i++ ){
      //register_new_variable<CCVariable<double> >(*i);
      //}

      //for (auto iter = m_StrainRateUD_names.begin(); iter != m_StrainRateUD_names.end(); iter++ ){
      //register_new_variable<CCVariable<double> >(*iter);
      //}
      //for (auto iter = m_StrainRateUD_names.begin(); iter != m_StrainRateUD_names.end(); iter++ ){
      //register_new_variable<CCVariable<double> >(*iter);
      //}

    }

  //---------------------------------------------------------------------------------
  void
    MultifractalSGS::register_initialize( std::vector<AFC::VariableInformation>&
        variable_registry , const bool packed_tasks){

      for (auto iter = m_SgsStress_names.begin(); iter != m_SgsStress_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry, _task_name );
      }
      //  register_variable( m_t_vis_name, AFC::COMPUTES, variable_registry );

      // register Velocity Delta
    }

  //---------------------------------------------------------------------------------
  void
    MultifractalSGS::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
      // Subgrid stress
      SFCXVariable<double>& ucell_xSgsStress = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_xSgsStress"));
      SFCXVariable<double>& ucell_ySgsStress = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_ySgsStress"));
      SFCXVariable<double>& ucell_zSgsStress = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_zSgsStress"));
      ucell_xSgsStress.initialize(0.0);
      ucell_ySgsStress.initialize(0.0);
      ucell_zSgsStress.initialize(0.0);

      SFCYVariable<double>& vcell_xSgsStress = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_xSgsStress"));
      SFCYVariable<double>& vcell_ySgsStress = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_ySgsStress"));
      SFCYVariable<double>& vcell_zSgsStress = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_zSgsStress"));
      vcell_xSgsStress.initialize(0.0);
      vcell_ySgsStress.initialize(0.0);
      vcell_zSgsStress.initialize(0.0);

      SFCZVariable<double>& wcell_xSgsStress = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_xSgsStress"));
      SFCZVariable<double>& wcell_ySgsStress = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_ySgsStress"));
      SFCZVariable<double>& wcell_zSgsStress = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_zSgsStress"));
      wcell_xSgsStress.initialize(0.0);
      wcell_ySgsStress.initialize(0.0);
      wcell_zSgsStress.initialize(0.0);


    }

  //---------------------------------------------------------------------------------
  void
    MultifractalSGS::register_timestep_init( std::vector<AFC::VariableInformation>&
        variable_registry , const bool packed_tasks){

      for (auto iter = m_SgsStress_names.begin(); iter != m_SgsStress_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry, _task_name );
      }

      //for (auto iter = m_VelDelta_names.begin(); iter != m_VelDelta_names.end(); iter++ ){
      //register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry, _task_name );
      //}

      // register_variable( m_t_vis_name, AFC::COMPUTES, variable_registry );

    }

  //---------------------------------------------------------------------------------
  void
    MultifractalSGS::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
      // Subgrid stress
      SFCXVariable<double>& ucell_xSgsStress = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_xSgsStress"));
      SFCXVariable<double>& ucell_ySgsStress = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_ySgsStress"));
      SFCXVariable<double>& ucell_zSgsStress = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_zSgsStress"));

      SFCYVariable<double>& vcell_xSgsStress = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_xSgsStress"));
      SFCYVariable<double>& vcell_ySgsStress = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_ySgsStress"));
      SFCYVariable<double>& vcell_zSgsStress = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_zSgsStress"));

      SFCZVariable<double>& wcell_xSgsStress = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_xSgsStress"));
      SFCZVariable<double>& wcell_ySgsStress = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_ySgsStress"));
      SFCZVariable<double>& wcell_zSgsStress = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_zSgsStress"));

      //CCVariable<double>& mu_sgc = *(tsk_info->get_uintah_field<CCVariable<double> >(m_t_vis_name));

    }

  //---------------------------------------------------------------------------------
  void
    MultifractalSGS::register_timestep_eval( std::vector<AFC::VariableInformation>&
        variable_registry, const int time_substep , const bool packed_tasks){
      register_variable( Ux_face_name,    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( Uy_face_name,    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( Uz_face_name,    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( Vx_face_name,    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( Vy_face_name,    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( Vz_face_name,    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( Wx_face_name,    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( Wy_face_name,    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( Wz_face_name,    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( m_u_vel_name,    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( m_v_vel_name,    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( m_w_vel_name,    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);


      register_variable( "uustress",    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( "uvstress",    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( "uwstress",    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( "vustress",    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( "vvstress",    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( "vwstress",    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( "wustress",    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( "wvstress",    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( "wwstress",    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep);
      register_variable( "density",     ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::OLDDW, variable_registry,time_substep );
      //register_variable( "volFraction",   ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::OLDDW, variable_registry,time_substep );
      // UPDATE USER DEFINE VARIABLES
      // register Velocity Delta
      for (auto iter = m_VelDelta_names.begin(); iter != m_VelDelta_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry, _task_name );

      }

      for (auto iter = m_SgsStress_names.begin(); iter != m_SgsStress_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::MODIFIES, variable_registry, _task_name );
      }
      //register_variable( m_u_vel_name, AFC::REQUIRES, Nghost_cells, AFC::LATEST, variable_registry, time_substep);
      //register_variable( m_v_vel_name, AFC::REQUIRES, Nghost_cells, AFC::LATEST, variable_registry, time_substep);
      //register_variable( m_w_vel_name, AFC::REQUIRES, Nghost_cells, AFC::LATEST, variable_registry, time_substep);

      //register_variable( m_cc_u_vel_name, AFC::REQUIRES, Nghost_cells, AFC::LATEST, variable_registry, time_substep);
      //register_variable( m_cc_v_vel_name, AFC::REQUIRES, Nghost_cells, AFC::LATEST, variable_registry, time_substep);
      //register_variable( m_cc_w_vel_name, AFC::REQUIRES, Nghost_cells, AFC::LATEST, variable_registry, time_substep);

      //register_variable( m_t_vis_name, AFC::MODIFIES ,  variable_registry, time_substep );

    }

  //---------------------------------------------------------------------------------
  void
    MultifractalSGS::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
      Vector Dx=patch->dCell();
      double dx=Dx.x(); double dy=Dx.y(); double dz=Dx.z();

      double LegthScales=pow(Dx.x()*Dx.y()*Dx.z(),1.0/3.0);
      double filter = pow(Dx.x()*Dx.y()*Dx.z(),1.0/3.0);
      double ratio_threshold= 3.0;
      double limDevotic=.3; //1.e-5; //0.65;

      constSFCXVariable<double>& ucell_xvel_face = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >(Ux_face_name));
      constSFCXVariable<double>& ucell_yvel_face = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >(Uy_face_name));
      constSFCXVariable<double>& ucell_zvel_face = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >(Uz_face_name));
      constSFCYVariable<double>& vcell_xvel_face = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >(Vx_face_name));
      constSFCYVariable<double>& vcell_yvel_face = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >(Vy_face_name));
      constSFCYVariable<double>& vcell_zvel_face = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >(Vz_face_name));
      constSFCZVariable<double>& wcell_xvel_face = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >(Wx_face_name));
      constSFCZVariable<double>& wcell_yvel_face = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >(Wy_face_name));
      constSFCZVariable<double>& wcell_zvel_face = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >(Wz_face_name));
      constSFCXVariable<double>& U_ctr =  *(tsk_info->get_const_uintah_field<constSFCXVariable<double> > (m_u_vel_name));
      constSFCYVariable<double>& V_ctr =  *(tsk_info->get_const_uintah_field<constSFCYVariable<double> > (m_v_vel_name));
      constSFCZVariable<double>& W_ctr =  *(tsk_info->get_const_uintah_field<constSFCZVariable<double> > (m_w_vel_name));
      //
      constSFCXVariable<double>& uustress = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("uustress"));
      constSFCXVariable<double>& uvstress = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("uvstress"));
      constSFCXVariable<double>& uwstress = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("uwstress"));
      //
      constSFCYVariable<double>& vustress = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("vustress"));
      constSFCYVariable<double>& vvstress = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("vvstress"));
      constSFCYVariable<double>& vwstress = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("vwstress"));
      //
      constSFCZVariable<double>& wustress = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("wustress"));
      constSFCZVariable<double>& wvstress = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("wvstress"));
      constSFCZVariable<double>& wwstress = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("wwstress"));

      // UD
      constSFCXVariable<double>& uD_ctr             = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("uD_ctr"));
      constSFCXVariable<double>& ucell_XvelD        = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("ucell_XvelD"));
      constSFCXVariable<double>& ucell_YvelD        = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("ucell_YvelD"));
      constSFCXVariable<double>& ucell_ZvelD        = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("ucell_ZvelD"));

      constSFCYVariable<double>& vD_ctr             = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("vD_ctr"));
      constSFCYVariable<double>& vcell_XvelD        = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("vcell_XvelD"));
      constSFCYVariable<double>& vcell_YvelD        = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("vcell_YvelD"));
      constSFCYVariable<double>& vcell_ZvelD        = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("vcell_ZvelD"));

      constSFCZVariable<double>& wD_ctr             = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("wD_ctr"));
      constSFCZVariable<double>& wcell_XvelD        = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("wcell_XvelD"));
      constSFCZVariable<double>& wcell_YvelD        = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("wcell_YvelD"));
      constSFCZVariable<double>& wcell_ZvelD        = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("wcell_ZvelD"));
      //U2D
      constSFCXVariable<double>& u2D_ctr            = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("u2D_ctr"));
      constSFCYVariable<double>& v2D_ctr            = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("v2D_ctr"));
      constSFCZVariable<double>& w2D_ctr            = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("w2D_ctr"));

      SFCXVariable<double>& ucell_xSgsStress = tsk_info->get_uintah_field_add<SFCXVariable<double> >("ucell_xSgsStress");
      SFCXVariable<double>& ucell_ySgsStress = tsk_info->get_uintah_field_add<SFCXVariable<double> >("ucell_ySgsStress");
      SFCXVariable<double>& ucell_zSgsStress = tsk_info->get_uintah_field_add<SFCXVariable<double> >("ucell_zSgsStress");

      SFCYVariable<double>& vcell_xSgsStress = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_xSgsStress"));
      SFCYVariable<double>& vcell_ySgsStress = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_ySgsStress"));
      SFCYVariable<double>& vcell_zSgsStress = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_zSgsStress"));

      SFCZVariable<double>& wcell_xSgsStress = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_xSgsStress"));
      SFCZVariable<double>& wcell_ySgsStress = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_ySgsStress"));
      SFCZVariable<double>& wcell_zSgsStress = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_zSgsStress"));

      constCCVariable<double>& density = tsk_info->get_const_uintah_field_add<constCCVariable<double> >("density");

      //constSFCXVariable<double>& uVel = tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >(m_u_vel_name);
      //std::vector<CCVariable<double>* > var_ref;
      //int ii = 0;
      //for ( auto i = m_stressNames.begin(); i != m_stressNames.end(); i++){
      //var_ref[ii] = task_info->get_uintah_field_add<CCVariable<double> >(*i);
      //}

      Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );

      Uintah::parallel_for( range, [&](int i, int j, int k){

          std::vector<double> UD(12,0.0);
          std::vector<double> sums(12,0.0);
          std::vector<double> rhoUD(12,0.0);
          std::vector<double> StrainUR(6,0.0);
          std::vector<double> StrainUD(6,0.0);
          std::vector<double> StrainU2D(6,0.0);
          std::vector<double> strain_ratio(6,0.0);
          std::vector<double> Sum(36,0.0);
          std::vector<double> tau(36,0.0);

          double mu = m_molecular_visc / density(i,j,k);

          //   calculating the Strain tensor
          // FILTERED strain components on the faces of the u-cell
          Strain_calc( U_ctr,    V_ctr,   W_ctr,  i, j, k,  StrainUR,dx,dy,dz  );
          // DELTA strain components on the faces of the u-cell
          Strain_calc( uD_ctr,  vD_ctr,  wD_ctr,  i, j, k,  StrainUD,dx,dy,dz  );
          // 2DELTA strain components on the faces of the u-cell
          Strain_calc( u2D_ctr, v2D_ctr, w2D_ctr, i, j, k,  StrainU2D,dx,dy,dz  );

          //      end for calculting ud and u2d
          double strainR_magn=       sqrt(StrainUR[1-1]*StrainUR[1-1]+StrainUR[2-1]*StrainUR[2-1]+StrainUR[3-1]*StrainUR[3-1]
              +2.0*(StrainUR[4-1]*StrainUR[4-1]+StrainUR[5-1]*StrainUR[5-1]+StrainUR[6-1]*StrainUR[6-1]));
          //
          // double strainUD_magn=       sqrt(StrainUD[1-1]*StrainUD[1-1]+StrainUD[2-1]*StrainUD[2-1]+StrainUD[3-1]*StrainUD[3-1]
          //                           +2.0*(StrainUD[4-1]*StrainUD[4-1]+StrainUD[5-1]*StrainUD[5-1]+StrainUD[6-1]*StrainUD[6-1]));
          //
          // double strainU2D_magn=     sqrt(StrainU2D[1-1]*StrainU2D[1-1]+StrainU2D[2-1]*StrainU2D[2-1]+StrainU2D[3-1]*StrainU2D[3-1]
          //                           +2.0*(StrainU2D[4-1]*StrainU2D[4-1]+StrainU2D[5-1]*StrainU2D[5-1]+StrainU2D[6-1]*StrainU2D[6-1]));

          // calcuting strain ratio
          for (unsigned int iter=0 ;iter < StrainUD.size(); iter++)
          {
            //strain_ratio[iter] = StrainUD[iter]==0 ? ratio_threshold : std::abs(StrainU2D[iter]/StrainUD[iter]);
            strain_ratio[iter] = std::abs(StrainU2D[iter]/(StrainUD[iter]+1e-10));
          }
          // calcuting C_sgs_G and U_sgs/rhoUsgs
          double sgs_scales=0; double factorN=0; double value=0; double Re_g=0.0;

          // JEREMY: These velocities aren't at the same location. Does this matter???
          double velmagn=std::sqrt((uD_ctr(i,j,k)*uD_ctr(i,j,k) + vD_ctr(i,j,k)*vD_ctr(i,j,k)+wD_ctr(i,j,k)*wD_ctr(i,j,k)  ));

          // paper 2004(ii), eq.15 to calculate the coefficient, which is in the header file
          double C_sgs_G= sgsVelCoeff(mu,LegthScales,strainR_magn,dx, dy, dz, sgs_scales,factorN, value,Re_g);
          //double C_sgs_G= sgsVelCoeff(mu,LegthScales,velmagn,dx, dy, dz, sgs_scales,factorN, value,Re_g);
          //if (i==2&&j==2&&k==2)
          //{ std::cout<<"\n StrainUD"<<strainUD_magn<<"Re_g="<<Re_g<<"C_sgs_G="<< C_sgs_G<<"mu="<<mu<<"LegthScale="<<LegthScales <<"dx="<<dx<<"factorN="<<factorN<<"sgs_scales="<<sgs_scales<<"value="<<value<<"\n";
          //}

          // at u-cell xface
          // 1:i=1 j=1
          //velmagn= std::abs(ucell_XvelD(i,j,k) );
          //velmagn=std::sqrt(uD_ctr(i,j,k)*uD_ctr(i,j,k) + 0.25*(vD_ctr(i-1,j,k)+vD_ctr(i-1,j+1,k))*(vD_ctr(i-1,j,k)+vD_ctr(i-1,j+1,k)) + 0.25*(wD_ctr(i-1,j,k)+wD_ctr(i-1,j,k+1) )*(wD_ctr(i-1,j,k)+ wD_ctr(i-1,j,k+1) ) );
          //C_sgs_G= sgsVelCoeff(mu,LegthScales,velmagn,dx, dy, dz, sgs_scales,factorN, value,Re_g);
          // paper 2004(II)-eq.14
          tau[0]= ucell_xvel_face(i,j,k)*ucell_xvel_face(i,j,k);
          tau[1]= ucell_xvel_face(i,j,k)*ucell_XvelD(i,j,k)*C_sgs_G;
          tau[2]= ucell_XvelD(i,j,k)*C_sgs_G*ucell_xvel_face(i,j,k);
          tau[3]= ucell_XvelD(i,j,k)*C_sgs_G*ucell_XvelD(i,j,k)*C_sgs_G;

          // at v-cell yface
          // 2:i=2 j=2
          //velmagn=std::sqrt(0.25*(uD_ctr(i,j-1,k)+uD_ctr(i+1,j-1,k))*(uD_ctr(i,j-1,k)+uD_ctr(i+1,j-1,k)) + vD_ctr(i,j,k)*vD_ctr(i,j,k)+ 0.25*(wD_ctr(i,j-1,k)+wD_ctr(i,j-1,k+1) )*(wD_ctr(i,j-1,k)+ wD_ctr(i,j-1,k+1) ) );
          //velmagn= std::abs(vcell_YvelD(i,j,k));
          //C_sgs_G= sgsVelCoeff(mu,LegthScales,velmagn,dx, dy, dz, sgs_scales,factorN, value,Re_g);
          tau[4]= vcell_yvel_face(i,j,k)*vcell_yvel_face(i,j,k);
          tau[5]= vcell_yvel_face(i,j,k)*vcell_YvelD(i,j,k)*C_sgs_G;
          tau[6]= vcell_YvelD(i,j,k)*C_sgs_G*vcell_yvel_face(i,j,k);
          tau[7]= vcell_YvelD(i,j,k)*C_sgs_G*vcell_YvelD(i,j,k)*C_sgs_G;

          // at w-cell wface
          // 3:i=3 j=3
          //velmagn=std::sqrt(0.25*(uD_ctr(i,j,k-1)+uD_ctr(i+1,j,k-1))*(uD_ctr(i,j,k-1)+uD_ctr(i+1,j,k-1)) + 0.25*(vD_ctr(i,j,k-1)+vD_ctr(i,j+1,k-1))*(vD_ctr(i,j,k-1)+vD_ctr(i,j+1,k-1))+ wD_ctr(i,j,k)*wD_ctr(i,j,k)  );
          //velmagn= std::abs(wcell_ZvelD(i,j,k));
          //C_sgs_G= sgsVelCoeff(mu,LegthScales,velmagn,dx, dy, dz, sgs_scales,factorN, value,Re_g);
          tau[8]=  wcell_zvel_face(i,j,k)*wcell_zvel_face(i,j,k);
          tau[9]=  wcell_zvel_face(i,j,k)*wcell_ZvelD(i,j,k)*C_sgs_G;
          tau[10]= wcell_ZvelD(i,j,k)*C_sgs_G*wcell_zvel_face(i,j,k);
          tau[11]= wcell_ZvelD(i,j,k)*C_sgs_G*wcell_ZvelD(i,j,k)*C_sgs_G;


          // 4:rhoU*V at uv-node: TauUV
          // i=1 j=2
          //velmagn=std::sqrt( vcell_XvelD(i,j,k)*vcell_XvelD(i,j,k) +ucell_YvelD(i,j,k)*ucell_YvelD(i,j,k)+0.25*0.25*(ucell_ZvelD(i,j,k)+ ucell_ZvelD(i,j-1,k)  + ucell_ZvelD(i,j,k+1) + ucell_ZvelD(i,j-1,k+1) )*(ucell_ZvelD(i,j,k)+ ucell_ZvelD(i,j-1,k)+ ucell_ZvelD(i,j,k+1) + ucell_ZvelD(i,j-1,k+1) ));
          //velmagn= std::abs(vcell_XvelD(i,j,k)+ucell_YvelD(i,j,k))*0.5;
          //C_sgs_G= sgsVelCoeff(mu,LegthScales,velmagn,dx, dy, dz, sgs_scales,factorN, value,Re_g);
          tau[12]= vcell_xvel_face(i,j,k)*ucell_yvel_face(i,j,k);
          tau[13]= vcell_xvel_face(i,j,k)*ucell_YvelD(i,j,k)*C_sgs_G;
          tau[14]= vcell_XvelD(i,j,k)*ucell_yvel_face(i,j,k)*C_sgs_G;
          tau[15]= vcell_XvelD(i,j,k)*C_sgs_G*ucell_YvelD(i,j,k)*C_sgs_G;

          // 5:rhoU*W at vw-node: TauUW
          // i=1 j=3
          //velmagn=std::sqrt( wcell_XvelD(i,j,k)*wcell_XvelD(i,j,k) + 0.25*0.25*(ucell_YvelD(i,j,k)+ ucell_YvelD(i,j+1,k)  + ucell_YvelD(i,j,k-1) + ucell_YvelD(i,j+1,k-1) )*(ucell_YvelD(i,j,k)+ ucell_YvelD(i,j+1,k)  + ucell_YvelD(i,j,k-1) + ucell_YvelD(i,j+1,k-1) )+ ucell_ZvelD(i,j,k)*ucell_ZvelD(i,j,k) );
          //velmagn= std::abs( ucell_ZvelD(i,j,k) + wcell_XvelD(i,j,k) )*0.5;
          //C_sgs_G= sgsVelCoeff(mu,LegthScales,velmagn,dx, dy, dz, sgs_scales,factorN, value,Re_g);
          tau[16]= wcell_xvel_face(i,j,k)*ucell_zvel_face(i,j,k);
          tau[17]= wcell_xvel_face(i,j,k)*ucell_ZvelD(i,j,k)*C_sgs_G;
          tau[18]= wcell_XvelD(i,j,k)*C_sgs_G*ucell_zvel_face(i,j,k);
          tau[19]= wcell_XvelD(i,j,k)*C_sgs_G*ucell_ZvelD(i,j,k)*C_sgs_G;


          // 6:rhoV*U at vu-node: TauVU
          // i=2 j=1
          //velmagn= std::abs( ucell_YvelD(i,j,k) + vcell_XvelD(i,j,k) )*0.5;
          //velmagn=std::sqrt( vcell_XvelD(i,j,k)*vcell_XvelD(i,j,k) +ucell_YvelD(i,j,k)*ucell_YvelD(i,j,k)+0.25*0.25*(vcell_ZvelD(i,j,k)+ vcell_ZvelD(i-1,j,k)  + vcell_ZvelD(i-1,j,k+1) + vcell_ZvelD(i,j,k+1) )*(vcell_ZvelD(i,j,k)+ vcell_ZvelD(i-1,j,k)  + vcell_ZvelD(i-1,j,k+1) + vcell_ZvelD(i,j,k+1) ));
          //C_sgs_G= sgsVelCoeff(mu,LegthScales,velmagn,dx, dy, dz, sgs_scales,factorN, value,Re_g);
          tau[20]= ucell_yvel_face(i,j,k)*vcell_xvel_face(i,j,k);
          tau[21]= ucell_yvel_face(i,j,k)*vcell_XvelD(i,j,k)*C_sgs_G;
          tau[22]= ucell_YvelD(i,j,k)*vcell_xvel_face(i,j,k)*C_sgs_G;
          tau[23]= ucell_YvelD(i,j,k)*C_sgs_G*vcell_XvelD(i,j,k)*C_sgs_G;


          // 7:rhoV*W at vw-node: TauVW
          // i=2 j=3
          //velmagn= std::abs( wcell_YvelD(i,j,k) + vcell_ZvelD(i,j,k) )*0.5;
          //velmagn=std::sqrt( 0.25*0.25*(vcell_XvelD(i,j,k)+ vcell_XvelD(i,j,k-1) +vcell_XvelD(i+1,j,k)+ vcell_XvelD(i+1,j,k-1) )*(vcell_XvelD(i,j,k)+ vcell_XvelD(i,j,k-1) +vcell_XvelD(i+1,j,k)+ vcell_XvelD(i+1,j,k-1) )+wcell_YvelD(i,j,k)*wcell_YvelD(i,j,k) +vcell_ZvelD(i,j,k)*vcell_ZvelD(i,j,k));
          //C_sgs_G= sgsVelCoeff(mu,LegthScales,velmagn,dx, dy, dz, sgs_scales,factorN, value,Re_g);
          tau[24]= wcell_yvel_face(i,j,k)*vcell_zvel_face(i,j,k);
          tau[25]= wcell_yvel_face(i,j,k)*vcell_ZvelD(i,j,k)*C_sgs_G;
          tau[26]= wcell_YvelD(i,j,k)*C_sgs_G*vcell_zvel_face(i,j,k);
          tau[27]= wcell_YvelD(i,j,k)*C_sgs_G*vcell_ZvelD(i,j,k)*C_sgs_G;

          // 8:rhoW*U at vw-node: TauWU
          // i=1 j=3
          //velmagn=std::sqrt( wcell_XvelD(i,j,k)*wcell_XvelD(i,j,k) + 0.25*0.25*(wcell_YvelD(i,j,k)+ wcell_YvelD(i,j+1,k)+ wcell_YvelD(i-1,j,k) + wcell_YvelD(i-1,j+1,k) )*(wcell_YvelD(i,j,k)+ wcell_YvelD(i,j+1,k)+ wcell_YvelD(i-1,j,k) + wcell_YvelD(i-1,j+1,k) )+ ucell_ZvelD(i,j,k)*ucell_ZvelD(i,j,k) );
          //velmagn= std::abs( ucell_ZvelD(i,j,k) + wcell_XvelD(i,j,k) )*0.5;
          //C_sgs_G= sgsVelCoeff(mu,LegthScales,velmagn,dx, dy, dz, sgs_scales,factorN, value,Re_g);
          tau[28]= ucell_zvel_face(i,j,k)*wcell_xvel_face(i,j,k);
          tau[29]= ucell_zvel_face(i,j,k)*wcell_XvelD(i,j,k)*C_sgs_G;
          tau[30]= ucell_ZvelD(i,j,k)*C_sgs_G*wcell_xvel_face(i,j,k);
          tau[31]= ucell_ZvelD(i,j,k)*C_sgs_G*wcell_XvelD(i,j,k)*C_sgs_G;

          // 9:rhoW*V at vw-node: TauWV
          // i=3 j=2
          //velmagn= std::abs( vcell_ZvelD(i,j,k) + wcell_YvelD(i,j,k) )*0.5;
          //velmagn=std::sqrt( 0.25*0.25*(wcell_XvelD(i,j,k)+ wcell_XvelD(i+1,j,k) +wcell_XvelD(i,j-1,k)+ wcell_XvelD(i+1,j-1,k) )*(wcell_XvelD(i,j,k)+ wcell_XvelD(i+1,j,k) +wcell_XvelD(i,j-1,k)+ wcell_XvelD(i+1,j-1,k) )+wcell_YvelD(i,j,k)*wcell_YvelD(i,j,k) +vcell_ZvelD(i,j,k)*vcell_ZvelD(i,j,k));
          //C_sgs_G= sgsVelCoeff(mu,LegthScales,velmagn,dx, dy, dz, sgs_scales,factorN, value,Re_g);
          tau[32]= vcell_zvel_face(i,j,k)*wcell_yvel_face(i,j,k);
          tau[33]= vcell_zvel_face(i,j,k)*wcell_YvelD(i,j,k)*C_sgs_G;
          tau[34]= vcell_ZvelD(i,j,k)*C_sgs_G*wcell_yvel_face(i,j,k);
          tau[35]= vcell_ZvelD(i,j,k)*C_sgs_G*wcell_YvelD(i,j,k)*C_sgs_G;
          // final Stress with limitors
          // remove trace term by term
          //if (i==3&&j==2&&k==2)
          //{ std::cout<<"\n uuFaceX="<< tau[0] <<" uvFaceX="<< tau[12]<<" uwFaceX="<< tau[16]<<"\n";
          //std::cout<<"\n vuFaceY="<< tau[20] <<"vvFaceY="<< tau[4]<<"  vwFaceY="<< tau[24]<<"\n";
          //std::cout<<"\n wuFaceZ="<< tau[28] <<"wvFaceZ="<< tau[32]<<" wwFaceZ="<< tau[8]<<"\n";
          //}

          double trace =0;
          for (unsigned int iter=0 ;iter < tau.size(); iter++)
          { Sum[iter]=tau[iter];   }

          //if(i==2&&j==2&&k==2)
          //{std::cout<<" Stress="<<" i= "<<i<<" j= "<<j<<" k="<<k<<"\n";
          //std::cout<<"\n before limiter Sum(2,2,2) output="<<std::scientific<<Sum[0]<<","<<Sum[1]<<","<<Sum[2]<<","<<Sum[3]<<","<<Sum[4]<<","<<Sum[5]<<","<<Sum[6]<<","<<Sum[7]<<","<<Sum[8]<<","<<Sum[9]<<","<<Sum[10]<<","<<Sum[11]<<"\n";
          //std::cout<<" Sum12="<<Sum[12]<<","<<Sum[13]<<","<<Sum[14]<<","<<Sum[15]<<","<<Sum[16]<<","<<Sum[17]<<","<<Sum[18]<<","<<Sum[19]<<","<<Sum[20]<<","<<Sum[21]<<","<<Sum[22]<<","<<Sum[23]<<","<<"\n";
          //}

          //if(i==3&&j==2&&k==2)
          //{std::cout<<" Stress="<<" i= "<<i<<" j= "<<j<<" k="<<k<<"\n";
          //std::cout<<"\n before limiter Sum(3,2,2) output="<<std::scientific<<Sum[0]<<","<<Sum[1]<<","<<Sum[2]<<","<<Sum[3]<<","<<Sum[4]<<","<<Sum[5]<<","<<Sum[6]<<","<<Sum[7]<<","<<Sum[8]<<","<<Sum[9]<<","<<Sum[10]<<","<<Sum[11]<<"\n";
          //std::cout<<" Sum12="<<Sum[12]<<","<<Sum[13]<<","<<Sum[14]<<","<<Sum[15]<<","<<Sum[16]<<","<<Sum[17]<<","<<Sum[18]<<","<<Sum[19]<<","<<Sum[20]<<","<<Sum[21]<<","<<Sum[22]<<","<<Sum[23]<<","<<"\n";
          //}
          trace=1.0/3.0*(Sum[1-1]+Sum[5-1]+Sum[9-1]);
          Sum[1-1]=Sum[1-1]-trace;
          Sum[5-1]=Sum[5-1]-trace;
          Sum[9-1]=Sum[9-1]-trace;

          trace=1.0/3.0*(Sum[2-1]+Sum[6-1]+Sum[10-1]);
          Sum[2-1]= Sum[2-1]-trace;
          Sum[6-1]= Sum[6-1]-trace;
          Sum[10-1]=Sum[10-1]-trace;

          trace=1.0/3.0*(Sum[3-1]+Sum[7-1]+Sum[11-1]);
          Sum[3-1]= Sum[3-1]-trace;
          Sum[7-1]= Sum[7-1]-trace;
          Sum[11-1]=Sum[11-1]-trace;

          trace=1.0/3.0*(Sum[4-1]+Sum[8-1]+Sum[12-1]);
          Sum[4-1]= Sum[4-1]-trace;
          Sum[8-1]= Sum[8-1]-trace;
          Sum[12-1]=Sum[12-1]-trace;

          //filter resolved-resolved scale
          filterOperator(uustress, i, j, k,Sum[0] );
          filterOperator(uvstress, i, j, k,Sum[12] );
          filterOperator(uwstress, i, j, k,Sum[16] );
          if((uvstress(i,j,k)!= tau[12]) || (uwstress(i,j,k)!= tau[16]) )
          { std::cout<< "uu resoved stress is not right "<< "\n";
          }

          filterOperator(vustress, i, j, k,Sum[20] );
          filterOperator(vvstress, i, j, k,Sum[4] );
          filterOperator(vwstress, i, j, k,Sum[24] );

          filterOperator(wustress, i, j, k,Sum[28] );
          filterOperator(wvstress, i, j, k,Sum[32] );
          filterOperator(wwstress, i, j, k,Sum[8] );

          //filter begin with this one
          double Lij=0.0;
          // at u-cell xface
          Lij=strain_ratio[1-1]/ratio_threshold;
          ////    Sum[1-1]= (strain_ratio[1-1]<ratio_threshold && StrainUR[1-1]*Sum[1-1]>0  )  ?  Lij*Sum[1-1]  : Sum[2-1] ;
          Sum[2-1]= (strain_ratio[1-1]<ratio_threshold && StrainUR[1-1]*Sum[2-1]>0  )  ?  Lij*Sum[2-1]  : Sum[2-1] ;
          Sum[3-1]= (strain_ratio[1-1]<ratio_threshold && StrainUR[1-1]*Sum[3-1]>0  )  ?  Lij*Sum[3-1]  : Sum[3-1] ;
          Sum[4-1]= (strain_ratio[1-1]<ratio_threshold && StrainUR[1-1]*Sum[4-1]>0  )  ?  Lij*Sum[4-1]  : Sum[4-1] ;

          //// at v-cell yface
          Lij=strain_ratio[2-1]/ratio_threshold;
          ////Sum[5-1]= (strain_ratio[2-1]<ratio_threshold && StrainUR[2-1]*Sum[5-1]>0  )  ?  Lij*Sum[5-1]  : Sum[5-1] ;
          Sum[6-1]= (strain_ratio[2-1]<ratio_threshold && StrainUR[2-1]*Sum[6-1]>0  )  ?  Lij*Sum[6-1]  : Sum[6-1] ;
          Sum[7-1]= (strain_ratio[2-1]<ratio_threshold && StrainUR[2-1]*Sum[7-1]>0  )  ?  Lij*Sum[7-1]  : Sum[7-1] ;
          Sum[8-1]= (strain_ratio[2-1]<ratio_threshold && StrainUR[2-1]*Sum[8-1]>0  )  ?  Lij*Sum[8-1]  : Sum[8-1] ;

          //// at w-cell zface
          Lij=strain_ratio[3-1]/ratio_threshold;
          ////Sum[9-1]=  (strain_ratio[9-1]<ratio_threshold && StrainUR[3-1]*Sum[9-1]>0  )  ?   Lij*Sum[9-1]  :  Sum[9-1] ;
          Sum[10-1]= (strain_ratio[3-1]<ratio_threshold && StrainUR[3-1]*Sum[10-1]>0  )  ?  Lij*Sum[10-1]  : Sum[10-1] ;
          Sum[11-1]= (strain_ratio[3-1]<ratio_threshold && StrainUR[3-1]*Sum[11-1]>0  )  ?  Lij*Sum[11-1]  : Sum[11-1] ;
          Sum[12-1]= (strain_ratio[3-1]<ratio_threshold && StrainUR[3-1]*Sum[12-1]>0  )  ?  Lij*Sum[12-1]  : Sum[12-1] ;

          //// Return the trace to diagonal components
          ////trace=1.0/3.0*(tau[2-1]+tau[6-1]+tau[10-1]);
          ////Sum[2-1]= Sum[2-1]+trace;
          ////Sum[6-1]= Sum[6-1]+trace;
          ////Sum[10-1]=Sum[10-1]+trace;

          ////trace=1.0/3.0*(tau[3-1]+tau[7-1]+tau[11-1]);
          ////Sum[3-1]= Sum[3-1]+trace;
          ////Sum[7-1]= Sum[7-1]+trace;
          ////Sum[11-1]=Sum[11-1]+trace;

          ////trace=1.0/3.0*(tau[4-1]+tau[8-1]+tau[12-1]);
          ////Sum[4-1]= Sum[4-1]+trace;
          ////Sum[8-1]= Sum[8-1]+trace;
          ////Sum[12-1]=Sum[12-1]+trace;


          //// divotical elements
          //// TauUV& TauVU
          Lij=strain_ratio[4-1]/ratio_threshold;
          // Sum[12]= (strain_ratio[4-1]<ratio_threshold && StrainUR[4-1]*Sum[12]>0  )  ?  Lij*Sum[12]  : Sum[20] ;
          Sum[13]= (strain_ratio[4-1]<ratio_threshold && StrainUR[4-1]*Sum[13]>0  )  ?  Lij*Sum[13]  : Sum[13] ;
          Sum[14]= (strain_ratio[4-1]<ratio_threshold && StrainUR[4-1]*Sum[14]>0  )  ?  Lij*Sum[14]  : Sum[14] ;
          Sum[15]= (strain_ratio[4-1]<ratio_threshold && StrainUR[4-1]*Sum[15]>0  )  ?  Lij*Sum[15]  : Sum[15] ;

          // Sum[20]= (strain_ratio[4-1]<ratio_threshold && StrainUR[4-1]*Sum[20]>0  )  ?  Lij*Sum[20]  : Sum[20] ;
          Sum[21]= (strain_ratio[4-1]<ratio_threshold && StrainUR[4-1]*Sum[21]>0  )  ?  Lij*Sum[21]  : Sum[21] ;
          Sum[22]= (strain_ratio[4-1]<ratio_threshold && StrainUR[4-1]*Sum[22]>0  )  ?  Lij*Sum[22]  : Sum[22] ;
          Sum[23]= (strain_ratio[4-1]<ratio_threshold && StrainUR[4-1]*Sum[23]>0  )  ?  Lij*Sum[23]  : Sum[23] ;

          //// TauWV& TauVW
          Lij=strain_ratio[5-1]/ratio_threshold;
          // Sum[24]= (strain_ratio[5-1]<ratio_threshold && StrainUR[5-1]*Sum[24]>0  )  ? Lij*Sum[24]  : Sum[24] ;
          Sum[25]= (strain_ratio[5-1]<ratio_threshold && StrainUR[5-1]*Sum[25]>0  )  ? Lij*Sum[25]  : Sum[25] ;
          Sum[26]= (strain_ratio[5-1]<ratio_threshold && StrainUR[5-1]*Sum[26]>0  )  ? Lij*Sum[26]  : Sum[26] ;
          Sum[27]= (strain_ratio[5-1]<ratio_threshold && StrainUR[5-1]*Sum[27]>0  )  ? Lij*Sum[27]  : Sum[27] ;

          // Sum[32]= (strain_ratio[5-1]<ratio_threshold && StrainUR[5-1]*Sum[32]>0  )  ? Lij*Sum[32]  : Sum[32] ;
          Sum[33]= (strain_ratio[5-1]<ratio_threshold && StrainUR[5-1]*Sum[33]>0  )  ? Lij*Sum[33]  : Sum[33] ;
          Sum[34]= (strain_ratio[5-1]<ratio_threshold && StrainUR[5-1]*Sum[34]>0  )  ? Lij*Sum[34]  : Sum[34] ;
          Sum[35]= (strain_ratio[5-1]<ratio_threshold && StrainUR[5-1]*Sum[35]>0  )  ? Lij*Sum[35]  : Sum[35] ;

          ////TauUW & TauWU
          Lij=strain_ratio[6-1]/ratio_threshold;
          // Sum[16]= (strain_ratio[6-1]<ratio_threshold && StrainUR[6-1]*Sum[16]>0  )  ? Lij*Sum[16]  : Sum[16] ;
          Sum[17]= (strain_ratio[6-1]<ratio_threshold && StrainUR[6-1]*Sum[17]>0  )  ? Lij*Sum[17]  : Sum[17] ;
          Sum[18]= (strain_ratio[6-1]<ratio_threshold && StrainUR[6-1]*Sum[18]>0  )  ? Lij*Sum[18]  : Sum[18] ;
          Sum[19]= (strain_ratio[6-1]<ratio_threshold && StrainUR[6-1]*Sum[19]>0  )  ? Lij*Sum[19]  : Sum[19] ;

          // Sum[28]= (strain_ratio[6-1]<ratio_threshold && StrainUR[6-1]*Sum[28]>0  )  ? Lij*Sum[28]  : Sum[28] ;
          Sum[29]= (strain_ratio[6-1]<ratio_threshold && StrainUR[6-1]*Sum[29]>0  )  ? Lij*Sum[29]  : Sum[29] ;
          Sum[30]= (strain_ratio[6-1]<ratio_threshold && StrainUR[6-1]*Sum[30]>0  )  ? Lij*Sum[30]  : Sum[30] ;
          Sum[31]= (strain_ratio[6-1]<ratio_threshold && StrainUR[6-1]*Sum[31]>0  )  ? Lij*Sum[31]  : Sum[31] ;

          // test more filter;
          //Sum[1-1]= ( StrainUR[1-1]*Sum[1-1]>0  )  ?  limDevotic*Sum[1-1]  : Sum[2-1] ;
          //Sum[5-1]= ( StrainUR[2-1]*Sum[5-1]>0  )  ?  limDevotic*Sum[5-1]  : Sum[5-1] ;
          //Sum[9-1]= ( StrainUR[3-1]*Sum[9-1]>0  )  ?  limDevotic*Sum[9-1]  : Sum[9-1] ;

          //direct diagonial limiter
          // direct limiter paper 2004(ii),eq28
          //Sum[1-1]= ( StrainUR[1-1]*Sum[1-1]>0  )  ? limDevotic*Sum[1-1]  : Sum[2-1] ;
          // // determining the backscatter paper2004(ii.), eq.24
          // Sum[2-1]= ( StrainUR[1-1]*Sum[2-1]>0  )  ? limDevotic*Sum[2-1]  : Sum[2-1] ;
          // Sum[3-1]= ( StrainUR[1-1]*Sum[3-1]>0  )  ? limDevotic*Sum[3-1]  : Sum[3-1] ;
          // //Sum[4-1]= ( StrainUR[1-1]*Sum[4-1]>0  )  ? (limDevotic+0.1)*Sum[4-1]  : Sum[4-1] ;
          // Sum[4-1]= ( StrainUR[1-1]*Sum[4-1]>0  )  ? (limDevotic)*Sum[4-1]  : Sum[4-1] ;
          //
          // // at v-cell yface
          // Lij=strain_ratio[2-1]/ratio_threshold;
          // //Sum[5-1]= ( StrainUR[2-1]*Sum[5-1]>0  )  ? limDevotic*Sum[5-1]  : Sum[5-1] ;
          // Sum[6-1]= ( StrainUR[2-1]*Sum[6-1]>0  )  ? limDevotic*Sum[6-1]  : Sum[6-1] ;
          // Sum[7-1]= ( StrainUR[2-1]*Sum[7-1]>0  )  ? limDevotic*Sum[7-1]  : Sum[7-1] ;
          // //Sum[8-1]= ( StrainUR[2-1]*Sum[8-1]>0  )  ? (limDevotic+0.1)*Sum[8-1]  : Sum[8-1] ;
          // Sum[8-1]= ( StrainUR[2-1]*Sum[8-1]>0  )  ? (limDevotic)*Sum[8-1]  : Sum[8-1] ;
          //
          // // at w-cell zface
          // Lij=strain_ratio[3-1]/ratio_threshold;
          // //Sum[9-1]=  ( StrainUR[3-1]*Sum[9-1]>0  ) ? limDevotic*Sum[9-1]  :  Sum[9-1] ;
          // Sum[10-1]= ( StrainUR[3-1]*Sum[10-1]>0 ) ? limDevotic*Sum[10-1]  : Sum[10-1] ;
          // Sum[11-1]= ( StrainUR[3-1]*Sum[11-1]>0 ) ? limDevotic*Sum[11-1]  : Sum[11-1] ;
          // //Sum[12-1]= ( StrainUR[3-1]*Sum[12-1]>0 ) ? (limDevotic+0.1)*Sum[12-1]  : Sum[12-1] ;
          // Sum[12-1]= ( StrainUR[3-1]*Sum[12-1]>0 ) ? (limDevotic)*Sum[12-1]  : Sum[12-1] ;
          //
          // ////  direct divogetic limiter
          // Sum[12]= ( StrainUR[4-1]*Sum[12]>0  )  ?  limDevotic*Sum[12]  : Sum[12] ;
          // Sum[13]= ( StrainUR[4-1]*Sum[13]>0  )  ?  limDevotic*Sum[13]  : Sum[13] ;
          // Sum[14]= ( StrainUR[4-1]*Sum[14]>0  )  ?  limDevotic*Sum[14]  : Sum[14] ;
          // Sum[15]= ( StrainUR[4-1]*Sum[15]>0  )  ?  (limDevotic)*Sum[15]  : Sum[15] ;
          // //Sum[15]= ( StrainUR[4-1]*Sum[15]>0  )  ?  (limDevotic+0.1)*Sum[15]  : Sum[15] ;
          //
          // Sum[20]= ( StrainUR[4-1]*Sum[20]>0  )  ?  limDevotic*Sum[20]  : Sum[20] ;
          // Sum[21]= ( StrainUR[4-1]*Sum[21]>0  )  ?  limDevotic*Sum[21]  : Sum[21] ;
          // Sum[22]= ( StrainUR[4-1]*Sum[22]>0  )  ?  limDevotic*Sum[22]  : Sum[22] ;
          // //Sum[23]= ( StrainUR[4-1]*Sum[23]>0  )  ?  (limDevotic+0.1)*Sum[23]  : Sum[23] ;
          // Sum[23]= ( StrainUR[4-1]*Sum[23]>0  )  ?  (limDevotic)*Sum[23]  : Sum[23] ;
          //
          //   //TauWV&
          // Sum[24]= ( StrainUR[5-1]*Sum[24]>0  )  ?  limDevotic*Sum[24]  : Sum[24] ;
          // Sum[25]= ( StrainUR[5-1]*Sum[25]>0  )  ?  limDevotic*Sum[25]  : Sum[25] ;
          // Sum[26]= ( StrainUR[5-1]*Sum[26]>0  )  ?  limDevotic*Sum[26]  : Sum[26] ;
          // Sum[27]= ( StrainUR[5-1]*Sum[27]>0  )  ?  (limDevotic)*Sum[27]  : Sum[27] ;
          // //Sum[27]= ( StrainUR[5-1]*Sum[27]>0  )  ?  (limDevotic+0.1)*Sum[27]  : Sum[27] ;
          //
          // Sum[32]= ( StrainUR[5-1]*Sum[32]>0  )  ?  limDevotic*Sum[32]  : Sum[32] ;
          // Sum[33]= ( StrainUR[5-1]*Sum[33]>0  )  ?  limDevotic*Sum[33]  : Sum[33] ;
          // Sum[34]= ( StrainUR[5-1]*Sum[34]>0  )  ?  limDevotic*Sum[34]  : Sum[34] ;
          // //Sum[35]= ( StrainUR[5-1]*Sum[35]>0  )  ?  (limDevotic+0.1)*Sum[35]  : Sum[35] ;
          // Sum[35]= ( StrainUR[5-1]*Sum[35]>0  )  ?  (limDevotic)*Sum[35]  : Sum[35] ;
          //
          //  //TauUW &
          // Sum[16]= ( StrainUR[6-1]*Sum[16]>0  )  ?  limDevotic*Sum[16]  : Sum[16] ;
          // Sum[17]= ( StrainUR[6-1]*Sum[17]>0  )  ?  limDevotic*Sum[17]  : Sum[17] ;
          // Sum[18]= ( StrainUR[6-1]*Sum[18]>0  )  ?  limDevotic*Sum[18]  : Sum[18] ;
          // //Sum[19]= ( StrainUR[6-1]*Sum[19]>0  )  ?  (limDevotic+0.1)*Sum[19]  : Sum[19] ;
          // Sum[19]= ( StrainUR[6-1]*Sum[19]>0  )  ?  (limDevotic)*Sum[19]  : Sum[19] ;
          //
          // Sum[28]= ( StrainUR[6-1]*Sum[28]>0  )  ?  limDevotic*Sum[28]  : Sum[28] ;
          // Sum[29]= ( StrainUR[6-1]*Sum[29]>0  )  ?  limDevotic*Sum[29]  : Sum[29] ;
          // Sum[30]= ( StrainUR[6-1]*Sum[30]>0  )  ?  limDevotic*Sum[30]  : Sum[30] ;
          // //Sum[31]= ( StrainUR[6-1]*Sum[31]>0  )  ?  (limDevotic+0.1)*Sum[31]  : Sum[31] ;
          // Sum[31]= ( StrainUR[6-1]*Sum[31]>0  )  ?  (limDevotic)*Sum[31]  : Sum[31] ;


          // remove the trace for the diagonal term after the limiter
          //trace=1.0/3.0*(Sum[1-1]+Sum[5-1]+Sum[9-1]);
          //Sum[1-1]=Sum[1-1]-trace;
          //Sum[5-1]=Sum[5-1]-trace;
          //Sum[9-1]=Sum[9-1]-trace;

          //trace=1.0/3.0*(Sum[2-1]+Sum[6-1]+Sum[10-1]);
          //Sum[2-1]= Sum[2-1]-trace;
          //Sum[6-1]= Sum[6-1]-trace;
          //Sum[10-1]=Sum[10-1]-trace;

          //trace=1.0/3.0*(Sum[3-1]+Sum[7-1]+Sum[11-1]);
          //Sum[3-1]= Sum[3-1]-trace;
          //Sum[7-1]= Sum[7-1]-trace;
          //Sum[11-1]=Sum[11-1]-trace;

          //trace=1.0/3.0*(Sum[4-1]+Sum[8-1]+Sum[12-1]);
          //Sum[4-1]= Sum[4-1]-trace;
          //Sum[8-1]= Sum[8-1]-trace;
          //Sum[12-1]=Sum[12-1]-trace;

          //if(i==2&&j==2&&k==2)
          //{std::cout<<" Stress="<<" i= "<<i<<" j= "<<j<<" k="<<k<<"\n";
          //std::cout<<"\n After limiter Sum(2,2,2) output="<<std::scientific<<Sum[0]<<","<<Sum[1]<<","<<Sum[2]<<","<<Sum[3]<<","<<Sum[4]<<","<<Sum[5]<<","<<Sum[6]<<","<<Sum[7]<<","<<Sum[8]<<","<<Sum[9]<<","<<Sum[10]<<","<<Sum[11]<<"\n";
          //std::cout<<" Sum12="<<Sum[12]<<","<<Sum[13]<<","<<Sum[14]<<","<<Sum[15]<<","<<Sum[16]<<","<<Sum[17]<<","<<Sum[18]<<","<<Sum[19]<<","<<Sum[20]<<","<<Sum[21]<<","<<Sum[22]<<","<<Sum[23]<<","<<"\n";
          //}

          // without uu resolved
          //ucell_xSgsStress(i,j,k) =Sum[2-1] +Sum[3-1] +Sum[4-1] ; //  TauUU
          //vcell_ySgsStress(i,j,k) =Sum[6-1] +Sum[7-1] +Sum[8-1] ;//TauVV
          //wcell_zSgsStress(i,j,k) =Sum[10-1]+Sum[11-1]+Sum[12-1]; //TauWW
          ////at U-V nodes
          ////TauRhoUV at u-v node
          //ucell_ySgsStress(i,j,k) = Sum[13]+Sum[14]+Sum[15];
          ////TauRhoUW at u-w node
          //ucell_zSgsStress(i,j,k) = Sum[17]+Sum[18]+Sum[19];

          //// TauRhoVU at v-u nodes
          //vcell_xSgsStress(i,j,k) = Sum[21]+Sum[22]+Sum[23];
          ////TauRhoVW at v-w node
          //vcell_zSgsStress(i,j,k) = Sum[25]+Sum[26]+Sum[27];

          ////TauRhoWU at w-u node
          //wcell_xSgsStress(i,j,k) = Sum[29]+Sum[30]+Sum[31];

          ////TauRhoWV at w-v node
          //wcell_ySgsStress(i,j,k) = Sum[33]+Sum[34]+Sum[35];

          // with uu resolved not removal diagonal
          //ucell_xSgsStress(i,j,k) =Sum[1-1]+Sum[2-1] +Sum[3-1] +Sum[4-1]  ; //  TauUU
          //vcell_ySgsStress(i,j,k) =Sum[5-1]+Sum[6-1] +Sum[7-1] +Sum[8-1]  ;//TauVV
          //wcell_zSgsStress(i,j,k) =Sum[9-1]+Sum[10-1]+Sum[11-1]+Sum[12-1] ; //TauWW
          ////at U-V nodes
          ////TauRhoUV at u-v node
          //ucell_ySgsStress(i,j,k) = Sum[12]+Sum[13]+Sum[14]+Sum[15];
          ////TauRhoUW at u-w node
          //ucell_zSgsStress(i,j,k) = Sum[16]+Sum[17]+Sum[18]+Sum[19];

          //// TauRhoVU at v-u nodes
          //vcell_xSgsStress(i,j,k) = Sum[20]+Sum[21]+Sum[22]+Sum[23];
          ////TauRhoVW at v-w node
          //vcell_zSgsStress(i,j,k) = Sum[24]+Sum[25]+Sum[26]+Sum[27];

          ////TauRhoWU at w-u node
          //wcell_xSgsStress(i,j,k) = Sum[28]+Sum[29]+Sum[30]+Sum[31];

          ////TauRhoWV at w-v node
          //wcell_ySgsStress(i,j,k) = Sum[32]+Sum[33]+Sum[34]+Sum[35];

          // with uu resolved couple convection
          // compute the subgrid stress from the paper 2004(ii),eq8
          ucell_xSgsStress(i,j,k) =Sum[1-1]+Sum[2-1] +Sum[3-1] +Sum[4-1] -tau[0] ; //  TauUU
          vcell_ySgsStress(i,j,k) =Sum[5-1]+Sum[6-1] +Sum[7-1] +Sum[8-1] -tau[4] ;//TauVV
          wcell_zSgsStress(i,j,k) =Sum[9-1]+Sum[10-1]+Sum[11-1]+Sum[12-1]-tau[8] ; //TauWW
          //at U-V nodes
          //TauRhoUV at u-v node
          ucell_ySgsStress(i,j,k) = Sum[12]+Sum[13]+Sum[14]+Sum[15]-tau[12];
          //TauRhoUW at u-w node
          ucell_zSgsStress(i,j,k) = Sum[16]+Sum[17]+Sum[18]+Sum[19]-tau[16];

          // TauRhoVU at v-u nodes
          vcell_xSgsStress(i,j,k) = Sum[20]+Sum[21]+Sum[22]+Sum[23]-tau[20];
          //TauRhoVW at v-w node
          vcell_zSgsStress(i,j,k) = Sum[24]+Sum[25]+Sum[26]+Sum[27]-tau[24];

          //TauRhoWU at w-u node
          wcell_xSgsStress(i,j,k) = Sum[28]+Sum[29]+Sum[30]+Sum[31]-tau[28];

          //TauRhoWV at w-v node
          wcell_ySgsStress(i,j,k) = Sum[32]+Sum[33]+Sum[34]+Sum[35]-tau[32];


      });

    }
} //namespace Uintah
