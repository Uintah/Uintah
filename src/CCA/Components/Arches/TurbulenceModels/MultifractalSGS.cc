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
    }

  //---------------------------------------------------------------------------------
  void
    MultifractalSGS::register_initialize( std::vector<AFC::VariableInformation>&
        variable_registry , const bool packed_tasks){

      for (auto iter = m_SgsStress_names.begin(); iter != m_SgsStress_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
      }

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
        register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
      }

    }

  //---------------------------------------------------------------------------------
  void
    MultifractalSGS::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

      // Unused - creating a compiler warning
      // SFCXVariable<double>& ucell_xSgsStress = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_xSgsStress"));
      // SFCXVariable<double>& ucell_ySgsStress = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_ySgsStress"));
      // SFCXVariable<double>& ucell_zSgsStress = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_zSgsStress"));
      // SFCYVariable<double>& vcell_xSgsStress = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_xSgsStress"));
      // SFCYVariable<double>& vcell_ySgsStress = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_ySgsStress"));
      // SFCYVariable<double>& vcell_zSgsStress = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_zSgsStress"));
      // SFCZVariable<double>& wcell_xSgsStress = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_xSgsStress"));
      // SFCZVariable<double>& wcell_ySgsStress = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_ySgsStress"));
      // SFCZVariable<double>& wcell_zSgsStress = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_zSgsStress"));

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
      register_variable( "density",     ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::OLDDW, variable_registry,time_substep );
      // UPDATE USER DEFINE VARIABLES
      // register Velocity Delta
      for (auto iter = m_VelDelta_names.begin(); iter != m_VelDelta_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry, m_task_name );

      }

      for (auto iter = m_SgsStress_names.begin(); iter != m_SgsStress_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::MODIFIES, variable_registry, m_task_name );
      }

    }

  //---------------------------------------------------------------------------------
  void
    MultifractalSGS::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
      Vector Dx=patch->dCell();
      double dx=Dx.x(); double dy=Dx.y(); double dz=Dx.z();

      double LegthScales=pow(Dx.x()*Dx.y()*Dx.z(),1.0/3.0);
      double ratio_threshold = 2.0;

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

          double strainUD_magn = sqrt(StrainUD[1-1]*StrainUD[1-1]+StrainUD[2-1]*StrainUD[2-1]+StrainUD[3-1]*StrainUD[3-1]
                                 + 2.0*(StrainUD[4-1]*StrainUD[4-1]+StrainUD[5-1]*StrainUD[5-1]+StrainUD[6-1]*StrainUD[6-1]));


          // calcuting strain ratio
          for (unsigned int iter=0 ;iter < StrainUD.size(); iter++)
          {
            //strain_ratio[iter] = StrainUD[iter]==0 ? ratio_threshold : std::abs(StrainU2D[iter]/StrainUD[iter]);
            strain_ratio[iter] = std::abs(StrainU2D[iter]/(StrainUD[iter]+1e-10));
          }
          // calcuting C_sgs_G and U_sgs/rhoUsgs
          double sgs_scales=0; double factorN=0; double value=0; double Re_g=0.0;

          // paper 2004(ii), eq.15 to calculate the coefficient, which is in the header file
          double C_sgs_G = sgsVelCoeff(mu,LegthScales,strainUD_magn,dx, dy, dz, sgs_scales,factorN, value,Re_g);

          // paper 2004(II)-eq.14

          // at u-cell xface
          tau[0]= ucell_xvel_face(i,j,k)*ucell_xvel_face(i,j,k);
          tau[1]= ucell_xvel_face(i,j,k)*ucell_XvelD(i,j,k)*C_sgs_G;
          tau[2]= ucell_XvelD(i,j,k)*C_sgs_G*ucell_xvel_face(i,j,k);
          tau[3]= ucell_XvelD(i,j,k)*C_sgs_G*ucell_XvelD(i,j,k)*C_sgs_G;

          // at v-cell yface
          tau[4]= vcell_yvel_face(i,j,k)*vcell_yvel_face(i,j,k);
          tau[5]= vcell_yvel_face(i,j,k)*vcell_YvelD(i,j,k)*C_sgs_G;
          tau[6]= vcell_YvelD(i,j,k)*C_sgs_G*vcell_yvel_face(i,j,k);
          tau[7]= vcell_YvelD(i,j,k)*C_sgs_G*vcell_YvelD(i,j,k)*C_sgs_G;

          // at w-cell wface
          tau[8]=  wcell_zvel_face(i,j,k)*wcell_zvel_face(i,j,k);
          tau[9]=  wcell_zvel_face(i,j,k)*wcell_ZvelD(i,j,k)*C_sgs_G;
          tau[10]= wcell_ZvelD(i,j,k)*C_sgs_G*wcell_zvel_face(i,j,k);
          tau[11]= wcell_ZvelD(i,j,k)*C_sgs_G*wcell_ZvelD(i,j,k)*C_sgs_G;

          // 4:rhoU*V at uv-node: TauUV
          tau[12]= vcell_xvel_face(i,j,k)*ucell_yvel_face(i,j,k);
          tau[13]= vcell_xvel_face(i,j,k)*ucell_YvelD(i,j,k)*C_sgs_G;
          tau[14]= vcell_XvelD(i,j,k)*ucell_yvel_face(i,j,k)*C_sgs_G;
          tau[15]= vcell_XvelD(i,j,k)*C_sgs_G*ucell_YvelD(i,j,k)*C_sgs_G;

          // 5:rhoU*W at vw-node: TauUW
          tau[16]= wcell_xvel_face(i,j,k)*ucell_zvel_face(i,j,k);
          tau[17]= wcell_xvel_face(i,j,k)*ucell_ZvelD(i,j,k)*C_sgs_G;
          tau[18]= wcell_XvelD(i,j,k)*C_sgs_G*ucell_zvel_face(i,j,k);
          tau[19]= wcell_XvelD(i,j,k)*C_sgs_G*ucell_ZvelD(i,j,k)*C_sgs_G;

          // 6:rhoV*U at vu-node: TauVU
          tau[20]= ucell_yvel_face(i,j,k)*vcell_xvel_face(i,j,k);
          tau[21]= ucell_yvel_face(i,j,k)*vcell_XvelD(i,j,k)*C_sgs_G;
          tau[22]= ucell_YvelD(i,j,k)*vcell_xvel_face(i,j,k)*C_sgs_G;
          tau[23]= ucell_YvelD(i,j,k)*C_sgs_G*vcell_XvelD(i,j,k)*C_sgs_G;

          // 7:rhoV*W at vw-node: TauVW
          tau[24]= wcell_yvel_face(i,j,k)*vcell_zvel_face(i,j,k);
          tau[25]= wcell_yvel_face(i,j,k)*vcell_ZvelD(i,j,k)*C_sgs_G;
          tau[26]= wcell_YvelD(i,j,k)*C_sgs_G*vcell_zvel_face(i,j,k);
          tau[27]= wcell_YvelD(i,j,k)*C_sgs_G*vcell_ZvelD(i,j,k)*C_sgs_G;

          // 8:rhoW*U at vw-node: TauWU
          tau[28]= ucell_zvel_face(i,j,k)*wcell_xvel_face(i,j,k);
          tau[29]= ucell_zvel_face(i,j,k)*wcell_XvelD(i,j,k)*C_sgs_G;
          tau[30]= ucell_ZvelD(i,j,k)*C_sgs_G*wcell_xvel_face(i,j,k);
          tau[31]= ucell_ZvelD(i,j,k)*C_sgs_G*wcell_XvelD(i,j,k)*C_sgs_G;

          // 9:rhoW*V at vw-node: TauWV
          tau[32]= vcell_zvel_face(i,j,k)*wcell_yvel_face(i,j,k);
          tau[33]= vcell_zvel_face(i,j,k)*wcell_YvelD(i,j,k)*C_sgs_G;
          tau[34]= vcell_ZvelD(i,j,k)*C_sgs_G*wcell_yvel_face(i,j,k);
          tau[35]= vcell_ZvelD(i,j,k)*C_sgs_G*wcell_YvelD(i,j,k)*C_sgs_G;

          for (unsigned int iter=0 ;iter < tau.size(); iter++)
          { Sum[iter]=tau[iter];   }

          double Lij=0.0;
          // at u-cell xface
          Lij=strain_ratio[1-1]/ratio_threshold;
          Sum[2-1]= (strain_ratio[1-1]<ratio_threshold && StrainUR[1-1]*Sum[2-1]>0  )  ?  Lij*Sum[2-1]  : Sum[2-1] ;
          Sum[3-1]= (strain_ratio[1-1]<ratio_threshold && StrainUR[1-1]*Sum[3-1]>0  )  ?  Lij*Sum[3-1]  : Sum[3-1] ;
          Sum[4-1]= (strain_ratio[1-1]<ratio_threshold && StrainUR[1-1]*Sum[4-1]>0  )  ?  Lij*Sum[4-1]  : Sum[4-1] ;

          //// at v-cell yface
          Lij=strain_ratio[2-1]/ratio_threshold;
          Sum[6-1]= (strain_ratio[2-1]<ratio_threshold && StrainUR[2-1]*Sum[6-1]>0  )  ?  Lij*Sum[6-1]  : Sum[6-1] ;
          Sum[7-1]= (strain_ratio[2-1]<ratio_threshold && StrainUR[2-1]*Sum[7-1]>0  )  ?  Lij*Sum[7-1]  : Sum[7-1] ;
          Sum[8-1]= (strain_ratio[2-1]<ratio_threshold && StrainUR[2-1]*Sum[8-1]>0  )  ?  Lij*Sum[8-1]  : Sum[8-1] ;

          //// at w-cell zface
          Lij=strain_ratio[3-1]/ratio_threshold;
          Sum[10-1]= (strain_ratio[3-1]<ratio_threshold && StrainUR[3-1]*Sum[10-1]>0  )  ?  Lij*Sum[10-1]  : Sum[10-1] ;
          Sum[11-1]= (strain_ratio[3-1]<ratio_threshold && StrainUR[3-1]*Sum[11-1]>0  )  ?  Lij*Sum[11-1]  : Sum[11-1] ;
          Sum[12-1]= (strain_ratio[3-1]<ratio_threshold && StrainUR[3-1]*Sum[12-1]>0  )  ?  Lij*Sum[12-1]  : Sum[12-1] ;

          //// TauUV& TauVU
          Lij=strain_ratio[4-1]/ratio_threshold;
          Sum[13]= (strain_ratio[4-1]<ratio_threshold && StrainUR[4-1]*Sum[13]>0  )  ?  Lij*Sum[13]  : Sum[13] ;
          Sum[14]= (strain_ratio[4-1]<ratio_threshold && StrainUR[4-1]*Sum[14]>0  )  ?  Lij*Sum[14]  : Sum[14] ;
          Sum[15]= (strain_ratio[4-1]<ratio_threshold && StrainUR[4-1]*Sum[15]>0  )  ?  Lij*Sum[15]  : Sum[15] ;
          Sum[21]= (strain_ratio[4-1]<ratio_threshold && StrainUR[4-1]*Sum[21]>0  )  ?  Lij*Sum[21]  : Sum[21] ;
          Sum[22]= (strain_ratio[4-1]<ratio_threshold && StrainUR[4-1]*Sum[22]>0  )  ?  Lij*Sum[22]  : Sum[22] ;
          Sum[23]= (strain_ratio[4-1]<ratio_threshold && StrainUR[4-1]*Sum[23]>0  )  ?  Lij*Sum[23]  : Sum[23] ;

          //// TauWV& TauVW
          Lij=strain_ratio[5-1]/ratio_threshold;
          Sum[25]= (strain_ratio[5-1]<ratio_threshold && StrainUR[5-1]*Sum[25]>0  )  ? Lij*Sum[25]  : Sum[25] ;
          Sum[26]= (strain_ratio[5-1]<ratio_threshold && StrainUR[5-1]*Sum[26]>0  )  ? Lij*Sum[26]  : Sum[26] ;
          Sum[27]= (strain_ratio[5-1]<ratio_threshold && StrainUR[5-1]*Sum[27]>0  )  ? Lij*Sum[27]  : Sum[27] ;
          Sum[33]= (strain_ratio[5-1]<ratio_threshold && StrainUR[5-1]*Sum[33]>0  )  ? Lij*Sum[33]  : Sum[33] ;
          Sum[34]= (strain_ratio[5-1]<ratio_threshold && StrainUR[5-1]*Sum[34]>0  )  ? Lij*Sum[34]  : Sum[34] ;
          Sum[35]= (strain_ratio[5-1]<ratio_threshold && StrainUR[5-1]*Sum[35]>0  )  ? Lij*Sum[35]  : Sum[35] ;

          ////TauUW & TauWU
          Lij=strain_ratio[6-1]/ratio_threshold;
          Sum[17]= (strain_ratio[6-1]<ratio_threshold && StrainUR[6-1]*Sum[17]>0  )  ? Lij*Sum[17]  : Sum[17] ;
          Sum[18]= (strain_ratio[6-1]<ratio_threshold && StrainUR[6-1]*Sum[18]>0  )  ? Lij*Sum[18]  : Sum[18] ;
          Sum[19]= (strain_ratio[6-1]<ratio_threshold && StrainUR[6-1]*Sum[19]>0  )  ? Lij*Sum[19]  : Sum[19] ;
          Sum[29]= (strain_ratio[6-1]<ratio_threshold && StrainUR[6-1]*Sum[29]>0  )  ? Lij*Sum[29]  : Sum[29] ;
          Sum[30]= (strain_ratio[6-1]<ratio_threshold && StrainUR[6-1]*Sum[30]>0  )  ? Lij*Sum[30]  : Sum[30] ;
          Sum[31]= (strain_ratio[6-1]<ratio_threshold && StrainUR[6-1]*Sum[31]>0  )  ? Lij*Sum[31]  : Sum[31] ;

          // TauUU
          ucell_xSgsStress(i,j,k) =Sum[1-1]+Sum[2-1] +Sum[3-1] +Sum[4-1];
          // TauVV
          vcell_ySgsStress(i,j,k) =Sum[5-1]+Sum[6-1] +Sum[7-1] +Sum[8-1];
          // TauWW
          wcell_zSgsStress(i,j,k) =Sum[9-1]+Sum[10-1]+Sum[11-1]+Sum[12-1];
          // TauUV TauVU
          ucell_ySgsStress(i,j,k) = Sum[12]+Sum[13]+Sum[14]+Sum[15];
          vcell_xSgsStress(i,j,k) = Sum[20]+Sum[21]+Sum[22]+Sum[23];
          // TauUW TauWU
          ucell_zSgsStress(i,j,k) = Sum[16]+Sum[17]+Sum[18]+Sum[19];
          wcell_xSgsStress(i,j,k) = Sum[28]+Sum[29]+Sum[30]+Sum[31];
          // TauVW TauWV
          vcell_zSgsStress(i,j,k) = Sum[24]+Sum[25]+Sum[26]+Sum[27];
          wcell_ySgsStress(i,j,k) = Sum[32]+Sum[33]+Sum[34]+Sum[35];
      });

    }
} //namespace Uintah
