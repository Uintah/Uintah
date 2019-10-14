#include <CCA/Components/Arches/TurbulenceModels/FractalUD.h>
#include <CCA/Components/Arches/GridTools.h>

namespace Uintah{

  typedef ArchesFieldContainer AFC;

  //---------------------------------------------------------------------------------
  FractalUD::FractalUD( std::string task_name, int matl_index ) :
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
    m_VelDelta_names.resize(24);
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
    //create uv resolved scale stress
    m_VelDelta_names[15] = "uustress";
    m_VelDelta_names[16] = "uvstress";
    m_VelDelta_names[17] = "uwstress";
    m_VelDelta_names[18] = "vustress";
    m_VelDelta_names[19] = "vvstress";
    m_VelDelta_names[20] = "vwstress";
    m_VelDelta_names[21] = "wustress";
    m_VelDelta_names[22] = "wvstress";
    m_VelDelta_names[23] = "wwstress";

  }

  //---------------------------------------------------------------------------------
  FractalUD::~FractalUD()
  {}

  //---------------------------------------------------------------------------------
  void
    FractalUD::problemSetup( ProblemSpecP& db ){

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
          msg << "ERROR: Constant FractalUD: problemSetup(): Zero viscosity specified \n"
            << "       in <PhysicalConstants> section of input file." << std::endl;
          throw InvalidValue(msg.str(),__FILE__,__LINE__);
        }
      } else {
        std::stringstream msg;
        msg << "ERROR: Constant FractalUD: problemSetup(): Missing <PhysicalConstants> \n"
          << "       section in input file!" << std::endl;
        throw InvalidValue(msg.str(),__FILE__,__LINE__);
      }

    }

  //---------------------------------------------------------------------------------
  void
    FractalUD::create_local_labels(){


      //U-CELL LABELS:
      register_new_variable<SFCXVariable<double> >(m_VelDelta_names[0]);
      register_new_variable<SFCXVariable<double> >(m_VelDelta_names[1]);
      register_new_variable<SFCXVariable<double> >(m_VelDelta_names[2]);
      register_new_variable<SFCXVariable<double> >(m_VelDelta_names[3]);
      // V-cell
      register_new_variable<SFCYVariable<double> >(m_VelDelta_names[4]);
      register_new_variable<SFCYVariable<double> >(m_VelDelta_names[5]);
      register_new_variable<SFCYVariable<double> >(m_VelDelta_names[6]);
      register_new_variable<SFCYVariable<double> >(m_VelDelta_names[7]);
      // W-cell
      register_new_variable<SFCZVariable<double> >(m_VelDelta_names[8]);
      register_new_variable<SFCZVariable<double> >(m_VelDelta_names[9]);
      register_new_variable<SFCZVariable<double> >(m_VelDelta_names[10]);
      register_new_variable<SFCZVariable<double> >(m_VelDelta_names[11]);

      // U2D - each u v w cell center
      register_new_variable<SFCXVariable<double> >(m_VelDelta_names[12]);
      register_new_variable<SFCYVariable<double> >(m_VelDelta_names[13]);
      register_new_variable<SFCZVariable<double> >(m_VelDelta_names[14]);

      //create x-psi flux
      register_new_variable<SFCXVariable<double> >(m_VelDelta_names[15]);
      register_new_variable<SFCXVariable<double> >(m_VelDelta_names[16]);
      register_new_variable<SFCXVariable<double> >(m_VelDelta_names[17]);
      //create y-psi flux
      register_new_variable<SFCYVariable<double> >(m_VelDelta_names[18]);
      register_new_variable<SFCYVariable<double> >(m_VelDelta_names[19]);
      register_new_variable<SFCYVariable<double> >(m_VelDelta_names[20]);
      //create z-psi flux
      register_new_variable<SFCZVariable<double> >(m_VelDelta_names[21]);
      register_new_variable<SFCZVariable<double> >(m_VelDelta_names[22]);
      register_new_variable<SFCZVariable<double> >(m_VelDelta_names[23]);


    }

  //---------------------------------------------------------------------------------
  void
    FractalUD::register_initialize( std::vector<AFC::VariableInformation>&
        variable_registry , const bool packed_tasks){

      //  register_variable( m_t_vis_name, AFC::COMPUTES, variable_registry );

      // register Velocity Delta
      for (auto iter = m_VelDelta_names.begin(); iter != m_VelDelta_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry, m_task_name  );
      }
    }

  //---------------------------------------------------------------------------------
  void
    FractalUD::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

      // CCVariable<double>& mu_sgc = *(tsk_info->get_uintah_field<CCVariable<double> >(m_t_vis_name));
      //mu_sgc.initialize(0.0);
      SFCXVariable<double>& uD_ctr          = *(tsk_info->get_uintah_field<SFCXVariable<double> >("uD_ctr"));
      SFCXVariable<double>& ucell_XvelD     = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_XvelD"));
      SFCXVariable<double>& ucell_YvelD     = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_YvelD"));
      SFCXVariable<double>& ucell_ZvelD     = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_ZvelD"));

      SFCYVariable<double>& vD_ctr          = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vD_ctr"));
      SFCYVariable<double>& vcell_XvelD     = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_XvelD"));
      SFCYVariable<double>& vcell_YvelD     = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_YvelD"));
      SFCYVariable<double>& vcell_ZvelD     = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_ZvelD"));

      SFCZVariable<double>& wD_ctr          = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wD_ctr"));
      SFCZVariable<double>& wcell_XvelD     = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_XvelD"));
      SFCZVariable<double>& wcell_YvelD     = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_YvelD"));
      SFCZVariable<double>& wcell_ZvelD     = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_ZvelD"));
      //U2D
      // SFCXVariable<double>& u2D_ctr    = *(tsk_info->get_uintah_field<SFCXVariable<double> >("u2D_ctr"));
      // SFCYVariable<double>& v2D_ctr   =  *(tsk_info->get_uintah_field<SFCYVariable<double> >("v2D_ctr"));
      // SFCZVariable<double>& w2D_ctr    = *(tsk_info->get_uintah_field<SFCZVariable<double> >("w2D_ctr"));
      uD_ctr.initialize(0.0);
      ucell_XvelD.initialize(0.0);
      ucell_YvelD.initialize(0.0);
      ucell_ZvelD.initialize(0.0);
      vD_ctr.initialize(0.0);
      vcell_XvelD.initialize(0.0);
      vcell_YvelD.initialize(0.0);
      vcell_ZvelD.initialize(0.0);
      wD_ctr.initialize(0.0);
      wcell_XvelD.initialize(0.0);
      wcell_YvelD.initialize(0.0);
      wcell_ZvelD.initialize(0.0);
      //
      SFCXVariable<double>& uustress    = *(tsk_info->get_uintah_field<SFCXVariable<double> >("uustress"));
      SFCXVariable<double>& uvstress   =  *(tsk_info->get_uintah_field<SFCXVariable<double> >("uvstress"));
      SFCXVariable<double>& uwstress    = *(tsk_info->get_uintah_field<SFCXVariable<double> >("uwstress"));
      //
      SFCYVariable<double>& vustress    = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vustress"));
      SFCYVariable<double>& vvstress   =  *(tsk_info->get_uintah_field<SFCYVariable<double> >("vvstress"));
      SFCYVariable<double>& vwstress    = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vwstress"));
      //
      SFCZVariable<double>& wustress    = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wustress"));
      SFCZVariable<double>& wvstress   =  *(tsk_info->get_uintah_field<SFCZVariable<double> >("wvstress"));
      SFCZVariable<double>& wwstress    = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wwstress"));
      uustress.initialize(0.0);
      uvstress.initialize(0.0);
      uwstress.initialize(0.0);
      vustress.initialize(0.0);
      vvstress.initialize(0.0);
      vwstress.initialize(0.0);
      wustress.initialize(0.0);
      wvstress.initialize(0.0);
      wwstress.initialize(0.0);




    }

  //---------------------------------------------------------------------------------
  void
    FractalUD::register_timestep_init( std::vector<AFC::VariableInformation>&
        variable_registry , const bool packed_tasks){
    }

  //---------------------------------------------------------------------------------
  void
    FractalUD::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
    }

  //---------------------------------------------------------------------------------
  void
    FractalUD::register_timestep_eval( std::vector<AFC::VariableInformation>&
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
      // UPDATE USER DEFINE VARIABLES
      // register Velocity Delta
      for (auto iter = m_VelDelta_names.begin(); iter != m_VelDelta_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry, time_substep,m_task_name );
      }

    }

  //---------------------------------------------------------------------------------
  void
    FractalUD::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

      constSFCXVariable<double>& uFaceX =tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >(Ux_face_name);
      constSFCXVariable<double>& uFaceY =tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >(Uy_face_name);
      constSFCXVariable<double>& uFaceZ =tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >(Uz_face_name);
      constSFCYVariable<double>& vFaceX =tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >(Vx_face_name);
      constSFCYVariable<double>& vFaceY =tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >(Vy_face_name);
      constSFCYVariable<double>& vFaceZ =tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >(Vz_face_name);
      constSFCZVariable<double>& wFaceX =tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >(Wx_face_name);
      constSFCZVariable<double>& wFaceY =tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >(Wy_face_name);
      constSFCZVariable<double>& wFaceZ =tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >(Wz_face_name);
      constSFCXVariable<double>& U_ctr = tsk_info->get_const_uintah_field_add<constSFCXVariable<double> > (m_u_vel_name);
      constSFCYVariable<double>& V_ctr = tsk_info->get_const_uintah_field_add<constSFCYVariable<double> > (m_v_vel_name);
      constSFCZVariable<double>& W_ctr = tsk_info->get_const_uintah_field_add<constSFCZVariable<double> > (m_w_vel_name);

      // UD
      SFCXVariable<double>& uD_ctr          = tsk_info->get_uintah_field_add<SFCXVariable<double> >(m_VelDelta_names[0]);
      SFCXVariable<double>& ucell_XvelD     = tsk_info->get_uintah_field_add<SFCXVariable<double> >("ucell_XvelD");
      SFCXVariable<double>& ucell_YvelD     = tsk_info->get_uintah_field_add<SFCXVariable<double> >("ucell_YvelD");
      SFCXVariable<double>& ucell_ZvelD     = tsk_info->get_uintah_field_add<SFCXVariable<double> >("ucell_ZvelD");

      SFCYVariable<double>& vD_ctr          = tsk_info->get_uintah_field_add<SFCYVariable<double> >("vD_ctr");
      SFCYVariable<double>& vcell_XvelD     = tsk_info->get_uintah_field_add<SFCYVariable<double> >("vcell_XvelD");
      SFCYVariable<double>& vcell_YvelD     = tsk_info->get_uintah_field_add<SFCYVariable<double> >("vcell_YvelD");
      SFCYVariable<double>& vcell_ZvelD     = tsk_info->get_uintah_field_add<SFCYVariable<double> >("vcell_ZvelD");

      SFCZVariable<double>& wD_ctr          = tsk_info->get_uintah_field_add<SFCZVariable<double> >("wD_ctr");
      SFCZVariable<double>& wcell_XvelD     = tsk_info->get_uintah_field_add<SFCZVariable<double> >("wcell_XvelD");
      SFCZVariable<double>& wcell_YvelD     = tsk_info->get_uintah_field_add<SFCZVariable<double> >("wcell_YvelD");
      SFCZVariable<double>& wcell_ZvelD     = tsk_info->get_uintah_field_add<SFCZVariable<double> >("wcell_ZvelD");
      //U2D
      SFCXVariable<double>& u2D_ctr         = tsk_info->get_uintah_field_add<SFCXVariable<double> >("u2D_ctr");
      SFCYVariable<double>& v2D_ctr         = tsk_info->get_uintah_field_add<SFCYVariable<double> >("v2D_ctr");
      SFCZVariable<double>& w2D_ctr         = tsk_info->get_uintah_field_add<SFCZVariable<double> >("w2D_ctr");
      //
      SFCXVariable<double>& uustress    = *(tsk_info->get_uintah_field<SFCXVariable<double> >("uustress"));
      SFCXVariable<double>& uvstress   =  *(tsk_info->get_uintah_field<SFCXVariable<double> >("uvstress"));
      SFCXVariable<double>& uwstress    = *(tsk_info->get_uintah_field<SFCXVariable<double> >("uwstress"));
      //
      SFCYVariable<double>& vustress    = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vustress"));
      SFCYVariable<double>& vvstress   =  *(tsk_info->get_uintah_field<SFCYVariable<double> >("vvstress"));
      SFCYVariable<double>& vwstress    = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vwstress"));
      //
      SFCZVariable<double>& wustress    = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wustress"));
      SFCZVariable<double>& wvstress   =  *(tsk_info->get_uintah_field<SFCZVariable<double> >("wvstress"));
      SFCZVariable<double>& wwstress    = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wwstress"));

      std::vector<double>  dum   ={4.6296296296297682e-03,1.8518518518518452e-02,4.6296296296296207e-03,1.8518518518518462e-02,
        7.4074074074074112e-02,1.8518518518518504e-02,4.6296296296296502e-03,1.8518518518518507e-02,
        4.6296296296296623e-03,1.8518518518518434e-02,7.4074074074074139e-02,1.8518518518518511e-02,
        7.4074074074074125e-02,2.9629629629629628e-01,7.4074074074074070e-02,1.8518518518518545e-02,
        7.4074074074074084e-02,1.8518518518518483e-02,4.6296296296296294e-03,1.8518518518518500e-02,
        4.6296296296296554e-03,1.8518518518518511e-02,7.4074074074074084e-02,1.8518518518518497e-02,
        4.6296296296296398e-03,1.8518518518518500e-02,4.6296296296296658e-03 };
      std::vector< std::vector<std::vector<double>>> cell_index(3,std::vector<std::vector<double>>(3,std::vector<double>(3,0.0)));
      std::vector< std::vector<std::vector<double>>> LegData(3,std::vector<std::vector<double>>(3,std::vector<double>(3,0.0)));
      int index_cv=-1;
      for(int ii:{-1,0,1} ){
        for(int jj:{-1,0,1}){
          for(int kk:{-1,0,1}){
            index_cv=index_cv+1;
            cell_index[ii+1][jj+1][kk+1]=index_cv;
          } // end kk
        } // end jj
      } // end ii
      int index=0;
      for(int ii:{-1,0,1} ){
        for(int jj:{-1,0,1}){
          for(int kk:{-1,0,1}){
            index=cell_index[ii+1][jj+1][kk+1];
            LegData[ii+1][jj+1][kk+1]=dum[index];
          } // end kk
        } // end jj
      } // end ii

      Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );

      Uintah::parallel_for( range, [&](int i, int j, int k){
          double sums=0.0;
          //ucell-center
          // filter the velocity with Legendre filter( 2004 paper(I)-eq.28)
          // filtered velocity on the staggered location.
          LegScaleSepU(U_ctr,  i,j,k,  u2D_ctr(i,j,k),uD_ctr(i,j,k),       LegData );
          //ucell-xface
          LegScaleSepU(uFaceX, i,j,k,  sums ,         ucell_XvelD(i,j,k),  LegData );
          //ucell-yface
          LegScaleSepU(uFaceY, i,j,k,  sums ,         ucell_YvelD(i,j,k),  LegData );
          //ucell-zface
          LegScaleSepU(uFaceZ, i,j,k,  sums ,         ucell_ZvelD(i,j,k),  LegData );
          //vcell-center
          LegScaleSepU(V_ctr,  i,j,k,  v2D_ctr(i,j,k),vD_ctr(i,j,k),       LegData );
          //vcell-xface
          LegScaleSepU(vFaceX, i,j,k,  sums ,         vcell_XvelD(i,j,k),  LegData );
          //vcell-yface
          LegScaleSepU(vFaceY, i,j,k,  sums ,         vcell_YvelD(i,j,k),  LegData );
          //vcell-zface
          LegScaleSepU(vFaceZ, i,j,k,  sums ,         vcell_ZvelD(i,j,k),  LegData );
          //wcell-center
          LegScaleSepU(W_ctr,  i,j,k,  w2D_ctr(i,j,k),wD_ctr(i,j,k),       LegData );
          //wcell-xface
          LegScaleSepU(wFaceX, i,j,k,  sums ,         wcell_XvelD(i,j,k),  LegData );
          //wcell-yface
          LegScaleSepU(wFaceY, i,j,k,  sums ,         wcell_YvelD(i,j,k),  LegData );
          //wcell-zface
          LegScaleSepU(wFaceZ, i,j,k,  sums ,         wcell_ZvelD(i,j,k),  LegData );

          uustress(i,j,k)  = uFaceX(i,j,k)*uFaceX(i,j,k);
          uvstress(i,j,k)  = vFaceX(i,j,k)*uFaceY(i,j,k);
          uwstress(i,j,k)  = wFaceX(i,j,k)*uFaceZ(i,j,k);

          vustress(i,j,k)  =uFaceY(i,j,k)*vFaceX(i,j,k);
          vvstress(i,j,k)  =vFaceY(i,j,k)*vFaceY(i,j,k);
          vwstress(i,j,k)  =wFaceY(i,j,k)*vFaceZ(i,j,k);

          wustress(i,j,k)  =uFaceZ(i,j,k)*wFaceX(i,j,k);
          wvstress(i,j,k)  =vFaceZ(i,j,k)*wFaceY(i,j,k);
          wwstress(i,j,k)  =wFaceZ(i,j,k)*wFaceZ(i,j,k);

      });

    }
} //namespace Uintah
