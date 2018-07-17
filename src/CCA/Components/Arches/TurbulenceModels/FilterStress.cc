#include <CCA/Components/Arches/TurbulenceModels/FilterStress.h>
#include <CCA/Components/Arches/GridTools.h>

namespace Uintah{

  typedef ArchesFieldContainer AFC;

  //---------------------------------------------------------------------------------
  FilterStress::FilterStress( std::string task_name, int matl_index ) :
    TaskInterface( task_name, matl_index )
  {
    U_ctr_name = "uVelocitySPBC";
    V_ctr_name = "vVelocitySPBC";
    W_ctr_name = "wVelocitySPBC";
    // Create SGS stress
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
    // Filter SGS stress
    m_FilterStress_names.resize(9);
    m_FilterStress_names[0] = "ucell_xFilterStress";
    m_FilterStress_names[1] = "ucell_yFilterStress";
    m_FilterStress_names[2] = "ucell_zFilterStress";
    m_FilterStress_names[3] = "vcell_xFilterStress";
    m_FilterStress_names[4] = "vcell_yFilterStress";
    m_FilterStress_names[5] = "vcell_zFilterStress";
    m_FilterStress_names[6] = "wcell_xFilterStress";
    m_FilterStress_names[7] = "wcell_yFilterStress";
    m_FilterStress_names[8] = "wcell_zFilterStress";


  }

  //---------------------------------------------------------------------------------
  FilterStress::~FilterStress()
  {}

  //---------------------------------------------------------------------------------
  void
    FilterStress::problemSetup( ProblemSpecP& db ){

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
          msg << "ERROR: Constant FilterStress: problemSetup(): Zero viscosity specified \n"
            << "       in <PhysicalConstants> section of input file." << std::endl;
          throw InvalidValue(msg.str(),__FILE__,__LINE__);
        }
      } else {
        std::stringstream msg;
        msg << "ERROR: Constant FilterStress: problemSetup(): Missing <PhysicalConstants> \n"
          << "       section in input file!" << std::endl;
        throw InvalidValue(msg.str(),__FILE__,__LINE__);
      }

    }

  //---------------------------------------------------------------------------------
  void
    FilterStress::create_local_labels(){

      //U-CELL LABELS:
      register_new_variable<SFCXVariable<double> >(m_FilterStress_names[0]);
      register_new_variable<SFCXVariable<double> >(m_FilterStress_names[1]);
      register_new_variable<SFCXVariable<double> >(m_FilterStress_names[2]);
      // V-cell
      register_new_variable<SFCYVariable<double> >(m_FilterStress_names[3]);
      register_new_variable<SFCYVariable<double> >(m_FilterStress_names[4]);
      register_new_variable<SFCYVariable<double> >(m_FilterStress_names[5]);
      // W-cell
      register_new_variable<SFCZVariable<double> >(m_FilterStress_names[6]);
      register_new_variable<SFCZVariable<double> >(m_FilterStress_names[7]);
      register_new_variable<SFCZVariable<double> >(m_FilterStress_names[8]);


    }

  //---------------------------------------------------------------------------------
  void
    FilterStress::register_initialize( std::vector<AFC::VariableInformation>&
        variable_registry , const bool packed_tasks){
      for (auto iter = m_FilterStress_names.begin(); iter != m_FilterStress_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry, _task_name  );
      }

      //  register_variable( m_t_vis_name, AFC::COMPUTES, variable_registry );

      // register Velocity Delta
    }

  //---------------------------------------------------------------------------------
  void
    FilterStress::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

      SFCXVariable<double>&  ucell_xFilterStress     = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_xFilterStress"));
      SFCXVariable<double>&  ucell_yFilterStress     = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_yFilterStress"));
      SFCXVariable<double>&  ucell_zFilterStress     = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_zFilterStress"));
      SFCYVariable<double>&  vcell_xFilterStress     = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_xFilterStress"));
      SFCYVariable<double>&  vcell_yFilterStress     = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_yFilterStress"));
      SFCYVariable<double>&  vcell_zFilterStress     = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_zFilterStress"));
      SFCZVariable<double>&  wcell_xFilterStress     = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_xFilterStress"));
      SFCZVariable<double>&  wcell_yFilterStress     = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_yFilterStress"));
      SFCZVariable<double>&  wcell_zFilterStress     = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_zFilterStress"));
      ucell_xFilterStress.initialize(0.0);
      ucell_yFilterStress.initialize(0.0);
      ucell_zFilterStress.initialize(0.0);
      vcell_xFilterStress.initialize(0.0);
      vcell_yFilterStress.initialize(0.0);
      vcell_zFilterStress.initialize(0.0);
      wcell_xFilterStress.initialize(0.0);
      wcell_yFilterStress.initialize(0.0);
      wcell_zFilterStress.initialize(0.0);

    }

  //---------------------------------------------------------------------------------
  void
    FilterStress::register_timestep_init( std::vector<AFC::VariableInformation>&
        variable_registry , const bool packed_tasks){
      //for (auto iter = m_VelDelta_names.begin(); iter != m_VelDelta_names.end(); iter++ ){
      //register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry, _task_name );
      //}
      for (auto iter = m_FilterStress_names.begin(); iter != m_FilterStress_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry, _task_name  );
      }

      // register_variable( m_t_vis_name, AFC::COMPUTES, variable_registry );

    }

  //---------------------------------------------------------------------------------
  void
    FilterStress::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

      SFCXVariable<double>&  ucell_xFilterStress     = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_xFilterStress"));
      SFCXVariable<double>&  ucell_yFilterStress     = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_yFilterStress"));
      SFCXVariable<double>&  ucell_zFilterStress     = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_zFilterStress"));
      SFCYVariable<double>&  vcell_xFilterStress     = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_xFilterStress"));
      SFCYVariable<double>&  vcell_yFilterStress     = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_yFilterStress"));
      SFCYVariable<double>&  vcell_zFilterStress     = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_zFilterStress"));
      SFCZVariable<double>&  wcell_xFilterStress     = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_xFilterStress"));
      SFCZVariable<double>&  wcell_yFilterStress     = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_yFilterStress"));
      SFCZVariable<double>&  wcell_zFilterStress     = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_zFilterStress"));

    }

  //---------------------------------------------------------------------------------
  void
    FilterStress::register_timestep_eval( std::vector<AFC::VariableInformation>&
        variable_registry, const int time_substep , const bool packed_tasks){
      //register_variable( Ux_face_name,    ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::LATEST, variable_registry,time_substep);
      //register_variable( "density",       ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::OLDDW, variable_registry,time_substep );
      //register_variable( "volFraction",   ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::OLDDW, variable_registry,time_substep );
      // UPDATE USER DEFINE VARIABLES
      // register Velocity Delta
      for (auto iter = m_SgsStress_names.begin(); iter != m_SgsStress_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry,time_substep );
      }

      for (auto iter = m_FilterStress_names.begin(); iter != m_FilterStress_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::MODIFIES, variable_registry, _task_name  );
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
    FilterStress::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
      //std::cout<<"zhou stupid 1 \n"<<std::endl;

      // Subgrid stress
      constSFCXVariable<double>& ucell_xSgsStress = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("ucell_xSgsStress"));
      constSFCXVariable<double>& ucell_ySgsStress = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("ucell_ySgsStress"));
      constSFCXVariable<double>& ucell_zSgsStress = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("ucell_zSgsStress"));
      constSFCYVariable<double>& vcell_xSgsStress = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("vcell_xSgsStress"));
      constSFCYVariable<double>& vcell_ySgsStress = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("vcell_ySgsStress"));
      constSFCYVariable<double>& vcell_zSgsStress = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("vcell_zSgsStress"));
      constSFCZVariable<double>& wcell_xSgsStress = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("wcell_xSgsStress"));
      constSFCZVariable<double>& wcell_ySgsStress = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("wcell_ySgsStress"));
      constSFCZVariable<double>& wcell_zSgsStress = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("wcell_zSgsStress"));

      SFCXVariable<double>&  ucell_xFilterStress     = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_xFilterStress"));
      SFCXVariable<double>&  ucell_yFilterStress     = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_yFilterStress"));
      SFCXVariable<double>&  ucell_zFilterStress     = *(tsk_info->get_uintah_field<SFCXVariable<double> >("ucell_zFilterStress"));
      SFCYVariable<double>&  vcell_xFilterStress     = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_xFilterStress"));
      SFCYVariable<double>&  vcell_yFilterStress     = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_yFilterStress"));
      SFCYVariable<double>&  vcell_zFilterStress     = *(tsk_info->get_uintah_field<SFCYVariable<double> >("vcell_zFilterStress"));
      SFCZVariable<double>&  wcell_xFilterStress     = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_xFilterStress"));
      SFCZVariable<double>&  wcell_yFilterStress     = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_yFilterStress"));
      SFCZVariable<double>&  wcell_zFilterStress     = *(tsk_info->get_uintah_field<SFCZVariable<double> >("wcell_zFilterStress"));



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
            //LegData[ii+1][jj+1][kk+1]=dum[index];
            //std::cout<<index_cv<<"\n"<<"i="<<ii <<" j="<<jj <<" k="<<kk<<"Legdata="<<LegData[ii+1][jj+1][kk+1]<<"\n";
          } // end kk
        } // end jj
      } // end ii
      int index=0;
      for(int ii:{-1,0,1} ){
        for(int jj:{-1,0,1}){
          for(int kk:{-1,0,1}){
            index=cell_index[ii+1][jj+1][kk+1];
            LegData[ii+1][jj+1][kk+1]=dum[index];
            //     std::cout<<index<<"\n"<<"i="<<ii <<" j="<<jj <<" k="<<kk<<"Legdata="<<LegData[ii+1][jj+1][kk+1]<<"\n";
          } // end kk
        } // end jj
      } // end ii

      Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );

      // Uintah::parallel_for( range, [&](int i, int j, int k){
      //     double sums=0.0;
      //    //ucell-center
      //   //if(i==2&& j==2 && k==2)
      //   //{  std::cout<<" Uctr_at_cell_center="<<" i= "<<i<<" j= "<<j<<" k="<<k<<"\n";
      //     //std::cout<<"U_resolved(2,2,2)="<<U_ctr(i,j,k) <<" UD(2,2,2)="<<uD_ctr(i,j,k)<<"U2D="<<u2D_ctr(i,j,k)<<"\n";
      //   //}
      //    //ucell-xface
      //   LegScaleSepU(ucell_xSgsStress, i,j,k,      ucell_xFilterStress(i,j,k),  LegData );
      //    //ucell-yface
      //   LegScaleSepU(ucell_ySgsStress, i,j,k,      ucell_yFilterStress(i,j,k),  LegData );
      //   //ucell-zface
      //   LegScaleSepU(ucell_zSgsStress, i,j,k,      ucell_zFilterStress(i,j,k),  LegData );
      //
      //   LegScaleSepU(vcell_xSgsStress, i,j,k,      vcell_xFilterStress(i,j,k),  LegData );
      //   LegScaleSepU(vcell_ySgsStress, i,j,k,      vcell_yFilterStress(i,j,k),  LegData );
      //   LegScaleSepU(vcell_zSgsStress, i,j,k,      vcell_zFilterStress(i,j,k),  LegData );
      //
      //   LegScaleSepU(wcell_xSgsStress, i,j,k,      wcell_xFilterStress(i,j,k),  LegData );
      //   LegScaleSepU(wcell_ySgsStress, i,j,k,      wcell_yFilterStress(i,j,k),  LegData );
      //   LegScaleSepU(wcell_zSgsStress, i,j,k,      wcell_zFilterStress(i,j,k),  LegData );
      //
      // });

    }
} //namespace Uintah
