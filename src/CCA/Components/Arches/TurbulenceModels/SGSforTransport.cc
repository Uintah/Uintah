#include <CCA/Components/Arches/TurbulenceModels/SGSforTransport.h>

namespace Uintah{

  //--------------------------------------------------------------------------------------------------
  SGSforTransport::SGSforTransport( std::string task_name, int matl_index ) :
    TaskInterface( task_name, matl_index ) {
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
      // register source terms
      m_fmom_source_names.resize(3);
      m_fmom_source_names[0]="FractalXSrc";
      m_fmom_source_names[1]="FractalYSrc";
      m_fmom_source_names[2]="FractalZSrc";
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

  //--------------------------------------------------------------------------------------------------
  SGSforTransport::~SGSforTransport(){
  }

  //--------------------------------------------------------------------------------------------------
  void
    SGSforTransport::problemSetup( ProblemSpecP& db ){

      using namespace Uintah::ArchesCore;
      // u, v , w velocities
      m_u_vel_name = parse_ups_for_role( UVELOCITY, db, "uVelocitySPBC" );
      m_v_vel_name = parse_ups_for_role( VVELOCITY, db, "vVelocitySPBC" );
      m_w_vel_name = parse_ups_for_role( WVELOCITY, db, "wVelocitySPBC" );
      m_density_name = parse_ups_for_role( DENSITY, db, "density" );

      m_rhou_vel_name = "x-mom";
      m_rhov_vel_name = "y-mom";
      m_rhow_vel_name = "z-mom" ;

      m_cc_u_vel_name = m_u_vel_name + "_cc";
      m_cc_v_vel_name = m_v_vel_name + "_cc";
      m_cc_w_vel_name = m_w_vel_name + "_cc";


    }

  //--------------------------------------------------------------------------------------------------
  void
    SGSforTransport::create_local_labels(){
      // U-cell labels
      register_new_variable<SFCXVariable<double> >("FractalXSrc");
      //V-CELL LABELS:
      register_new_variable<SFCYVariable<double> >("FractalYSrc");
      //W-CELL LABELS:
      register_new_variable<SFCZVariable<double> >("FractalZSrc");


    }

  //--------------------------------------------------------------------------------------------------
  void
    SGSforTransport::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>&
        variable_registry , const bool packed_tasks){
      for (auto iter = m_fmom_source_names.begin(); iter != m_fmom_source_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry, _task_name );
      }

      //register_variable( "FractalXSrc", ArchesFieldContainer::COMPUTES ,  variable_registry,  _task_name, packed_tasks);
      //register_variable( "FractalYSrc", ArchesFieldContainer::COMPUTES ,  variable_registry,  _task_name, packed_tasks);
      //register_variable( "FractalZSrc", ArchesFieldContainer::COMPUTES ,  variable_registry,  _task_name, packed_tasks);
    }

  //--------------------------------------------------------------------------------------------------
  void
    SGSforTransport::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
      SFCXVariable<double>&  FractalXSrc= tsk_info->get_uintah_field_add<SFCXVariable<double> >("FractalXSrc");
      SFCYVariable<double>&  FractalYSrc= tsk_info->get_uintah_field_add<SFCYVariable<double> >("FractalYSrc");
      SFCZVariable<double>&  FractalZSrc= tsk_info->get_uintah_field_add<SFCZVariable<double> >("FractalZSrc");

      FractalXSrc.initialize(0.0);
      FractalYSrc.initialize(0.0);
      FractalZSrc.initialize(0.0);
    }
  //--------------------------------------------------------------------------------------------------
  void
    SGSforTransport::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>&
        variable_registry , const bool packed_tasks){
      for (auto iter = m_fmom_source_names.begin(); iter != m_fmom_source_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::COMPUTES, variable_registry, _task_name );
      }
      //register_variable( "FractalXSrc", ArchesFieldContainer::COMPUTES ,  variable_registry,  _task_name, packed_tasks);
      //register_variable( "FractalYSrc", ArchesFieldContainer::COMPUTES ,  variable_registry,  _task_name, packed_tasks);
      //register_variable( "FractalZSrc", ArchesFieldContainer::COMPUTES ,  variable_registry,  _task_name, packed_tasks);

    }

  //--------------------------------------------------------------------------------------------------
  void
    SGSforTransport::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
      SFCXVariable<double>&  FractalXSrc= tsk_info->get_uintah_field_add<SFCXVariable<double> >("FractalXSrc");
      SFCYVariable<double>&  FractalYSrc= tsk_info->get_uintah_field_add<SFCYVariable<double> >("FractalYSrc");
      SFCZVariable<double>&  FractalZSrc= tsk_info->get_uintah_field_add<SFCZVariable<double> >("FractalZSrc");
      //SFCXVariable<double>&  srcx= tsk_info->get_uintah_field_add<SFCXVariable<double> >("FractalXSrc");
      //SFCYVariable<double>&  srcy= tsk_info->get_uintah_field_add<SFCYVariable<double> >("FractalYSrc");
      //SFCZVariable<double>&  srcz= tsk_info->get_uintah_field_add<SFCZVariable<double> >("FractalZSrc");

      //srcx.initialize(0.0);
      //srcy.initialize(0.0);
      //srcz.initialize(0.0);

    }

  //--------------------------------------------------------------------------------------------------
  void
    SGSforTransport::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>&
        variable_registry, const int time_substep , const bool packed_tasks){

      for (auto iter = m_FilterStress_names.begin(); iter != m_FilterStress_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry, _task_name );
      }

      for (auto iter = m_SgsStress_names.begin(); iter != m_SgsStress_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::REQUIRES, 2, ArchesFieldContainer::NEWDW, variable_registry, _task_name );
      }
      for (auto iter = m_fmom_source_names.begin(); iter != m_fmom_source_names.end(); iter++ ){
        register_variable( *iter, ArchesFieldContainer::MODIFIES, variable_registry, _task_name );
      }

      //register_variable( "FractalXSrc", ArchesFieldContainer::COMPUTES ,  variable_registry,  _task_name, packed_tasks);
      //register_variable( "FractalYSrc", ArchesFieldContainer::COMPUTES ,  variable_registry,  _task_name, packed_tasks);
      //register_variable( "FractalZSrc", ArchesFieldContainer::COMPUTES ,  variable_registry,  _task_name, packed_tasks);


    }

  //--------------------------------------------------------------------------------------------------
  void
    SGSforTransport::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
      Vector Dx=patch->dCell();
      //  double dx=Dx.x(); double dy=Dx.y(); double dz=Dx.z();
      double vol = Dx.x()*Dx.y()*Dx.z();
      double Area_NS =Dx.x()*Dx.z(); 
      double Area_EW =Dx.y()*Dx.z(); 
      double Area_TB =Dx.x()*Dx.y();
      double densitygas=1.0;//kg/m3
      double cellvol=1.0/(Dx.x()*Dx.y()*Dx.z());
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
      // Subgrid stress
      constSFCXVariable<double>& ucell_xFilterStress = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("ucell_xFilterStress"));
      constSFCXVariable<double>& ucell_yFilterStress = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("ucell_yFilterStress"));
      constSFCXVariable<double>& ucell_zFilterStress = *(tsk_info->get_const_uintah_field<constSFCXVariable<double> >("ucell_zFilterStress"));
      constSFCYVariable<double>& vcell_xFilterStress = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("vcell_xFilterStress"));
      constSFCYVariable<double>& vcell_yFilterStress = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("vcell_yFilterStress"));
      constSFCYVariable<double>& vcell_zFilterStress = *(tsk_info->get_const_uintah_field<constSFCYVariable<double> >("vcell_zFilterStress"));
      constSFCZVariable<double>& wcell_xFilterStress = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("wcell_xFilterStress"));
      constSFCZVariable<double>& wcell_yFilterStress = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("wcell_yFilterStress"));
      constSFCZVariable<double>& wcell_zFilterStress = *(tsk_info->get_const_uintah_field<constSFCZVariable<double> >("wcell_zFilterStress"));
      //SFCXVariable<double>&  srcx= tsk_info->get_uintah_field_add<SFCXVariable<double> >("FractalXSrc");
      //SFCYVariable<double>&  srcy= tsk_info->get_uintah_field_add<SFCYVariable<double> >("FractalYSrc");
      //SFCZVariable<double>&  srcz= tsk_info->get_uintah_field_add<SFCZVariable<double> >("FractalZSrc");
      SFCXVariable<double>&  FractalXSrc= tsk_info->get_uintah_field_add<SFCXVariable<double> >("FractalXSrc");
      SFCYVariable<double>&  FractalYSrc= tsk_info->get_uintah_field_add<SFCYVariable<double> >("FractalYSrc");
      SFCZVariable<double>&  FractalZSrc= tsk_info->get_uintah_field_add<SFCZVariable<double> >("FractalZSrc");

      //FractalXSrc.initialize(0.0);
      //FractalYSrc.initialize(0.0);
      //FractalZSrc.initialize(0.0);


      Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );

      Uintah::parallel_for( range, [&](int i, int j, int k){

          //         X-momentum
          //FractalXSrc(i,j,k)=-cellvol*((ucell_xFilterStress(i+1,j,k)-ucell_xFilterStress(i,j,k))*Area_EW+(ucell_yFilterStress(i,j+1,k) -ucell_yFilterStress(i,j,k))*Area_NS+(ucell_zFilterStress(i,j,k+1)-ucell_zFilterStress(i,j,k))*Area_TB);
          //// Y-momentum
          //FractalYSrc(i,j,k)=-cellvol*((vcell_xFilterStress(i+1,j,k)-vcell_xFilterStress(i,j,k))*Area_EW+(vcell_yFilterStress(i,j+1,k) -vcell_yFilterStress(i,j,k))*Area_NS+(vcell_zFilterStress(i,j,k+1)-vcell_zFilterStress(i,j,k))*Area_TB);
          //// Z-momentum
          //FractalZSrc(i,j,k)=-cellvol*((wcell_xFilterStress(i+1,j,k)-wcell_xFilterStress(i,j,k))*Area_EW+(wcell_yFilterStress(i,j+1,k) -wcell_yFilterStress(i,j,k))*Area_NS+(wcell_zFilterStress(i,j,k+1)-wcell_zFilterStress(i,j,k))*Area_TB);

          // compute the source term for the finite volume method, not using the overbar filter again.
          //   X-momentum
          FractalXSrc(i,j,k)=-densitygas*cellvol*((ucell_xSgsStress(i+1,j,k)-ucell_xSgsStress(i,j,k))*Area_EW+(ucell_ySgsStress(i,j+1,k) -ucell_ySgsStress(i,j,k))*Area_NS+(ucell_zSgsStress(i,j,k+1)-ucell_zSgsStress(i,j,k))*Area_TB);
          //// Y-momentum
          FractalYSrc(i,j,k)=-densitygas*cellvol*((vcell_xSgsStress(i+1,j,k)-vcell_xSgsStress(i,j,k))*Area_EW+(vcell_ySgsStress(i,j+1,k) -vcell_ySgsStress(i,j,k))*Area_NS+(vcell_zSgsStress(i,j,k+1)-vcell_zSgsStress(i,j,k))*Area_TB);
          //// Z-momentum
          FractalZSrc(i,j,k)=-densitygas*cellvol*((wcell_xSgsStress(i+1,j,k)-wcell_xSgsStress(i,j,k))*Area_EW+(wcell_ySgsStress(i,j+1,k) -wcell_ySgsStress(i,j,k))*Area_NS+(wcell_zSgsStress(i,j,k+1)-wcell_zSgsStress(i,j,k))*Area_TB);

          //if(i==32&&j==32&&k==32)
          //{std::cout<<" unfilter stress="<<" i= "<<i<<" j= "<<j<<" k="<<k<<"\n"; 
          //std::cout<<"\n ucell_xSgsStress(i,j,k) After="<<ucell_xSgsStress(i,j,k) <<"\n";
          //std::cout<<"\n ucell_ySgsStress(i,j,k) After="<<ucell_ySgsStress(i,j,k) <<"\n";
          //std::cout<<"\n ucell_zSgsStress(i,j,k) After="<<ucell_zSgsStress(i,j,k) <<"\n";
          //std::cout<<"\n vcell_ySgsStress(i,j,k) After="<<vcell_ySgsStress(i,j,k)  <<"\n";
          //std::cout<<"\n vcell_xSgsStress(i,j,k) After="<<vcell_xSgsStress(i,j,k) <<"\n";
          //std::cout<<"\n vcell_zSgsStress(i,j,k) After="<<vcell_zSgsStress(i,j,k) <<"\n";
          //std::cout<<"\n wcell_zSgsStress(i,j,k) After="<<wcell_zSgsStress(i,j,k) <<"\n";
          //std::cout<<"\n wcell_xSgsStress(i,j,k) After="<<wcell_xSgsStress(i,j,k) <<"\n";
          //std::cout<<"\n wcell_ySgsStress(i,j,k) After="<<wcell_ySgsStress(i,j,k) <<"\n";
          //}
          //if(i==2&&j==2&&k==2)
          //{std::cout<<"  filter stress="<<" i= "<<i<<" j= "<<j<<" k="<<k<<"\n"; 
          //std::cout<<"\n ucell_xFilterStress(i,j,k) After="<<ucell_xFilterStress(i,j,k) <<"\n";
          //std::cout<<"\n ucell_yFilterStress(i,j,k) After="<<ucell_yFilterStress(i,j,k) <<"\n";
          //std::cout<<"\n ucell_zFilterStress(i,j,k) After="<<ucell_zFilterStress(i,j,k) <<"\n";
          //std::cout<<"\n vcell_yFilterStress(i,j,k) After="<<vcell_yFilterStress(i,j,k)  <<"\n";
          //std::cout<<"\n vcell_xFilterStress(i,j,k) After="<<vcell_xFilterStress(i,j,k) <<"\n";
          //std::cout<<"\n vcell_zFilterStress(i,j,k) After="<<vcell_zFilterStress(i,j,k) <<"\n";
          //std::cout<<"\n wcell_zFilterStress(i,j,k) After="<<wcell_zFilterStress(i,j,k) <<"\n";
          //std::cout<<"\n wcell_xFilterStress(i,j,k) After="<<wcell_xFilterStress(i,j,k) <<"\n";
          //std::cout<<"\n wcell_yFilterStress(i,j,k) After="<<wcell_yFilterStress(i,j,k) <<"\n";
          //}

          //if (i==3&& j==2 && k==2)
          //{ std::cout<<"src:"<<"\n";
          //std::cout<<"src(i,j,k)[0]= "<<FractalXSrc(i,j,k)/cellvol<<" src(i,j,k)[1]="<<FractalYSrc(i,j,k)/cellvol<<" Src(i,j,k)[2]="<<FractalZSrc(i,j,k)/cellvol <<"\n";
          //}

      });


    }

} //namespace Uintah
