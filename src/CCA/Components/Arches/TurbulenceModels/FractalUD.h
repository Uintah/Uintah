#ifndef Uintah_Component_Arches_FractalUD_h
#define Uintah_Component_Arches_FractalUD_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class FractalUD : public TaskInterface {

    public:

      FractalUD( std::string task_name, int matl_index );
      ~FractalUD();

      TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

      TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

      TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

      TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

      TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

      void problemSetup( ProblemSpecP& db );

      void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

      void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

      void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

      void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){}

      template <typename ExecSpace, typename MemSpace>
      void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

      template <typename ExecSpace, typename MemSpace>
      void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

      template <typename ExecSpace, typename MemSpace>
      void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

      template <typename ExecSpace, typename MemSpace>
      void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

      void create_local_labels();
      // Legendre scale separation to get the node and center U2D 
      // U means U cell, x means X direction velocity
      void LegScaleSepU(const Array3<double> &Vel, int i, int j, int k , double &sums , double &UD,
          std::vector< std::vector<std::vector<double>>> &LegData )
      {  
        sums=0.0;
        for(int kk:{-1,0,1} ){
          for(int jj:{-1,0,1}){
            for(int ii:{-1,0,1}){
              //index=index+1;
              //LegData[ii][jj][kk]=dum[index];
              sums =sums+LegData[ii+1][jj+1][kk+1]*Vel(i+ii,j+jj,k+kk); //UR at ctr
            } // end kk
          } // end jj
        } // end ii
        // Get UD velocity
        UD =Vel(i,j,k) -sums ; //UR at ctr
      }; // end for LegScaleSep functions

      class Builder : public TaskInterface::TaskBuilder {

        public:

          Builder( std::string task_name, int matl_index ) : m_task_name(task_name), m_matl_index(matl_index){}
          ~Builder(){}

          FractalUD* build()
          { return scinew FractalUD( m_task_name, m_matl_index ); }

        private:

          std::string m_task_name;
          int m_matl_index;
      };

    private:

      std::string U_ctr_name ;
      std::string V_ctr_name ;
      std::string W_ctr_name ;

      std::string Ux_face_name ;
      std::string Uy_face_name ;
      std::string Uz_face_name ;
      std::string Vx_face_name ;
      std::string Vy_face_name ;
      std::string Vz_face_name ;
      std::string Wx_face_name ;
      std::string Wy_face_name ;
      std::string Wz_face_name ;
      // create new UD variables at the velocity cell center 
      std::string UD_ctr_name ;
      std::string VD_ctr_name ;
      std::string WD_ctr_name ;

      std::string U2D_ctr_name ;
      std::string V2D_ctr_name ;
      std::string W2D_ctr_name ;

      std::vector<std::string> m_VelDelta_names;

      // old_wale model 
      std::string m_u_vel_name;
      std::string m_v_vel_name;
      std::string m_w_vel_name;

      std::string m_cc_u_vel_name;
      std::string m_cc_v_vel_name;
      std::string m_cc_w_vel_name;
      double m_Cs; //Wale constant
      double m_molecular_visc;
      std::string m_t_vis_name;

      int Nghost_cells;

  };
}
#endif
