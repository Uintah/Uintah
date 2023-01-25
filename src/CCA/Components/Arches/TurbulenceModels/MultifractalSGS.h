#ifndef Uintah_Component_Arches_MultifractalSGS_h
#define Uintah_Component_Arches_MultifractalSGS_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class MultifractalSGS : public TaskInterface {

    public:

      MultifractalSGS( std::string task_name, int matl_index );
      ~MultifractalSGS();

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
      void Strain_calc( const Array3<double> &U,  const Array3<double>  &V, const Array3<double>  & W,
          int i, int j, int k ,      std::vector<double> &StrainOut , double &dx, double &dy, double dz)
      { //double c1=0.0; double c2=1.0;
        double c1=27.0/24.0; double c2=1.0/24.0;
        double dudy=0.0; double dvdx=0.0; double dvdz=0.0; double dwdy=0.0; double dwdx=0.0; double dudz=0.0;
        dudy=        (c1*(U(i,j,k)-U(i,j-1,k))-c2*(U(i,j+1,k)-U(i,j-2,k)))/dy;
        dvdx=        (c1*(V(i,j,k)-V(i-1,j,k))-c2*(V(i+1,j,k)-V(i-2,j,k)))/dx;
        dvdz=        (c1*(V(i,j,k)-V(i,j,k-1))-c2*(V(i,j,k+1)-V(i,j,k-2)))/dz;
        dwdy=        (c1*(W(i,j,k)-W(i,j-1,k))-c2*(W(i,j+1,k)-W(i,j-2,k)))/dy;
        dwdx=        (c1*(W(i,j,k)-W(i-1,j,k))-c2*(W(i+1,j,k)-W(i-2,j,k)))/dx;
        dudz=        (c1*(U(i,j,k)-U(i,j,k-1))-c2*(U(i,j,k+1)-U(i,j,k-2)))/dz;
        StrainOut[0]=(c1*(U(i,j,k)-U(i-1,j,k))-c2*(U(i+1,j,k)-U(i-2,j,k)))/dx;
        StrainOut[1]=(c1*(V(i,j,k)-V(i,j-1,k))-c2*(V(i,j+1,k)-V(i,j-2,k)))/dy;
        StrainOut[2]=(c1*(W(i,j,k)-W(i,j,k-1))-c2*(W(i,j,k+1)-W(i,j,k-2)))/dz;
        StrainOut[3]=0.5*(dudy+dvdx);
        StrainOut[4]=0.5*(dvdz+dwdy) ;
        StrainOut[5]=0.5*(dudz+dwdx);

      };// end for calcuting Strain for ud u2d u_ctr

      double sgsVelCoeff( double &Visc, double &MeshSize,  double &StrainD,
          double dx, double dy, double dz,
          double &sgs_scales,double &factorN, double &value, double &Re_g)
      { //double base_scale=0.5;
        double cascade_iters= 0.0;
        //double factorN=0.0;
        Re_g=StrainD*MeshSize*MeshSize/Visc;
        Re_g=Re_g*Re_g*Re_g;
        Re_g=sqrt(sqrt(Re_g));
        sgs_scales=0.089285714285714*Re_g;
        cascade_iters= sgs_scales >1.0 ? std::log2(sgs_scales) : 0.0 ;
        factorN= (cascade_iters>0) ? exp2(-2.0/3.0 * cascade_iters)*sqrt(exp2(4.0/3.0*cascade_iters)-1) : 0.0;
        value =0.75;
        double C_sgs_A=value*factorN;
        return C_sgs_A;
      };// end for calculating the velocity coeffficient

      //Build instructions for this (MultifractalSGS) class.

      void filterOperator(const Array3<double> &var, int i, int j, int k , double &F_var )
      {
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


        F_var=0.0;
        for ( int m = -1; m <= 1; m++ ){
          for ( int n = -1; n <= 1; n++ ){
            for ( int l = -1; l <= 1; l++ ){
              F_var += LegData[m+1][n+1][l+1]* var(i+m,j+n,k+l);
            }
          }
        }
      }; // end for filter

      class Builder : public TaskInterface::TaskBuilder {

        public:

          Builder( std::string task_name, int matl_index ) : m_task_name(task_name), m_matl_index(matl_index){}
          ~Builder(){}

          MultifractalSGS* build()
          { return scinew MultifractalSGS( m_task_name, m_matl_index ); }

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
      std::vector<std::string> m_SgsStress_names;

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
