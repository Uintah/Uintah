#ifndef Uintah_Component_Arches_FilterStress_h
#define Uintah_Component_Arches_FilterStress_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class FilterStress : public TaskInterface {

public:

    FilterStress( std::string task_name, int matl_index );
    ~FilterStress();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();
  // Legendre scale separation to get the node and center U2D 
  // U means U cell, x means X direction velocity
    void LegScaleSepU(const Array3<double> &Vel, int i, int j, int k , double &sums , 
                      std::vector< std::vector<std::vector<double>>> &LegData )
         {  
           sums=0.0;
       int  it=0;
       int  jt=0;
       int  kt=0;
	         for(int kk:{-1,0,1} ){
     	        for(int jj:{-1,0,1}){
		             for(int ii:{-1,0,1}){
		      //index=index+1;
			//LegData[ii][jj][kk]=dum[index];
	                it=ii+1;
		              jt=jj+1;
			            kt=kk+1;
         		        sums =sums+LegData[it][jt][kt]*Vel(i+ii,j+jj,k+kk); //UR at ctr
                    //if (i==2&&j==2&&k==2)
                    //{   std::cout<<"\n"<< "Leg verification ii="<<ii<<" jj= "<<jj<< " kk= "<<kk<< " U =LegData[it][jt][kt]*U(i+ii,j+jj,k+kk)="<<LegData[it][jt][kt]<<" Vel = "<<Vel(i+ii,j+jj,k+kk)<<" sum= "<<sums<<"\n"  ; //WR at wu nodes
                    //}
                     } // end kk
		             } // end jj
	     } // end ii
	// Get UD velocity
     //     UD =Vel(i,j,k) -sums ; //UR at ctr
        //if(i==2&& j==2 && k==2)
        //{  std::cout<<" tau="<<" i= "<<i<<" j= "<<j<<" k="<<k<<"\n"; 
          //std::cout<<"U_resolved(2,2,2)="<<Vel(i,j,k) <<" UD(2,2,2)="<<UD<<"U2D="<<sums<<"\n";
        //}
        //if(i==3&& j==2 && k==2)
        //{  std::cout<<" nearby ="<<" i= "<<i<<" j= "<<j<<" k="<<k<<"\n"; 
          //std::cout<<"U_resolved(3,2,2)="<<Vel(i,j,k) <<" UD(3,2,2)="<<UD<<"U2D="<<sums<<"\n";
        //}
      }; // end for LegScaleSep functions

    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      FilterStress* build()
      { return scinew FilterStress( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;
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
    
   std::vector<std::string> m_FilterStress_names;
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
