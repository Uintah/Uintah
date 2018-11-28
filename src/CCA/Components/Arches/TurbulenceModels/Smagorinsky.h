#ifndef Uintah_Component_Arches_Smagorinsky_h
#define Uintah_Component_Arches_Smagorinsky_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class Smagorinsky : public TaskInterface {

public:

    Smagorinsky( std::string task_name, int matl_index );
    ~Smagorinsky();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry  , const bool packed_tasks );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry  , const bool packed_tasks );

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep  , const bool packed_tasks );

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep  , const bool packed_tasks ){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

    void compute_smag_viscos(double&, const Array3<double>&, const Array3<double>&, const Array3<double>&,
                             const Array3<double>&, const Array3<double>&, const Array3<double>&, const Vector&,
                             int, int , int );

    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      Smagorinsky* build()
      { return scinew Smagorinsky( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;
    };

private:

    std::string m_u_vel_name;
    std::string m_v_vel_name;
    std::string m_w_vel_name;

    std::string m_cc_u_vel_name;
    std::string m_cc_v_vel_name;
    std::string m_cc_w_vel_name;
    double m_Cs; //Smagorinsky constant
    double m_molecular_visc;
    std::string m_t_vis_name;

    int Nghost_cells;

  }; // class Smagorinsky
} // namespace Uintah

#endif
