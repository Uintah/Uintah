#ifndef Uintah_Component_Arches_MassFlowRate_h
#define Uintah_Component_Arches_MassFlowRate_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

/**
 *  \file     SpeciesTransport.h
 *  \class    MassFlowRate
 *  \author   Jebin Elias
 *  \date     April 28, 2018          v1
 *
 *  \brief    Compute gas and particle phase flow rates across inlet boundaries.
 *            Utility to be used in conjuction with type specification at the
 *            boundary.
 */

//------------------------------------------------------------------------------
namespace Uintah{

  class MassFlowRate : public TaskInterface {

  public:

    MassFlowRate( std::string task_name, int matl_index );
    ~MassFlowRate();

    void problemSetup( ProblemSpecP& db );

    void create_local_labels();

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){};

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){};

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){};

    class Builder : public TaskInterface::TaskBuilder {

    public:

      Builder( std::string task_name, int matl_index ) : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      MassFlowRate* build()
      { return scinew MassFlowRate( m_task_name, m_matl_index ); }

    private:

      std::string m_task_name;
      int m_matl_index;

    };

  private:

    int m_Nenv;
    bool particleMethod_bool;

    std::string m_g_uVel_name;
    std::string m_g_vVel_name;
    std::string m_g_wVel_name;

    std::string m_p_uVel_base_name;
    std::string m_p_vVel_base_name;
    std::string m_p_wVel_base_name;

    std::string m_w_base_name;
    std::string m_RC_base_name;
    std::string m_CH_base_name;

    std::vector<std::string > m_p_uVel_names;
    std::vector<std::string > m_p_vVel_names;
    std::vector<std::string > m_p_wVel_names;

    std::vector<std::string > m_w_names;
    std::vector<std::string > m_RC_names;
    std::vector<std::string > m_CH_names;

    std::vector<double> m_w_scaling_constant;
    std::vector<double> m_RC_scaling_constant;
    std::vector<double> m_CH_scaling_constant;

    std::vector<double> m_p_uVel_scaling_constant;
    std::vector<double> m_p_vVel_scaling_constant;
    std::vector<double> m_p_wVel_scaling_constant;

    void register_massFlowRate( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks );

    void eval_massFlowRate( const Patch* patch, ArchesTaskInfoManager* tsk_info );

  };

}
#endif
