#ifndef Uintah_Component_Arches_CO_h
#define Uintah_Component_Arches_CO_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Arches/TransportEqns/Discretization_new.h>

namespace Uintah{

  class BoundaryCondition_new;
  class CO : public TaskInterface {

public:

    typedef std::vector<ArchesFieldContainer::VariableInformation> VIVec;

    CO( std::string task_name, int matl_index );
    ~CO();

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( VIVec& variable_registry , const bool pack_tasks);

    void register_timestep_init( VIVec& variable_registry , const bool packed_tasks);

    void register_restart_initialize( VIVec& variable_registry , const bool packed_tasks);

    void register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( VIVec& variable_registry, const int time_substep , const bool packed_tasks){}

    template <typename ExecSpace, typename MemSpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    /** @details This model computes carbon monoxide as a sum of the equilibrum CO and a defect CO.
    CO = CO_equil + defect
    y = ym + d
    or d = y - ym
    accordingly,
    d rho*d
    ________ = RHS_y/V + r_y - (RHS_ym/V + r_ym)
      dt

      the reaction rate for CO "r_y" is defined as follows:

                                [rho*ym]^(t+1) - [rho*y]^(t)
             { for T > T_crit    ___________________________    - RHS_y/V
             {                               dt
      r_y =  {
             { for T < T_crit   r_a

             where r_a = A * CO^a * H2O^b * O2^c * exp( -E/(RT) )

    One can then compute the update for d (and consequently y) as follows:
    [rho*d]^(t+1) - [rho*d]^(t)                              D rho*ym
    ___________________________  = [RHS_y]^t/V + [r_y]^t -   ________
               dt                                               Dt
                                      (S1)        (S2)         (S3)

                           [rho*ym]^(t+1) - [rho*ym]^(t)
             where S3 =      ___________________________
                                        dt
    Then we get:
                  1
    [d]^(t+1) = ______ * ( [rho*d]^(t) + dt * ( S1 + S2 - S3))
             [rho]^(t+1)

    [y]^(t+1) = [ym]^(t+1) + [d]^(t+1)

    It is trivial to show mathematically that when T > Tcrit
    [d]^(t+1) = 0
    and [y]^(t+1) = [ym]^(t+1)
    Otherwise S1, S2, and S2 need to be computed.

    The following algorithm is used to update y and d:
    if T > T_crit
      d=0 , y=0
    else
      Step 1: compute S1 (all variables are at time t)
      Step 2: Compute S2 (all variables are at time t)
      Step 3: compute S3 (variables at t and t+1)
      Step 4: update d to time t+1
      Step 5: update y to time t+1
    **/
    template <typename ExecSpace, typename MemSpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    void create_local_labels();


    //Build instructions for this (CO) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index )
        : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      CO* build()
      { return scinew CO( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

private:

    Discretization_new* m_disc;
    BoundaryCondition_new* m_boundary_condition;
    const double m_Rgas{1.9872041};      // Kcol/K/mol
    const double m_MW_CO{28.01};         // g/mol
    const double m_MW_H2O{18.01528};     // ""
    const double m_MW_O2{31.9988};       // ""
    double m_st_O2;
    double m_st_H2O;
    double m_a;
    double m_b;
    double m_c;
    double m_A;
    double m_Ea;
    double m_prNo;
    double m_T_crit;

    bool m_new_on_restart{false};
    std::string m_conv_scheme;
    std::string m_CO_model_name;
    std::string m_CO_diff_name;
    std::string m_CO_conv_name;
    std::string m_defect_name;
    std::string m_rate_name;
    std::string m_rho_table_name;
    std::string m_temperature_table_name;
    std::string m_CO_table_name;
    std::string m_H2O_table_name;
    std::string m_O2_table_name;
    std::string m_MW_table_name;
    std::string m_u_vel;
    std::string m_v_vel;
    std::string m_w_vel;
    std::string m_area_frac;
    std::string m_turb_visc;
    std::string m_vol_frac;

  };
}
#endif
