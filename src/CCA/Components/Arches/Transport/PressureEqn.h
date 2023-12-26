#ifndef Uintah_Component_Arches_PressureEqn_h
#define Uintah_Component_Arches_PressureEqn_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <Core/Parallel/Portability.h>

namespace Uintah{

  class SolverInterface;

  class PressureEqn : public TaskInterface {

public:

    PressureEqn( std::string task_name, int matl_index, MaterialManagerP materialManager );
    ~PressureEqn();

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool pack_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks);

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    template <typename ExecSpace, typename MemSpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    void create_local_labels();

    //Build instructions for this (PressureEqn) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, MaterialManagerP materialManager ) : m_task_name(task_name),
               m_matl_index(matl_index), m_materialManager(materialManager){}
      ~Builder(){}

      PressureEqn* build()
      { return scinew PressureEqn( m_task_name, m_matl_index, m_materialManager ); }

      private:

      std::string m_task_name;
      int m_matl_index;
      MaterialManagerP m_materialManager;

    };

    void solve( const LevelP& level, SchedulerP& sched, const int time_substep );

    void set_solver( SolverInterface* solver ){ m_hypreSolver = solver; }

    void setup_solver( ProblemSpecP& db );


    void sched_Initialize( const LevelP& level, SchedulerP& sched );

    void sched_restartInitialize( const LevelP& level, SchedulerP& sched );

private:

    std::string m_eps_name;
    std::string m_xmom_name;
    std::string m_ymom_name;
    std::string m_zmom_name;
    std::string m_pressure_name;
    std::string m_density_name;
    std::string m_drhodt_name;

    bool m_enforceSolvability{false};
    bool m_use_mms_drhodt;
    bool m_standAlone{false};

    SolverInterface* m_hypreSolver;

    MaterialManagerP m_materialManager;

    IntVector m_periodic_vector;
    bool do_custom_arches_linear_solve{false};

 void
 sched_custom( const LevelP              & level
                ,       SchedulerP       & sched
                , const MaterialSet      * matls
                , const VarLabel         * A_label
                ,       Task::WhichDW      which_A_dw
                , const VarLabel         * x_label
                ,       bool               modifies_X
                , const VarLabel         * b_label
                ,       Task::WhichDW      which_b_dw
                , const VarLabel         * guess_label
                ,       Task::WhichDW      which_guess_dw
                ,       int               time_substep 
                );

 template <typename ExecSpace, typename MemSpace>
 void
 blindGuessToLinearSystem(const PatchSubset* patches,
              const MaterialSubset* matls,
              OnDemandDataWarehouse* old_dw,
              OnDemandDataWarehouse* new_dw,
              UintahParams& uintahParams,
              ExecutionObject<ExecSpace, MemSpace>& execObj);
 int indx{0};
 int cg_ghost{3};   /// number of ghost cells required due to reducing tasks
 int cg_n_iter{30};  /// number of cg iterations

 int total_rb_switch;  
 int d_stencilWidth{1}  ;        // stencilWidth  of jacobi block
 int d_blockSize{1};             // size of jacobi block

 const VarLabel* d_residualLabel;
 const VarLabel* d_littleQLabel;
 const VarLabel* d_bigZLabel;
 const VarLabel* d_smallPLabel;
                           const VarLabel* ALabel;
                           const VarLabel* xLabel;
                           const VarLabel* bLabel;
                           const VarLabel* guess;

std::vector< const VarLabel *> d_corrSumLabel{};                    /// reduction computing correction factor
std::vector< const VarLabel *> d_convMaxLabel{};                    /// reduction checking for convergence1
std::vector< const VarLabel *> d_resSumLabel{};                     /// reduction computing sum of residuals

std::vector<const VarLabel *> d_precMLabel{};                       /// fields used to store preconditioner

#define num_prec_elem 4 // number of fields used to store the preconditioner matrix
 enum uintah_linear_solve_relaxType{ redBlack, jacobBlock, jacobi};
int  d_custom_relax_type;

public:

template <typename ExecSpace, typename MemSpace>
void
dummyTask(const PatchSubset* patches,
         const MaterialSubset* matls,
         OnDemandDataWarehouse* old_dw,
         OnDemandDataWarehouse* new_dw,
         UintahParams& uintahParams,
         ExecutionObject<ExecSpace, MemSpace>& execObj,int iter){ // use to force scheduler copies
                                                                   }

template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT>
void
precondition_relax(ExecutionObject<ExecSpace, MemSpace>& execObj,
                   struct1DArray<grid_CT,num_prec_elem>& precMatrix,
                   grid_CT& residual, 
                   grid_T& bigZ,
                   const IntVector &idxLo,
                   const IntVector &idxHi, 
                   int rb_switch,
                   const Patch* patch );

template <typename ExecSpace, typename MemSpace>
void
cg_init1(const PatchSubset* patches,
         const MaterialSubset* matls,
         OnDemandDataWarehouse* old_dw,
         OnDemandDataWarehouse* new_dw,
         UintahParams& uintahParams,
         ExecutionObject<ExecSpace, MemSpace>& execObj, int rk_step);

template <typename ExecSpace, typename MemSpace>
void
cg_init2(const PatchSubset* patches,
         const MaterialSubset* matls,
         OnDemandDataWarehouse* old_dw,
         OnDemandDataWarehouse* new_dw,
         UintahParams& uintahParams,
         ExecutionObject<ExecSpace, MemSpace>& execObj, int iter,  int rk_step);

template <typename ExecSpace, typename MemSpace>
void
cg_task1(const PatchSubset* patches,
         const MaterialSubset* matls,
         OnDemandDataWarehouse* old_dw,
         OnDemandDataWarehouse* new_dw,
         UintahParams& uintahParams,
         ExecutionObject<ExecSpace, MemSpace>& execObj,int iter);

template <typename ExecSpace, typename MemSpace>
void
cg_task2(const PatchSubset* patches,
         const MaterialSubset* matls,
         OnDemandDataWarehouse* old_dw,
         OnDemandDataWarehouse* new_dw,
         UintahParams& uintahParams,
         ExecutionObject<ExecSpace, MemSpace>& execObj,int iter);

template <typename ExecSpace, typename MemSpace>
void
cg_task3(const PatchSubset* patches,
         const MaterialSubset* matls,
         OnDemandDataWarehouse* old_dw,
         OnDemandDataWarehouse* new_dw,
         UintahParams& uintahParams,
         ExecutionObject<ExecSpace, MemSpace>& execObj,int iter);

template <typename ExecSpace, typename MemSpace>
void
cg_task4(const PatchSubset* patches,
         const MaterialSubset* matls,
         OnDemandDataWarehouse* old_dw,
         OnDemandDataWarehouse* new_dw,
         UintahParams& uintahParams,
         ExecutionObject<ExecSpace, MemSpace>& execObj,int iter);

};
}
#endif
