/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/**
 * @file CCA/Components/PhaseField/Applications/Heat.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_Applications_Heat_h
#define Packages_Uintah_CCA_Components_PhaseField_Applications_Heat_h

#include <CCA/Components/PhaseField/Util/Definitions.h>
#include <CCA/Components/PhaseField/Util/Expressions.h>
#include <CCA/Components/PhaseField/Util/BlockRangeIO.h>
#include <CCA/Components/PhaseField/DataTypes/HeatProblem.h>
#include <CCA/Components/PhaseField/DataTypes/SubProblemsP.h>
#include <CCA/Components/PhaseField/DataTypes/SubProblems.h> // must be included after SubProblemsP where swapbyte soverride is defined
#include <CCA/Components/PhaseField/DataTypes/Variable.h>
#include <CCA/Components/PhaseField/DataTypes/ScalarField.h>
#include <CCA/Components/PhaseField/DataTypes/VectorField.h>
#include <CCA/Components/PhaseField/Applications/Application.h>
#include <CCA/Components/PhaseField/Factory/Implementation.h>
#include <CCA/Components/PhaseField/Views/View.h>
#include <CCA/Components/PhaseField/Views/FDView.h>
#include <CCA/Components/PhaseField/DataWarehouse/DWView.h>
#include <CCA/Components/PhaseField/AMR/AMRInterpolator.h>
#include <CCA/Components/PhaseField/AMR/AMRRestrictor.h>

#include <Core/Util/DebugStream.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/PerPatchVars.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/Regridder.h>

#define PhaseField_Heat_DBG_MATRIX 0
#define PhaseField_Heat_DBG_DERIVATIVES 0

namespace Uintah
{
namespace PhaseField
{

/// Debugging stream for component schedulings
static DebugStream cout_heat_scheduling ( "HEAT SCHEDULING", false );

/**
 * @brief Heat PhaseField applications
 *
 * Implements a Finite Difference solver for the simulation of the heat
 * diffusion model
 * \f[
 * \dot u = \alpha \nabla^2 u
 * \f]
 * with initial data
 * \f[
 * u_{|t=0} = \prod_{d} \cos ( \alpha x_d );
 * \f]
 * where \f$d\f$ ranges over the problem dimension.
 *
 * The model parameters are:
 * - \f$ \alpha \f$   thermal diffusivity
 *
 * @todo templetize FAC/nonFAC
 * @todo check for more than 2 amr levels
 *
 * @tparam VAR type of variable representation
 * @tparam DIM problem dimensions
 * @tparam STN finite-difference stencil
 * @tparam AMR whether to use adaptive mesh refinement
 */
template<VarType VAR, DimType DIM, StnType STN, bool AMR = false>
class Heat
    : public Application<VAR, DIM, STN, AMR>
    , public Implementation<Heat<VAR, DIM, STN, AMR>, UintahParallelComponent, const ProcessorGroup *, const MaterialManagerP, int>
{
private: // STATIC MEMBERS

    /// Index for solution
    static constexpr size_t U = 0;

    /// Number of ghost elements required by STN (on the same level)
    static constexpr int FGN = get_stn<STN>::ghosts;

    /// Type of ghost elements required by VAR and STN (on coarser level)
    static constexpr Ghost::GhostType FGT = FGN ? get_var<VAR>::ghost_type : Ghost::None;

    /// Number of ghost elements required by STN (on coarser level)
    /// @remark this should depend on FCI bc type but if fixed for simplicity
    static constexpr int CGN = 1;

    /// Type of ghost elements required by VAR and STN (on the same level)
    static constexpr Ghost::GhostType CGT = CGN ? get_var<VAR>::ghost_type : Ghost::None;

    /// Problem material index (only one SimpleMaterial)
    static constexpr int material = 0;

    /// Interpolation type for refinement
    static constexpr FCIType C2F = ( VAR == CC ) ? I0 : I1; // TODO make template parameter

    /// Restriction type for coarsening
    static constexpr FCIType F2C = ( VAR == CC ) ? I1 : I0; // TODO make template parameter

public: // STATIC MEMBERS

    /// Class name as used by ApplicationFactory
    static const std::string Name;

protected: // MEMBERS

    /// Output streams for debugging
    DebugStream dbg_out1, dbg_out2, dbg_out3, dbg_out4;

    // Labels for variables to be stored into the DataWarehouse
    const VarLabel * u_label, * delta_u_label, * error_u_label;
    const VarLabel * u_normL2_label, * error_normL2_label;
    const VarLabel * subproblems_label;

#ifdef PhaseField_Heat_DBG_DERIVATIVES
    std::array<const VarLabel *, DIM> du_label, error_du_label;
    std::array<const VarLabel *, DIM> ddu_label, error_ddu_label;
    const VarLabel * u_normH10_label, * error_normH10_label;
    const VarLabel * u_normH20_label, * error_normH20_label;
#endif

    /// Wether to perform comparisons with analytical solution
    bool test;

    /// Time step size
    double delt;

    /// Non-dimensional thermal diffusivity
    double alpha;

    /// Threshold for AMR
    double refine_threshold;

    /// Store which fine/coarse interface conditions to use on each variable
    std::map<std::string, FC> c2f;

    /// Flag for avoiding multiple reinitialization of subproblems after regridding
    bool is_first_schedule_refine;

public: // CONSTRUCTORS/DESTRUCTOR

    /**
     * @brief Constructor
     *
     * Intantiate an Heat application
     *
     * @param myWorld data structure to manage mpi processes
     * @param materialManager data structure to manage materials
     * @param verbosity constrols amount of debugging output
     */
    Heat (
        const ProcessorGroup * myWorld,
        const MaterialManagerP materialManager,
        int verbosity = 0
    );

    /**
     * @brief Destructor
     */
    virtual ~Heat();

    /// Prevent copy (and move) constructor
    Heat ( Heat const & ) = delete;

    /// Prevent copy (and move) assignment
    Heat & operator= ( Heat const & ) = delete;

protected: // SETUP

    /**
     * @brief Setup
     *
     * Initialize problem parameters with values from problem specifications
     *
     * @param params problem specifications parsed from input file
     * @param restart_prob_spec unused
     * @param grid unused
     */
    virtual void
    problemSetup (
        const ProblemSpecP & params,
        const ProblemSpecP & restart_prob_spec,
        GridP & grid
    ) override;

protected: // SCHEDULINGS

    /**
     * @brief Schedule the initialization tasks
     *
     * Specify all tasks to be performed at initial timestep to initialize
     * variables in the DataWarehouse
     *
     * @param level grid level to be initialized
     * @param sched scheduler to manage the tasks
     */
    virtual void
    scheduleInitialize (
        const LevelP & level,
        SchedulerP & sched
    ) override;

    /**
     * @brief Schedule task_initialize_subproblems (non AMR implementation)
     *
     * Defines the dependencies and output of the task which initializes the
     * subproblems allowing sched to control its execution order
     *
     * @param level grid level to be initialized
     * @param sched scheduler to manage the tasks
     */
    template < bool MG >
    typename std::enable_if < !MG, void >::type
    scheduleInitialize_subproblems (
        const LevelP & level,
        SchedulerP & sched
    );

    /**
     * @brief Schedule task_initialize_subproblems (AMR implementation)
     *
     * Defines the dependencies and output of the task which initializes the
     * subproblems allowing sched to control its execution order
     *
     * @param level grid level to be initialized
     * @param sched scheduler to manage the tasks
     */
    template < bool MG >
    typename std::enable_if < MG, void >::type
    scheduleInitialize_subproblems (
        const LevelP & level,
        SchedulerP & sched
    );

    /**
     * @brief Schedule task_initialize_solution (non AMR implementation)
     *
     * Defines the dependencies and output of the task which initializes the
     * subproblems allowing sched to control its execution order
     *
     * @param level grid level to be initialized
     * @param sched scheduler to manage the tasks
     */
    template < bool MG >
    typename std::enable_if < !MG, void >::type
    scheduleInitialize_solution (
        const LevelP & level,
        SchedulerP & sched
    );

    /**
     * @brief Schedule task_initialize_solution (AMR implementation)
     *
     * Defines the dependencies and output of the task which initializes the
     * subproblems allowing sched to control its execution order
     *
     * @param level grid level to be initialized
     * @param sched scheduler to manage the tasks
     */
    template < bool MG >
    typename std::enable_if < MG, void >::type
    scheduleInitialize_solution (
        const LevelP & level,
        SchedulerP & sched
    );

    /**
     * @brief Schedule the initialization tasks for restarting a simulation
     *
     * Specify all tasks to be performed at fist timestep after a stop to
     * initialize not saved variables to the DataWarehouse
     *
     * @remark only subproblems need to be reinitialized all other variables
     * should be retrieved from saved checkpoints
     *
     * @param level grid level to be initialized
     * @param sched scheduler to manage the tasks
     */
    virtual void
    scheduleRestartInitialize (
        const LevelP & level,
        SchedulerP & sched
    ) override;

    /**
     * @brief Schedule task_compute_stable_timestep
     *
     * Specify all tasks to be performed before each time advance to compute a
     * timestep size which ensures numerical stability
     *
     * @param level grid level to be initialized
     * @param sched scheduler to manage the tasks
     */
    virtual void
    scheduleComputeStableTimeStep (
        const LevelP & level,
        SchedulerP & sched
    ) override;

    /**
     * @brief Schedule the time advance tasks
     *
     * Specify all tasks to be performed at each timestep to update the
     * simulation variables in the DataWarehouse
     *
     * @param level grid level to be initialized
     * @param sched scheduler to manage the tasks
     */
    virtual void
    scheduleTimeAdvance (
        const LevelP & level,
        SchedulerP & sched
    ) override;

    /**
     * @brief Schedule task_time_advance_solution (non AMR implementation)
     *
     * Defines the dependencies and output of the task which updates the
     * subproblems allowing sched to control its execution order
     *
     * @param level grid level to be updated
     * @param sched scheduler to manage the tasks
     */
    template < bool MG >
    typename std::enable_if < !MG, void >::type
    scheduleTimeAdvance_subproblems (
        const LevelP & level,
        SchedulerP & sched
    );

    /**
     * @brief Schedule task_time_advance_solution (AMR implementation)
     *
     * Defines the dependencies and output of the task which updates the
     * subproblems allowing sched to control its execution order
     *
     * @param level grid level to be updated
     * @param sched scheduler to manage the tasks
     */
    template < bool MG >
    typename std::enable_if < MG, void >::type
    scheduleTimeAdvance_subproblems (
        const LevelP & level,
        SchedulerP & sched
    );

#ifdef PhaseField_Heat_DBG_DERIVATIVES
    /**
     * @brief Schedule task_time_advance_dbg_derivatives (non AMR implementation)
     *
     * Defines the dependencies and output of the task which updates the
     * subproblems allowing sched to control its execution order
     *
     * @param level grid level to be updated
     * @param sched scheduler to manage the tasks
     */
    template < bool MG >
    typename std::enable_if < !MG, void >::type
    scheduleTimeAdvance_dbg_derivatives (
        const LevelP & level,
        SchedulerP & sched
    );

    /**
     * @brief Schedule task_time_advance_dbg_derivatives (AMR implementation)
     *
     * Defines the dependencies and output of the task which updates the
     * subproblems allowing sched to control its execution order
     *
     * @param level grid level to be updated
     * @param sched scheduler to manage the tasks
     */
    template < bool MG >
    typename std::enable_if < MG, void >::type
    scheduleTimeAdvance_dbg_derivatives (
        const LevelP & level,
        SchedulerP & sched
    );

    /**
     * @brief Schedule task_time_advance_dbg_derivatives_error
     * (non AMR implementation)
     *
     * Defines the dependencies and output of the task which updates the
     * subproblems allowing sched to control its execution order
     *
     * @param level grid level to be updated
     * @param sched scheduler to manage the tasks
     */
    template < bool MG >
    typename std::enable_if < !MG, void >::type
    scheduleTimeAdvance_dbg_derivatives_error (
        const LevelP & level,
        SchedulerP & sched
    );

    /**
     * @brief Schedule task_time_advance_dbg_derivatives_error
     * (AMR implementation)
     *
     * Defines the dependencies and output of the task which updates the
     * subproblems allowing sched to control its execution order
     *
     * @param level grid level to be updated
     * @param sched scheduler to manage the tasks
     */
    template < bool MG >
    typename std::enable_if < MG, void >::type
    scheduleTimeAdvance_dbg_derivatives_error (
        const LevelP & level,
        SchedulerP & sched
    );
#endif

    /**
     * @brief Schedule task_time_advance_solution (non AMR implementation)
     *
     * Defines the dependencies and output of the task which updates psi
     * and u allowing sched to control its execution order
     *
     * @param level grid level to be updated
     * @param sched scheduler to manage the tasks
     */
    template < bool MG >
    typename std::enable_if < !MG, void >::type
    scheduleTimeAdvance_solution (
        const LevelP & level,
        SchedulerP & sched
    );

    /**
     * @brief Schedule task_time_advance_solution (AMR implementation)
     *
     * Defines the dependencies and output of the task which updates psi
     * and u allowing sched to control its execution order
     *
     * @param level grid level to be updated
     * @param sched scheduler to manage the tasks
     */
    template < bool MG >
    typename std::enable_if < MG, void >::type
    scheduleTimeAdvance_solution (
        const LevelP & level,
        SchedulerP & sched
    );

    /**
     * @brief Schedule task_time_advance_solution_error
     *
     * Defines the dependencies and output of the task which updates the
     * subproblems allowing sched to control its execution order
     *
     * @param level grid level to be updated
     * @param sched scheduler to manage the tasks
     */
    void
    scheduleTimeAdvance_solution_error (
        const LevelP & level,
        SchedulerP & sched
    );

    /**
     * @brief Schedule the refinement tasks
     *
     * Specify all tasks to be performed after an AMR regrid in order to populate
     * variables in the DataWarehouse at newly created patches
     *
     * @remark If regridding happens at initial time step scheduleInitialize is
     * called instead
     *
     * @param new_patches patches to be populated
     * @param sched scheduler to manage the tasks
     */
    virtual void
    scheduleRefine (
        const PatchSet * new_patches,
        SchedulerP & sched
    ) override;

    /**
     * @brief Schedule task_initialize_subproblems after regridding
     *
     * Defines the dependencies and output of the task which initializes the
     * subproblems allowing sched to control its execution order
     *
     * @remark subproblems need to be reinitialized on all patches because
     * even preexisting patches may have different neighbors
     *
     * @param new_patches patches to be populated
     * @param sched scheduler to manage the tasks
     */
    void
    scheduleRefine_subproblems (
        const PatchSet * new_patches,
        SchedulerP & sched
    );

    /**
     * @brief Schedule task_refine_solution
     *
     * Defines the dependencies and output of the task which interpolates the
     * solution from the coarser level to each one of the new_patches
     * allowing sched to control its execution order
     *
     * @param new_patches patches to be populated
     * @param sched scheduler to manage the tasks
     */
    void
    scheduleRefine_solution (
        const PatchSet * new_patches,
        SchedulerP & sched
    );

    /**
     * @brief Schedule the refinement tasks
     *
     * Do nothing
     *
     * @param level_fine unused
     * @param sched unused
     * @param need_old_coarse unused
     * @param need_new_coarse unused
     */
    virtual void
    scheduleRefineInterface (
        const LevelP & level_fine,
        SchedulerP & sched,
        bool need_old_coarse,
        bool need_new_coarse
    ) override;

    /**
     * @brief Schedule the time coarsen tasks
     *
     * Specify all tasks to be performed after each timestep to restrict the
     * computed variables from finer to coarser levels
     *
     * @param level_coarse level to be updated
     * @param sched scheduler to manage the tasks
     */
    virtual void
    scheduleCoarsen (
        const LevelP & level_coarse,
        SchedulerP & sched
    ) override;

    /**
     * @brief Schedule task_coarsen_solution
     *
     * Defines the dependencies and output of the task which restrict the
     * solution to level_coarse from its finer level allowing sched to control
     * its execution order
     *
     * @param level_coarse level to be updated
     * @param sched scheduler to manage the tasks
     */
    void
    scheduleCoarsen_solution (
        const LevelP & level_coarse,
        SchedulerP & sched
    );

    /**
     * @brief Schedule the error estimate tasks
     *
     * Specify all tasks to be performed before each timestep to estimate the
     * spatial discretization error on the solution update in order to decide
     * where to refine the grid
     *
     * @param level level to check
     * @param sched scheduler to manage the tasks
     */
    virtual void
    scheduleErrorEstimate (
        const LevelP & level,
        SchedulerP & sched
    ) override;

    /**
     * @brief Schedule task_error_estimate_solution (coarsest level implementation)
     *
     * Defines the dependencies and output of the task which estimates the
     * spatial discretization error allowing sched to controvaluel its execution order
     *
     * @param level level to check
     * @param sched scheduler to manage the tasks
     */
    template < bool MG >
    typename std::enable_if < !MG, void >::type
    scheduleErrorEstimate_solution (
        const LevelP & level,
        SchedulerP & sched
    );

    /**
     * @brief Schedule task_error_estimate_grad_psi (refinement level implementation)
     *
     * Defines the dependencies and output of the task which estimates the
     * spatial discretization error allowing sched to control its execution order
     *
     * @param level level to check
     * @param sched scheduler to manage the tasks
     */
    template < bool MG >
    typename std::enable_if < MG, void >::type
    scheduleErrorEstimate_solution (
        const LevelP & level,
        SchedulerP & sched
    );

    /**
     * @brief Schedule the initial error estimate tasks
     *
     * Specify all tasks to be performed before the first timestep to estimate
     * the spatial discretization error on the solution update in order to decide
     * where to refine the grid
     *
     * @remark forward to scheduleErrorEstimate
     *
     * @param level level to check
     * @param sched scheduler to manage the tasks
     */
    virtual void
    scheduleInitialErrorEstimate (
        const LevelP & level,
        SchedulerP & sched
    ) override;

protected: // TASKS

    /**
     * @brief Initialize subproblems task
     *
     * Create the SubProblems for each one of the patches and save it to dw_new
     *
     * @param myworld data structure to manage mpi processes
     * @param patches list of patches to be initialized
     * @param matls unused
     * @param dw_old unused
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_initialize_subproblems (
        const ProcessorGroup * myworld,
        const PatchSubset * patches,
        const MaterialSubset * matls,
        DataWarehouse * dw_old,
        DataWarehouse * dw_new
    );

    /**
     * @brief Initialize solution task
     *
     * Allocate and save variables for psi and u for each one of the patches
     * and save them to dw_new
     * @remark initialize also anisotropy terms to 0
     *
     * @param myworld data structure to manage mpi processes
     * @param patches list of patches to be initialized
     * @param matls unused
     * @param dw_old unused
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_initialize_solution (
        const ProcessorGroup * myworld,
        const PatchSubset * patches,
        const MaterialSubset * matls,
        DataWarehouse * dw_old,
        DataWarehouse * dw_new
    );

    /**
     * @brief Compute timestep task
     *
     * Puts into the new DataWarehouse the constant value specified in input (delt)
     * of the timestep
     *
     * @param myworld data structure to manage mpi processes
     * @param patches list of patches to be initialized
     * @param matls unused
     * @param dw_old unused
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_compute_stable_timestep (
        const ProcessorGroup * myworld,
        const PatchSubset * patches,
        const MaterialSubset * matls,
        DataWarehouse * dw_old,
        DataWarehouse * dw_new
    );

    /**
     * @brief Advance subproblems task
     *
     * Move SubProblems for each one of the patches and from dw_old to dw_new
     * or, if not found in dw_old (after regrid), create new subproblems in dw_new
     *
     * @param myworld data structure to manage mpi processes
     * @param patches list of patches to be initialized
     * @param matls unused
     * @param dw_old DataWarehouse for previous timestep
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_time_advance_subproblems (
        const ProcessorGroup * myworld,
        const PatchSubset * patches,
        const MaterialSubset * matls,
        DataWarehouse * dw_old,
        DataWarehouse * dw_new
    );

#ifdef PhaseField_Heat_DBG_DERIVATIVES

    /**
     * @brief Advance derivatives task (debug)
     *
     * Computes value of u derivatives using the solution at the previous
     * timestep (for debugging purpose)
     *
     * @param myworld data structure to manage mpi processes
     * @param patches list of patches to be initialized
     * @param matls unused
     * @param dw_old DataWarehouse for previous timestep
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_time_advance_dbg_derivatives (
        const ProcessorGroup * myworld,
        const PatchSubset * patches,
        const MaterialSubset * matls,
        DataWarehouse * dw_old,
        DataWarehouse * dw_new
    );

    /**
     * @brief Advance derivatives error task (debug)
     *
     * Computes the error in the approximation of u derivatives
     * comparing them with their analytical expressions (for debugging purpose)
     *
     * @param myworld data structure to manage mpi processes
     * @param patches list of patches to be initialized
     * @param matls unused
     * @param dw_old DataWarehouse for previous timestep
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_time_advance_dbg_derivatives_error (
        const ProcessorGroup * myworld,
        const PatchSubset * patches,
        const MaterialSubset * matls,
        DataWarehouse * dw_old,
        DataWarehouse * dw_new
    );
#endif

    /**
     * @brief Advance solution task
     *
     * Computes new value of u using the value of the solution and at
     * previous timestep
     *
     * @param myworld data structure to manage mpi processes
     * @param patches list of patches to be initialized
     * @param matls unused
     * @param dw_old DataWarehouse for previous timestep
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_time_advance_solution (
        const ProcessorGroup * myworld,
        const PatchSubset * patches,
        const MaterialSubset * matls,
        DataWarehouse * dw_old,
        DataWarehouse * dw_new
    );

    /**
     * @brief Advance solution error task (test)
     *
     * Computes error in u approximation using the analytical solution
     *
     * @remark test must be set to true in input
     *
     * @param myworld data structure to manage mpi processes
     * @param patches list of patches to be initialized
     * @param matls unused
     * @param dw_old DataWarehouse for previous timestep
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_time_advance_solution_error (
        const ProcessorGroup * myworld,
        const PatchSubset * patches,
        const MaterialSubset * matls,
        DataWarehouse * dw_old,
        DataWarehouse * dw_new
    );

    /**
     * @brief Refine solution task
     *
     * Computes interpolated value of u on new refined patched
     *
     * @param myworld data structure to manage mpi processes
     * @param patches_fine list of patches to be initialized
     * @param matls unused
     * @param dw_old unused
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_refine_solution (
        const ProcessorGroup * myworld,
        const PatchSubset * patches_fine,
        const MaterialSubset * matls,
        DataWarehouse * dw_old,
        DataWarehouse * dw_new
    );

    /**
     * @brief Coarsen solution task
     *
     * Restricted value of u from refined regions to coarse patches
     * underneath
     *
     * @param myworld data structure to manage mpi processes
     * @param patches_coarse list of patches to be updated
     * @param matls unused
     * @param dw_old unused
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_coarsen_solution (
        const ProcessorGroup * myworld,
        const PatchSubset * patches_coarse,
        const MaterialSubset * matls,
        DataWarehouse * dw_old,
        DataWarehouse * dw_new
    );

    /**
     * @brief ErrorEstimate solution task
     *
     * Computes the gradient of the solution using its value at the previous
     * timestep and set refinement flag where it is above the threshold given
     * in input
     *
     * @param myworld data structure to manage mpi processes
     * @param patches list of patches to be initialized
     * @param matls unused
     * @param dw_old DataWarehouse for previous timestep
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_error_estimate_solution (
        const ProcessorGroup * myworld,
        const PatchSubset * patches,
        const MaterialSubset * matls,
        DataWarehouse * dw_old,
        DataWarehouse * dw_new
    );

protected: // IMPLEMENTATIONS

    /**
     * @brief Initialize solution implementation
     *
     * compute initial condition for u at a given grid position
     *
     * @param id grid index
     * @param patch grid patch
     * @param L domain width
     * @param[out] u view of the solution field in the new dw
     */
    void
    initialize_solution (
        const IntVector & id,
        const Patch * patch,
        const double & L,
        View < ScalarField<double> > & u
    );

#ifdef PhaseField_Heat_DBG_DERIVATIVES

    /**
     * @brief Advance derivatives implementation
     *
     * Computes value of u derivatives using the solution at the previous
     * timestep at a given grid position (for debugging purpose)
     *
     * @param id grid index
     * @param u_old view of the solution field in the old dw
     * @param[out] du view of the solution derivatives vector field in the new dw
     * @param[out] ddu view of the solution derivatives vector field in the new dw
     */
    void
    time_advance_dbg_derivatives (
        const IntVector & id,
        const FDView < ScalarField<const double>, STN > & u_old,
        View < VectorField<double, DIM> > & du,
        View < VectorField<double, DIM> > & ddu
    );

    /**
     * @brief Advance derivatives error implementation
     *
     * Computes error in the approximation of u derivatives comparing them with
     * their analytical expressions at a given grid position
     * (for debugging purpose)
     *
     * @param id grid index
     * @param patch grid patch
     * @param L domain width
     * @param t simulation time
     * @param du view of the solution derivatives vector field in the new dw
     * @param ddu view of the solution derivatives vector field in the new dw
     * @param[out] error_du view of the solution derivatives error vector field in the new dw
     * @param[out] error_ddu view of the solution derivatives error vector field in the new dw
     * @param[out] u_normH10 L2-norm of the solution 1st order derivatives vector
     * @param[out] u_normH20 L2-norm of the solution 2nd order derivatives vector
     * @param[out] error_normH10 L2-norm of the solution 1st order derivatives error vector
     * @param[out] error_normH20 L2-norm of the solution 2nd order derivatives error vector
     */
    void
    time_advance_dbg_derivatives_error (
        const IntVector & id,
        const Patch * patch,
        const double & L,
        const double & t,
        View < VectorField<const double, DIM> > & du,
        View < VectorField<const double, DIM> > & ddu,
        View < VectorField<double, DIM> > & error_du,
        View < VectorField<double, DIM> > & error_ddu,
        double & u_normH10,
        double & u_normH20,
        double & error_normH10,
        double & error_normH20
    );
#endif

    /**
     * @brief Advance solution implementation
     *
     * compute new value for u at a given grid position using the value of the
     * solution and at previous timestep
     *
     * @param id grid index
     * @param u_old view of the solution field in the old dw
     * @param[out] u_new view of the solution field in the new dw
     */
    void
    time_advance_solution (
        const IntVector & id,
        const FDView < ScalarField<const double>, STN > & u_old,
        View < ScalarField<double> > & u_new
    );

    /**
     * @brief Advance solution error task (test)
     *
     * compute error in u approximation at a given grid position using the
     * analytical solution
     *
     * @param id grid index
     * @param patch grid patch
     * @param L domain width
     * @param t simulation time
     * @param u view of the newly computed solution field in the new dw
     * @param[out] delta_u view of the local error (difference between computed and
     * analytical solution at each grid position)
     * @param[out] error_u interpolation error (L2 norm over the range of each
     * grid position of the difference between computed and
     * analytical solution at each grid position)
     * @param[out] u_normL2 L2-norm (global) of the solution vector
     * @param[out] error_normL2 L2-norm (global) of the solution error vector
     *
     */
    void
    time_advance_solution_error (
        const IntVector & id,
        const Patch * patch,
        const double & L,
        const double & t,
        View < ScalarField<const double> > & u,
        View < ScalarField<double> > & delta_u,
        View < ScalarField<double> > & error_u,
        double & u_normL2,
        double & error_normL2
    );

    /**
     * @brief Refine solution implementation
     *
     * Computes interpolated value of u at a given grid position

     * @param id_fine fine grid index
     * @param u_coarse_interp interpolator of the aolution field on the coarse level
     * @param[out] u_fine view of the aolution field on the fine level
     */
    void
    refine_solution (
        const IntVector id_fine,
        const View < ScalarField<const double> > & u_coarse_interp,
        View < ScalarField<double> > & u_fine
    );

    /**
     * @brief Coarsen solution implementation
     *
     * Computes restricted value of u at a given grid position

     * @param id_coarse coarse grid index
     * @param u_fine_restr restrictor of the aolution field on the fine level
     * @param[out] u_coarse view of the aolution field on the coarse level
     */
    void
    coarsen_solution (
        const IntVector id_coarse,
        const View < ScalarField<const double> > & u_fine_restr,
        View < ScalarField<double> > & u_coarse
    );

    /**
     * @brief ErrorEstimate solution implementation (cell centered implementation)
     *
     * Computes the gradient of the phase field using its value at the previous
     * timestep and set refinement flag where it is above the threshold given
     * in input
     *
     * @param id grid index
     * @param u view of the solution the old dw
     * @param[out] refine_flag view of refine flag (grid field) in the new dw
     * @param[out] refine_patch flag for patch refinement
     */
    template < VarType V >
    typename std::enable_if < V == CC, void >::type
    error_estimate_solution (
        const IntVector id,
        FDView < ScalarField<const double>, STN > & u,
        View < ScalarField<int> > & refine_flag,
        bool & refine_patch
    );

    /**
     * @brief ErrorEstimate solution implementation (vertex based implementation)
     *
     * Computes the gradient of the phase field using its value at the previous
     * timestep and set refinement flag where it is above the threshold given
     * in input
     *
     * @param id grid index
     * @param u view of the solution in the old dw
     * @param[out] refine_flag view of refine flag (grid field) in the new dw
     * @param[out] refine_patch flag for patch refinement
     */
    template < VarType V >
    typename std::enable_if < V == NC, void >::type
    error_estimate_solution (
        const IntVector id,
        FDView < ScalarField<const double>, STN > & u,
        View < ScalarField<int> > & refine_flag,
        bool & refine_patch
    );

}; // class Heat

// CONSTRUCTORS/DESTRUCTOR

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
Heat<VAR, DIM, STN, AMR>::Heat (
    const ProcessorGroup * myworld,
    const MaterialManagerP materialManager,
    int verbosity
) : Application<VAR, DIM, STN, AMR> ( myworld, materialManager )
    ,  dbg_out1 ( "Heat", verbosity > 0 )
    ,  dbg_out2 ( "Heat", verbosity > 1 )
    ,  dbg_out3 ( "Heat", verbosity > 2 )
    ,  dbg_out4 ( "Heat", verbosity > 3 )
{
    u_label = VarLabel::create ( "u", Variable<VAR, double>::getTypeDescription() );
    subproblems_label = VarLabel::create ( "subproblems", Variable < PP, SubProblems < HeatProblem<VAR, STN> > >::getTypeDescription() );
    delta_u_label = VarLabel::create ( "delta_u", Variable<VAR, double>::getTypeDescription() );
    error_u_label = VarLabel::create ( "error_u", Variable<VAR, double>::getTypeDescription() );
    u_normL2_label = VarLabel::create ( "u_normL2", sum_vartype::getTypeDescription() );
    error_normL2_label = VarLabel::create ( "error_normL2", sum_vartype::getTypeDescription() );

#ifdef PhaseField_Heat_DBG_DERIVATIVES
    du_label[X] = VarLabel::create ( "ux", Variable<VAR, double>::getTypeDescription() );
    ddu_label[X] = VarLabel::create ( "uxx", Variable<VAR, double>::getTypeDescription() );
    error_du_label[X] = VarLabel::create ( "error_ux", Variable<VAR, double>::getTypeDescription() );
    error_ddu_label[X] = VarLabel::create ( "error_uxx", Variable<VAR, double>::getTypeDescription() );
    if ( DIM > D1 )
    {
        du_label[Y] = VarLabel::create ( "uy", Variable<VAR, double>::getTypeDescription() );
        ddu_label[Y] = VarLabel::create ( "uyy", Variable<VAR, double>::getTypeDescription() );
        error_du_label[Y] = VarLabel::create ( "error_uy", Variable<VAR, double>::getTypeDescription() );
        error_ddu_label[Y] = VarLabel::create ( "error_uyy", Variable<VAR, double>::getTypeDescription() );
    }
    if ( DIM > D2 )
    {
        du_label[Z] = VarLabel::create ( "uz", Variable<VAR, double>::getTypeDescription() );
        ddu_label[Z] = VarLabel::create ( "uzz", Variable<VAR, double>::getTypeDescription() );
        error_du_label[Z] = VarLabel::create ( "error_uz", Variable<VAR, double>::getTypeDescription() );
        error_ddu_label[Z] = VarLabel::create ( "error_uzz", Variable<VAR, double>::getTypeDescription() );
    }
    u_normH10_label = VarLabel::create ( "u_normH10", sum_vartype::getTypeDescription() );
    u_normH20_label = VarLabel::create ( "u_normH20", sum_vartype::getTypeDescription() );
    error_normH10_label = VarLabel::create ( "error_normH10", sum_vartype::getTypeDescription() );
    error_normH20_label = VarLabel::create ( "error_normH20", sum_vartype::getTypeDescription() );
#endif
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
Heat<VAR, DIM, STN, AMR>::~Heat()
{
    VarLabel::destroy ( u_label );
    VarLabel::destroy ( subproblems_label );
    VarLabel::destroy ( delta_u_label );
    VarLabel::destroy ( error_u_label );
    VarLabel::destroy ( u_normL2_label );
    VarLabel::destroy ( error_normL2_label );
#ifdef PhaseField_Heat_DBG_DERIVATIVES
    for ( size_t D = 0; D < DIM; ++D )
    {
        VarLabel::destroy ( du_label[D] );
        VarLabel::destroy ( ddu_label[D] );
        VarLabel::destroy ( error_du_label[D] );
        VarLabel::destroy ( error_ddu_label[D] );
    }
    VarLabel::destroy ( u_normH10_label );
    VarLabel::destroy ( u_normH20_label );
    VarLabel::destroy ( error_normH10_label );
    VarLabel::destroy ( error_normH20_label );
#endif
}

// SETUP

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::problemSetup (
    const ProblemSpecP & params,
    const ProblemSpecP &,
    GridP &
)
{
    this->m_scheduler->overrideVariableBehavior ( subproblems_label->getName(), false, false, false, true, true );
    this->m_materialManager->registerSimpleMaterial ( scinew SimpleMaterial() );

    ProblemSpecP heat = params->findBlock ( "PhaseField" );
    heat->require ( "delt", delt );
    heat->require ( "alpha", alpha );
    heat->getWithDefault ( "test", test, false );

    std::string scheme;
    heat->getWithDefault ( "scheme", scheme, "forward_euler" );
    if ( scheme != "forward_euler" )
        SCI_THROW ( InternalError ( "\n ERROR: Implicit time scheme requires HYPRE\n", __FILE__, __LINE__ ) );

    if ( AMR )
    {
        this->setLockstepAMR ( true );

        c2f[u_label->getName()] = ( VAR == CC ) ? FC::FC0 : FC::FC1;

        heat->require ( "refine_threshold", refine_threshold );
        ProblemSpecP amr, regridder, fci;
        if ( ! ( amr = params->findBlock ( "AMR" ) ) ) return;
        if ( ! ( fci = amr->findBlock ( "FineCoarseInterfaces" ) ) ) return;
        if ( ! ( fci = fci->findBlock ( "FCIType" ) ) ) return;
        do
        {
            std::string label, var;
            fci->getAttribute ( "label", label );
            fci->getAttribute ( "var", var );
            c2f[label] = str_to_fc ( var );
        }
        while ( fci = fci->findNextBlock ( "FCIType" ) );
    }

}

// SCHEDULINGS

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::scheduleInitialize (
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleInitialize" << std::endl;

    scheduleInitialize_subproblems<AMR> ( level, sched );
    scheduleInitialize_solution<AMR> ( level, sched );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < !MG, void >::type
Heat<VAR, DIM, STN, AMR>::scheduleInitialize_subproblems (
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleInitialize_subproblems" << std::endl;

    Task * task = scinew Task ( "Heat::task_initialize_subproblems", this, &Heat::task_initialize_subproblems );
    task->computes ( subproblems_label );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

/**
 * @remark we need to schedule all levels before task_error_estimate_solution to avoid
 * the error "Failure finding [subproblems , coarseLevel, MI: none, NewDW
 * (mapped to dw index 1), ####] for Heat::task_error_estimate_solution",
 * on patch #, Level #, on material #, resource (rank): #" while compiling the
 * TaskGraph
 */

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < MG, void >::type
Heat<VAR, DIM, STN, AMR>::scheduleInitialize_subproblems (
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleInitialize_subproblems" << std::endl;

    // since the SimulationController is calling this scheduler starting from
    // the finest level we schedule only on the finest level
    if ( level->hasFinerLevel() ) return;

    GridP grid = level->getGrid();
    for ( int l = 0; l < grid->numLevels(); ++l )
        scheduleInitialize_subproblems < !MG > ( grid->getLevel ( l ), sched );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < !MG, void >::type
Heat<VAR, DIM, STN, AMR>::scheduleInitialize_solution (
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleInitialize_solution" << std::endl;
    Task * task = scinew Task ( "Heat::task_initialize_solution", this, &Heat::task_initialize_solution );
    task->computes ( u_label );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

/**
 * @remark we need to schedule all levels before task_error_estimate_solution to avoid
 * the error "Failure finding [u , coarseLevel, MI: none, NewDW
 * (mapped to dw index 1), ####] for Heat::task_error_estimate_solution",
 * on patch #, Level #, on material #, resource (rank): #" while compiling the
 * TaskGraph
 */
template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < MG, void >::type
Heat<VAR, DIM, STN, AMR>::scheduleInitialize_solution (
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleInitialize_solution" << std::endl;

    // since the SimulationController is calling this scheduler starting from
    // the finest level we schedule only on the finest level
    if ( level->hasFinerLevel() ) return;

    GridP grid = level->getGrid();
    for ( int l = 0; l < grid->numLevels(); ++l )
        scheduleInitialize_solution < !MG > ( grid->getLevel ( l ), sched );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::scheduleRestartInitialize (
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleRestartInitialize" << std::endl;
    scheduleInitialize_subproblems<AMR> ( level, sched );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::scheduleComputeStableTimeStep (
    const LevelP & level,
    SchedulerP & sched
)
{
    Task * task = scinew Task ( "Heat::task_compute_stable_timestep ", this, &Heat::task_compute_stable_timestep );
    task->computes ( this->getDelTLabel(), level.get_rep() );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );

    is_first_schedule_refine = false;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::scheduleTimeAdvance (
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleTimeAdvance" << std::endl;

    scheduleTimeAdvance_subproblems<AMR> ( level, sched );
#ifdef PhaseField_Heat_DBG_DERIVATIVES
    scheduleTimeAdvance_dbg_derivatives<AMR> ( level, sched );
    if ( test ) scheduleTimeAdvance_dbg_derivatives_error<AMR> ( level, sched );
#endif

    scheduleTimeAdvance_solution<AMR> ( level, sched );
    if ( test ) scheduleTimeAdvance_solution_error ( level, sched );
};

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < !MG, void >::type
Heat<VAR, DIM, STN, AMR>::scheduleTimeAdvance_subproblems (
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleTimeAdvance_subproblems" << std::endl;

    Task * task = scinew Task ( "Heat::task_time_advance_subproblems", this, &Heat::task_time_advance_subproblems );
    task->requires ( Task::OldDW, subproblems_label, Ghost::None, 0 );
    task->computes ( subproblems_label );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < MG, void >::type
Heat<VAR, DIM, STN, AMR>::scheduleTimeAdvance_subproblems (
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleTimeAdvance_subproblems" << std::endl;

    if ( level->hasCoarserLevel() ) return;

    GridP grid = level->getGrid();
    for ( int l = 0; l < grid->numLevels(); ++l )
        scheduleTimeAdvance_subproblems < !MG > ( grid->getLevel ( l ), sched );
}

#ifdef PhaseField_Heat_DBG_DERIVATIVES
template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < !MG, void >::type
Heat<VAR, DIM, STN, AMR>::scheduleTimeAdvance_dbg_derivatives (
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleTimeAdvance_dbg_derivatives" << std::endl;

    Task * task = scinew Task ( "Heat::task_time_advance_dbg_derivatives", this, &Heat::task_time_advance_dbg_derivatives );
    task->requires ( Task::OldDW, subproblems_label, Ghost::None, 0 );
    task->requires ( Task::OldDW, u_label, FGT, FGN );
    for ( size_t D = 0; D < DIM; ++D )
    {
        task->computes ( du_label[D] );
        task->computes ( ddu_label[D] );
    }
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < MG, void >::type
Heat<VAR, DIM, STN, AMR>::scheduleTimeAdvance_dbg_derivatives (
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleTimeAdvance_dbg_derivatives" << std::endl;

    if ( !level->hasCoarserLevel() ) scheduleTimeAdvance_dbg_derivatives < !MG > ( level, sched );
    else
    {
        Task * task = scinew Task ( "Heat::task_time_advance_dbg_derivatives", this, &Heat::task_time_advance_dbg_derivatives );
        task->requires ( Task::OldDW, subproblems_label, Ghost::None, 0 );
        task->requires ( Task::OldDW, subproblems_label, nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
        task->requires ( Task::OldDW, u_label, FGT, FGN );
        task->requires ( Task::OldDW, u_label, nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
        for ( size_t D = 0; D < DIM; ++D )
        {
            task->computes ( du_label[D] );
            task->computes ( ddu_label[D] );
        }
        sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
    }
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < !MG, void >::type
Heat<VAR, DIM, STN, AMR>::scheduleTimeAdvance_dbg_derivatives_error (
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleTimeAdvance_dbg_derivatives_error" << std::endl;

    Task * task = scinew Task ( "Heat::task_time_advance_dbg_derivatives_error", this, &Heat::task_time_advance_dbg_derivatives_error );
    task->requires ( Task::NewDW, subproblems_label, Ghost::None, 0 );
    for ( size_t D = 0; D < DIM; ++D )
    {
        task->requires ( Task::NewDW, du_label[D], Ghost::None, 0 );
        task->requires ( Task::NewDW, ddu_label[D], Ghost::None, 0 );
        task->computes ( error_du_label[D] );
        task->computes ( error_ddu_label[D] );
    }
    task->computes ( u_normH10_label );
    task->computes ( error_normH10_label );
    task->computes ( u_normH20_label );
    task->computes ( error_normH20_label );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < MG, void >::type
Heat<VAR, DIM, STN, AMR>::scheduleTimeAdvance_dbg_derivatives_error (
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleTimeAdvance_dbg_derivatives_error" << std::endl;

    if ( !level->hasCoarserLevel() ) scheduleTimeAdvance_dbg_derivatives_error < !MG > ( level, sched );
    else
    {
        Task * task = scinew Task ( "Heat::task_time_advance_dbg_derivatives_error", this, &Heat::task_time_advance_dbg_derivatives_error );
        task->requires ( Task::NewDW, subproblems_label, Ghost::None, 0 );
        task->requires ( Task::NewDW, subproblems_label, nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
        for ( size_t D = 0; D < DIM; ++D )
        {
            task->requires ( Task::NewDW, du_label[D], Ghost::None, 0 );
            task->requires ( Task::NewDW, ddu_label[D], Ghost::None, 0 );
            task->computes ( error_du_label[D] );
            task->computes ( error_ddu_label[D] );
        }
        task->computes ( u_normH10_label );
        task->computes ( error_normH10_label );
        task->computes ( u_normH20_label );
        task->computes ( error_normH20_label );
        sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
    }
}
#endif

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < !MG, void >::type
Heat<VAR, DIM, STN, AMR>::scheduleTimeAdvance_solution (
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleTimeAdvance_solution " << std::endl;

    Task * task = scinew Task ( "Heat::task_time_advance_solution", this, &Heat::task_time_advance_solution );
    task->requires ( Task::OldDW, subproblems_label, Ghost::None, 0 );
    task->requires ( Task::OldDW, u_label, FGT, FGN );
    task->computes ( u_label );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < MG, void >::type
Heat<VAR, DIM, STN, AMR>::scheduleTimeAdvance_solution (
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleTimeAdvance_solution" << std::endl;

    if ( !level->hasCoarserLevel() ) scheduleTimeAdvance_solution < !MG > ( level, sched );
    else
    {
        Task * task = scinew Task ( "Heat::task_time_advance_solution", this, &Heat::task_time_advance_solution );
        task->requires ( Task::OldDW, subproblems_label, Ghost::None, 0 );
        task->requires ( Task::OldDW, subproblems_label, nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
        task->requires ( Task::OldDW, u_label, FGT, FGN );
        task->requires ( Task::OldDW, u_label, nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
        task->computes ( u_label );
        sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
    }
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::scheduleTimeAdvance_solution_error (
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleTimeAdvance_solution_error" << std::endl;

    Task * task = scinew Task ( "Heat::task_time_advance_solution_error", this, &Heat::task_time_advance_solution_error );
    task->requires ( Task::NewDW, u_label, Ghost::None, 0 );
    task->computes ( delta_u_label );
    task->computes ( error_u_label );
    task->computes ( u_normL2_label );
    task->computes ( error_normL2_label );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::scheduleRefine
(
    const PatchSet * new_patches,
    SchedulerP & sched
)
{
    // we need to create subproblems for new patches.
    // moreover, since SchedulerCommon::copyDataToNewGrid is not copying PerPatch
    // variable to the new grid (which is fine since the geometry -thus the subproblems-
    // has changed) we need to schedule their creation within scheduleRefine/scheduleRefineInterface
    // since this tasks are compiled separately from those scheduled by scheduleTimeAdvance
    cout_heat_scheduling << "scheduleRefine" << std::endl;

    const Level * level = getLevel ( new_patches );

    if ( !is_first_schedule_refine )
    {
        scheduleRefine_subproblems ( new_patches, sched );
        is_first_schedule_refine = true;
    };

    // no need to refine on coarser level
    if ( level->hasCoarserLevel() )
        scheduleRefine_solution ( new_patches, sched );
}

/**
 * @remark we need to schedule all levels before task_error_estimate_solution to avoid
 * the error "Failure finding [subproblems , coarseLevel, MI: none, NewDW
 * (mapped to dw index 1), ####] for Heat::task_error_estimate_solution",
 * on patch #, Level #, on material #, resource (rank): #" while compiling the
 * TaskGraph
 */
template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::scheduleRefine_subproblems (
    const PatchSet * new_patches,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleRefine_update_subproblems" << std::endl;

    const GridP & grid = getLevel ( new_patches )->getGrid();
    for ( int l = 0; l < grid->numLevels(); ++l )
    {
        Task * task = scinew Task ( "Heat::task_refine_update_subproblems", this, &Heat::task_initialize_subproblems );
        task->computes ( subproblems_label );
        sched->addTask ( task, grid->getLevel ( l )->eachPatch(), this->m_materialManager->allMaterials() );
    }
}
template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::scheduleRefine_solution (
    const PatchSet * new_patches,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleRefine_solution" << std::endl;

        Task * task = scinew Task ( "Heat::task_refine_solution", this, &Heat::task_refine_solution );
        task->requires ( Task::NewDW, u_label, nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
        task->requires ( Task::NewDW, subproblems_label, Ghost::None, 0 );
        task->requires ( Task::NewDW, subproblems_label, nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
        task->computes ( u_label );
        sched->addTask ( task, new_patches, this->m_materialManager->allMaterials() );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::scheduleRefineInterface (
    const LevelP & /*level_fine*/,
    SchedulerP & /*sched*/,
    bool /*need_old_coarse*/,
    bool /*need_new_coarse*/
)
{};

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::scheduleCoarsen
(
    const LevelP & level_coarse,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleCoarsen" << std::endl;

    scheduleCoarsen_solution ( level_coarse, sched );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::scheduleCoarsen_solution (
    const LevelP & level_coarse,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleCoarsen" << std::endl;

    Task * task = scinew Task ( "Heat::task_coarsen_solution", this, &Heat::task_coarsen_solution );
    task->requires ( Task::NewDW, u_label, nullptr, Task::FineLevel, nullptr, Task::NormalDomain, Ghost::None, 0 );
    task->requires ( Task::NewDW, subproblems_label, Ghost::None, 0 );
    task->requires ( Task::NewDW, subproblems_label, nullptr, Task::FineLevel, nullptr, Task::NormalDomain, Ghost::None, 0 );
    task->modifies ( u_label );
    sched->addTask ( task, level_coarse->eachPatch(), this->m_materialManager->allMaterials() );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::scheduleErrorEstimate
(
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleErrorEstimate" << std::endl;

    scheduleErrorEstimate_solution<AMR> ( level, sched );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < !MG, void >::type
Heat<VAR, DIM, STN, AMR>::scheduleErrorEstimate_solution (
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleErrorEstimate_solution" << std::endl;

    Task * task = scinew Task ( "Heat::task_error_estimate_solution", this, &Heat::task_error_estimate_solution );
    task->requires ( Task::NewDW, subproblems_label, Ghost::None, 0 );
    task->requires ( Task::NewDW, u_label, FGT, FGN );
    task->modifies ( this->m_regridder->getRefineFlagLabel(), this->m_regridder->refineFlagMaterials() );
    task->modifies ( this->m_regridder->getRefinePatchFlagLabel(), this->m_regridder->refineFlagMaterials() );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < MG, void >::type
Heat<VAR, DIM, STN, AMR>::scheduleErrorEstimate_solution (
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleErrorEstimate_solution" << std::endl;

    if ( !level->hasCoarserLevel() ) scheduleErrorEstimate_solution < !MG > ( level, sched );
    else
    {
        Task * task = scinew Task ( "Heat::task_error_estimate_solution", this, &Heat::task_error_estimate_solution );
        task->requires ( Task::NewDW, subproblems_label, Ghost::None, 0 );
        task->requires ( Task::NewDW, subproblems_label, nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
        task->requires ( Task::NewDW, u_label, FGT, FGN );
        task->requires ( Task::NewDW, u_label, nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
        task->modifies ( this->m_regridder->getRefineFlagLabel(), this->m_regridder->refineFlagMaterials() );
        task->modifies ( this->m_regridder->getRefinePatchFlagLabel(), this->m_regridder->refineFlagMaterials() );
        sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
    }
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void Heat<VAR, DIM, STN, AMR>::scheduleInitialErrorEstimate
(
    const LevelP & level,
    SchedulerP & sched
)
{
    cout_heat_scheduling << "scheduleInitialErrorEstimate" << std::endl;

    scheduleErrorEstimate ( level, sched );
}

// TASKS

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::task_initialize_subproblems (
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset *,
    DataWarehouse *,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();

    dbg_out1 << myrank << "==== Heat::task_initialize_subproblems ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << myrank << "== Patch: " << *patch << " Level: " << patch->getLevel()->getIndex() << std::endl;

        Variable < PP, SubProblems < HeatProblem<VAR, STN> > > subproblems;
        subproblems.setData ( scinew SubProblems < HeatProblem<VAR, STN> > ( this, u_label, subproblems_label, material, patch, &c2f ) );
        dw_new->put ( subproblems, subproblems_label, material, patch );
    }

    dbg_out2 << myrank << std::endl;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::task_initialize_solution (
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset *,
    DataWarehouse *,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();

    dbg_out1 << myrank << "==== Heat::task_initialize_solution ====" << std::endl;

    BBox box;
    getLevel ( patches )->getGrid()->getSpatialRange ( box );
    Vector L = box.max().asVector();

    ASSERTMSG ( DIM < D2 || L[Y] == L[X], "grid geometry must be a square" );
    ASSERTMSG ( DIM < D3 || L[Z] == L[X], "grid geometry must be a cube" );

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << myrank << "== Patch: " << *patch << " Level: " << patch->getLevel()->getIndex() << std::endl;

        BlockRange range ( this->get_range ( patch ) );
        dbg_out3 << myrank << "= Iterating over range " << range << std::endl;

        DWView < ScalarField<double>, VAR, DIM > u ( dw_new, u_label, material, patch );
        parallel_for ( range, [patch, &L, &u, this] ( int i, int j, int k )->void { initialize_solution ( {i, j, k}, patch, L[X], u ); } );
    }

    dbg_out2 << myrank << std::endl;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::task_compute_stable_timestep (
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset *,
    DataWarehouse *,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();

    dbg_out1 << myrank << "==== Heat::task_compute_stable_timestep ====" << std::endl;
    dw_new->put ( delt_vartype ( delt ), this->getDelTLabel(), getLevel ( patches ) );
    dbg_out2 << myrank << std::endl;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::task_time_advance_subproblems (
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset *,
    DataWarehouse * dw_old,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();

    dbg_out1 << myrank << "==== Heat::task_time_advance_subproblems ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << myrank << "== Patch: " << *patch << " Level: " << patch->getLevel()->getIndex() << std::endl;

        if ( dw_old->exists ( subproblems_label, material, patch ) )
        {
            Variable < PP, SubProblems < HeatProblem<VAR, STN> > > subproblems;
            dw_old->get ( subproblems, subproblems_label, material, patch );
            dw_new->put ( subproblems, subproblems_label, material, patch );
            dbg_out4 << "subproblems moved from OldDW to NewDW" << std::endl;
        }
        else // after a regrid all patches are new thus subproblems does not exists in old db
            // not bad since we want re recompute them!
        {
            Variable < PP, SubProblems < HeatProblem<VAR, STN> > > subproblems;
            subproblems.setData ( scinew SubProblems < HeatProblem<VAR, STN> > ( this, u_label, subproblems_label, material, patch, &c2f ) );
            dw_new->put ( subproblems, subproblems_label, material, patch );
            dbg_out4 << "subproblems initialized in NewDW" << std::endl;
        }
    }

    dbg_out2 << myrank << std::endl;
}

#ifdef PhaseField_Heat_DBG_DERIVATIVES
template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::task_time_advance_dbg_derivatives (
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset *,
    DataWarehouse * dw_old,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();

    dbg_out1 << myrank << "==== Heat::task_time_advance_dbg_derivatives ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << myrank << "== Patch: " << *patch << " Level: " << patch->getLevel()->getIndex() << std::endl;

        DWView < VectorField<double, DIM>, VAR, DIM > du ( dw_new, du_label, material, patch );
        DWView < VectorField<double, DIM>, VAR, DIM > ddu ( dw_new, ddu_label, material, patch );

        Variable < PP, SubProblems < HeatProblem<VAR, STN> > > subproblems;
        dw_new->get ( subproblems, subproblems_label, material, patch );
        auto problems = subproblems.get().get_rep();

        for ( const auto & p : *problems )
        {
            dbg_out3 << myrank << "= Iterating over " << p << std::endl;
            FDView < ScalarField<const double>, STN > & u_old = p.template get_fd_view<U> ( dw_old );
            parallel_for ( p.get_range(), [patch, &u_old, &du, &ddu, this] ( int i, int j, int k )->void { time_advance_dbg_derivatives ( {i, j, k}, u_old, du, ddu ); } );
        }
    }

    dbg_out2 << myrank << std::endl;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::task_time_advance_dbg_derivatives_error
(
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset * /*matls*/,
    DataWarehouse * dw_old,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();

    dbg_out1 << myrank << "==== Heat::task_time_advance_dbg_derivatives_error ====" << std::endl;

    std::array<double, 4> norms {{ 0., 0., 0., 0. }}; // { u_normH10, u_normH20, error_normH10, error_normH20 }

    simTime_vartype simTimeVar;
    dw_old->get ( simTimeVar, VarLabel::find ( simTime_name ) );
    double simTime = simTimeVar;

    BBox box;
    getLevel ( patches )->getGrid()->getSpatialRange ( box );
    Vector L = box.max() - box.min();

    ASSERTMSG ( DIM < D2 || L[Y] == L[X], "grid geometry must be a square" );
    ASSERTMSG ( DIM < D3 || L[Z] == L[X], "grid geometry must be a cube" );

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << myrank << "== Patch: " << *patch << std::endl;

        DWView < VectorField<const double, DIM>, VAR, DIM > du ( dw_new, du_label, material, patch );
        DWView < VectorField<const double, DIM>, VAR, DIM > ddu ( dw_new, ddu_label, material, patch );

        DWView < VectorField<double, DIM>, VAR, DIM > error_du ( dw_new, error_du_label, material, patch );
        DWView < VectorField<double, DIM>, VAR, DIM > error_ddu ( dw_new, error_ddu_label, material, patch );

        BlockRange range ( this->get_range ( patch ) );
        dbg_out3 << "= Iterating over range " << range << std::endl;

        parallel_reduce_sum (
            range,
            [patch, &simTime, &L, &du, &ddu, &error_du, &error_ddu, this] ( int i, int j, int k, std::array<double, 4> & norms )->void { time_advance_dbg_derivatives_error ( {i, j, k}, patch, simTime, L[0], du, ddu, error_du, error_ddu, norms[0], norms[1], norms[2], norms[3] ); },
            norms
        );
    }

    dw_new->put ( sum_vartype ( norms[0] ), u_normH10_label );
    dw_new->put ( sum_vartype ( norms[1] ), u_normH20_label );
    dw_new->put ( sum_vartype ( norms[2] ), error_normH10_label );
    dw_new->put ( sum_vartype ( norms[3] ), error_normH20_label );

    dbg_out2 << myrank << std::endl;
}
#endif

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::task_time_advance_solution (
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset *,
    DataWarehouse * dw_old,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();

    dbg_out1 << myrank << "==== Heat::task_time_advance_solution ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << myrank << "== Patch: " << *patch << " Level: " << patch->getLevel()->getIndex() << std::endl;

        DWView < ScalarField<double>, VAR, DIM > u_new ( dw_new, u_label, material, patch );

        Variable < PP, SubProblems < HeatProblem<VAR, STN> > > subproblems;
        dw_new->get ( subproblems, subproblems_label, material, patch );
        auto problems = subproblems.get().get_rep();

        for ( const auto & p : *problems )
        {
            dbg_out3 << myrank << "= Iterating over " << p << std::endl;

            FDView < ScalarField<const double>, STN > & u_old = p.template get_fd_view<U> ( dw_old );
            parallel_for ( p.get_range(), [patch, &u_old, &u_new, this] ( int i, int j, int k )->void { time_advance_solution ( {i, j, k}, u_old, u_new ); } );
        }
    }

    dbg_out2 << myrank << std::endl;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::task_time_advance_solution_error
(
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset * /*matls*/,
    DataWarehouse * dw_old,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();

    dbg_out1 << myrank << "==== Heat::task_time_advance_solution_error ====" << std::endl;

    std::array<double, 2> norms {{ 0., 0. }}; // { u_normL2, error_normL2 }

    simTime_vartype simTimeVar;
    dw_old->get ( simTimeVar, VarLabel::find ( simTime_name ) );
    double simTime = simTimeVar;

    BBox box;
    getLevel ( patches )->getGrid()->getSpatialRange ( box );
    Vector L = box.max() - box.min();

    ASSERTMSG ( DIM < D2 || L[Y] == L[X], "grid geometry must be a square" );
    ASSERTMSG ( DIM < D3 || L[Z] == L[X], "grid geometry must be a cube" );

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << myrank << "== Patch: " << *patch << std::endl;

        DWView < ScalarField<const double>, VAR, DIM > u ( dw_new, u_label, material, patch );

        DWView < ScalarField<double>, VAR, DIM > delta_u ( dw_new, delta_u_label, material, patch );
        DWView < ScalarField<double>, VAR, DIM > error_u ( dw_new, error_u_label, material, patch );

        BlockRange range ( this->get_range ( patch ) );
        dbg_out3 << "= Iterating over range " << range << std::endl;

        parallel_reduce_sum ( range, [patch, &simTime, &L, &u, &delta_u, &error_u, this] ( int i, int j, int k, std::array<double, 2> & norms )->void { time_advance_solution_error ( {i, j, k}, patch, simTime, L[X], u, delta_u, error_u, norms[0], norms[1] ); }, norms );
    }

    dw_new->put ( sum_vartype ( norms[0] ), u_normL2_label );
    dw_new->put ( sum_vartype ( norms[1] ), error_normL2_label );

    dbg_out2 << myrank << std::endl;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::task_refine_solution
(
    const ProcessorGroup * myworld,
    const PatchSubset * patches_fine,
    const MaterialSubset * /*matls*/,
    DataWarehouse * /*dw_old*/,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();

    dbg_out1 << myrank << "==== Heat::task_refine_solution ====" << std::endl;

    for ( int p = 0; p < patches_fine->size(); ++p )
    {
        const Patch * patch_fine = patches_fine->get ( p );
        dbg_out2 << myrank << "== Fine Patch: " << *patch_fine << " Level: " << patch_fine->getLevel()->getIndex() << std::endl;

        DWView < ScalarField<double>, VAR, DIM > u_fine ( dw_new, u_label, material, patch_fine );

        AMRInterpolator < HeatProblem<VAR, STN>, U, C2F > u_coarse_interp ( dw_new, u_label, subproblems_label, material, patch_fine );

        BlockRange range_fine ( this->get_range ( patch_fine ) );
        dbg_out3 << myrank << "= Iterating over fine range" << range_fine << std::endl;
        parallel_for ( range_fine, [&u_coarse_interp, &u_fine, this] ( int i, int j, int k )->void { refine_solution ( {i, j, k}, u_coarse_interp, u_fine ); } );
    }

    dbg_out2 << myrank << std::endl;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::task_coarsen_solution (
    const ProcessorGroup * myworld,
    const PatchSubset * patches_coarse,
    const MaterialSubset * /*matls*/,
    DataWarehouse * /*dw_old*/,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();

    dbg_out1 << myrank << "==== Heat::task_coarsen_solution " << std::endl;

    for ( int p = 0; p < patches_coarse->size(); ++p )
    {
        const Patch * patch_coarse = patches_coarse->get ( p );
        dbg_out2 << myrank << "== Coarse Patch: " << *patch_coarse << " Level: " << patch_coarse->getLevel()->getIndex() << std::endl;

        DWView < ScalarField<double>, VAR, DIM > u_coarse ( dw_new, u_label, material, patch_coarse );

        AMRRestrictor < HeatProblem<VAR, STN>, U, F2C > u_fine_restr ( dw_new, u_label, subproblems_label, material, patch_coarse, false );

        for ( const auto & region : u_fine_restr.get_support() )
        {
            dbg_out3 << myrank << "= Iterating over coarse cells region " << region << std::endl;
            BlockRange range_coarse (
                Max ( region.getLow(), this->get_low ( patch_coarse ) ),
                Min ( region.getHigh(), this->get_high ( patch_coarse ) )
            );

            parallel_for ( range_coarse, [&u_fine_restr, &u_coarse, this] ( int i, int j, int k )->void { coarsen_solution ( {i, j, k}, u_fine_restr, u_coarse ); } );
        }
    }

    dbg_out2 << myrank << std::endl;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::task_error_estimate_solution
(
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset * /*matls*/,
    DataWarehouse * /*dw_old*/,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();

    dbg_out1 << myrank << "==== Heat::task_error_estimate_solution " << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << myrank << "== Patch: " << *patch << " Level: " << patch->getLevel()->getIndex() << std::endl;

        Variable<PP, PatchFlag> refine_patch_flag;
        dw_new->get ( refine_patch_flag, this->m_regridder->getRefinePatchFlagLabel(), material, patch );

        PatchFlag * patch_flag_refine = refine_patch_flag.get().get_rep();

        bool refine_patch = false;

        DWView < ScalarField<int>, CC, DIM > refine_flag ( dw_new, this->m_regridder->getRefineFlagLabel(), material, patch );

        Variable < PP, SubProblems < HeatProblem<VAR, STN> > > subproblems;
        dw_new->get ( subproblems, subproblems_label, material, patch );

        auto problems = subproblems.get().get_rep();

        for ( const auto & p : *problems )
        {
            dbg_out3 << myrank << "= Iterating over " << p << std::endl;

            FDView < ScalarField<const double>, STN > & u = p.template get_fd_view<U> ( dw_new );
            parallel_reduce_sum ( p.get_range(), [&u, &refine_flag, &refine_patch, this] ( int i, int j, int k, bool & refine_patch )->void { error_estimate_solution<VAR> ( {i, j, k}, u, refine_flag, refine_patch ); }, refine_patch );
        }

        if ( refine_patch )
        {
            dbg_out3 << myrank << "= Setting refine flag" << std::endl;
            patch_flag_refine->set();
        }
    }

    dbg_out2 << myrank << std::endl;
}

// IMPLEMENTATIONS

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::initialize_solution (
    const IntVector & id,
    Patch const * patch,
    const double & L,
    View < ScalarField<double> > & u
)
{
    Vector v ( this->get_position ( patch, id ).asVector() );

    // BUG workaround
    std::stringstream ss;
    ss << v << std::endl;

    double a = M_PI_2 / L;
    u[id] = 1.;
    for ( size_t d = 0; d < DIM; ++d )
        u[id] *= cos ( a * v[d] );
}

#ifdef PhaseField_Heat_DBG_DERIVATIVES
template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::time_advance_dbg_derivatives (
    const IntVector & id,
    const FDView < ScalarField<const double>, STN > & u_old,
    View < VectorField<double, DIM> > & du,
    View < VectorField<double, DIM> > & ddu
)
{
    du[X][id] = u_old.dx ( id );
    ddu[X][id] = u_old.dxx ( id );
    if ( DIM > D1 )
    {
        du[Y][id] = u_old.dy ( id );
        ddu[Y][id] = u_old.dyy ( id );
    }
    if ( DIM > D2 )
    {
        du[Z][id] = u_old.dz ( id );
        ddu[Z][id] = u_old.dzz ( id );
    }
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::time_advance_dbg_derivatives_error (
    const IntVector & id,
    const Patch * patch,
    const double & t,
    const double & L,
    View < VectorField<const double, DIM> > & du,
    View < VectorField<const double, DIM> > & ddu,
    View < VectorField<double, DIM> > & error_du,
    View < VectorField<double, DIM> > & error_ddu,
    double & u_normH10,
    double & u_normH20,
    double & error_normH10,
    double & error_normH20
)
{
    const Level  * level ( patch->getLevel() );
    if ( level->hasFinerLevel() && level->getFinerLevel()->containsCell ( level->mapCellToFiner ( id ) ) )
    {
        for ( size_t i = 0; i < DIM; ++i )
        {
            error_du[i][id] = 0.;
            error_ddu[i][id] = 0.;
        }
    }
    else
    {
        Vector v ( this->get_position ( patch, id ).asVector() );
        Vector dCell ( level->dCell() );

        // BUG workaround
        std::stringstream ss;
        ss << v << std::endl;

        double a = M_PI_2 / L;
        double da2 = a * a * static_cast<double> ( DIM );
        double area = 1.;
        double int_u = exp ( -da2 * ( t + delt ) );
        double int_u2 = exp ( -2.* da2 * ( t + delt ) );
        double laplacian = 0;
        for ( size_t i = 0; i < DIM; ++i )
        {
            double vi0 = v[i] - 0.5 * dCell[i];
            double vi1 = v[i] + 0.5 * dCell[i];
            area *= dCell[i];
            int_u *= ( sin ( a * vi1 ) - sin ( a * vi0 ) ) / a;
            int_u2 *= ( vi1 - vi0 ) / 2. + ( sin ( 2.*a * vi1 ) - sin ( 2.*a * vi0 ) ) / ( 4. * a );
            laplacian += ddu[i][id];
        }

        for ( size_t i = 0; i < DIM; ++i )
        {
            double int_diu = a * exp ( -da2 * ( t + delt ) );
            double int_diu2 = int_diu * int_diu;

            for ( size_t j = 0; j < DIM; ++j )
            {
                double vj0 = v[j] - 0.5 * dCell[j];
                double vj1 = v[j] + 0.5 * dCell[j];
                int_diu *= ( j == i ) ? ( cos ( a * vj1 ) - cos ( a * vj0 ) ) / a
                           : ( sin ( a * vj1 ) - sin ( a * vj0 ) ) / a;
                int_diu2 *= ( j == i ) ?
                            ( vj1 - vj0 ) / 2. - ( sin ( 2.*a * vj1 ) - sin ( 2.*a * vj0 ) ) / ( 4. * a ) :
                            ( vj1 - vj0 ) / 2. + ( sin ( 2.*a * vj1 ) - sin ( 2.*a * vj0 ) ) / ( 4. * a );
            }

            double int_die2 = area * du[i][id] * du[i][id] - 2. * du[i][id] * int_diu + int_diu2;
            double int_ddie2 = area * ddu[i][id] * ddu[i][id] + 2. * a * a * ddu[i][id] * int_u + a * a * a * a * int_u2;

            error_du[i][id] = sqrt ( int_die2 / area );
            error_ddu[i][id] = sqrt ( int_ddie2 / area );

            u_normH10 += int_diu2;
            error_normH10 += int_die2;
        }
        u_normH20 += da2 * da2 * int_u2;

        error_normH20 += area * laplacian * laplacian + 2. * da2 * laplacian * int_u + da2 * da2 * int_u2;
    }
}
#endif

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::time_advance_solution (
    const IntVector & id,
    const FDView < ScalarField<const double>, STN > & u_old,
    View < ScalarField<double> > & u_new
)
{
    double delta_u = delt * alpha * u_old.laplacian ( id );
    u_new[id] = u_old[id] + delta_u;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::time_advance_solution_error
(
    const IntVector & id,
    const Patch * patch,
    const double & t,
    const double & L,
    View < ScalarField<const double> > & u,
    View < ScalarField<double> > & delta_u,
    View < ScalarField<double> > & error_u,
    double & u_normL2,
    double & error_normL2
)
{
    const Level  * level ( patch->getLevel() );
    if ( level->hasFinerLevel() && level->getFinerLevel()->containsCell ( level->mapCellToFiner ( id ) ) )
    {
        error_u[id] = 0;
        delta_u[id] = 0;
    }
    else
    {
        Vector v ( this->get_position ( patch, id ).asVector() );
        Vector dCell ( level->dCell() );

        // BUG workaround
        std::stringstream ss;
        ss << v << std::endl;

        double a = M_PI_2 / L;
        double da2 = a * a * static_cast<double> ( DIM );
        double area = 1.;
        double del_u = exp ( -da2 * ( t + delt ) );
        double int_u = del_u;
        double int_u2 = del_u * del_u;

        for ( size_t i = 0; i < DIM; ++i )
        {
            double vi0 = v[i] - 0.5 * dCell[i];
            double vi1 = v[i] + 0.5 * dCell[i];
            area *= dCell[i];
            del_u *= cos ( a * v[i] );;
            int_u *= ( sin ( a * vi1 ) - sin ( a * vi0 ) ) / a;
            int_u2 *= ( vi1 - vi0 ) / 2. + ( sin ( 2.*a * vi1 ) - sin ( 2.*a * vi0 ) ) / ( 4. * a );
        }

        double int_e2 = area * u[id] * u[id] - 2. * u[id] * int_u + int_u2;

        delta_u[id] = del_u - u[id];
        error_u[id] = sqrt ( int_e2 / area );

        u_normL2 += int_u2;
        error_normL2 += int_e2;
    }
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::refine_solution
(
    const IntVector id_fine,
    const View < ScalarField<const double> > & u_coarse_interp,
    View < ScalarField<double> > & u_fine
)
{
    u_fine[id_fine] = u_coarse_interp[id_fine];
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
Heat<VAR, DIM, STN, AMR>::coarsen_solution
(
    const IntVector id_coarse,
    const View < ScalarField<const double> > & u_fine_restr,
    View < ScalarField<double> > & u_coarse
)
{
    u_coarse[id_coarse] = u_fine_restr[id_coarse];
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template<VarType V>
typename std::enable_if < V == CC, void >::type
Heat<VAR, DIM, STN, AMR>::error_estimate_solution
(
    const IntVector id,
    FDView < ScalarField<const double>, STN > & u,
    View < ScalarField<int> > & refine_flag,
    bool & refine_patch
)
{
    bool refine = false;
    auto grad = u.gradient ( id );
    double err2 = 0;
    for ( size_t d = 0; d < DIM; ++d )
        err2 += grad[d] * grad[d];
    refine = err2 > refine_threshold * refine_threshold;
    refine_flag[id] = refine;
    refine_patch |= refine;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template<VarType V>
typename std::enable_if < V == NC, void >::type
Heat<VAR, DIM, STN, AMR>::error_estimate_solution
(
    const IntVector id,
    FDView < ScalarField<const double>, STN > & u,
    View < ScalarField<int> > & refine_flag,
    bool & refine_patch
)
{
    bool refine = false;
    auto grad = u.gradient ( id );
    double err2 = 0;
    for ( size_t d = 0; d < DIM; ++d )
        err2 += grad[d] * grad[d];
    refine = err2 > refine_threshold * refine_threshold;
    if ( refine_flag.is_defined_at ( id ) ) refine_flag[id] = refine;
    for ( size_t d = 0; d < DIM; ++d )
    {
        IntVector id0 ( id );
        id0[d] -= 1;
        if ( refine_flag.is_defined_at ( id0 ) ) refine_flag[id0] = refine;
    }

    refine_patch |= refine;
}

} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_Applications_Heat_h
