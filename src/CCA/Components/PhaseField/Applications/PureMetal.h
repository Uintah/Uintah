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
 * @file CCA/Components/PhaseField/Applications/PureMetal.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_Applications_PureMetal_h
#define Packages_Uintah_CCA_Components_PhaseField_Applications_PureMetal_h

#include <CCA/Components/PhaseField/Util/Definitions.h>
#include <CCA/Components/PhaseField/Util/Expressions.h>
#include <CCA/Components/PhaseField/Util/BlockRangeIO.h>
#include <CCA/Components/PhaseField/DataTypes/PureMetalProblem.h>
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
#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/Regridder.h>
#include <Core/Grid/Variables/PerPatchVars.h>

namespace Uintah
{
namespace PhaseField
{

/// Debugging stream for component schedulings
static DebugStream cout_pure_metal_scheduling ( "PURE METAL SCHEDULING", false );

/**
 * @brief PureMetal PhaseField applications
 *
 * Implements a Finite Difference solver for the simulation of the anisotropic
 * solidification of an under-cooled metal around a solid seed using the model
 * from
 *
 * A, Karma and W.-J. Rappel,
 * "Phase-Field Method for Computationally Efficient Modeling of Solidification
 * with Arbitrary Interface Kinetics",
 * Phys.Rev.E, 1996.
 *
 * Phase-field equation for \f$\psi : \Omega \to [-1. 1]\f$
 * \f[
 * \tau_0 A^2 \dot \psi = \psi (1-\psi^2) - \lambda u (\psi^2 - 1)^2
 *                      + W_0^2 A^2 \nabla^2 \psi
 *                      + W_0^2 (A^2_x - \partial_y B_{xy} - \partial_y B_{xz} \psi_x
 *                      + W_0^2 (A^2_y + \partial_y B_{xy} - \partial_y B_{yz} \psi_y
 *                      + W_0^2 (A^2_y + \partial_y B_{xz} + \partial_y B_{yz} \psi_z
 * \f]
 * (for 3D problems all quantities with \f$z\f$ in the subscript are null)
 *
 * non-dimensional temperature equation for \f$u:\Omega \to \mathbb R\f$
 * \f[
 * \dot u = \alpha \nabla^2 u + 1/2 \dot \psi
 * \f]
 *
 * with the following anisotropy functions
 * \f[ A    = 1 - 3 \epsilon + 4\epsilon \sum n_i^4 \f]
 * \f[ B_ij = 16 \epsilon A \psi_i \psi_j (\psi_i^2-\psi_j^2) / |\nabla\psi|^4 \f]
 * where \f$ n=\nabla\psi/|\nabla\psi \f$
 *
 * The model parameters are:
 * - \f$ \lambda \f$  coupling parameter
 * - \f$ W_0 \f$      width scaling (\f$ W = W_0 A \f$ is the interface width)
 * - \f$ \tau_0 \f$   time scaling (\f$ \tau = \tau_0 A^2 \f$ is the characteristic
 *                    time of attachment of atoms at the interface)
 * - \f$ \epsilon \f$ anisotropy strength
 * - \f$ \alpha \f$   thermal diffusivity
 *
 * The following no-dimensionalizations are performed
 * - \f$ x \mapsto x/W_0 \f$
 * - \f$ t \mapsto t/\tau_0 \f$
 * - \f$ \alpha \mapsto \alpha\tau_0/W_0^2 \f$
 *
 * @tparam VAR type of variable representation
 * @tparam DIM problem dimensions
 * @tparam STN finite-difference stencil
 * @tparam AMR whether to use adaptive mesh refinement
 */
template<VarType VAR, DimType DIM, StnType STN, bool AMR = false>
class PureMetal
    : public Application<VAR, DIM, STN, AMR>
    , public Implementation<PureMetal<VAR, DIM, STN, AMR>, UintahParallelComponent, const ProcessorGroup *, const MaterialManagerP, int>
{
    /// Problem material index (only one SimpleMaterial)
    static constexpr int material = 0;

    /// Number of anisotropy functions B
    static constexpr size_t BSZ = combinations<DIM, 2>::value;

    /// Indices for anisotropy functions B
    static constexpr size_t XY = 0, XZ = 1, YZ = 2;

    /// Number of ghost elements required by STN (on the same level)
    static constexpr int FGN = get_stn<STN>::ghosts;

    /// Type of ghost elements required by VAR and STN (on coarser level)
    static constexpr Ghost::GhostType FGT = FGN ? get_var<VAR>::ghost_type : Ghost::None;

    /// Number of ghost elements required by STN (on coarser level)
    /// @remark this should depend on FCI bc type but if fixed for simplicity
    static constexpr int CGN = 1;

    /// Type of ghost elements required by VAR and STN (on the same level)
    static constexpr Ghost::GhostType CGT = CGN ? get_var<VAR>::ghost_type : Ghost::None;

    /// Interpolation type for refinement
    static constexpr FCIType C2F = ( VAR == CC ) ? I0 : I1; // TODO make template parameter

    /// Restriction type for coarsening
    static constexpr FCIType F2C = ( VAR == CC ) ? I1 : I0; // TODO make template parameter

    /// If grad_psi_norm2 is less than tol than psi is considered constant when computing anisotropy terms
    static constexpr double tol = 1.e-6;

    /// Index of variables within PureMetalProblem
    static constexpr size_t PSI = 0; ///< Index for phase-field
    static constexpr size_t U = 1;   ///< Index for non-dimensional temperature field
    static constexpr size_t A2 = 2;  ///< Index for the square of the anisotropy function
    static constexpr size_t B = 3;   ///< Index for the anisotropy terms B_ij

public: // STATIC MEMBERS

    /// Class name as used by ApplicationFactory
    static const std::string Name;

protected: // MEMBERS

    /// Output streams for debugging
    DebugStream dbg_out1, dbg_out2, dbg_out3, dbg_out4;

    // Labels for variables to be stored into the DataWarehouse
    const VarLabel * subproblems_label;
    const VarLabel * psi_label, * u_label;
    const VarLabel * grad_psi_norm2_label;
    const VarLabel * a_label, * a2_label;
    std::array<const VarLabel *, DIM> grad_psi_label;
    std::array<const VarLabel *, BSZ> b_label;

    /// Time step size
    double delt;

    /// Coupling parameter
    double lambda;

    /// Non-dimensional thermal diffusivity
    double alpha;

    /// Anisotropy strength
    double epsilon;

    /// Initial interface widths
    double gamma_psi, gamma_u;

    /// Initial seed radius
    double r0;

    /// Initial undercooling
    double delta;

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
     * Intantiate a PureMetal application
     *
     * @param myWorld data structure to manage mpi processes
     * @param materialManager data structure to manage materials
     * @param verbosity constrols amount of debugging output
     */
    PureMetal (
        const ProcessorGroup * myWorld,
        const MaterialManagerP materialManager,
        int verbosity = 0
    );

    /**
     * @brief Destructor
     */
    virtual ~PureMetal();

    /// Prevent copy (and move) constructor
    PureMetal ( const PureMetal & ) = delete;

    /// Prevent copy (and move) assignment
    PureMetal & operator= ( const PureMetal & ) = delete;

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
     * @brief Schedule task_initialize_grad_psi (non AMR implementation)
     *
     * Defines the dependencies and output of the task which initializes psi
     * and grad_psi_norm2 allowing sched to control its execution order.
     * Initialize also the anisotropy terms to 0
     *
     * @param level grid level to be initialized
     * @param sched scheduler to manage the tasks
     */
    template < bool MG >
    typename std::enable_if < !MG, void >::type
    scheduleInitialize_grad_psi (
        const LevelP & level,
        SchedulerP & sched
    );

    /**
     * @brief Schedule task_initialize_grad_psi (AMR implementation)
     *
     * Does nothing since grad_psi and grad_psi_norm2 are computed by
     * error_estimate_grad_psi.
     *
     * @param level unused
     * @param sched unused
     */
    template < bool MG >
    typename std::enable_if < MG, void >::type
    scheduleInitialize_grad_psi (
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

    /**
     * @brief Schedule task_time_advance_grad_psi (non AMR implementation)
     *
     * Defines the dependencies and output of the task which updates grad psi
     * and grad_psi_norm2 allowing sched to control its execution order
     *
     * @param level grid level to be updated
     * @param sched scheduler to manage the tasks
     */
    template < bool MG >
    typename std::enable_if < !MG, void >::type
    scheduleTimeAdvance_grad_psi (
        const LevelP & level,
        SchedulerP & sched
    );

    /**
     * @brief Schedule task_time_advance_grad_psi (AMR implementation)
     *
     * Defines the dependencies and output of the task which updates grad psi
     * and grad_psi_norm2 allowing sched to control its execution order
     *
     * @remark does nothing since it grad_psi and grad_psi_norm2 are already
     * computed by error_estimate_grad_psi
     *
     * @param level unused
     * @param sched unused
     */
    template < bool MG >
    typename std::enable_if < MG, void >::type
    scheduleTimeAdvance_grad_psi (
        const LevelP & level,
        SchedulerP & sched
    );

    /**
     * @brief Schedule task_time_advance_anisotropy_terms
     *
     * Defines the dependencies and output of the task which updates the
     * anisotropy terms allowing sched to control its execution order
     *
     * @param level grid level to be updated
     * @param sched scheduler to manage the tasks
     *
     * @remark since no derivative is involved there is no need to have two
     * different implementation for non AMR and non AMR cases
     */
    void
    scheduleTimeAdvance_anisotropy_terms (
        const LevelP & level,
        SchedulerP & sched
    );

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
     * @param grid grid to be populated
     * @param sched scheduler to manage the tasks
     */
    void
    scheduleRefine_subproblems (
        const GridP & grid,
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
     * @brief Schedule task_refine_grad_psi
     *
     * Defines the dependencies and output of the task which computes the
     * derivatives of psi on each one of the new_patches allowing sched to
     * control its execution order
     *
     * @param new_patches patches to be populated
     * @param sched scheduler to manage the tasks
     */
    void
    scheduleRefine_grad_psi (
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
     * @brief Schedule task_error_estimate_grad_psi (coarsest level implementation)
     *
     * Defines the dependencies and output of the task which estimates the
     * spatial discretization error allowing sched to control its execution order
     *
     * @param level level to check
     * @param sched scheduler to manage the tasks
     */
    template < bool MG >
    typename std::enable_if < !MG, void >::type
    scheduleErrorEstimate_grad_psi (
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
    scheduleErrorEstimate_grad_psi (
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
     * @brief Initialize grad psi task
     *
     * Allocate and save variable for psi derivatives for each one of the patches
     * and save it to dw_new
     * @remark initialize also anisotropy terms to 0
     *
     * @param myworld data structure to manage mpi processes
     * @param patches list of patches to be initialized
     * @param matls unused
     * @param dw_old unused
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_initialize_grad_psi (
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

    /**
     * @brief Advance grad_psi task
     *
     * Computes the gradient of the phase field using its value at the previous
     * timestep
     *
     * @remark this task is scheduled only if AMR is disabled since the the
     * gradient of the phase field for AMR simulations is already computed by
     * task_error_estimate_grad_psi
     *
     * @param myworld data structure to manage mpi processes
     * @param patches list of patches to be initialized
     * @param matls unused
     * @param dw_old DataWarehouse for previous timestep
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_time_advance_grad_psi (
        const ProcessorGroup * myworld,
        const PatchSubset * patches,
        const MaterialSubset * matls,
        DataWarehouse * dw_old,
        DataWarehouse * dw_new
    );

    /**
     * @brief Advance anisotropy terms task
     *
     * Computes A, A2 and B using the value of grad_psi at the previous timestep
     *
     * @param myworld data structure to manage mpi processes
     * @param patches list of patches to be initialized
     * @param matls unused
     * @param dw_old DataWarehouse for previous timestep
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_time_advance_anisotropy_terms (
        const ProcessorGroup * myworld,
        const PatchSubset * patches,
        const MaterialSubset * matls,
        DataWarehouse * dw_old,
        DataWarehouse * dw_new
    );

    /**
     * @brief Advance solution task
     *
     * Computes new value of psi and u using the newly computed anisotropy terms
     * together with the value of the solution and grad_psi at the previous timestep
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
     * @brief Refine solution task
     *
     * Computes interpolated value of u and psi on new refined patched
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
     * @brief Refine grad psi task
     *
     * Computes value of grad psi on new refined patched from the interpolated
     * values of psi
     *
     * @param myworld data structure to manage mpi processes
     * @param patches_fine list of patches to be initialized
     * @param matls unused
     * @param dw_old unused
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_refine_grad_psi (
        const ProcessorGroup * myworld,
        const PatchSubset * patches_fine,
        const MaterialSubset * matls,
        DataWarehouse * dw_old,
        DataWarehouse * dw_new
    );

    /**
     * @brief Coarsen solution task
     *
     * Restricted value of u and psi from refined regions to coarse patches
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
     * @brief ErrorEstimate grad_psi task
     *
     * Computes the gradient of the phase field using its value at the previous
     * timestep and set refinement flag where it is above the threshold given
     * in input
     *
     * @remark this task replaces task_time_advance_grad_psi for AMR simulations
     *
     * @param myworld data structure to manage mpi processes
     * @param patches list of patches to be initialized
     * @param matls unused
     * @param dw_old DataWarehouse for previous timestep
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_error_estimate_grad_psi (
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
     * compute initial condition for psi and u at a given grid position
     *
     * @param id grid index
     * @param patch grid patch
     * @param[out] psi view of the phase field in the new dw
     * @param[out] u view of the temperature field in the new dw
     */
    void
    initialize_solution (
        const IntVector & id,
        const Patch * patch,
        View < ScalarField<double> > & psi,
        View < ScalarField<double> > & u
    );

    /**
     * @brief Advance grad_psi implementation
     *
     * compute new value for grad_psi at a given grid position using its value
     * at the previous timestep
     *
     * @param id grid index
     * @param psi view of the phase field in the old dw
     * @param[out] grad_psi view of the phase gradient field in the new dw
     * @param[out] grad_psi_norm2 view of the norm of the phase gradient field
     * in the new dw
     */
    void
    time_advance_grad_psi (
        const IntVector & id,
        FDView < ScalarField<const double>, STN > & psi,
        View < VectorField<double, DIM > > & grad_psi,
        View < ScalarField<double> > & grad_psi_norm2
    );

    /**
     * @brief Advance anisotropy terms implementation
     *
     * compute new value for grad_psi at a given grid position using its value
     * at the previous timestep
     *
     * @param id grid index
     * @param grad_psi view of the phase gradient field in the old dw
     * @param grad_psi_norm2 view of the norm of the phase gradient field
     * in the old dw
     * @param[out] a view of A in the new dw
     * @param[out] a2 view of A2 in the new dw
     * @param[out] b view of B in the new dw
     */
    void
    time_advance_anisotropy_terms (
        const IntVector & id,
        View < VectorField<const double, DIM> > & grad_psi,
        View < ScalarField<const double> > & grad_psi_norm2,
        View < ScalarField<double> > & a,
        View < ScalarField<double> > & a2,
        View < VectorField<double, BSZ> > & b
    );

    /**
     * @brief Advance solution terms implementation
     *
     * compute new value for psi and u at a given grid position using the newly
     * computed anisotropy terms together with the value of the solution and
     * grad_psi at the previous timestep
     *
     * @param id grid index
     * @param psi_old view of the phase field in the old dw
     * @param u_old view of the temperature field in the old dw
     * @param grad_psi view of the phase gradient field in the old dw
     * @param a view of A in the new dw
     * @param a2 view of A2 in the new dw
     * @param b view of B in the new dw
     * @param[out] psi_new view of the phase field in the new dw
     * @param[out] u_new view of the temperature field in the new dw
     */
    void
    time_advance_solution (
        const IntVector & id,
        FDView < ScalarField<const double>, STN > & psi_old,
        FDView < ScalarField<const double>, STN > & u_old,
        View < VectorField<const double, DIM> > & grad_psi,
        View < ScalarField<const double> > & a,
        FDView < ScalarField<const double>, STN > & a2,
        FDView < VectorField<const double, BSZ>, STN > & b,
        View < ScalarField<double> > & psi_new,
        View < ScalarField<double> > & u_new
    );

    /**
     * @brief Refine solution implementation
     *
     * Computes interpolated value of u and psi at a given grid position

     * @param id_fine fine grid index
     * @param psi_coarse_interp interpolator of the phase field on the coarse level
     * @param u_coarse_interp interpolator of the temperature field on the coarse level
     * @param[out] psi_fine view of the phase field on the fine level
     * @param[out] u_fine view of the temperature field on the fine level
     */
    void
    refine_solution (
        const IntVector id_fine,
        const View < ScalarField<const double> > & psi_coarse_interp,
        const View < ScalarField<const double> > & u_coarse_interp,
        View < ScalarField<double> > & psi_fine,
        View < ScalarField<double> > & u_fine
    );

    /**
     * @brief Coarsen solution implementation
     *
     * Computes restricted value of u and psi at a given grid position

     * @param id_coarse coarse grid index
     * @param psi_fine_restr restrictor of the phase field on the fine level
     * @param u_fine_restr restrictor of the temperature field on the fine level
     * @param[out] psi_coarse view of the phase field on the coarse level
     * @param[out] u_coarse view of the temperature field on the coarse level
     */
    void
    coarsen_solution (
        const IntVector id_coarse,
        const View < ScalarField<const double> > & psi_fine_restr,
        const View < ScalarField<const double> > & u_fine_restr,
        View < ScalarField<double> > & psi_coarse,
        View < ScalarField<double> > & u_coarse
    );

    /**
     * @brief ErrorEstimate grad_psi implementation
     *
     * Computes the gradient of the phase field using its value at the previous
     * timestep and set refinement flag where it is above the threshold given
     * in input
     *
     * @param id grid index
     * @param psi view of the phase field in the old dw
     * @param[out] grad_psi view of the phase gradient field in the new dw
     * @param[out] grad_psi_norm2 view of the norm of the phase gradient field
     * in the new dw
     * @param[out] refine_flag view of refine flag (grid field) in the new dw
     * @param[out] refine_patch flag for patch refinement
     */
    void
    error_estimate_grad_psi (
        const IntVector & id,
        FDView < ScalarField<const double>, STN > & psi,
        View < VectorField<double, DIM> > & grad_psi,
        View < ScalarField<double> > & grad_psi_norm2,
        View< ScalarField<int> > & refine_flag,
        bool & refine_patch
    );

}; // class PureMetal

// CONSTRUCTORS/DESTRUCTOR

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
PureMetal<VAR, DIM, STN, AMR>::PureMetal (
    const ProcessorGroup * myworld,
    MaterialManagerP const materialManager,
    int verbosity
) : Application<VAR, DIM, STN, AMR> ( myworld, materialManager ),
    dbg_out1 ( "PureMetal", verbosity > 0 ),
    dbg_out2 ( "PureMetal", verbosity > 1 ),
    dbg_out3 ( "PureMetal", verbosity > 2 ),
    dbg_out4 ( "PureMetal", verbosity > 3 )
{
    subproblems_label = VarLabel::create ( "subproblems", Variable< PP, SubProblems < PureMetalProblem<VAR, STN> > >::getTypeDescription() );
    psi_label = VarLabel::create ( "psi", Variable<VAR, double>::getTypeDescription() );
    u_label = VarLabel::create ( "u", Variable<VAR, double>::getTypeDescription() );
    grad_psi_norm2_label = VarLabel::create ( "grad_psi_norm2", Variable<VAR, double>::getTypeDescription() );
    a_label = VarLabel::create ( "A", Variable<VAR, double>::getTypeDescription() );
    a2_label = VarLabel::create ( "A2", Variable<VAR, double>::getTypeDescription() );

    grad_psi_label[X] = VarLabel::create ( "psi_x", Variable<VAR, double>::getTypeDescription() );
    if ( DIM > D1 )
    {
        grad_psi_label[Y] = VarLabel::create ( "psi_y", Variable<VAR, double>::getTypeDescription() );
        b_label[XY] = VarLabel::create ( "Bxy", Variable<VAR, double>::getTypeDescription() );
    }
    if ( DIM > D2 )
    {
        grad_psi_label[Z] = VarLabel::create ( "psi_z", Variable<VAR, double>::getTypeDescription() );
        b_label[XZ] = VarLabel::create ( "Bxz", Variable<VAR, double>::getTypeDescription() );
        b_label[YZ] = VarLabel::create ( "Byz", Variable<VAR, double>::getTypeDescription() );
    }
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
PureMetal<VAR, DIM, STN, AMR>::~PureMetal()
{
    VarLabel::destroy ( subproblems_label );
    VarLabel::destroy ( psi_label );
    VarLabel::destroy ( u_label );
    VarLabel::destroy ( grad_psi_norm2_label );
    VarLabel::destroy ( a_label );
    VarLabel::destroy ( a2_label );
    for ( size_t d = 0; d < DIM; ++d )
        VarLabel::destroy ( grad_psi_label[d] );
    for ( size_t d = 0; d < BSZ; ++d )
        VarLabel::destroy ( b_label[d] );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::problemSetup (
    const ProblemSpecP & params,
    const ProblemSpecP & /*restart_prob_spec*/,
    GridP & /*grid*/
)
{
    // PerPatch variables are not copied automatically even if a task for copying
    // them is scheduled, we need to prevent such task to be scheduled
    this->m_scheduler->overrideVariableBehavior ( subproblems_label->getName(), false, false, false, true, true );

    // register default material
    this->m_materialManager->registerSimpleMaterial ( scinew SimpleMaterial() );

    // read model parameters
    ProblemSpecP pure_metal = params->findBlock ( "PhaseField" );
    pure_metal->require ( "delt", delt );
    pure_metal->require ( "alpha", alpha );
    pure_metal->require ( "R0", r0 );
    pure_metal->require ( "Delta", delta );
    pure_metal->require ( "epsilon", epsilon );
    pure_metal->getWithDefault ( "gamma_psi", gamma_psi, 1. );
    pure_metal->getWithDefault ( "gamma_u", gamma_u, 1. );

    // coupling parameter
    lambda = alpha / 0.6267;

    if ( AMR )
    {
        this->setLockstepAMR ( true );

        // read amr parameters
        pure_metal->require ( "refine_threshold", refine_threshold );

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
        while ( ( fci = fci->findNextBlock ( "FCIType" ) ) );
    }
}

// SCHEDULINGS

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::scheduleInitialize (
    const LevelP & level,
    SchedulerP & sched
)
{
    scheduleInitialize_subproblems<AMR> ( level, sched );
    scheduleInitialize_solution<AMR> ( level, sched );
    scheduleInitialize_grad_psi<AMR> ( level, sched );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < !MG, void >::type
PureMetal<VAR, DIM, STN, AMR>::scheduleInitialize_subproblems (
    const LevelP & level,
    SchedulerP & sched
)
{
    Task * task = scinew Task ( "PureMetal::task_initialize_subproblems", this, &PureMetal::task_initialize_subproblems );
    task->computes ( subproblems_label );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

/**
 * @remark we need to schedule all levels before task_error_estimate_grad_psi to
 * avoid the error "Failure finding [subproblems , coarseLevel, MI: none, NewDW
 * (mapped to dw index 1), ####] for PureMetal::task_error_estimate_grad_psi",
 * on patch #, Level #, on material #, resource (rank): #" while compiling the
 * TaskGraph
 */

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < MG, void >::type
PureMetal<VAR, DIM, STN, AMR>::scheduleInitialize_subproblems (
    const LevelP & level,
    SchedulerP & sched
)
{
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
PureMetal<VAR, DIM, STN, AMR>::scheduleInitialize_solution (
    const LevelP & level,
    SchedulerP & sched
)
{
    Task * task = scinew Task ( "PureMetal::task_initialize_solution", this, &PureMetal::task_initialize_solution );
    task->computes ( psi_label );
    task->computes ( u_label );
    task->computes ( a_label );
    task->computes ( a2_label );
    for ( size_t d = 0; d < BSZ; ++d )
        task->computes ( b_label[d] );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < !MG, void >::type
PureMetal<VAR, DIM, STN, AMR>::scheduleInitialize_grad_psi (
    const LevelP & level,
    SchedulerP & sched
)
{
    Task * task = scinew Task ( "PureMetal::task_initialize_grad_psi", this, &PureMetal::task_initialize_grad_psi );
    task->requires ( Task::NewDW, subproblems_label, Ghost::None, 0 );
    task->requires ( Task::NewDW, psi_label, FGT, FGN );
    task->computes ( grad_psi_norm2_label );
    for ( size_t d = 0; d < DIM; ++d )
        task->computes ( grad_psi_label[d] );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < MG, void >::type
PureMetal<VAR, DIM, STN, AMR>::scheduleInitialize_grad_psi (
    const LevelP & /*level*/,
    SchedulerP & /*sched*/
)
{}

/**
 * @remark we need to schedule all levels before task_error_estimate_grad_psi to
 * avoid the error "Failure finding [u , coarseLevel, MI: none, NewDW
 * (mapped to dw index 1), ####] for PureMetal::task_error_estimate_grad_psi",
 * on patch #, Level #, on material #, resource (rank): #" while compiling the
 * TaskGraph
 */
template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < MG, void >::type
PureMetal<VAR, DIM, STN, AMR>::scheduleInitialize_solution (
    const LevelP & level,
    SchedulerP & sched
)
{
    // since the SimulationController is calling this scheduler starting from
    // the finest level we schedule only on the finest level
    if ( level->hasFinerLevel() ) return;

    GridP grid = level->getGrid();
    for ( int l = 0; l < grid->numLevels(); ++l )
        scheduleInitialize_solution < !MG > ( grid->getLevel ( l ), sched );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::scheduleRestartInitialize (
    const LevelP & level,
    SchedulerP & sched
)
{
    scheduleInitialize_subproblems<AMR> ( level, sched );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void PureMetal<VAR, DIM, STN, AMR>::scheduleComputeStableTimeStep ( LevelP const & level, SchedulerP & sched )
{
    Task * task = scinew Task ( "PureMetal::task_compute_stable_timestep", this, &PureMetal::task_compute_stable_timestep );
    task->computes ( this->getDelTLabel(), level.get_rep() );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );

    // reset flag here since TaskGraph compiler call scheduleComputeStableTimeStep
    // after each regridding
    is_first_schedule_refine = true;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::scheduleTimeAdvance (
    const LevelP & level,
    SchedulerP & sched
)
{
    scheduleTimeAdvance_subproblems<AMR> ( level, sched );
    scheduleTimeAdvance_grad_psi<AMR> ( level, sched );
    scheduleTimeAdvance_anisotropy_terms ( level, sched );
    scheduleTimeAdvance_solution<AMR> ( level, sched );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < !MG, void >::type
PureMetal<VAR, DIM, STN, AMR>::scheduleTimeAdvance_subproblems (
    const LevelP & level,
    SchedulerP & sched
)
{
    Task * task = scinew Task ( "PureMetal::task_time_advance_subproblems", this, &PureMetal::task_time_advance_subproblems );
    task->requires ( Task::OldDW, subproblems_label, Ghost::None, 0 );
    task->computes ( subproblems_label );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < MG, void >::type
PureMetal<VAR, DIM, STN, AMR>::scheduleTimeAdvance_subproblems (
    const LevelP & level,
    SchedulerP & sched
)
{
    if ( level->hasCoarserLevel() ) return;

    GridP grid = level->getGrid();
    for ( int l = 0; l < grid->numLevels(); ++l )
        scheduleTimeAdvance_subproblems < !MG > ( grid->getLevel ( l ), sched );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < !MG, void >::type
PureMetal<VAR, DIM, STN, AMR>::scheduleTimeAdvance_grad_psi (
    const LevelP & level,
    SchedulerP & sched
)
{
    Task * task = scinew Task ( "PureMetal::task_time_advance_grad_psi", this, &PureMetal::task_time_advance_grad_psi );
    task->requires ( Task::OldDW, subproblems_label, Ghost::None, 0 );
    task->requires ( Task::OldDW, psi_label, FGT, FGN );
    task->computes ( grad_psi_norm2_label );
    for ( size_t d = 0; d < DIM; ++d )
        task->computes ( grad_psi_label[d] );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < MG, void >::type
PureMetal<VAR, DIM, STN, AMR>::scheduleTimeAdvance_grad_psi (
    const LevelP & ,
    SchedulerP &
)
{
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::scheduleTimeAdvance_anisotropy_terms (
    const LevelP & level,
    SchedulerP & sched
)
{
    Task * task = scinew Task ( "PureMetal::task_time_advance_anisotropy_terms", this, &PureMetal::task_time_advance_anisotropy_terms );
    task->requires ( Task::OldDW, grad_psi_norm2_label, Ghost::None, 0 );
    for ( size_t d = 0; d < DIM; ++d )
        task->requires ( Task::OldDW, grad_psi_label[d], Ghost::None, 0 );
    task->computes ( a_label );
    task->computes ( a2_label );
    for ( size_t d = 0; d < BSZ; ++d )
        task->computes ( b_label[d] );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < !MG, void >::type
PureMetal<VAR, DIM, STN, AMR>::scheduleTimeAdvance_solution (
    const LevelP & level,
    SchedulerP & sched
)
{
    Task * task = scinew Task ( "PureMetal::task_time_advance_solution", this, &PureMetal::task_time_advance_solution );
    task->requires ( Task::OldDW, subproblems_label, Ghost::None, 0 );
    task->requires ( Task::OldDW, psi_label, FGT, FGN );
    task->requires ( Task::OldDW, u_label, FGT, FGN );
    for ( size_t d = 0; d < DIM; ++d )
        task->requires ( Task::OldDW, grad_psi_label[d], Ghost::None, 0 );
    task->requires ( Task::NewDW, a_label, Ghost::None, 0 );
    task->requires ( Task::NewDW, a2_label, FGT, FGN );
    for ( size_t d = 0; d < BSZ; ++d )
        task->requires ( Task::NewDW, b_label[d], FGT, FGN );
    task->computes ( psi_label );
    task->computes ( u_label );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < MG, void >::type
PureMetal<VAR, DIM, STN, AMR>::scheduleTimeAdvance_solution (
    const LevelP & level,
    SchedulerP & sched
)
{
    if ( !level->hasCoarserLevel() ) scheduleTimeAdvance_solution < !MG > ( level, sched );
    else
    {
        Task * task = scinew Task ( "PureMetal::task_time_advance_solution", this, &PureMetal::task_time_advance_solution );
        task->requires ( Task::OldDW, subproblems_label, Ghost::None, 0 );
        task->requires ( Task::OldDW, subproblems_label, nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
        task->requires ( Task::OldDW, psi_label, FGT, FGN );
        task->requires ( Task::OldDW, psi_label, nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
        for ( size_t d = 0; d < DIM; ++d )
            task->requires ( Task::OldDW, grad_psi_label[d], Ghost::None, 0 );
        task->requires ( Task::OldDW, u_label, nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
        task->requires ( Task::OldDW, u_label, FGT, FGN );
        task->requires ( Task::OldDW, u_label, nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
        task->requires ( Task::NewDW, a_label, Ghost::None, 0 );
        task->requires ( Task::NewDW, a2_label, FGT, FGN );
        task->requires ( Task::NewDW, a2_label, nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
        for ( size_t d = 0; d < BSZ; ++d )
        {
            task->requires ( Task::NewDW, b_label[d], FGT, FGN );
            task->requires ( Task::NewDW, b_label[d], nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
        }
        task->computes ( psi_label );
        task->computes ( u_label );
        sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
    }
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::scheduleRefine
(
    const PatchSet * new_patches,
    SchedulerP & sched
)
{
    cout_pure_metal_scheduling << "scheduleRefine on: " << *new_patches << std::endl;

    const Level * level = getLevel ( new_patches );
    const GridP & grid = level->getGrid();

    // we need to create subproblems for new patches.
    // moreover, since SchedulerCommon::copyDataToNewGrid is not copying PerPatch
    // variable to the new grid (which is fine since the geometry -thus the subproblems-
    // has changed) we need to schedule their creation within scheduleRefine/scheduleRefineInterface
    // since this tasks are compiled separately from those scheduled by scheduleTimeAdvance
    if ( is_first_schedule_refine )
    {
        scheduleRefine_subproblems ( grid, sched );
        is_first_schedule_refine = false;
    };

    // no need to refine on coarser level
    if ( level->hasCoarserLevel() )
    {
        scheduleRefine_solution ( new_patches, sched );
        scheduleRefine_grad_psi ( new_patches, sched );
    }
}

/**
 * @remark we need to schedule all levels before task_error_estimate_grad_psi
 * to avoid the error "Failure finding [subproblems , coarseLevel, MI: none, NewDW
 * (mapped to dw index 1), ####] for PureMetal::task_error_estimate_grad_psi",
 * on patch #, Level #, on material #, resource (rank): #" while compiling the
 * TaskGraph
 */
template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::scheduleRefine_subproblems (
    const GridP & grid,
    SchedulerP & sched
)
{
    cout_pure_metal_scheduling << "scheduleRefine_subproblems" << std::endl;

    for ( int l = 0; l < grid->numLevels(); ++l )
    {
        Task * task = scinew Task ( "PureMetal::task_initialize_subproblems", this, &PureMetal::task_initialize_subproblems );
        task->computes ( subproblems_label );
        sched->addTask ( task, grid->getLevel ( l )->eachPatch(), this->m_materialManager->allMaterials() );
    }
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::scheduleRefine_solution (
    const PatchSet * patches,
    SchedulerP & sched
)
{
    cout_pure_metal_scheduling << "scheduleRefine_solution on: " << *patches << std::endl;

    Task * task = scinew Task ( "PureMetal::task_refine_solution", this, &PureMetal::task_refine_solution );
    task->requires ( Task::NewDW, psi_label, nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
    task->requires ( Task::NewDW, u_label, nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
    task->requires ( Task::NewDW, subproblems_label, Ghost::None, 0 );
    task->requires ( Task::NewDW, subproblems_label, nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
    task->computes ( psi_label );
    task->computes ( u_label );
    sched->addTask ( task, patches, this->m_materialManager->allMaterials() );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::scheduleRefine_grad_psi (
    const PatchSet * patches,
    SchedulerP & sched
)
{
    cout_pure_metal_scheduling << "scheduleRefine_grad_psi on: " << *patches << std::endl;

    Task * task = scinew Task ( "PureMetal::task_refine_grad_psi", this, &PureMetal::task_refine_grad_psi );
    task->requires ( Task::NewDW, subproblems_label, Ghost::None, 0 );
    task->requires ( Task::NewDW, psi_label, nullptr, Task::ThisLevel, nullptr, Task::NormalDomain, FGT, FGN );
    task->computes ( grad_psi_norm2_label );
    for ( size_t d = 0; d < DIM; ++d )
        task->computes ( grad_psi_label[d] );
    sched->addTask ( task, patches, this->m_materialManager->allMaterials() );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::scheduleRefineInterface (
    const LevelP & /*level_fine*/,
    SchedulerP & /*sched*/,
    bool /*need_old_coarse*/,
    bool /*need_new_coarse*/
)
{};

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::scheduleCoarsen
(
    const LevelP & level_coarse,
    SchedulerP & sched
)
{
    scheduleCoarsen_solution ( level_coarse, sched );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::scheduleCoarsen_solution (
    const LevelP & level_coarse,
    SchedulerP & sched
)
{
    Task * task = scinew Task ( "PureMetal::task_coarsen_solution", this, &PureMetal::task_coarsen_solution );
    task->requires ( Task::NewDW, psi_label, nullptr, Task::FineLevel, nullptr, Task::NormalDomain, Ghost::None, 0 );
    task->requires ( Task::NewDW, u_label, nullptr, Task::FineLevel, nullptr, Task::NormalDomain, Ghost::None, 0 );
    task->requires ( Task::NewDW, subproblems_label, Ghost::None, 0 );
    task->requires ( Task::NewDW, subproblems_label, nullptr, Task::FineLevel, nullptr, Task::NormalDomain, Ghost::None, 0 );
    task->modifies ( psi_label );
    task->modifies ( u_label );
    sched->addTask ( task, level_coarse->eachPatch(), this->m_materialManager->allMaterials() );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::scheduleErrorEstimate
(
    const LevelP & level,
    SchedulerP & sched
)
{
    scheduleErrorEstimate_grad_psi<AMR> ( level, sched );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < !MG, void >::type
PureMetal<VAR, DIM, STN, AMR>::scheduleErrorEstimate_grad_psi (
    const LevelP & level,
    SchedulerP & sched
)
{
    Task * task = scinew Task ( "PureMetal::task_error_estimate_grad_psi", this, &PureMetal::task_error_estimate_grad_psi );
    task->requires ( Task::NewDW, subproblems_label, Ghost::None, 0 );
    task->requires ( Task::NewDW, psi_label, FGT, FGN );
    task->modifies ( this->m_regridder->getRefineFlagLabel(), this->m_regridder->refineFlagMaterials() );
    task->modifies ( this->m_regridder->getRefinePatchFlagLabel(), this->m_regridder->refineFlagMaterials() );
    task->computes ( grad_psi_norm2_label );
    for ( size_t d = 0; d < DIM; ++d )
        task->computes ( grad_psi_label[d] );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
template < bool MG >
typename std::enable_if < MG, void >::type
PureMetal<VAR, DIM, STN, AMR>::scheduleErrorEstimate_grad_psi (
    const LevelP & level,
    SchedulerP & sched
)
{
    if ( !level->hasCoarserLevel() ) scheduleErrorEstimate_grad_psi < !MG > ( level, sched );
    else
    {
        Task * task = scinew Task ( "PureMetal::task_error_estimate_grad_psi", this, &PureMetal::task_error_estimate_grad_psi );
        task->requires ( Task::NewDW, subproblems_label, Ghost::None, 0 );
        task->requires ( Task::NewDW, subproblems_label, nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
        task->requires ( Task::NewDW, psi_label, FGT, FGN );
        task->requires ( Task::NewDW, psi_label, nullptr, Task::CoarseLevel, nullptr, Task::NormalDomain, CGT, CGN );
        task->modifies ( this->m_regridder->getRefineFlagLabel(), this->m_regridder->refineFlagMaterials() );
        task->modifies ( this->m_regridder->getRefinePatchFlagLabel(), this->m_regridder->refineFlagMaterials() );
        task->computes ( grad_psi_norm2_label );
        for ( size_t d = 0; d < DIM; ++d )
            task->computes ( grad_psi_label[d] );
        sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
    }
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void PureMetal<VAR, DIM, STN, AMR>::scheduleInitialErrorEstimate
(
    const LevelP & level,
    SchedulerP & sched
)
{
    scheduleErrorEstimate ( level, sched );
}

// TASKS

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::task_initialize_subproblems (
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset * /*matls*/,
    DataWarehouse * /*dw_old*/,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();
    dbg_out1 << myrank << "==== PureMetal::task_initialize_subproblems ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << myrank << "== Patch: " << *patch << " Level: " << patch->getLevel()->getIndex() << std::endl;

        Variable < PP, SubProblems < PureMetalProblem<VAR, STN> > > subproblems;
        subproblems.setData ( scinew SubProblems < PureMetalProblem<VAR, STN> > ( this, psi_label, u_label, a2_label, b_label, subproblems_label, material, patch, &c2f ) );
        dw_new->put ( subproblems, subproblems_label, material, patch );
    }

    dbg_out2 << myrank << std::endl;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::task_initialize_solution (
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset * /*matls*/,
    DataWarehouse * /*dw_old*/,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();
    dbg_out1 << myrank << "==== PureMetal::task_initialize_solution ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << myrank << "== Patch: " << *patch << std::endl;

        // Allocate solution variables into the new DataWarehouse
        DWView < ScalarField<double>, VAR, DIM > psi ( dw_new, psi_label, material, patch );
        DWView < ScalarField<double>, VAR, DIM > u ( dw_new, u_label, material, patch );

        // Get patch range
        BlockRange range ( this->get_range ( patch ) );
        dbg_out3 << myrank << "= Iterating over range " << range << std::endl;

        // Initialize solution variables in range
        parallel_for ( range, [patch, &psi, &u, this] ( int i, int j, int k )->void { initialize_solution ( {i, j, k}, patch, psi, u ); } );

        // Allocate anisotropy terms variables into the new DataWarehouse
        DWView < ScalarField<double>, VAR, DIM > a ( dw_new, a_label, material, patch );
        DWView < ScalarField<double>, VAR, DIM > a2 ( dw_new, a2_label, material, patch );
        DWView < VectorField<double, BSZ>, VAR, DIM > b ( dw_new, b_label, material, patch );

        // Initialize anisotropy terms variables
        a.initialize ( 0. );
        a2.initialize ( 0. );
        b.initialize ( 0. );
    }

    dbg_out2 << myrank << std::endl;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::task_initialize_grad_psi (
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset * /*matls*/,
    DataWarehouse * /*dw_old*/,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();
    dbg_out1 << myrank << "==== PureMetal::task_initialize_grad_psi ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << myrank << "== Patch: " << *patch << std::endl;

        // Allocate grad_psi and grad_psi_norm2 variables into the new DataWarehouse
        DWView < ScalarField<double>, VAR, DIM > grad_psi_norm2 ( dw_new, grad_psi_norm2_label, material, patch );
        DWView < VectorField<double, DIM>, VAR, DIM > grad_psi ( dw_new, grad_psi_label, material, patch );

        // Retrieve subproblems from the DataWarehouse
        Variable < PP, SubProblems < PureMetalProblem<VAR, STN> > > subproblems;
        dw_new->get ( subproblems, subproblems_label, material, patch );
        const auto * problems = subproblems.get().get_rep();

        // Iterate over each subproblem
        for ( const auto & p : *problems )
        {
            // Get a view of psi that implements finite differences approximations
            dbg_out3 << myrank << "= Iterating over " << p << std::endl;
            FDView < ScalarField<const double>, STN > & psi = p.template get_fd_view<PSI> ( dw_new );

            // Compute psi derivatives on subproblem range
            parallel_for ( p.get_range(), [patch, &psi, &grad_psi, &grad_psi_norm2, this] ( int i, int j, int k )->void { time_advance_grad_psi ( {i, j, k}, psi, grad_psi, grad_psi_norm2 ); } );
        }
    }

    dbg_out2 << myrank << std::endl;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::task_compute_stable_timestep (
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset *,
    DataWarehouse *,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();
    dbg_out1 << myrank << "==== PureMetal::task_compute_stable_timestep ====" << std::endl;

    dw_new->put ( delt_vartype ( delt ), this->getDelTLabel(), getLevel ( patches ) );

    dbg_out2 << myrank << std::endl;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::task_time_advance_subproblems (
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset *,
    DataWarehouse * dw_old,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();

    dbg_out1 << myrank << "==== PureMetal::task_time_advance_subproblems ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << myrank << "== Patch: " << *patch << " Level: " << patch->getLevel()->getIndex() << std::endl;

        if ( dw_old->exists ( subproblems_label, material, patch ) )
        {
            Variable < PP, SubProblems < PureMetalProblem<VAR, STN> > > subproblems;
            dw_old->get ( subproblems, subproblems_label, material, patch );
            dw_new->put ( subproblems, subproblems_label, material, patch );
            dbg_out4 << "subproblems moved from OldDW to NewDW" << std::endl;
        }
        else // after a regrid all patches are new thus subproblems does not exists in old db
            // not bad since we want re recompute them!
        {
            auto * problems = scinew SubProblems < PureMetalProblem<VAR, STN> > ( this, psi_label, u_label, a2_label, b_label, subproblems_label, material, patch, &c2f );

            Variable < PP, SubProblems < PureMetalProblem<VAR, STN> > > subproblems;
            subproblems.setData ( problems );
            dw_new->put ( subproblems, subproblems_label, material, patch );
            dbg_out4 << "subproblems initialized in NewDW" << std::endl;
        }
    }

    dbg_out2 << myrank << std::endl;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::task_time_advance_grad_psi (
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset *,
    DataWarehouse * dw_old,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();
    dbg_out1 << myrank << "==== PureMetal::task_time_advance_grad_psi ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << myrank << "== Patch: " << *patch << std::endl;

        DWView < ScalarField<double>, VAR, DIM > grad_psi_norm2 ( dw_new, grad_psi_norm2_label, material, patch );
        DWView < VectorField<double, DIM>, VAR, DIM > grad_psi ( dw_new, grad_psi_label, material, patch );

        Variable < PP, SubProblems < PureMetalProblem<VAR, STN> > > subproblems;
        dw_new->get ( subproblems, subproblems_label, material, patch );
        auto problems = subproblems.get().get_rep();

        for ( const auto & p : *problems )
        {
            dbg_out3 << myrank << "= Iterating over " << p << std::endl;
            FDView < ScalarField<const double>, STN > & psi = p.template get_fd_view<PSI> ( dw_old );
            parallel_for ( p.get_range(), [patch, &psi, &grad_psi, &grad_psi_norm2, this] ( int i, int j, int k )->void { time_advance_grad_psi ( {i, j, k}, psi, grad_psi, grad_psi_norm2 ); } );
        }
    }

    dbg_out2 << myrank << std::endl;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::task_time_advance_anisotropy_terms (
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset *,
    DataWarehouse * dw_old,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();
    dbg_out1 << myrank << "==== PureMetal::task_time_advance_anisotropy_terms ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << myrank << "== Patch: " << *patch << std::endl;

        DWView < ScalarField<const double>, VAR, DIM > grad_psi_norm2 ( dw_old, grad_psi_norm2_label, material, patch );
        DWView < VectorField<const double, DIM>, VAR, DIM > grad_psi ( dw_old, grad_psi_label, material, patch );

        DWView < ScalarField<double>, VAR, DIM > a ( dw_new, a_label, material, patch );
        DWView < ScalarField<double>, VAR, DIM > a2 ( dw_new, a2_label, material, patch );
        DWView < VectorField<double, BSZ>, VAR, DIM > b ( dw_new, b_label, material, patch );

        BlockRange range ( this->get_range ( patch ) );
        dbg_out3 << myrank << "= Iterating over range " << range << std::endl;
        parallel_for ( range, [&grad_psi, &grad_psi_norm2, &a, &a2, &b, this] ( int i, int j, int k )->void { time_advance_anisotropy_terms ( {i, j, k}, grad_psi, grad_psi_norm2, a, a2, b ); } );
    }

    dbg_out2 << myrank << std::endl;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::task_time_advance_solution (
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset *,
    DataWarehouse * dw_old,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();
    dbg_out1 << myrank << "==== PureMetal::task_time_advance_solution ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << myrank << "== Patch: " << *patch << std::endl;

        DWView < VectorField<const double, DIM>, VAR, DIM > grad_psi ( dw_old, grad_psi_label, material, patch );
        DWView < ScalarField<const double>, VAR, DIM > a ( dw_new, a_label, material, patch );

        DWView < ScalarField<double>, VAR, DIM > psi_new ( dw_new, psi_label, material, patch );
        DWView < ScalarField<double>, VAR, DIM > u_new ( dw_new, u_label, material, patch );

        Variable < PP, SubProblems < PureMetalProblem<VAR, STN> > > subproblems;
        dw_new->get ( subproblems, subproblems_label, material, patch );
        auto problems = subproblems.get().get_rep();

        for ( const auto & p : *problems )
        {
            dbg_out3 << myrank << "= Iterating over " << p << std::endl;
            FDView < ScalarField<const double>, STN > & psi_old = p.template get_fd_view<PSI> ( dw_old );
            FDView < ScalarField<const double>, STN > & u_old = p.template get_fd_view<U> ( dw_old );
            FDView < ScalarField<const double>, STN > & a2 = p.template get_fd_view<A2> ( dw_new );
            FDView < VectorField<const double, BSZ>, STN > b = p.template get_fd_view<B> ( dw_new );
            parallel_for ( p.get_range(), [&psi_old, &u_old, &grad_psi, &a, &a2, &b, &psi_new, &u_new, this] ( int i, int j, int k )->void { time_advance_solution ( {i, j, k}, psi_old, u_old, grad_psi, a, a2, b, psi_new, u_new ); } );
        }
    }

    dbg_out2 << myrank << std::endl;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::task_refine_solution
(
    const ProcessorGroup * myworld,
    const PatchSubset * patches_fine,
    const MaterialSubset * ,
    DataWarehouse * ,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();

    dbg_out1 << myrank << "==== PureMetal::task_refine_solution ====" << std::endl;

    for ( int p = 0; p < patches_fine->size(); ++p )
    {
        const Patch * patch_fine = patches_fine->get ( p );
        dbg_out2 << myrank << "== Fine Patch: " << *patch_fine << " Level: " << patch_fine->getLevel()->getIndex() << std::endl;

        DWView < ScalarField<double>, VAR, DIM > psi_fine ( dw_new, psi_label, material, patch_fine );
        DWView < ScalarField<double>, VAR, DIM > u_fine ( dw_new, u_label, material, patch_fine );

        AMRInterpolator < PureMetalProblem<VAR, STN>, PSI, C2F > psi_coarse_interp ( dw_new, psi_label, subproblems_label, material, patch_fine );
        AMRInterpolator < PureMetalProblem<VAR, STN>, U, C2F > u_coarse_interp ( dw_new, u_label, subproblems_label, material, patch_fine );

        BlockRange range_fine ( this->get_range ( patch_fine ) );
        dbg_out3 << myrank << "= Iterating over fine range" << range_fine << std::endl;
        parallel_for ( range_fine, [&psi_coarse_interp, &u_coarse_interp, &psi_fine, &u_fine, this] ( int i, int j, int k )->void { refine_solution ( {i, j, k}, psi_coarse_interp, u_coarse_interp, psi_fine, u_fine ); } );
    }

    dbg_out2 << myrank << std::endl;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::task_refine_grad_psi
(
    const ProcessorGroup * myworld,
    const PatchSubset * patches_fine,
    const MaterialSubset * ,
    DataWarehouse * ,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();
    dbg_out1 << myrank << "==== PureMetal::task_refine_grad_psi ====" << std::endl;

    for ( int p = 0; p < patches_fine->size(); ++p )
    {
        const Patch * patch_fine = patches_fine->get ( p );
        dbg_out2 << myrank << "== Fine Patch: " << *patch_fine << " Level: " << patch_fine->getLevel()->getIndex() << std::endl;

        DWView < ScalarField<double>, VAR, DIM > grad_psi_norm2 ( dw_new, grad_psi_norm2_label, material, patch_fine );
        DWView < VectorField<double, DIM>, VAR, DIM > grad_psi ( dw_new, grad_psi_label, material, patch_fine );

        Variable < PP, SubProblems < PureMetalProblem<VAR, STN> > > subproblems;
        dw_new->get ( subproblems, subproblems_label, material, patch_fine );
        auto problems = subproblems.get().get_rep();

        for ( const auto & p : *problems )
        {
            dbg_out3 << myrank << "= Iterating over " << p << std::endl;
            FDView < ScalarField<const double>, STN > & psi = p.template get_fd_view<PSI> ( dw_new );
            parallel_for ( p.get_range(), [patch_fine, &psi, &grad_psi, &grad_psi_norm2, this] ( int i, int j, int k )->void { time_advance_grad_psi ( {i, j, k}, psi, grad_psi, grad_psi_norm2 ); } );
        }
    }

    dbg_out2 << myrank << std::endl;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::task_coarsen_solution (
    const ProcessorGroup * myworld,
    const PatchSubset * patches_coarse,
    const MaterialSubset * ,
    DataWarehouse * ,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();
    dbg_out1 << myrank << "==== PureMetal::task_coarsen_solution " << std::endl;

    for ( int p = 0; p < patches_coarse->size(); ++p )
    {
        const Patch * patch_coarse = patches_coarse->get ( p );
        dbg_out2 << myrank << "== Coarse Patch: " << *patch_coarse << " Level: " << patch_coarse->getLevel()->getIndex() << std::endl;

        DWView < ScalarField<double>, VAR, DIM > psi_coarse ( dw_new, psi_label, material, patch_coarse );
        DWView < ScalarField<double>, VAR, DIM > u_coarse ( dw_new, u_label, material, patch_coarse );

        AMRRestrictor < PureMetalProblem<VAR, STN>, PSI, F2C > psi_fine_restr ( dw_new, psi_label, subproblems_label, material, patch_coarse, false );
        AMRRestrictor < PureMetalProblem<VAR, STN>, U, F2C > u_fine_restr ( dw_new, u_label, subproblems_label, material, patch_coarse, false );

        for ( const auto & region : u_fine_restr.get_support() )
        {
            dbg_out3 << myrank << "= Iterating over coarse cells region " << region << std::endl;
            BlockRange range_coarse (
                Max ( region.getLow(), this->get_low ( patch_coarse ) ),
                Min ( region.getHigh(), this->get_high ( patch_coarse ) )
            );

            parallel_for ( range_coarse, [&psi_fine_restr, &u_fine_restr, &psi_coarse, &u_coarse, this] ( int i, int j, int k )->void { coarsen_solution ( {i, j, k}, psi_fine_restr, u_fine_restr, psi_coarse, u_coarse ); } );
        }
    }

    dbg_out2 << myrank << std::endl;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::task_error_estimate_grad_psi
(
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset * ,
    DataWarehouse * ,
    DataWarehouse * dw_new
)
{
    int myrank = myworld->myRank();

    dbg_out1 << myrank << "==== PureMetal::task_error_estimate_grad_psi " << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << myrank << "== Patch: " << *patch << " Level: " << patch->getLevel()->getIndex() << std::endl;

        PerPatch<PatchFlagP> refine_patch_flag;
        dw_new->get ( refine_patch_flag, this->m_regridder->getRefinePatchFlagLabel(), material, patch );

        PatchFlag * patch_flag_refine = refine_patch_flag.get().get_rep();

        bool refine_patch = false;

        DWView < ScalarField<double>, VAR, DIM > grad_psi_norm2 ( dw_new, grad_psi_norm2_label, material, patch );
        DWView < VectorField<double, DIM>, VAR, DIM > grad_psi ( dw_new, grad_psi_label, material, patch );
        DWView < ScalarField<int>, CC, DIM > refine_flag ( dw_new, this->m_regridder->getRefineFlagLabel(), material, patch );

        Variable<PP, SubProblems < PureMetalProblem<VAR, STN> > > subproblems;
        dw_new->get ( subproblems, subproblems_label, material, patch );

        auto problems = subproblems.get().get_rep();

        for ( const auto & p : *problems )
        {
            dbg_out3 << myrank << "= Iterating over " << p << std::endl;
            FDView < ScalarField<const double>, STN > & psi = p.template get_fd_view<PSI> ( dw_new );
            parallel_reduce_sum ( p.get_range(), [&psi, &grad_psi, &grad_psi_norm2, &refine_flag, &refine_patch, this] ( int i, int j, int k, bool & refine_patch )->void { error_estimate_grad_psi ( {i, j, k}, psi, grad_psi, grad_psi_norm2, refine_flag, refine_patch ); }, refine_patch );
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
PureMetal<VAR, DIM, STN, AMR>::initialize_solution (
    const IntVector & id,
    const Patch * patch,
    View < ScalarField<double> > & psi,
    View < ScalarField<double> > & u
)
{
    Vector v ( this->get_position ( patch, id ).asVector() );

    // BUG workaround
    std::stringstream ss;
    ss << v << std::endl;

    double r2 = 0;

    for ( size_t d = 0; d < DIM; ++d )
        r2 += v[d] * v[d];

    double tmp = r2 - r0 * r0;

    psi[id] = - tanh ( gamma_psi * tmp );
    u[id] = -delta * ( 1. + tanh ( gamma_u * tmp ) ) / 2.;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::time_advance_grad_psi (
    const IntVector & id,
    FDView < ScalarField<const double>, STN > & psi,
    View < VectorField<double, DIM> > & grad_psi,
    View < ScalarField<double> > & grad_psi_norm2
)
{
    auto grad = psi.gradient ( id );

    grad_psi_norm2[id] = 0;
    for ( size_t d = 0; d < DIM; ++d )
    {
        grad_psi[d][id] = grad[d];
        grad_psi_norm2[id] += grad[d] * grad[d];
    }
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void PureMetal<VAR, DIM, STN, AMR>::time_advance_anisotropy_terms (
    const IntVector & id,
    View < VectorField<const double, DIM> > & grad_psi,
    View < ScalarField<const double> > & grad_psi_norm2,
    View < ScalarField<double> > & a,
    View < ScalarField<double> > & a2,
    View < VectorField<double, BSZ> > & b
)
{
    double tmp = 1. + epsilon;
    double n2 = grad_psi_norm2[id];
    if ( n2 < tol )
    {
        a[id] = tmp;
        a2[id] = tmp * tmp;
        for ( size_t d = 0; d < BSZ; ++d )
            b[d][id] = 0.;
    }
    else
    {
        double n4 = n2 * n2;

        double tmp4 = 0.;
        double grad[DIM], grad2[DIM];
        for ( size_t d = 0; d < BSZ; ++d )
        {
            grad[d] = grad_psi[d][id];
            grad2[d] = grad[d] * grad[d];
            tmp4 += grad2[d] * grad2[d];
        }
        tmp4 *= 4. / n4;

        double tmp = 1. + epsilon * ( tmp4 - 3. );
        a[id] = tmp;
        a2[id] = tmp * tmp;
        if ( DIM > 1 )
        {
            b[XY][id] = 16. * epsilon * tmp * ( grad[X] * grad[Y] )  * ( grad2[X] - grad2[Y] ) / n4;
        }
        if ( DIM > 2 ) // Compile time if
        {
            b[XZ][id] = 16. * epsilon * tmp * ( grad[X] * grad[Z] ) * ( grad2[X] - grad2[Z] ) / n4;
            b[YZ][id] = 16. * epsilon * tmp * ( grad[Y] * grad[Z] ) * ( grad2[Y] - grad2[Z] ) / n4;
        }
    }
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::time_advance_solution (
    const IntVector & id,
    FDView < ScalarField<const double>, STN > & psi_old,
    FDView < ScalarField<const double>, STN > & u_old,
    View < VectorField<const double, DIM> > & grad_psi,
    View < ScalarField<const double> > & a,
    FDView < ScalarField<const double>, STN > & a2,
    FDView < VectorField<const double, BSZ>, STN > & b,
    View < ScalarField<double> > & psi_new,
    View < ScalarField<double> > & u_new
)
{
    double source = 1. - psi_old[id] * psi_old[id];
    source *= ( psi_old[id] - lambda * u_old[id] * source );

    double delta_psi = 0;

    if ( DIM == 2 )
    {
        delta_psi = delt * ( psi_old.laplacian ( id ) * a2[id]
                             + ( a2.dx ( id ) - b[XY].dy ( id ) ) * grad_psi[X][id]
                             + ( a2.dy ( id ) + b[XY].dx ( id ) ) * grad_psi[Y][id]
                             + source ) / a[id];

    }
    if ( DIM == 3 )
        delta_psi = delt * ( psi_old.laplacian ( id ) * a2[id]
                             + ( a2.dx ( id ) - b[XY].dy ( id ) - b[XZ].dz ( id ) ) * grad_psi[X][id]
                             + ( a2.dy ( id ) + b[XY].dx ( id ) - b[YZ].dz ( id ) ) * grad_psi[Y][id]
                             + ( a2.dz ( id ) + b[XZ].dx ( id ) + b[YZ].dy ( id ) ) * grad_psi[Z][id]
                             + source ) / a[id];

    double delta_u = delt * u_old.laplacian ( id ) * alpha + delta_psi / 2.;

    psi_new[id] = psi_old[id] + delta_psi;
    u_new[id] = u_old[id] + delta_u;
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::refine_solution
(
    const IntVector id_fine,
    const View < ScalarField<const double> > & psi_coarse_interp,
    const View < ScalarField<const double> > & u_coarse_interp,
    View < ScalarField<double> > & psi_fine,
    View < ScalarField<double> > & u_fine
)
{
    psi_fine[id_fine] = psi_coarse_interp[id_fine];
    u_fine[id_fine] = u_coarse_interp[id_fine];
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::coarsen_solution
(
    const IntVector id_coarse,
    const View < ScalarField<const double> > & psi_fine_restr,
    const View < ScalarField<const double> > & u_fine_restr,
    View < ScalarField<double> > & psi_coarse,
    View < ScalarField<double> > & u_coarse
)
{
    psi_coarse[id_coarse] = psi_fine_restr[id_coarse];
    u_coarse[id_coarse] = u_fine_restr[id_coarse];
}

template<VarType VAR, DimType DIM, StnType STN, bool AMR>
void
PureMetal<VAR, DIM, STN, AMR>::error_estimate_grad_psi (
    const IntVector & id,
    FDView < ScalarField<const double>, STN > & psi,
    View < VectorField<double, DIM> > & grad_psi,
    View < ScalarField<double> > & grad_psi_norm2,
    View< ScalarField<int> > & refine_flag,
    bool & refine_patch
)
{
    bool refine = false;
    auto grad = psi.gradient ( id );

    grad_psi_norm2[id] = 0;
    for ( size_t d = 0; d < DIM; ++d )
    {
        grad_psi[d][id] = grad[d];
        grad_psi_norm2[id] += grad[d] * grad[d];
    }

    refine = grad_psi_norm2[id] > refine_threshold * refine_threshold;
    refine_flag[id] = refine;
    refine_patch |= refine;
}

} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_Applications_PureMetal_h
