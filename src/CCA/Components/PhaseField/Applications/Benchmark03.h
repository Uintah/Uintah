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
 * @file CCA/Components/PhaseField/Applications/Benchmark03.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_Benchmark03_h
#define Packages_Uintah_CCA_Components_PhaseField_Benchmark03_h

#include <CCA/Components/PhaseField/Applications/Application.h>
#include <CCA/Components/PhaseField/Factory/Implementation.h>
#include <CCA/Components/PhaseField/Views/View.h>
#include <CCA/Components/PhaseField/DataWarehouse/DWView.h>

#include <Core/Grid/SimpleMaterial.h>

namespace Uintah
{
namespace PhaseField
{

/**
 * @brief Benchmark III application: 1D Cahn Hilliard
 *
 * \f[
 * \dot u = \epsilon^2 v^{\prime\prime} + \nabla^2 W^\prime(u)
 * \f]
 * \f[
 * v = u^{\prime\prime}
 * \f]
 * where
 * \f[
 *  W (u) = \frac{1}{4} (u^2 − 1)^2
 * \f]
 * on \f$[0, 2\pi]^2\f$ with initial data
 * \f[
 * u_{|t=0} = \cos(2x) + \exp[ \cos(x+1/10) ]/100
 * \f]
 * and with periodic boundary conditions.
 *
 * The model parameters are:
 * - \f$ \epsilon \f$   interface width
 *
 * @tparam VAR type of variable representation
 * @tparam STN finite-difference stencil
 */
template < VarType VAR, StnType STN >
class Benchmark03
    : public Application<VAR, D1, STN, false>
    , public Implementation<Benchmark03<VAR, STN>, UintahParallelComponent, const ProcessorGroup *, const MaterialManagerP, int>
{
    /// Problem material index (only one SimpleMaterial)
    static constexpr int material = 0;

    /// Problem dimension
    static constexpr DimType DIM = D1;

    /// Number of ghost elements required by STN
    static constexpr int GN = get_stn<STN>::ghosts;

    /// Type of ghost elements required by VAR and STN (on the same level)
    static constexpr Ghost::GhostType GT = GN ? get_var<VAR>::ghost_type : Ghost::None;

public: // STATIC MEMBERS

    /// Class name as used by ApplicationFactory
    static const std::string Name;

protected: // MEMBERS

    /// Output streams for debugging
    DebugStream dbg_out1, dbg_out2, dbg_out3, dbg_out4;

    // Labels for variables to be stored into the DataWarehouse
    const VarLabel * u_label, * v_label;
    const VarLabel * u0_label;
    const VarLabel * energy_label;

    /// Time step size
    double delt;

    /// Interface width
    double epsilon;

public: // CONSTRUCTORS/DESTRUCTOR

    /**
     * @brief Constructor
     *
     * Intantiate a Benchmark III application
     *
     * @param myworld data structure to manage mpi processes
     * @param materialManager data structure to manage materials
     * @param verbosity constrols amount of debugging output
     */
    Benchmark03 (
        const ProcessorGroup * myworld,
        const MaterialManagerP materialManager,
        int verbosity = 0
    );

    /**
     * @brief Destructor
     */
    virtual ~Benchmark03();

    /// Prevent copy (and move) constructor
    Benchmark03 ( Benchmark03 const & ) = delete;

    /// Prevent copy (and move) assignment
    Benchmark03 & operator= ( Benchmark03 const & ) = delete;

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
        ProblemSpecP const & params,
        ProblemSpecP const & restart_prob_spec,
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
        LevelP const & level,
        SchedulerP & sched
    ) override;

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
        LevelP const & level,
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
        LevelP const & level,
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
        LevelP const & level,
        SchedulerP & sched
    ) override;

    /**
     * @brief Schedule task_time_advance_solution
     *
     * Defines the dependencies and output of the tasks which updates the
     * solution allowing sched to control its execution order
     *
     * @param level grid level to be updated
     * @param sched scheduler to manage the tasks
     */
    void
    scheduleTimeAdvance_solution (
        const LevelP & level,
        SchedulerP & sched
    );

    /**
     * @brief Schedule task_time_advance_solution
     *
     * Defines the dependencies and output of the tasks which updates v
     * allowing sched to control its execution order
     *
     * @param level grid level to be updated
     * @param sched scheduler to manage the tasks
     */
    void
    scheduleTimeAdvance_v (
        LevelP const & level,
        SchedulerP & sched
    );

    /**
     * @brief Schedule task_time_advance_solution
     *
     * Defines the dependencies and output of the tasks which updates u
     * allowing sched to control its execution order
     *
     * @param level grid level to be updated
     * @param sched scheduler to manage the tasks
     */
    void
    scheduleTimeAdvance_u (
        LevelP const & level,
        SchedulerP & sched
    );

    /**
     * @brief Schedule task_time_advance_postprocess
     *
     * Defines the dependencies and output of the task which updates energy and
     * solution value at middle point allowing sched to control its execution order
     *
     * @param level grid level to be updated
     * @param sched scheduler to manage the tasks
     */
    void scheduleTimeAdvance_postprocess (
        LevelP const & level,
        SchedulerP & sched
    );

protected: // TASKS

    /**
     * @brief Initialize solution task
     *
     * Allocate and save variables u and v for each one of the patches
     * and save them to dw_new
     *
     * @param myworld data structure to manage mpi processes
     * @param patches list of patches to be initialized
     * @param matls unused
     * @param dw_old unused
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_initialize_solution (
        ProcessorGroup const * myworld,
        PatchSubset const * patches,
        MaterialSubset const * matls,
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
        ProcessorGroup const * myworld,
        PatchSubset const * patches,
        MaterialSubset const * matls,
        DataWarehouse * dw_old,
        DataWarehouse * dw_new
    );

    /**
     * @brief Advance solution task
     *
     * Computes new value of v using the value of the solution and at
     * previous timestep
     *
     * @param myworld data structure to manage mpi processes
     * @param patches list of patches to be initialized
     * @param matls unused
     * @param dw_old DataWarehouse for previous timestep
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_time_advance_v (
        ProcessorGroup const * myworld,
        PatchSubset const * patches,
        MaterialSubset const * matls,
        DataWarehouse * dw_old,
        DataWarehouse * dw_new
    );

    /**
     * @brief Advance solution task
     *
     * Computes new value of u using the value of the newly computed v and
     * the solution and at previous timestep
     *
     * @param myworld data structure to manage mpi processes
     * @param patches list of patches to be initialized
     * @param matls unused
     * @param dw_old DataWarehouse for previous timestep
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_time_advance_u (
        ProcessorGroup const * myworld,
        PatchSubset const * patches,
        MaterialSubset const * matls,
        DataWarehouse * dw_old,
        DataWarehouse * dw_new
    );

    /**
     * @brief Advance postprocess task
     *
     * Computes new value of the system energy and of the solution at middle point
     *
     * @param myworld data structure to manage mpi processes
     * @param patches list of patches to be initialized
     * @param matls unused
     * @param dw_old DataWarehouse for previous timestep
     * @param dw_new DataWarehouse to be initialized
     */
    void
    task_time_advance_postprocess (
        ProcessorGroup const * myworld,
        PatchSubset const * patches,
        MaterialSubset const * matls,
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
     * @param[out] u view of the solution field in the new dw
     */
    void
    initialize_solution (
        const IntVector & id,
        const Patch * patch,
        View< ScalarField<double> > & u
    );

    /**
     * @brief Advance solution implementation
     *
     * compute new value for v at a given grid position using the value of the
     * solution and at previous timestep
     *
     * @param id grid index
     * @param u_old view of the solution field in the old dw
     * @param[out] v_new view of v field in the new dw
     */
    virtual void
    time_advance_v (
        const IntVector & id,
        const FDView < ScalarField<const double>, STN > & u_old,
        View< ScalarField<double> > & v_new
    );

    /**
     * @brief Advance solution implementation
     *
     * compute new value for u at a given grid position using the value of the
     * solution and at previous timestep
     *
     * @param id grid index
     * @param u_old view of u field in the old dw
     * @param v_new  view of v field in the old dw
     * @param[out] u_new view of the solution field in the new dw
     */
    virtual void
    time_advance_u (
        const IntVector & id,
        const View< ScalarField<const double> > & u_old,
        const FDView < ScalarField<const double>, STN > & v_new,
        View< ScalarField<double> > & u_new
    );

    /**
     * @brief Advance system energy implementation
     *
     * compute new value for energy at a given grid position and accumulate it
     *
     * @param id grid index
     * @param patch grid patch
     * @param u_new view of the solution field in the new dw
     * @param[out] energy system energy
     */
    virtual void
    time_advance_postprocess_energy (
        const IntVector & id,
        const Patch * patch,
        const FDView < ScalarField<const double>, STN > & u_new,
        double & energy
    );

}; // class Benchmark03

// CONSTRUCTORS/DESTRUCTOR

template<VarType VAR, StnType STN>
Benchmark03<VAR, STN>::Benchmark03 (
    const ProcessorGroup * myworld,
    const MaterialManagerP materialManager,
    int verbosity
) : Application<VAR, DIM, STN> ( myworld, materialManager ),
    dbg_out1 ( "Benchmark03", verbosity > 0 ),
    dbg_out2 ( "Benchmark03", verbosity > 1 ),
    dbg_out3 ( "Benchmark03", verbosity > 2 ),
    dbg_out4 ( "Benchmark03", verbosity > 3 )
{
    u_label = VarLabel::create ( "u", Variable<VAR, double>::getTypeDescription() );
    v_label = VarLabel::create ( "v", Variable<VAR, double>::getTypeDescription() );
    u0_label = VarLabel::create ( "u0", sum_vartype::getTypeDescription() );
    energy_label = VarLabel::create ( "energy", sum_vartype::getTypeDescription() );
}

template<VarType VAR, StnType STN>
Benchmark03<VAR, STN>::~Benchmark03()
{
    VarLabel::destroy ( u_label );
    VarLabel::destroy ( v_label );
    VarLabel::destroy ( u0_label );
    VarLabel::destroy ( energy_label );
}

// SETUP

template<VarType VAR, StnType STN>
void
Benchmark03<VAR, STN>::problemSetup ( ProblemSpecP const & params, ProblemSpecP const &, GridP & )
{
    this->m_materialManager->registerSimpleMaterial ( scinew SimpleMaterial() );

    ProblemSpecP benchmark = params->findBlock ( "PhaseField" );
    benchmark->require ( "delt", delt );
    benchmark->require ( "epsilon", epsilon );
}

// SCHEDULINGS

template<VarType VAR, StnType STN>
void
Benchmark03<VAR, STN>::scheduleInitialize (
    const LevelP & level,
    SchedulerP & sched
)
{
    Task * task = scinew Task ( "Benchmark03::task_initialize_solution", this, &Benchmark03::task_initialize_solution );
    task->computes ( u_label );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

template<VarType VAR, StnType STN>
void
Benchmark03<VAR, STN>::scheduleRestartInitialize (
    const LevelP & level,
    SchedulerP & sched
)
{
}

template<VarType VAR, StnType STN>
void
Benchmark03<VAR, STN>::scheduleComputeStableTimeStep (
    const LevelP & level,
    SchedulerP & sched
)
{
    Task * task = scinew Task ( "Benchmark03::task_compute_stable_timestep ", this, &Benchmark03::task_compute_stable_timestep );
    task->computes ( this->getDelTLabel(), level.get_rep() );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

template<VarType VAR, StnType STN>
void
Benchmark03<VAR, STN>::scheduleTimeAdvance (
    const LevelP & level,
    SchedulerP & sched
)
{
    scheduleTimeAdvance_solution ( level, sched );
    scheduleTimeAdvance_postprocess ( level, sched );
};

template<VarType VAR, StnType STN>
void
Benchmark03<VAR, STN>::scheduleTimeAdvance_solution (
    const LevelP & level,
    SchedulerP & sched
)
{
    scheduleTimeAdvance_v ( level, sched );
    scheduleTimeAdvance_u ( level, sched );
}

template<VarType VAR, StnType STN>
void
Benchmark03<VAR, STN>::scheduleTimeAdvance_v (
    const LevelP & level,
    SchedulerP & sched
)
{
    Task * task = scinew Task ( "Benchmark03::task_time_advance_v", this, &Benchmark03<VAR, STN>::task_time_advance_v );
    task->requires ( Task::OldDW, u_label, GT, GN );
    task->computes ( v_label );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

template<VarType VAR, StnType STN>
void
Benchmark03<VAR, STN>::scheduleTimeAdvance_u (
    const LevelP & level,
    SchedulerP & sched
)
{
    Task * task = scinew Task ( "Benchmark03::task_time_advance_u", this, &Benchmark03<VAR, STN>::task_time_advance_u );
    task->requires ( Task::OldDW, u_label, Ghost::None, 0 );
    task->requires ( Task::NewDW, v_label, GT, GN );
    task->computes ( u_label );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

template<VarType VAR, StnType STN>
void Benchmark03<VAR, STN>::scheduleTimeAdvance_postprocess (
    const LevelP & level,
    SchedulerP & sched
)
{
    Task * task = scinew Task ( "Benchmark03::task_time_advance_postprocess", this, &Benchmark03<VAR, STN>::task_time_advance_postprocess );
    task->requires ( Task::NewDW, u_label, GT, GN );
    task->computes ( u0_label );
    task->computes ( energy_label );
    sched->addTask ( task, level->eachPatch(), this->m_materialManager->allMaterials() );
}

// TASKS

template<VarType VAR, StnType STN>
void
Benchmark03<VAR, STN>::task_initialize_solution (
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset *,
    DataWarehouse *,
    DataWarehouse * dw_new
)
{
    dbg_out1 << "==== Benchmark03::task_initialize_solution ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << "== Patch: " << *patch << std::endl;

        BlockRange range ( this->get_range ( patch ) );
        dbg_out3 << "= Iterating over range " << range << std::endl;

        DWView < ScalarField<double>, VAR, DIM > u_view ( dw_new, u_label, material, patch );
        parallel_for ( range, [patch, &u_view, this] ( int i, int j, int k )->void { initialize_solution ( {i, j, k}, patch, u_view ); } );
    }

    dbg_out2 << std::endl;
}

template<VarType VAR, StnType STN>
void
Benchmark03<VAR, STN>::task_compute_stable_timestep (
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset *,
    DataWarehouse *,
    DataWarehouse * dw_new
)
{
    dbg_out1 << "==== Benchmark03::task_compute_stable_timestep ====" << std::endl;
    dw_new->put ( delt_vartype ( delt ), this->getDelTLabel(), getLevel ( patches ) );
    dbg_out2 << std::endl;
}


template<VarType VAR, StnType STN>
void Benchmark03<VAR, STN>::task_time_advance_v (
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset *,
    DataWarehouse * dw_old,
    DataWarehouse * dw_new
)
{
    dbg_out1 << "==== Benchmark03<VAR,STN>::task_time_advance_v ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << "== Patch: " << *patch << std::endl;

        DWFDView < ScalarField<const double>, STN, VAR > u_old_view ( dw_old, u_label, material, patch );
        DWView < ScalarField<double>, VAR, DIM > v_new_view ( dw_new, v_label, material, patch );

        BlockRange range ( this->get_range ( patch ) );
        dbg_out3 << "= Iterating over range " << range << std::endl;

        parallel_for ( range, [patch, &u_old_view, &v_new_view, this] ( int i, int j, int k )->void { time_advance_v ( {i, j, k}, u_old_view, v_new_view ); } );
    }

    dbg_out2 << std::endl;
}

template<VarType VAR, StnType STN>
void Benchmark03<VAR, STN>::task_time_advance_u (
    const ProcessorGroup * myworld,
    const PatchSubset * patches,
    const MaterialSubset *,
    DataWarehouse * dw_old,
    DataWarehouse * dw_new
)
{
    dbg_out1 << "==== Benchmark03<VAR,STN>::task_time_advance_u ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << "== Patch: " << *patch << std::endl;

        DWView < ScalarField<const double>, VAR, DIM > u_old_view ( dw_old, u_label, material, patch );
        DWFDView < ScalarField<const double>, STN, VAR > v_new_view ( dw_new, v_label, material, patch );
        DWView < ScalarField<double>, VAR, DIM > u_new_view ( dw_new, u_label, material, patch );

        BlockRange range ( this->get_range ( patch ) );
        dbg_out3 << "= Iterating over range " << range << std::endl;

        parallel_for ( range, [patch, &u_old_view, &v_new_view, &u_new_view, this] ( int i, int j, int k )->void { time_advance_u ( {i, j, k}, u_old_view, v_new_view, u_new_view ); } );
    }

    dbg_out2 << std::endl;
}

template<VarType VAR, StnType STN>
void
Benchmark03<VAR, STN>::task_time_advance_postprocess (
    const ProcessorGroup *,
    const PatchSubset * patches,
    const MaterialSubset *,
    DataWarehouse * dw_old,
    DataWarehouse * dw_new
)
{
    dbg_out1 << "==== Benchmark03<VAR,STN>::task_time_advance_postprocess ====" << std::endl;

    for ( int p = 0; p < patches->size(); ++p )
    {
        const Patch * patch = patches->get ( p );
        dbg_out2 << "== Patch: " << *patch << std::endl;

        DWFDView < ScalarField<const double>, STN, VAR > u_view ( dw_new, u_label, material, patch );

        IntVector i0;
        if ( this->find_point ( patch, {M_PI, 0., 0.}, i0 ) )
        {
            double u0 = u_view[i0];
            if ( fabs ( u0 ) > 2 ) SCI_THROW ( AssertionFailed ( "\n ERROR: Unstable simulation\n", __FILE__, __LINE__ ) );
            dw_new->put ( sum_vartype ( u0 ), u0_label );
        }
        else
            dw_new->put ( sum_vartype ( 0. ), u0_label );

        double energy = 0.;

        BlockRange range ( this->get_range ( patch ) );
        dbg_out3 << "= Iterating over range " << range << std::endl;

        parallel_reduce_sum (
            range,
            [patch, &u_view, this] ( int i, int j, int k, double & energy )->void { time_advance_postprocess_energy ( {i, j, k}, patch, u_view, energy ); },
            energy
        );

        dw_new->put ( sum_vartype ( energy ), energy_label );
    }

    dbg_out2 << std::endl;
}

// IMPLEMENTATIONS

template<VarType VAR, StnType STN>
void Benchmark03<VAR, STN>::initialize_solution (
    const IntVector & id,
    Patch const * patch,
    View< ScalarField<double> > & u
)
{
    Vector v ( this->get_position ( patch, id ).asVector() );

    // BUG workaround
    std::stringstream ss;
    ss << v << std::endl;

    const double & x = v[0];
    u[id] = cos ( 2. * x ) + 0.01 * exp ( cos ( x + 0.1 ) );
}

template<VarType VAR, StnType STN>
void Benchmark03<VAR, STN>::time_advance_v (
    const IntVector & id,
    const FDView < ScalarField<const double>, STN > & u_old,
    View< ScalarField<double> > & v_new
)
{
    const double & u = u_old[id];
    double lap = u_old.laplacian ( id );
    double src = u * ( u * u - 1. );
    v_new[id] = - epsilon * epsilon * lap + src;
}

template<VarType VAR, StnType STN>
void Benchmark03<VAR, STN>::time_advance_u (
    const IntVector & id,
    const View< ScalarField<const double> > & u_old,
    const FDView < ScalarField<const double>, STN > & v_new,
    View< ScalarField<double> > & u_new
)
{
    const double & u = u_old[id];
    double delta_u = delt * v_new.laplacian ( id );
    u_new[id] = u + delta_u;
}

template<VarType VAR, StnType STN>
void Benchmark03<VAR, STN>::time_advance_postprocess_energy (
    const IntVector & id,
    const Patch * patch,
    const FDView < ScalarField<const double>, STN > & u_new,
    double & energy
)
{
    const double & u = u_new[id];
    auto grad = u_new.gradient ( id );
    double A = patch->getLevel()->dCell() [0] * patch->getLevel()->dCell() [1];
    energy += A * ( epsilon * epsilon * ( grad[0] * grad[0] + grad[1] * grad[1] ) / 2. + ( u * u * u * u - 2 * u * u + 1. ) / 4. );
}

} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_Benchmark03_h
