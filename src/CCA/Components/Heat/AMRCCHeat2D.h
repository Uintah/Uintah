/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef Packages_Uintah_CCA_Components_Heat_AMRCCHeat2D_h
#define Packages_Uintah_CCA_Components_Heat_AMRCCHeat2D_h

#include <CCA/Components/Heat/CCHeat2D.h>

namespace Uintah
{

class AMRCCHeat2D : public CCHeat2D
{
protected:
    using FlagVariable = CCVariable<int>;
#ifdef UINTAH_ENABLE_KOKKOS
    using FlagView = KokkosView3<int>;
#else
    using FlagView = FlagVariable & ;
#endif

    using CCHeat2D::get_view;
    static constexpr inline FlagView get_view ( FlagVariable & var );

    static inline IntVector map_to_coarser ( Level const * level, IntVector const & i );
    static inline IntVector map_to_finer ( Level const * level, IntVector const & i );

protected:
    double refine_threshold;

public:
    AMRCCHeat2D ( ProcessorGroup const * myworld, int verbosity = 0 );
    virtual ~AMRCCHeat2D ();

protected:
    AMRCCHeat2D ( AMRCCHeat2D const & ) = delete;
    AMRCCHeat2D & operator= ( AMRCCHeat2D const & ) = delete;

    virtual void problemSetup ( ProblemSpecP const & params, ProblemSpecP const & restart_prob_spec, GridP & grid, SimulationStateP & state ) override;
    virtual void scheduleTimeAdvance ( LevelP const & level, SchedulerP & ) override;
    virtual void scheduleRefine ( PatchSet const * patches, SchedulerP & sched ) override;
    virtual void scheduleRefineInterface ( LevelP const & /*level_fine*/, SchedulerP & /*sched*/, bool /*need_old_coarse*/, bool /*need_new_coarse*/ ) override {};
    virtual void scheduleCoarsen ( LevelP const & level_coarse, SchedulerP & sched ) override;
    virtual void scheduleErrorEstimate ( LevelP const & level_coarse, SchedulerP & sched ) override;
    virtual void scheduleInitialErrorEstimate ( LevelP const & level_coarse, SchedulerP & sched ) override;

    void scheduleTimeAdvance_forward_euler_refinement ( LevelP const & level, SchedulerP & );

    void task_forward_euler_time_advance_refinement ( ProcessorGroup const * myworld, PatchSubset const * patches_fine, MaterialSubset const * matls, DataWarehouse * dw_old, DataWarehouse * dw_new );
    void task_refine ( ProcessorGroup const * myworld, PatchSubset const * patches_fine, MaterialSubset const * matls, DataWarehouse * dw_old, DataWarehouse * dw_new );
    void task_coarsen ( ProcessorGroup const * myworld, PatchSubset const * patches_coarse, MaterialSubset const * matls, DataWarehouse * dw_old, DataWarehouse * dw_new );
    void task_error_estimate ( ProcessorGroup const * myworld, PatchSubset const * patches, MaterialSubset const * matls, DataWarehouse * dw_old, DataWarehouse * dw_new );

    void forward_euler_time_advance_refinement ( int i, int j, int k, Patch const * patch, ConstVariableView u_old, VariableView u_new, Patch::FaceType face, const std::string bc_kind[4], const double bc_value[4], ConstVariable u_coarse_old[4] );
    void refine ( int i_fine, int j_fine, int k_fine, Level const * level_fine, Level const * level_coarse, ConstVariableView u_coarse, VariableView u_fine );
    void coarsen ( int i_coarse, int j_coarse, int k_coarse, Level const * level_coarse, Level const * level_fine, ConstVariableView u_fine, VariableView u_coarse );
    void error_estimate ( int i, int j, int k, Patch const * patch, ConstVariableView u, FlagView flag_refine, bool & refine_patch, double const & refine_threshold );

    double interpolate_coarser ( const IntVector & cell_fine, Level const * level_fine, Level const * level_coarse, ConstVariableView u_coarse );
    double restrict_finer ( const IntVector & cell_coarse, Level const * level_coarse, Level const * level_fine, ConstVariableView u_fine );
}; // class AMRCCHeat2D

} // namespace Uintah

constexpr inline Uintah::AMRCCHeat2D::FlagView Uintah::AMRCCHeat2D::get_view ( FlagVariable & var )
{
#ifdef UINTAH_ENABLE_KOKKOS
    return var.getKokkosView();
#else
    return var;
#endif
}

inline Uintah::IntVector Uintah::AMRCCHeat2D::map_to_coarser ( Level const * level, IntVector const & i )
{
    return level->mapCellToCoarser ( i );
}

inline Uintah::IntVector Uintah::AMRCCHeat2D::map_to_finer ( Level const * level, IntVector const & i )
{
    return level->mapCellToFiner ( i );
}

#endif // Packages_Uintah_CCA_Components_Heat_AMRCCHeat2D_h
