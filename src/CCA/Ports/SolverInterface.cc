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



// I assume this comment in out of date-> These routines are not needed, can just delete this file...

#include <CCA/Ports/SolverInterface.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Ports/DataWarehouse.h>

using namespace Uintah;

namespace Uintah {

  template <typename FieldT>
  void
  SolverInterface::scheduleEnforceSolvability( const LevelP       & level,
                                                     SchedulerP   & sched,
                                               const MaterialSet  * matls,
                                               const VarLabel     * bLabel,
                                               const int            rkStage )
  {
    std::stringstream strRKStage;
    strRKStage << rkStage;
    const std::string varName = "poisson_rhs_integral" + strRKStage.str();
    VarLabel* rhsIntegralLabel = VarLabel::find(varName);
    if (rhsIntegralLabel == nullptr) {
      rhsIntegralLabel = VarLabel::create(varName, sum_vartype::getTypeDescription());
      varLabels_.push_back(rhsIntegralLabel);
    }
    
    Task* tskIntegral = scinew Task("SolverInterface::computeRHSIntegral" + strRKStage.str(),
                                    this, &SolverInterface::computeRHSIntegral<FieldT>, bLabel,
                                    rhsIntegralLabel);
    tskIntegral->computes( rhsIntegralLabel );
    tskIntegral->requires( Uintah::Task::NewDW, bLabel, Ghost::None, 0 );
    sched->addTask(tskIntegral, level->eachPatch(), matls);
    
    Task* tskSolvability = scinew Task("SolverInterface::enforceSolvability"+ strRKStage.str(),
                                       this, &SolverInterface::enforceSolvability<FieldT>, bLabel,
                                       rhsIntegralLabel);
    tskSolvability->requires( Uintah::Task::NewDW, rhsIntegralLabel );
    tskSolvability->modifies( bLabel );
    sched->addTask(tskSolvability, level->eachPatch(), matls);
  }

  template <typename FieldT>
  void
  SolverInterface::scheduleSetReferenceValue( const LevelP      & level,
                                                    SchedulerP  & sched,
                                              const MaterialSet * matls,
                                              const VarLabel    * xLabel,
                                              const int           rkStage,
                                              const IntVector     refCell,
                                              const double        refValue )
  {
    std::stringstream strRKStage;
    strRKStage << rkStage;
    const std::string varName = "poisson_ref_value_offset" + strRKStage.str();
    VarLabel* refValueLabel = VarLabel::find(varName);
    if (refValueLabel == nullptr) {
      refValueLabel = VarLabel::create(varName, sum_vartype::getTypeDescription());
      varLabels_.push_back(refValueLabel);
    }
    
    Task* tskFindDiff = scinew Task("SolverInterface::findRefValueDiff",
                                    this, &SolverInterface::findRefValueDiff<FieldT>, xLabel,
                                    refValueLabel, refCell, refValue);
    tskFindDiff->computes( refValueLabel );
    tskFindDiff->requires( Uintah::Task::NewDW, xLabel, Ghost::None, 0 );
    sched->addTask(tskFindDiff, level->eachPatch(), matls);
    
    Task* tskSetRefValue = scinew Task("SolverInterface::setRefValue",
                                       this, &SolverInterface::setRefValue<FieldT>, xLabel,
                                       refValueLabel);
    tskSetRefValue->requires( Uintah::Task::NewDW, refValueLabel );
    tskSetRefValue->modifies( xLabel );
    sched->addTask(tskSetRefValue, level->eachPatch(), matls);
  }

  //----------------------------------------------------------------------------------------------
  /**
   \brief The user picks a reference cell (cRef) and a
   desired reference value pRef. The pressure in the reference cell is pCell = p[cRef].
   Then, pCell + dp = pRef
   or dp = pRef - pCell
   This task computes dp and broadcasts it across all processors.
   We do this using a reduction variable because Uintah doesn't provide us with a nice
   interface for doing a broadcast.
   */
  template<typename FieldT>
  void
  SolverInterface::findRefValueDiff( const Uintah::ProcessorGroup *,
                                     const Uintah::PatchSubset    * patches,
                                     const Uintah::MaterialSubset * materials,
                                           Uintah::DataWarehouse  * old_dw,
                                           Uintah::DataWarehouse  * new_dw,
                                     const VarLabel               * xLabel,
                                           VarLabel               * refValueLabel,
                                     const IntVector                refCell,
                                     const double                   refValue )
  {
    for( int ip=0; ip<patches->size(); ++ip ) {
      const Uintah::Patch* const patch = patches->get(ip);
      for( int im=0; im<materials->size(); ++im ){
        // int matl = materials->get(im);
        FieldT x;
        new_dw->getModifiable(x, xLabel, im, patch);
        double refValueDiff = 0.0;
        if (patch->containsCell(refCell)) {
          const double cellValue = x[refCell];
          refValueDiff = refValue - cellValue;
        }
        new_dw->put( sum_vartype(refValueDiff), refValueLabel );
      }
    }
  }

  //----------------------------------------------------------------------------------------------
  /**
   \brief Computes the volume integral of the RHS of the Poisson equation: 1/V * int(rhs*dV)
   Since Uintah deals with uniform structured grids, the above equation can be simplified, discretely,
   to: 1/n * sum(rhs) where n is the total number of cell sin the domain.
   */
  template<typename FieldT>
  void
  SolverInterface::computeRHSIntegral( const Uintah::ProcessorGroup *,
                                       const Uintah::PatchSubset    * patches,
                                       const Uintah::MaterialSubset * materials,
                                             Uintah::DataWarehouse  * old_dw,
                                             Uintah::DataWarehouse  * new_dw,
                                       const VarLabel               * bLabel,
                                             VarLabel               * rhsIntegralLabel )
  {
    for( int ip=0; ip<patches->size(); ++ip ) {
      const Uintah::Patch* const patch = patches->get(ip);
      for( int im=0; im<materials->size(); ++im ){
        // int matl = materials->get(im);
        // compute integral of b
        FieldT b;
        new_dw->getModifiable(b, bLabel, im, patch);
        double rhsIntegral = 0.0;
        for(Uintah::CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
          IntVector iCell = *iter;
          rhsIntegral += b[iCell];
        }
        // divide by total volume.
        rhsIntegral /= patch->getLevel()->totalCells();
        new_dw->put( sum_vartype(rhsIntegral), rhsIntegralLabel );
      }
    }
  }

  //----------------------------------------------------------------------------------------------
  /**
   \brief This task adds dp (see above) to the pressure at all points in the domain to reflect
   the reference value specified by the user/developer.
   */
  template<typename FieldT>
  void
  SolverInterface::setRefValue( const Uintah::ProcessorGroup *,
                                const Uintah::PatchSubset    * patches,
                                const Uintah::MaterialSubset * materials,
                                      Uintah::DataWarehouse  * old_dw,
                                      Uintah::DataWarehouse  * new_dw,
                                const VarLabel               * xLabel,
                                      VarLabel               * refValueLabel )
  {
    // once we've computed the total integral, subtract it from the poisson rhs
    for( int ip=0; ip<patches->size(); ++ip ){
      const Patch* const patch = patches->get(ip);
      for( int im=0; im<materials->size(); ++im ){
        // int matl = materials->get(im);
        sum_vartype refValueDiff_;
        new_dw->get( refValueDiff_, refValueLabel );
        const double refValueDiff = refValueDiff_;
        FieldT x;
        new_dw->getModifiable(x, xLabel, im, patch);
        for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
          IntVector iCell = *iter;
          x[iCell] += refValueDiff;
        }
      }
    }
  }

  //----------------------------------------------------------------------------------------------
  /**
   \brief Modifies the RHS of the Poisson equation to satisfy the solvability condition
   on periodic problems.
   */
  template<typename FieldT>
  void
  SolverInterface::enforceSolvability( const Uintah::ProcessorGroup *,
                                       const Uintah::PatchSubset    * patches,
                                       const Uintah::MaterialSubset * materials,
                                             Uintah::DataWarehouse  * old_dw,
                                             Uintah::DataWarehouse  * new_dw,
                                       const VarLabel               * bLabel,
                                             VarLabel               * rhsIntegralLabel )
  {
    // once we've computed the total integral, subtract it from the poisson rhs
    for( int ip=0; ip<patches->size(); ++ip ){
      const Patch* const patch = patches->get(ip);
      for( int im=0; im<materials->size(); ++im ){
        // int matl = materials->get(im);
        sum_vartype rhsIntegral_;
        new_dw->get( rhsIntegral_, rhsIntegralLabel );
        double rhsIntegral = rhsIntegral_;
        FieldT b;
        new_dw->getModifiable(b, bLabel, im, patch);
        for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
          IntVector iCell = *iter;
          b[iCell] -= rhsIntegral;
        }
      }
    }
  }
  //----------------------------------------------------------------------------------------------

  // explicit instantiation


  #define INSTANTIATE_TYPES(VOLT) \
                                                                                                    \
  template void SolverInterface::scheduleEnforceSolvability<VOLT>( const LevelP & level,            \
                                                                   SchedulerP   & sched,            \
                                                                   const MaterialSet  * matls,      \
                                                                   const VarLabel     * bLabel,     \
                                                                   const int rkStage);              \
  template void SolverInterface::scheduleSetReferenceValue<VOLT>( const LevelP & level,             \
                                                                  SchedulerP   & sched,             \
                                                                  const MaterialSet  * matls,       \
                                                                  const VarLabel     * xLabel,      \
                                                                  const int rkStage,                \
                                                                  const IntVector refCell,          \
                                                                  const double refValue);           \
  template void SolverInterface::findRefValueDiff<VOLT>( const Uintah::ProcessorGroup*,             \
                                                         const Uintah::PatchSubset* patches,        \
                                                         const Uintah::MaterialSubset* materials,   \
                                                         Uintah::DataWarehouse* old_dw,             \
                                                         Uintah::DataWarehouse* new_dw,             \
                                                         const VarLabel         * xLabel,           \
                                                         VarLabel         * refValueLabel,          \
                                                         const IntVector refCell,                   \
                                                         const double refValue);                    \
  template void SolverInterface::computeRHSIntegral<VOLT>( const Uintah::ProcessorGroup*,           \
                                                           const Uintah::PatchSubset* patches,      \
                                                           const Uintah::MaterialSubset* materials, \
                                                           Uintah::DataWarehouse* old_dw,           \
                                                           Uintah::DataWarehouse* new_dw,           \
                                                           const VarLabel         * bLabel,         \
                                                           VarLabel* rhsIntegralLabel);             \
  template void SolverInterface::setRefValue<VOLT>( const Uintah::ProcessorGroup*,                  \
                                                    const Uintah::PatchSubset* patches,             \
                                                    const Uintah::MaterialSubset* materials,        \
                                                    Uintah::DataWarehouse* old_dw,                  \
                                                    Uintah::DataWarehouse* new_dw,                  \
                                                    const VarLabel         * xLabel,                \
                                                    VarLabel* refValueLabel);                       \
  template void SolverInterface::enforceSolvability<VOLT>( const Uintah::ProcessorGroup*,           \
                                                           const Uintah::PatchSubset* patches,      \
                                                           const Uintah::MaterialSubset* materials, \
                                                           Uintah::DataWarehouse* old_dw,           \
                                                           Uintah::DataWarehouse* new_dw,           \
                                                           const VarLabel         * bLabel,         \
                                                           VarLabel * rhsIntegralLabel);

  INSTANTIATE_TYPES(CCVariable<double>);
}
