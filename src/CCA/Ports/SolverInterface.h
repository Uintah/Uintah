/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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


#ifndef Packages_Uintah_CCA_Ports_SolverInterace_h
#define Packages_Uintah_CCA_Ports_SolverInterace_h

#include <Core/Parallel/UintahParallelPort.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/LevelP.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/Reductions.h>
#include <Core/Grid/Variables/ReductionVariable.h>

#include <string>


namespace Uintah {
  class VarLabel;
  class SolverParameters {
  public:
    SolverParameters() : useStencil4(false), solveOnExtraCells(false), residualNormalizationFactor(1), 
                        restartableTimestep(false), outputFileName("NULL") {}
    
    void setSolveOnExtraCells(bool s) {
      solveOnExtraCells = s;
    }
    
    bool getSolveOnExtraCells() const {
      return solveOnExtraCells;
    }
    
    void setUseStencil4(bool s) {
      useStencil4 = s;
    }
    
    bool getUseStencil4() const {
      return useStencil4;
    }
    
    void setResidualNormalizationFactor(double s) {
      residualNormalizationFactor = s;
    }
    
    double getResidualNormalizationFactor() const {
      return residualNormalizationFactor;
    }
    
    //If convergence fails call for the timestep to be restarted.
    void setRestartTimestepOnFailure(bool s){
      restartableTimestep=s;
    }
    
    bool getRestartTimestepOnFailure() const {
      return restartableTimestep;
    }
    
    // Used for outputting A, X & B to files
    void setOutputFileName(std::string s){
      outputFileName=s;
    }
    
    void getOutputFileName(std::vector<std::string>& fname) const {
      fname.push_back( "A" + outputFileName );
      fname.push_back( "b" + outputFileName );
      fname.push_back( "x" + outputFileName );
    }

    void setSetupFrequency(const int freq) {}

    int getSetupFrequency() const { return 1;}
        
    virtual ~SolverParameters() {}

  private:
    bool        useStencil4;
    bool        solveOnExtraCells;
    double      residualNormalizationFactor;
    bool        restartableTimestep;
    std::string outputFileName;
  };
  
  class SolverInterface : public UintahParallelPort {
  public:
    SolverInterface()
    {}
    virtual ~SolverInterface()
    {
      for (int i=0;  i < varLabels_.size(); ++i) {
        VarLabel::destroy(varLabels_[i]);
      }
    }

    virtual SolverParameters* readParameters( ProblemSpecP     & params,
					                                    const std::string      & name,
                                              SimulationStateP & state ) = 0;

    virtual void scheduleInitialize( const LevelP      & level,
                                           SchedulerP  & sched,
                                     const MaterialSet * matls) = 0;
                            
    virtual void scheduleSolve( const LevelP           & level,
                                      SchedulerP       & sched,
                                const MaterialSet      * matls,
                                const VarLabel         * A,
                                      Task::WhichDW      which_A_dw,
                                const VarLabel         * x,
                                      bool               modifies_x,
                                const VarLabel         * b,
                                      Task::WhichDW      which_b_dw,
                                const VarLabel         * guess,
                                      Task::WhichDW      which_guess_dw,
                                const SolverParameters * params,
                                      bool               modifies_hypre = false ) = 0;

    virtual std::string getName() = 0;
    
    //----------------------------------------------------------------------------------------------
    /**
     \brief Enforces solvability condition on periodic problems or in domains where boundary
     conditions on the Poisson system are zero Neumann (dp/dn = 0).
     \param bLabel Varlabel of the Poisson system right hand side (RHS). The RHS MUST live in the 
     newDW (i.e. be modifiable).
     The remaining parameters take the standard form of other Uintah tasks.
     \param rkStage: In a multistep integration scheme, Uintah is incapable of dealing
     with multiple reductions on the same variable. Hence the need for a stage number
     (e.g. rkStage) to create unique varlabels
     */
    template <typename FieldT>
    void scheduleEnforceSolvability( const LevelP & level,
                                     SchedulerP   & sched,
                                     const MaterialSet  * matls,
                                     const VarLabel     * bLabel,
                                     const int rkStage = 1)
    {
      std::stringstream strRKStage;
      strRKStage << rkStage;
      const std::string varName = "poisson_rhs_integral" + strRKStage.str();
      VarLabel* rhsIntegralLabel = VarLabel::find(varName);
      if (rhsIntegralLabel == NULL) {
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
    
    //----------------------------------------------------------------------------------------------
    /**
     \brief Set a reference pressure in the domain. The user picks a reference cell (cRef) and a 
     desired reference value pRef. The pressure in the reference cell is pCell = p[cRef].
     Then, pCell + dp = pRef
     or dp = pRef - pCell
     The value dp is the pressure difference that needs to be added to the pressure solution at ALL 
     points in the domain to adjust for the reference pressure.
     */
    template <typename FieldT>
    void scheduleSetReferenceValue( const LevelP & level,
                                    SchedulerP   & sched,
                                    const MaterialSet  * matls,
                                    const VarLabel     * xLabel,
                                    const int rkStage = 1,
                                    const IntVector refCell = IntVector(0,0,0),
                                    const double refValue = 0.0)
    {
      std::stringstream strRKStage;
      strRKStage << rkStage;
      const std::string varName = "poisson_ref_value_offset" + strRKStage.str();
      VarLabel* refValueLabel = VarLabel::find(varName);
      if (refValueLabel == NULL) {
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
    

  private:
    SolverInterface(const SolverInterface&);
    SolverInterface& operator=(const SolverInterface&);
    std::vector<VarLabel*> varLabels_;

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
    void findRefValueDiff( const Uintah::ProcessorGroup*,
                            const Uintah::PatchSubset* patches,
                            const Uintah::MaterialSubset* materials,
                            Uintah::DataWarehouse* old_dw,
                            Uintah::DataWarehouse* new_dw,
                            const VarLabel         * xLabel,
                          VarLabel         * refValueLabel,
                          const IntVector refCell,
                          const double refValue)
    {
      for( int ip=0; ip<patches->size(); ++ip ) {
        const Uintah::Patch* const patch = patches->get(ip);
        for( int im=0; im<materials->size(); ++im ){
          int matl = materials->get(im);
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
     \brief This task adds dp (see above) to the pressure at all points in the domain to reflect
     the reference value specified by the user/developer.
     */
    template<typename FieldT>
    void setRefValue( const Uintah::ProcessorGroup*,
                            const Uintah::PatchSubset* patches,
                            const Uintah::MaterialSubset* materials,
                            Uintah::DataWarehouse* old_dw,
                            Uintah::DataWarehouse* new_dw,
                            const VarLabel         * xLabel,
                            VarLabel* refValueLabel)
    {
      // once we've computed the difference needed to set the ref pressure, subtract it from all points
      for( int ip=0; ip<patches->size(); ++ip ){
        const Patch* const patch = patches->get(ip);
        for( int im=0; im<materials->size(); ++im ){
          int matl = materials->get(im);
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
     \brief Computes the volume integral of the RHS of the Poisson equation: 1/V * int(rhs*dV)
     Since Uintah deals with uniform structured grids, the above equation can be simplified, discretely,
     to: 1/n * sum(rhs) where n is the total number of cell sin the domain.
     */
    template<typename FieldT>
    void computeRHSIntegral( const Uintah::ProcessorGroup*,
                            const Uintah::PatchSubset* patches,
                            const Uintah::MaterialSubset* materials,
                            Uintah::DataWarehouse* old_dw,
                            Uintah::DataWarehouse* new_dw,
                            const VarLabel         * bLabel,
                            VarLabel* rhsIntegralLabel)
    {
      for( int ip=0; ip<patches->size(); ++ip ) {
        const Uintah::Patch* const patch = patches->get(ip);
        for( int im=0; im<materials->size(); ++im ){
          int matl = materials->get(im);
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
     \brief Modifies the RHS of the Poisson equation to satisfy the solvability condition
     on periodic problems.
     */
    template<typename FieldT>
    void enforceSolvability( const Uintah::ProcessorGroup*,
                            const Uintah::PatchSubset* patches,
                            const Uintah::MaterialSubset* materials,
                            Uintah::DataWarehouse* old_dw,
                            Uintah::DataWarehouse* new_dw,
                            const VarLabel         * bLabel,
                            VarLabel * rhsIntegralLabel)
    {
      // once we've computed the total integral, subtract it from the poisson rhs
      for( int ip=0; ip<patches->size(); ++ip ){
        const Patch* const patch = patches->get(ip);
        for( int im=0; im<materials->size(); ++im ){
          int matl = materials->get(im);
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
  };
}

#endif
