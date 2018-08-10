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


#ifndef Packages_Uintah_CCA_Ports_SolverInterace_h
#define Packages_Uintah_CCA_Ports_SolverInterace_h

#include <Core/Parallel/UintahParallelPort.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/LevelP.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SchedulerP.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <string>


namespace Uintah {

  class UintahParallelComponent;
  class VarLabel;
  
  //______________________________________________________________________
  //
  class SolverParameters   {
  
  public:
    SolverParameters() : useStencil4(false),
                         symmetric(true),
                         solveOnExtraCells(false),
                         residualNormalizationFactor(1),
                         recomputableTimeStep(false),
                         setupFrequency(1),
                         updateCoefFrequency(1),
                         outputFileName("nullptr"),
                         m_which_old_dw(Task::OldDW) {}
    
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

    void setSymmetric(bool s) {
      symmetric = s;
    }
    
    bool getSymmetric() const {
      return symmetric;
    }

    void setResidualNormalizationFactor(double s) {
      residualNormalizationFactor = s;
    }
    
    double getResidualNormalizationFactor() const {
      return residualNormalizationFactor;
    }
    
    // If convergence fails call for the time step to be recomputed.
    void setRecomputeTimeStepOnFailure(bool s) {
      recomputableTimeStep = s;
    }
    
    bool getRecomputeTimeStepOnFailure() const {
      return recomputableTimeStep;
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

    void setSetupFrequency(const int freq) {
      setupFrequency = freq;
    }
    
    int getSetupFrequency() const { 
      return setupFrequency;
    }

    void setUpdateCoefFrequency(const int freq) {
      updateCoefFrequency = freq;
    }
    int  getUpdateCoefFrequency() const { 
      return updateCoefFrequency;
    }
    
    // oldDW or ParentOldDW
    void setWhichOldDW(Task::WhichDW dw){
      ASSERT( ((dw == Task::OldDW ) || ( dw == Task::ParentOldDW)) );
      m_which_old_dw = dw;
    }
    
    Task::WhichDW getWhichOldDW() const {
      return m_which_old_dw;
    }    
    
    virtual ~SolverParameters() {}

  private:
    bool        useStencil4;
    bool        symmetric;
    bool        solveOnExtraCells;
    double      residualNormalizationFactor;
    bool        recomputableTimeStep;
    int         setupFrequency;        // delete matrix and recreate it and update coefficients. Needed if Stencil changes.
    int         updateCoefFrequency;   // do not modify matrix stencil/sparsity - only change values of coefficients
    std::string outputFileName;
    Task::WhichDW  m_which_old_dw;     // DataWarehouse either old_dw or parent_old_dw
  };
  
  //______________________________________________________________________
  
  class SolverInterface : public UintahParallelPort {

  public:
    SolverInterface() {}

    virtual ~SolverInterface()
    {
      for (size_t i = 0; i < varLabels_.size(); ++i ) {
        VarLabel::destroy( varLabels_[i] );
      }
    }

    // Methods for managing the components attached via the ports.
    virtual void setComponents( UintahParallelComponent *comp ) = 0;
    virtual void getComponents() = 0;
    virtual void releaseComponents() = 0;
  
    virtual void readParameters( ProblemSpecP & params,
                                 const std::string  & name ) = 0;
                            
    virtual SolverParameters * getParameters() = 0;                        

    virtual void scheduleInitialize( const LevelP      & level,
                                           SchedulerP  & sched,
                                     const MaterialSet * matls) = 0;

    virtual void scheduleRestartInitialize( const LevelP      & level,
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
                                      bool               isFirstSolve = true ) = 0;

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
    void scheduleEnforceSolvability( const LevelP      & level,
                                           SchedulerP  & sched,
                                     const MaterialSet * matls,
                                     const VarLabel    * bLabel,
                                     const int           rkStage );

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
    void scheduleSetReferenceValue( const LevelP       & level,
                                          SchedulerP   & sched,
                                    const MaterialSet  * matls,
                                    const VarLabel     * xLabel,
                                    const int            rkStage,
                                    const IntVector      refCell,
                                    const double         refValue );

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
    void findRefValueDiff( const Uintah::ProcessorGroup *,
                           const Uintah::PatchSubset    * patches,
                           const Uintah::MaterialSubset * materials,
                                 Uintah::DataWarehouse  * old_dw,
                                 Uintah::DataWarehouse  * new_dw,
                           const VarLabel               * xLabel,
                                 VarLabel               * refValueLabel,
                           const IntVector                refCell,
                           const double                   refValue );

    //----------------------------------------------------------------------------------------------
    /**
     \brief Computes the volume integral of the RHS of the Poisson equation: 1/V * int(rhs*dV)
     Since Uintah deals with uniform structured grids, the above equation can be simplified, discretely,
     to: 1/n * sum(rhs) where n is the total number of cell sin the domain.
     */
    template<typename FieldT>
    void computeRHSIntegral( const Uintah::ProcessorGroup *,
                             const Uintah::PatchSubset    * patches,
                             const Uintah::MaterialSubset * materials,
                                   Uintah::DataWarehouse  * old_dw,
                                   Uintah::DataWarehouse  * new_dw,
                             const VarLabel               * bLabel,
                                   VarLabel               * rhsIntegralLabel );
    //----------------------------------------------------------------------------------------------
    /**
     \brief This task adds dp (see above) to the pressure at all points in the domain to reflect
     the reference value specified by the user/developer.
     */
    template<typename FieldT>
    void setRefValue( const Uintah::ProcessorGroup *,
                      const Uintah::PatchSubset    * patches,
                      const Uintah::MaterialSubset * materials,
                            Uintah::DataWarehouse  * old_dw,
                            Uintah::DataWarehouse  * new_dw,
                      const VarLabel               * xLabel,
                            VarLabel               * refValueLabel );

    //----------------------------------------------------------------------------------------------
    /**
     \brief Modifies the RHS of the Poisson equation to satisfy the solvability condition
     on periodic problems.
     */
    template<typename FieldT>
    void enforceSolvability( const Uintah::ProcessorGroup *,
                             const Uintah::PatchSubset    * patches,
                             const Uintah::MaterialSubset * materials,
                                   Uintah::DataWarehouse  * old_dw,
                                   Uintah::DataWarehouse  * new_dw,
                             const VarLabel               * bLabel,
                                   VarLabel               * rhsIntegralLabel );
    //----------------------------------------------------------------------------------------------
  };
}

#endif
