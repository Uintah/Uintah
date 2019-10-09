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


#ifndef Packages_Uintah_CCA_Components_ontheflyAnalysis_findFragments_h
#define Packages_Uintah_CCA_Components_ontheflyAnalysis_findFragments_h

#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Ports/Output.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <list>
#include <vector>

namespace Uintah {


  class findFragments : public AnalysisModule {
  public:
    findFragments(ProblemSpecP       & prob_spec,
                  SimulationStateP   & sharedState,
		    Output             * dataArchiver);

    findFragments();

    virtual ~findFragments();

    virtual void problemSetup(const ProblemSpecP & prob_spec,
                              const ProblemSpecP & restart_prob_spec,
                              GridP              & grid,
                              SimulationStateP   & sharedState);

    virtual void outputProblemSpec(ProblemSpecP& ps){};

    virtual void scheduleInitialize(SchedulerP   & sched,
                                    const LevelP & level);

    virtual void scheduleRestartInitialize(SchedulerP  & sched,
                                           const LevelP& level);

    virtual void restartInitialize(){};

    virtual void scheduleDoAnalysis(SchedulerP   & sched,
                                    const LevelP & level);

    virtual void scheduleDoAnalysis_preReloc(SchedulerP & sched,
                                             const LevelP& level) {};

  private:

    void initialize( const ProcessorGroup *,
                     const PatchSubset    * patches,
                     const MaterialSubset *,
                     DataWarehouse        *,
                     DataWarehouse        * new_dw );

    void doAnalysis( const ProcessorGroup * pg,
                     const PatchSubset    * patches,
                     const MaterialSubset *,
                     DataWarehouse        *,
                     DataWarehouse        * new_dw );
                     
    void sched_sumLocalizedParticles(SchedulerP   & sched,
                                     const LevelP & level);

    void sumLocalizedParticles(const ProcessorGroup * pg,
                               const PatchSubset    * patches,
                               const MaterialSubset *,
                               DataWarehouse        *,
                               DataWarehouse        * new_dw);

    void sched_identifyFragments(SchedulerP   & sched,
                                 const LevelP & level);

    void identifyFragments(const ProcessorGroup * pg,
                           const PatchSubset    * patches,
                           const MaterialSubset *,
                           DataWarehouse        *,
                           DataWarehouse        * new_dw);

    std::list<IntVector> 
    intersection_of( const std::list<IntVector>& a, 
                     const std::list<IntVector>& b);
                                   
    void checkNeighborCells( const std::string    & desc,
                             std::list<IntVector> & cellList,
                             const IntVector      & cell, 
                             const Patch          * patch,
                             const int              fragID,
                             constCCVariable<int> & numLocalized,
                             CCVariable<int>      & fragmentID,
                             CCVariable<int>      & nTouched );
                             
    void sched_sumQ_inFragments( SchedulerP   & sched,
                                 const LevelP & level);


    void sumQ_inFragments( const ProcessorGroup * pg,
                           const PatchSubset    * patches,
                           const MaterialSubset *,
                           DataWarehouse        * old_dw,
                           DataWarehouse        * new_dw);


    void createFile( const std::string  & filename,
                     const VarLabel     * varLabel,
                     const int matl,
                     FILE*& fp );

    void createDirectory( std::string & planeName,
                          std::string & timestep,
                          const double  now,
                          std::string & levelIndex );

    // general labels
    class findFragmentsLabel {
    public:
      VarLabel* prevAnalysisTimeLabel;
      VarLabel* fileVarsStructLabel;
      VarLabel* fragmentIDLabel;
      VarLabel* maxFragmentIDLabel;
      VarLabel* nTouchedLabel;
      VarLabel* gMassLabel;
      VarLabel* pMassLabel;
      VarLabel* pLocalizedLabel;
      VarLabel* pXLabel;
      
      VarLabel* numLocalized_CCLabel;
    };

    findFragmentsLabel* d_lb;

    //__________________________________
    //
    double d_analysisFreq;
    double d_startTime;
    double d_stopTime;

    std::vector<VarLabel*> d_varLabels;

    SimulationStateP d_sharedState;
    Output*          d_dataArchiver;
    ProblemSpecP     d_prob_spec;
    std::set<std::string> d_isDirCreated;
    
    const Material*  d_matl;
    MaterialSet*     d_matl_set;
    const MaterialSubset*  d_matl_subSet;
  };
}

#endif
