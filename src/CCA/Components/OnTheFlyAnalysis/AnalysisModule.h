/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef Packages_Uintah_CCA_Ports_AnalysisModule_h
#define Packages_Uintah_CCA_Ports_AnalysisModule_h

#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/SchedulerP.h>

#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  class ApplicationInterface;
  class Output;
  class Scheduler;

  class Material;
  class VarLabel;

  class AnalysisModule : public UintahParallelComponent {
  public:

    AnalysisModule(const ProcessorGroup* myworld,
                   const MaterialManagerP materialManager,
                   const ProblemSpecP& module_spec);

    virtual ~AnalysisModule();

    // Methods for managing the components attached via the ports.
    virtual void setComponents( UintahParallelComponent *comp ) {};
    virtual void setComponents( ApplicationInterface *comp );
    virtual void getComponents();
    virtual void releaseComponents();

    virtual void problemSetup(const ProblemSpecP& prob_spec,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid,
                              std::vector<std::vector<const VarLabel* > > &PState,
                              std::vector<std::vector<const VarLabel* > > &PState_preReloc) = 0;

    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;


    virtual void scheduleInitialize(SchedulerP& sched,
                                    const LevelP& level) = 0;

    virtual void scheduleRestartInitialize(SchedulerP& sched,
                                           const LevelP& level) = 0;

    virtual void restartInitialize() = 0;

    virtual void scheduleDoAnalysis(SchedulerP& sched,
                                    const LevelP& level) = 0;

    virtual void scheduleDoAnalysis_preReloc(SchedulerP& sched,
                                             const LevelP& level) = 0;


    //__________________________________
    //  time related
    struct timeVars{
        double prevAnlysTime {-9e9};
        double nextAnlysTime {-9e9};
        double now           {-9e9};
        bool isItTime {false};
    };

    void sched_TimeVars( Task * t,
                         const LevelP   & level,
                         const VarLabel * prev_AnlysTimeLabel,
                         const bool addComputes );
                                                                       
    bool getTimeVars( DataWarehouse  * old_dw,
                      const Level    * level,
                      const VarLabel * prev_AnlysTime,
                      timeVars       & tv );
                   
    void putTimeVars( DataWarehouse  * new_dw,
                      const VarLabel * prev_AnlysTimeLabel,
                      timeVars tv);

    bool isItTime( DataWarehouse  * old_dw,
                   const Level    * level,
                   const VarLabel * prev_AnlysTime );



    //__________________________________
    //  Variables
  protected:
    ApplicationInterface*  m_application {nullptr};
    Output*                m_output      {nullptr};
    Scheduler*             m_scheduler   {nullptr};

    MaterialManagerP m_materialManager {nullptr};
    ProblemSpecP     m_module_spec     {nullptr};

    const VarLabel* m_timeStepLabel       {nullptr};
    const VarLabel* m_simulationTimeLabel {nullptr};
    const VarLabel* m_delTLabel           {nullptr};

    double     m_analysisFreq;               // analysis frequency: units 1/sec
    double     d_startTime{0.0};
    double     d_stopTime{DBL_MAX};
  };
}

#endif
