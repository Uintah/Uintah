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

#ifndef UINTAH_HOMEBREW_Component_PostProcess_H
#define UINTAH_HOMEBREW_Component_PostProcess_H

#include <CCA/Components/Application/ApplicationCommon.h>
#include <CCA/Components/PostProcessUda/Module.h>
#include <vector>

namespace Uintah {
  class LoadBalancer;
  class Module;


  class PostProcessUda : public ApplicationCommon {

  public:
    PostProcessUda( const ProcessorGroup * myworld,
                    const MaterialManagerP materialManager,
                    const std::string    & udaDir );

    virtual ~PostProcessUda();

    virtual void problemSetup( const ProblemSpecP & params,
                               const ProblemSpecP & restart_prob_spec,
                               GridP              & grid );

    virtual void scheduleInitialize( const LevelP & level,
                                     SchedulerP   & );

    virtual void scheduleRestartInitialize( const LevelP & level,
                                            SchedulerP   & ){};

    virtual void restartInitialize() {}

    virtual void scheduleComputeStableTimeStep( const LevelP &,
                                                SchedulerP   & );

    virtual void scheduleTimeAdvance( const LevelP & level,
                                      SchedulerP   & );


    virtual bool needRecompile( const GridP & grid );

    virtual void scheduleFinalizeTimestep(const LevelP & level,
                                          SchedulerP   &){};

    // stubs
    virtual void scheduleInitialErrorEstimate  ( const LevelP& , SchedulerP&  ){};
    virtual void scheduleCoarsen               ( const LevelP& , SchedulerP&  ){};
    virtual void scheduleRefine                ( const PatchSet*, SchedulerP& ){};
    virtual void scheduleRefineInterface       ( const LevelP& , SchedulerP& , bool, bool){};


    double getMaxTime();

    double getInitialTime();

    GridP getGrid();
  //______________________________________________________________________
  //
  private:
    PostProcessUda(const PostProcessUda&);
    PostProcessUda& operator=(const PostProcessUda&);


    void computeDelT(const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* /*old_dw*/,
                     DataWarehouse* new_dw);

    void sched_readDataArchive(const LevelP& level,
                               SchedulerP& sched);

    void readDataArchive(const ProcessorGroup*,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* /*old_dw*/,
                         DataWarehouse* new_dw);

    void finalizeTimestep(const ProcessorGroup*,
                          const PatchSubset*,
                          const MaterialSubset*,
                          DataWarehouse*,
                          DataWarehouse* );

    void doAnalysis(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw);
                    
    enum {NOTUSED = -9};
    std::string            d_udaDir;
    bool                   d_gridChanged;

    std::vector<int>       d_udaTimesteps;
    std::vector<int>       d_numMatls;
    std::vector<double>    d_udaTimes;
    std::vector<VarLabel*> d_udaSavedLabels;

    GridP                  d_oldGrid;
    DataArchive          * d_dataArchive = nullptr;

    int                    d_simTimestep = 0;

    std::vector<Module*> d_Modules;
  };
} // End namespace Uintah

#endif
