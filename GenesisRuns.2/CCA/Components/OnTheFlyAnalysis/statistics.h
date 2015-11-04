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


#ifndef Packages_Uintah_CCA_Components_ontheflyAnalysis_statistics_h
#define Packages_Uintah_CCA_Components_ontheflyAnalysis_statistics_h
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Ports/Output.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>

#include <vector>

namespace Uintah {


/**************************************

CLASS
   statistics

GENERAL INFORMATION

   statistics.h

   Todd Harman
   Department of Mechanical Engineering
   University of Utah


KEYWORDS
   statistics

DESCRIPTION
   This computes turbulence related statistical quantities


WARNING

****************************************/
  class statistics : public AnalysisModule {
  public:
    statistics(ProblemSpecP& prob_spec,
              SimulationStateP& sharedState,
		Output* dataArchiver);

    statistics();

    virtual ~statistics();

    virtual void problemSetup(const ProblemSpecP& prob_spec,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid,
                              SimulationStateP& sharedState);


    virtual void scheduleInitialize(SchedulerP& sched,
                                    const LevelP& level);

    virtual void restartInitialize();

    virtual void scheduleDoAnalysis(SchedulerP& sched,
                                    const LevelP& level);

    void scheduleDoAnalysis_preReloc(SchedulerP& sched,
                                     const LevelP& level) {};

  private:
    //__________________________________
    //  container to hold
    struct Qstats{
      std::string  name;
      int matl;
      VarLabel* Q_Label;
      VarLabel* Qsum_Label;
      VarLabel* QsumSqr_Label;
      VarLabel* Qmean_Label;
      VarLabel* QmeanSqr_Label;
      VarLabel* Qvariance_Label;
      const Uintah::TypeDescription* subtype;

      void print(){
        std::cout << name << " matl: " << matl << " subtype: " << subtype->getName() << "\n";
      };
    };

    //__________________________________
    //
    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset*,
                    DataWarehouse*,
                    DataWarehouse* new_dw);

    void doAnalysis(const ProcessorGroup* pg,
                    const PatchSubset* patches,
                    const MaterialSubset*,
                    DataWarehouse*,
                    DataWarehouse* new_dw);

    template <class T>
    void allocateAndZero( DataWarehouse* new_dw,
                          const Patch*    patch,
                          Qstats Q);

    template <class T>
    void computeStats( DataWarehouse* old_dw,
                       DataWarehouse* new_dw,
                       const Patch*   patch,
                       Qstats Q);

    //__________________________________
    // global constants
    std::vector< Qstats >  d_Qstats;


    SimulationStateP d_sharedState;
    Output* d_dataArchiver;
    ProblemSpecP d_prob_spec;
    const Material* d_matl;
    MaterialSet* d_matlSet;
    const MaterialSubset* d_matSubSet;
  };


}

#endif
