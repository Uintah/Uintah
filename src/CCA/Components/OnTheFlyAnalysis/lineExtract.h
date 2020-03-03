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


#ifndef Packages_Uintah_CCA_Components_ontheflyAnalysis_lineExtract_h
#define Packages_Uintah_CCA_Components_ontheflyAnalysis_lineExtract_h
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Output.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>

#include <map>
#include <vector>

namespace Uintah {


  class lineExtract : public AnalysisModule {
  public:
    lineExtract(const ProcessorGroup* myworld,
                const MaterialManagerP materialManager,
                const ProblemSpecP& module_spec);

    lineExtract();

    virtual ~lineExtract();

    virtual void problemSetup(const ProblemSpecP& prob_spec,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid,
                              std::vector<std::vector<const VarLabel* > > &PState,
                              std::vector<std::vector<const VarLabel* > > &PState_preReloc);

    virtual void outputProblemSpec(ProblemSpecP& ps){};

    virtual void scheduleInitialize(SchedulerP& sched,
                                    const LevelP& level);

    virtual void scheduleRestartInitialize(SchedulerP& sched,
                                           const LevelP& level);

    virtual void restartInitialize(){};

    virtual void scheduleDoAnalysis(SchedulerP& sched,
                                    const LevelP& level);

    virtual void scheduleDoAnalysis_preReloc(SchedulerP& sched,
                                    const LevelP& level) {};

  private:

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

    void createFile(std::string& filename, FILE*& fp);

    void createDirectory(std::string& lineName, std::string& levelIndex);

    void printHeader( FILE*& fp,
                      const Uintah::TypeDescription::Type myType);

    template< class D, class V >
    void fprintf_Arrays( FILE*& fp,
                         const IntVector& c,
                         const D&  doubleData,
                         const V&  VectorData);

    // general labels
    class lineExtractLabel {
    public:
      VarLabel* lastWriteTimeLabel;
      VarLabel* fileVarsStructLabel;
    };

    lineExtractLabel* ps_lb;

    struct line{
      std::string  name;
      Point   startPt;
      Point   endPt;
      double  stepSize;
      int loopDir;    // direction to loop over
    };

    //__________________________________
    // global constants
    std::vector<VarLabel*> d_varLabels;
    std::vector<int>       d_varMatl;
    std::vector<line*>     d_lines;
    int                    d_col_width = 16;    //  column width

    const Material  * d_matl;
    MaterialSet     * d_matl_set;
    MaterialSubset  * d_zero_matl;
    std::set<std::string> d_isDirCreated;
  };
}

#endif
