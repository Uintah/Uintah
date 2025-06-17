/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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
#include <Core/Grid/Variables/NCVariable.h>
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

    void isCommonBaseVarType();

    std::vector<std::string>desc( FILE*& fp );

    Point findCellOffset( const Level* level );

    void createFile( const std::string& filename,
                     FILE*& fp,
                     const Level* level );

    void printHeader(FILE*& fp,
                     const Vector dx );

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

    lineExtractLabel* m_lb;

    struct line{
      std::string  name;
      Point   startPt;
      Point   endPt;
      double  stepSize;
      int loopDir;    // direction to loop over
    };

    struct varProperty{
      const VarLabel* varLabel {nullptr};
      const std::string name {};
      const int matl;
      const TypeDescription* td;
      const TypeDescription::Type baseType;
      const TypeDescription::Type subType;
    };

    enum allVarLocations { CC, NC, SFCX, SFCY, SFCZ, mixed, undefined };             // where are the variables

    //__________________________________
    // global constants
    std::vector<line*>     m_liness;
    int                    m_col_width = 16;    //  column width
    TypeDescription::Type  m_allVarsBaseType {TypeDescription::Other};

    std::vector<varProperty> m_varProperties;   //

    const Material  * m_matl      {nullptr};
    MaterialSet     * m_matl_set  {nullptr};
  };
}

#endif
