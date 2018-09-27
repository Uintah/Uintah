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


#ifndef Packages_Uintah_CCA_Components_ontheflyAnalysis_planeAverage_h
#define Packages_Uintah_CCA_Components_ontheflyAnalysis_planeAverage_h
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Output.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/GridIterator.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>

#include <map>
#include <vector>

namespace Uintah {


/*______________________________________________________________________

GENERAL INFORMATION

   planeAverage.h

   This computes the spatial average over a plane in the domain

   Todd Harman
   Department of Mechanical Engineering
   University of Utah
______________________________________________________________________*/

  class planeAverage : public AnalysisModule {
  public:

    planeAverage(const ProcessorGroup   * myworld,
                 const MaterialManagerP   materialManager,
                 const ProblemSpecP     & module_spec);
    planeAverage();

    virtual ~planeAverage();

    virtual void problemSetup(const ProblemSpecP  & prob_spec,
                              const ProblemSpecP  & restart_prob_spec,
                              GridP               & grid,
                              std::vector<std::vector<const VarLabel* > > &PState,
                              std::vector<std::vector<const VarLabel* > > &PState_preReloc);

    virtual void outputProblemSpec(ProblemSpecP& ps){};

    virtual void scheduleInitialize(SchedulerP   & sched,
                                    const LevelP & level);

    virtual void scheduleRestartInitialize(SchedulerP   & sched,
                                           const LevelP & level);

    virtual void restartInitialize();

    virtual void scheduleDoAnalysis(SchedulerP   & sched,
                                    const LevelP & level);

    virtual void scheduleDoAnalysis_preReloc(SchedulerP   & sched,
                                             const LevelP & level) {};

  private:

    //__________________________________
    //  This is a wrapper to create a vector of objects of different types(aveVar)
    struct aveVarBase{
      public:
        VarLabel* label;
        int matl;
        int level;
        TypeDescription::Type baseType;
        TypeDescription::Type subType;
        
        virtual void reserve( const int ) = 0;
        
        // virtual templated functions are not allowed in C++11
        // instantiate the various types
        virtual  void getPlaneAve( std::vector<double>& ave ){};
        virtual  void getPlaneAve( std::vector<Vector>& ave ){};
        virtual  void setPlaneAve( std::vector<double>& ave ){};
        virtual  void setPlaneAve( std::vector<Vector>& ave ){}
    };

    //  It's simple and straight forward to use a double and vector class
    //  A templated class would be ideal but that involves too much C++ magic
    //  
    //__________________________________
    //  Class that holds the planar averages     double
    class aveVar_double: public aveVarBase{

      private:
        std::vector<double> sum;
        std::vector<double> weight;
        std::vector<double> ave;
        
      public:
        void reserve( const int n )
        {
          sum.reserve(n);
          weight.reserve(n);
          ave.reserve(n);
        }
        
        void getPlaneAve( std::vector<double>& me )  { me = ave; }
        void setPlaneAve( std::vector<double>& me )  { ave = me; } 

        ~aveVar_double(){}
    };
    
    //__________________________________
    //  Class that holds the planar averages      Vector
    class aveVar_Vector: public aveVarBase{

      private:
        std::vector<Vector> sum;
        std::vector<Vector> weight;
        std::vector<Vector> ave;

      public:
        void reserve( const int n )
        {
          sum.reserve(n);
          weight.reserve(n);
          ave.reserve(n);
        }
        
        void getPlaneAve( std::vector<Vector>& me )  { me = ave; }
        void setPlaneAve( std::vector<Vector>& me )  { ave = me; } 
        ~aveVar_Vector(){}
    };

    // Each element of the vector contains
    // 
    std::vector< std::shared_ptr< aveVarBase > > d_aveVars;

    
    //______________________________________________________________________
    //
    //
    bool isRightLevel( const int myLevel,
                       const int L_indx,
                       const LevelP& level);

    void initialize(const ProcessorGroup *,
                    const PatchSubset    * patches,
                    const MaterialSubset *,
                    DataWarehouse        *,
                    DataWarehouse        * new_dw);
                    
    void restartInitialize(const ProcessorGroup *,
                           const PatchSubset    * patches,
                           const MaterialSubset *,
                           DataWarehouse        *,
                           DataWarehouse        * new_dw);

    void computeAverage(const ProcessorGroup * pg,
                        const PatchSubset    * patches,
                        const MaterialSubset *,
                        DataWarehouse        * old_dw,
                        DataWarehouse        * new_dw);

    void doAnalysis(const ProcessorGroup  * pg,
                    const PatchSubset     * patches,
                    const MaterialSubset  *,
                    DataWarehouse         *,
                    DataWarehouse         * new_dw);

    void createFile(std::string & filename,
                    FILE*       & fp,
                    std::string & levelIndex);

    int createDirectory( mode_t mode,
                         const std::string & rootPath,
                         std::string       & path );

    template <class Tvar, class Ttype>
    void findAverage( DataWarehouse  * new_dw,
                      std::shared_ptr< aveVarBase > analyzeVars,
                      const Patch    * patch,
                      GridIterator     iter );


    void planeIterator( const GridIterator& patchIter,
                        IntVector & lo,
                        IntVector & hi );

    // general labels
    class planeAverageLabel {
    public:
      VarLabel* lastCompTimeLabel;
      VarLabel* fileVarsStructLabel;
    };

    planeAverageLabel* d_lb;

    //__________________________________
    // global constants always begin with "d_"
    double d_writeFreq;
    double d_startTime;
    double d_stopTime;

    const Material*  d_matl;
    MaterialSet*     d_matl_set;
    std::set<std::string> d_isDirCreated;
    MaterialSubset*  d_zero_matl;
    PatchSet*        d_zeroPatch;

    enum orientation { XY, XZ, YZ };        // plane orientation
    orientation d_planeOrientation;
  };
}

#endif
