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


#ifndef CCA_Components_ontheflyAnalysis_meanTurbFluxes_h
#define CCA_Components_ontheflyAnalysis_meanTurbFluxes_h

#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Components/OnTheFlyAnalysis/planeAverage.h>

namespace Uintah {

/*______________________________________________________________________

GENERAL INFORMATION

   meanTurbFluxes.h

   This module computes the mean turbulent fluxes on each plane in the domain
   {u'u'}^bar, {v'v'}^bar, {w'w'}^bar {u'v'}^bar, {v'w'}^bar {u'w'}^bar

   foreach Q ( T, P, scalar )
     ( {u'Q'}^bar(y), {v'Q'}^bar(y), {w'Q'}^bar(y) )
   end

   Todd Harman
   Department of Mechanical Engineering
   University of Utah
______________________________________________________________________*/



//______________________________________________________________________

  class meanTurbFluxes :  public AnalysisModule{
  public:

    meanTurbFluxes(const ProcessorGroup   * myworld,
                   const MaterialManagerP   materialManager,
                   const ProblemSpecP     & module_spec);

    meanTurbFluxes();

    virtual ~meanTurbFluxes();

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

    virtual void restartInitialize(){};

    virtual void scheduleDoAnalysis(SchedulerP   & sched,
                                    const LevelP & level);

    virtual void scheduleDoAnalysis_preReloc(SchedulerP   & sched,
                                             const LevelP & level) {};

  private:

    // remove this once C++14 is adopted
    template<typename T, typename ...Args>
    std::unique_ptr<T> make_unique( Args&& ...args );


    //______________________________________________________________________
    //          STRUCTS
    //  All variables except velocity
    struct Qvar{

      Qvar(){};

      Qvar( int m ) :matl(m)
      {
        matSubSet = scinew MaterialSubset();
        matSubSet->add( matl );
        matSubSet->addReference();
      };

      int matl;
      VarLabel * label          {nullptr};    // Q
      VarLabel * primeLabel     {nullptr};    // Q'
      VarLabel * turbFluxLabel  {nullptr};    // u'Q', v'Q', w'Q'
      MaterialSubset * matSubSet {nullptr};

      ~Qvar()
      {
        if(matSubSet && matSubSet->removeReference()){
          delete matSubSet;
        }
        VarLabel::destroy( primeLabel );
        VarLabel::destroy( turbFluxLabel );
      }
    };

    //__________________________________
    //  Velocity
    struct velocityVar: public Qvar{
      VarLabel * normalTurbStrssLabel {nullptr};  // u'u', v'v', w'w'
      VarLabel * shearTurbStrssLabel  {nullptr};  // u'v', v'w', w'u'

      ~velocityVar()
      {
        VarLabel::destroy( normalTurbStrssLabel );
        VarLabel::destroy( shearTurbStrssLabel );
      }
      std::string normalTurbStrssName  = "normalTurbStrss";
      std::string shearTurbStrssName   = "shearTurbStrss";
    };


    //______________________________________________________________________
    //          TASKS AND FUNCTIONS

    void sched_populateVerifyLabels( SchedulerP   & sched,
                                     const LevelP & level );

    void populateVerifyLabels(const ProcessorGroup * ,
                              const PatchSubset    * patches,         
                              const MaterialSubset * ,                
                              DataWarehouse        * ,          
                              DataWarehouse        * new_dw);
                              
    int findFilePositionOffset( const PatchSubset  * patches, 
                                const int nPlaneCellPerPatch,
                                const IntVector      pLo,
                                const IntVector      pHi);

    void sched_TurbFluctuations(SchedulerP   & sched,
                                const LevelP & level);

    void calc_TurbFluctuations(const ProcessorGroup  * ,
                               const PatchSubset    * patches,
                               const MaterialSubset * ,
                               DataWarehouse        * ,
                               DataWarehouse        * new_dw);

    template <class T>
    void calc_Q_prime( DataWarehouse       * new_dw,
                       const Patch         * patch,
                       std::shared_ptr<Qvar> Q );

    void sched_TurbFluxes(SchedulerP   & sched,
                          const LevelP & level);

    void calc_TurbFluxes(const ProcessorGroup * ,
                         const PatchSubset    * patches,
                         const MaterialSubset * ,
                         DataWarehouse        * ,
                         DataWarehouse        * new_dw);

    //______________________________________________________________________
    //          VARIABLES

    //__________________________________
    // global constants begin with "d_"
    std::vector< std::shared_ptr< Qvar > >  d_Qvars;
    std::shared_ptr< velocityVar >          d_velocityVar;

    MaterialSet*  d_matl_set;
    
    VarLabel* d_lastCompTimeLabel {nullptr};
    VarLabel* d_verifyScalarLabel {nullptr};  // labels for verification
    VarLabel* d_verifyVectorLabel {nullptr};

    planeAverage * d_planeAve_1;
    planeAverage * d_planeAve_2;
    
    IntVector d_monitorCell;             // Monitor this cells.  Used for debugging
    bool d_doVerification { false };
  };
}

#endif
