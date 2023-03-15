/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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


#ifndef Packages_Uintah_CCA_Components_ontheflyAnalysis_turbulentFluxes_h
#define Packages_Uintah_CCA_Components_ontheflyAnalysis_turbulentFluxes_h
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <vector>

namespace Uintah {


/**************************************

CLASS
   turbulentFluxes

GENERAL INFORMATION

   turbulentFluxes.h

   Todd Harman
   Department of Mechanical Engineering
   University of Utah


KEYWORDS
   turbulentFluxes

DESCRIPTION
   This computes turbulence fluxes



WARNING

****************************************/
  class turbulentFluxes : public AnalysisModule {
  public:
    turbulentFluxes( const ProcessorGroup  * myworld,
                     const MaterialManagerP materialManager,
                     const ProblemSpecP    & module_spec );

    turbulentFluxes();

    virtual ~turbulentFluxes();

    virtual void problemSetup(const ProblemSpecP& prob_spec,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid,
                              std::vector<std::vector<const VarLabel* > > &PState,
                              std::vector<std::vector<const VarLabel* > > &PState_preReloc);

    virtual void outputProblemSpec( ProblemSpecP& ps);

    virtual void scheduleInitialize( SchedulerP  & sched,
                                     const LevelP& level);

    virtual void scheduleRestartInitialize( SchedulerP  & sched,
                                            const LevelP& level);

    virtual void scheduleDoAnalysis( SchedulerP   & sched,
                                     const LevelP & level);

    virtual void scheduleDoAnalysis_preReloc( SchedulerP  & sched,
                                              const LevelP& level) {};

  private:

    //______________________________________________________________________
    //  container to hold
    struct Qvar{

      Qvar(){};

      Qvar( int m ) :matl(m)
      {
        matlSubset = scinew MaterialSubset();
        matlSubset->add( matl );
        matlSubset->addReference();
      };

      int matl;

      //__________________________________
      // labels associated with this variable
      VarLabel * Label              {nullptr};
      VarLabel * Qsum_Label         {nullptr};  // sum_overallTimesteps( Q )
      VarLabel * Q2sum_Label        {nullptr};  // sum_overallTimesteps( Q*Q )
      VarLabel * Qmean_Label        {nullptr};  // Q_sum  / nTimesteps
      VarLabel * Q2mean_Label       {nullptr};  // Q_sum2 / nTimesteps

      VarLabel * Qu_Qv_Qw_sum_Label {nullptr};  // sum_overallTimesteps( Q*u, Q*v, Q*w)
      VarLabel * Qu_Qv_Qw_mean_Label{nullptr};  // sum_Qu_Qv_Qw./nTimesteps

      // variance of Q <double>
      // double  (mean(Q * Q)  - mean(Q)*mean(Q)
      //
      // variance of Q <Vector>
      // Vector( (mean(Q.x * u)  - mean(Q.x)*mean(u)
      //         (mean(Q.y * v)  - mean(Q.y)*mean(v)
      //         (mean(Q.z * w)  - mean(Q.z)*mean(w)
      //
      // covariance of a double Q
      // Vector( (mean(Q * u)  - mean(Q)*mean(u)
      //         (mean(Q * v)  - mean(Q)*mean(v)
      //         (mean(Q * w)  - mean(Q)*mean(w)
      //
      // covariance of Q <Vector>
      // Vector( (mean(Q.x * v)  - mean(Q.x)*mean(v)
      //         (mean(Q.y * w)  - mean(Q.y)*mean(w)
      //         (mean(Q.z * u)  - mean(Q.z)*mean(u)

      VarLabel * variance_Label     {nullptr};
      VarLabel * covariance_Label   {nullptr};

      MaterialSubset * matlSubset   {nullptr};

      bool isInitialized            {false};  // have the summation variables initialized

      const Uintah::TypeDescription* subtype;

      //__________________________________
      // summation timesteps
      int  firstSumTimestep     {-9};          // first timestep
      int  nTimesteps           {-9};          // number of timesteps summation has occured
      bool isStatEnabled        {false};

      //__________________________________
      // utilities
      void print(){
        const std::string name = Label->getName();
        proc0cout << std::setw(15) <<  name <<  " matl: " << matl
                  << " subtype: " << subtype->getName()
                  << " summation started on timestep: " << firstSumTimestep
                  << " nTimesteps: " << nTimesteps
                  << " enabled/diabled: " << isStatEnabled << "\n";
      };

    };

    //__________________________________
    //  For the velocity variable
    struct velocityVar: public Qvar{

      velocityVar(){};

      velocityVar( int m ): Qvar(m){};

      std::string normalTurbStrssName  = "normalTurbStrss";
      std::string shearTurbStrssName   = "shearTurbStrss";
    };

    //__________________________________
    // alias
    using Qvar_ptr        = std::shared_ptr<Qvar>;
    using velocityVar_ptr = std::shared_ptr<velocityVar>;

    //__________________________________
    //  multiplication operators
    inline double multiply_Q_Q(const double a, const double b){
      return a*b ;
    }

    inline Vector multiply_Q_Q(const Vector a, const Vector b){
      return Vector( a.x()*b.x(),
                     a.y()*b.y(),
                     a.z()*b.z() );
    }

    inline Vector multiply_Q_Vel(const Vector a, const Vector b){
      return Vector( a.x()*b.y(),
                     a.y()*b.z(),
                     a.z()*b.x() );
    }

    inline Vector multiply_Q_Vel(const double a, const Vector b){
      return Vector( a*b.x(),
                     a*b.y(),
                     a*b.z() );
    }

    //__________________________________
    //
    void initialize(const ProcessorGroup  *,
                    const PatchSubset     * patches,
                    const MaterialSubset  *,
                    DataWarehouse         *,
                    DataWarehouse         * new_dw );

    void sched_Q_mean(SchedulerP   & sched,
                         const LevelP & level );

    void task_Q_mean( const ProcessorGroup  * ,
                      const PatchSubset    * patches,
                      const MaterialSubset * ,
                      DataWarehouse        * old_dw,
                      DataWarehouse        * new_dw );

    template<class T>
    void Q_mean( DataWarehouse * old_dw,
                 DataWarehouse * new_dw,
                 const Patch   * patch,
                 const Qvar_ptr  Q,
                 constCCVariable<Vector> vel,
                 const int curr_timestep );


    void sched_turbFluxes( SchedulerP   & sched,
                           const LevelP & level );

    void task_turbFluxes( const ProcessorGroup  * ,
                          const PatchSubset    * patches,
                          const MaterialSubset * ,
                          DataWarehouse        * old_dw,
                          DataWarehouse        * new_dw);

    template<class T>
    void turbFluxes( DataWarehouse * new_dw,
                     const Patch   * patch,
                     const Qvar_ptr  Q,
                     constCCVariable<Vector> velMean );

    //__________________________________
    //      Utilities
    VarLabel* createVarLabel( std::string naem,
                              const TypeDescription * td );

    template <class T>
    void allocateAndZero( DataWarehouse * new_dw,
                          const VarLabel* label,
                          const int       matl,
                          const Patch   * patch );
    template <class T>
    void allocateAndZeroAll( DataWarehouse * new_dw,
                             const Patch   * patch,
                             Qvar_ptr        Q);

    void carryForward_means( DataWarehouse    * old_dw,
                             DataWarehouse    * new_dw,
                             const PatchSubset* patches );

    void carryForward_variances( DataWarehouse    * old_dw,
                                 DataWarehouse    * new_dw,
                                 const PatchSubset* patches );

    void set_firstSumTimestep( Qvar_ptr  Q,
                               const int curr_timestep );

    //__________________________________
    // global constants
    IntVector            m_monitorCell;                 // debugging cell to monitor
    std::vector< Qvar_ptr >  m_Qvars;
    Qvar_ptr                 m_velVar;
    MaterialSet            * m_matlSet {nullptr};
    std::vector< std::string >  m_VarLabelNames;
  };
}

#endif
