#ifndef Uintah_Component_Arches_CQMOMEqn_h
#define Uintah_Component_Arches_CQMOMEqn_h
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/CQMOMEqnFactory.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <CCA/Components/Arches/Directives.h>

//==========================================================================

/**
 * @class CQMOMEqn
 * @author Alex Abboud, adapted from DQMOMEqn
 * @date May 2014
 *
 * @brief Transport equation class for a CQMOM moment
 *
 *
 */

namespace Uintah{
  
  //---------------------------------------------------------------------------
  // Builder
  class CQMOMEqn;
  class CQMOMEqnBuilder: public CQMOMEqnBuilderBase
  {
  public:
    CQMOMEqnBuilder( ArchesLabel* fieldLabels,
                    ExplicitTimeInt* timeIntegrator,
                    std::string eqnName );
    ~CQMOMEqnBuilder();
    
    EqnBase* build();
  private:
    
  };
  // End Builder
  //---------------------------------------------------------------------------
  
  class ArchesLabel;
  class ExplicitTimeInt;
  class CQMOMEqn:
  public EqnBase{
    
  public:
    
    CQMOMEqn( ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, std::string eqnName );
    
    ~CQMOMEqn();
    
    /** @brief Set any parameters from input file, initialize any constants, etc.. */
    void problemSetup(const ProblemSpecP& inputdb, int qn){};
    void problemSetup(const ProblemSpecP& inputdb);
    
    /** @brief not needed here. **/
    void assign_stage_to_sources(){};
    
    /** @brief Schedule a transport equation to be built and solved */
    void sched_evalTransportEqn( const LevelP&,
                                SchedulerP& sched, int timeSubStep );
    
    /** @brief Schedule the build for the terms needed in the transport equation */
    void sched_buildTransportEqn( const LevelP& level,
                                 SchedulerP& sched, int timeSubStep );
    /** @brief Actually build the transport equation */
    void buildTransportEqn(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset*,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw);
    
    /** @brief Schedule the solution the transport equation */
    void sched_solveTransportEqn(const LevelP& level,
                                 SchedulerP& sched, int timeSubStep );
    /** @brief Solve the transport equation */
    void solveTransportEqn(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset*,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw,
                           int timeSubStep);
    
    /** @brief Schedule the initialization of the variables */
    void sched_initializeVariables( const LevelP& level, SchedulerP& sched );
    
    /** @brief Actually initialize the variables at the begining of a time step */
    void initializeVariables( const ProcessorGroup* pc,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw );
    
    /** @brief schedule compute all source terms for this scalar eqn */
    void sched_computeSources( const LevelP& level, SchedulerP& schedi, int timeSubStep );
    
    /** @schedule compute all source terms for this scalar eqn */
    void computeSources( const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw );
    
    /** @brief Apply boundary conditions */
    template <class phiType> void computeBCs( const Patch* patch, std::string varName, phiType& phi ){
      d_boundaryCond->setScalarValueBC( 0, patch, phi, varName );
    };
    
    /** @brief Schedule the cleanup after this equation. */
    void sched_cleanUp( const LevelP&, SchedulerP& sched );
    
    /** @brief Time averaging method required by base class. This method is empty (not needed) at the moment */
    void sched_timeAve( const LevelP& level, SchedulerP& sched, int timeSubStep ){};
    
    /** @brief advanced clipping method required by base class. This method is empty (not needed) at the moment */
    void sched_advClipping( const LevelP& level, SchedulerP& sched, int timeSubStep );
    
    
    // --------------------------------------
    // Access functions:
    
    /** @brief Set the time integrator. */
    inline void setTimeInt( ExplicitTimeInt* timeIntegrator ) { d_timeIntegrator = timeIntegrator;}
    
    /** @brief Return the list of models associated with this equation. */
    inline const std::vector<std::string> getModelsList(){ return d_models; };
    
    /** @brief Return the VarLabel for this equation's source term. */
    inline const VarLabel* getSourceLabel(){ return d_sourceLabel; };
    
    /** @brief Return the VarLabel for this equation */
    inline const VarLabel* getMomentLabel(){ return d_momentLabel; };
    
    /** @brief return the moment index vector of this equation */
    inline const std::vector<int> getMomentIndex(){ return momentIndex; };
    
    /** @brief Get the small clipping value (for weights only). */
    inline double getSmallClip(){
      if( clip.activated && clip.do_low ) {
        double small = clip.low + clip.tol;
        return small;
      } else {
        return 0.0;
      }
    };

    
  private:
    
    const VarLabel* d_sourceLabel;       // create one summed source label and use in build
    std::vector<int> momentIndex;        //moment index for this transport equation, needed for convective and soruce closure

    const VarLabel* d_momentLabel;       //Label for the moment of this transport equation
    int M;                               //number of internal coordiantes
        
    std::vector<std::string> d_models;   ///< This is the list of models for this internal coordinate
    std::vector<std::string> d_sources;
    bool d_addExtraSources; 
    double d_w_small;               ///< Value of "small" weights
    bool d_normalized;
    bool d_usePartVel;             //determine whether to use particle velocities, or fluid velocities for convection
    
  }; // class CQMOMEqn
} // namespace Uintah

#endif
