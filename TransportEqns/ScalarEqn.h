#ifndef Uintah_Component_Arches_ScalarEqn_h
#define Uintah_Component_Arches_ScalarEqn_h

#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>

#define YDIM
#define ZDIM
//==========================================================================

/**
* @class ScalarEqn
* @author Jeremy Thornock
* @date Oct 16, 2008
*
* @brief Transport equation class for a CCVariable scalar
*
*
*/

namespace Uintah{

//---------------------------------------------------------------------------
// Builder 
class ScalarEqn; 
class CCScalarEqnBuilder: public EqnBuilder
{
public:
  CCScalarEqnBuilder( ArchesLabel* fieldLabels, 
                      ExplicitTimeInt* timeIntegrator, 
                      string eqnName );
  ~CCScalarEqnBuilder();

  EqnBase* build(); 
private:

}; 
// End Builder
//---------------------------------------------------------------------------

class ArchesLabel; 
class BoundaryCond; 
class ExplicitTimeInt; 
class SourceTerm; 
class ScalarEqn: 
public EqnBase{

public: 

  ScalarEqn( ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName );

  ~ScalarEqn();

  /** @brief Set any parameters from input file, initialize any constants, etc.. */
  void problemSetup(const ProblemSpecP& inputdb);
  void problemSetup(const ProblemSpecP& inputdb, int qn){};
  
  /** @brief Schedule a transport equation to be built and solved */
  void sched_evalTransportEqn( const LevelP&, 
                               SchedulerP& sched, int timeSubStep);

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

  /** @brief Compute all source terms for this scalar eqn */
  void sched_computeSources( const LevelP& level, SchedulerP& sched, int timeSubStep);

  /** @brief Compute the convective terms */ 
  template <class fT, class oldPhiT> void
  computeConv(const Patch* patch, fT& Fconv, oldPhiT& oldPhi, 
              constSFCXVariable<double>& uVel, constSFCYVariable<double>& vVel, 
              constSFCZVariable<double>& wVel);

  /** @brief Compute the diffusion terms */
  template <class fT, class oldPhiT, class lambdaT> 
  void computeDiff( const Patch* patch, fT& Fdiff, 
                    oldPhiT& oldPhi, lambdaT& lambda );

  /** @brief Apply boundary conditions */
  template <class phiType> void computeBCs( const Patch* patch, string varName, phiType& phi );

  /** @brief Schedule the cleanup after this equation. */ 
  void sched_cleanUp( const LevelP&, SchedulerP& sched ); 
  /** @brief Actually clean up after the equation. This just reinitializes 
             source term booleans so that the code can determine if the source
             term label should be allocated or just retrieved from the data 
             warehouse. */ 
  void cleanUp( const ProcessorGroup* pc, 
                const PatchSubset* patches, 
                const MaterialSubset* matls, 
                DataWarehouse* old_dw, 
                DataWarehouse* new_dw  ); 

  // Access functions:
  /** @brief Sets the boundary condition object. */ 
  inline void setBoundaryCond( BoundaryCond* boundaryCond ) {
  d_boundaryCond = boundaryCond; 
  }
  /** @brief Sets the time integrator. */ 
  inline void setTimeInt( ExplicitTimeInt* timeIntegrator ) {
  d_timeIntegrator = timeIntegrator; 
  }

  struct FaceValues {
    double e; 
    double w; 
    double n; 
    double s; 
    double t; 
    double b; 
    double p; 
  };  

  /** @brief Interpolate a point to face values for the respective cv. */
  template <class phiT, class interpT> void
  interpPtoF( phiT& phi, const IntVector c, interpT& F ); 
  /** @brief Take a gradient of a variable to result in a face value for a respective cv. */
  template <class phiT, class gradT> void
  gradPtoF( phiT& phi, const IntVector c, const Patch* p, gradT& G ); 

  inline const vector<string> getSourcesList(){
    return d_sources; };

private:

  BoundaryCond* d_boundaryCond;
  vector<std::string> d_sources;

  double d_turbPrNo;

}; // class ScalarEqn
} // namespace Uintah

#endif


