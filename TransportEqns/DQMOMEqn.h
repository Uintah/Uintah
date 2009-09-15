#ifndef Uintah_Component_Arches_DQMOMEqn_h
#define Uintah_Component_Arches_DQMOMEqn_h

#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

#define YDIM
#define ZDIM
//==========================================================================

/**
* @class DQMOMEqn
* @author Jeremy Thornock
* @date Oct 16, 2008
*
* @brief Transport equation class for a DQMOM scalar (weight or weighted 
*        abscissa)
*
*
*/

namespace Uintah{

//---------------------------------------------------------------------------
// Builder 
class DQMOMEqn; 
class DQMOMEqnBuilder: public DQMOMEqnBuilderBase
{
public:
  DQMOMEqnBuilder( ArchesLabel* fieldLabels, 
                   ExplicitTimeInt* timeIntegrator, 
                   string eqnName );
  ~DQMOMEqnBuilder();

  EqnBase* build(); 
private:

}; 
// End Builder
//---------------------------------------------------------------------------

class ArchesLabel; 
class ExplicitTimeInt; 
class DQMOMEqn: 
public EqnBase{

public: 

  DQMOMEqn( ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName );

  ~DQMOMEqn();

  /** @brief Set any parameters from input file, initialize any constants, etc.. */
  void problemSetup(const ProblemSpecP& inputdb, int qn);
  void problemSetup(const ProblemSpecP& inputdb){};

  
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

  /** @brief Compute all source terms for this scalar eqn */
  void sched_computeSources( const LevelP& level, SchedulerP& sched);

  /** @brief Computes the internal coordinate value by dividing by the weight */
  void sched_getAbscissaValues( const LevelP& level, SchedulerP& sched );
  void getAbscissaValues( const ProcessorGroup* pc, 
                    const PatchSubset* patches, 
                    const MaterialSubset* matls, 
                    DataWarehouse* old_dw, 
                    DataWarehouse* new_dw );



  /** @brief Compute the convective terms */ 
  template <class fT, class oldPhiT> void
  computeConv(const Patch* patch, fT& Fconv, oldPhiT& oldPhi, 
              constSFCXVariable<double>& uVel, constSFCYVariable<double>& vVel, 
              constSFCZVariable<double>& wVel, constCCVariable<Vector>& partVel);

  /** @brief Compute the diffusion terms */
  template <class fT, class oldPhiT, class lambdaT> 
  void computeDiff( const Patch* patch, fT& Fdiff, 
                    oldPhiT& oldPhi, lambdaT& lambda );

  /** @brief Apply boundary conditions */
  template <class phiType> void computeBCs( const Patch* patch, string varName, phiType& phi ){
    d_boundaryCond->setScalarValueBC( 0, patch, phi, varName );
  };

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

  /** @brief Clips the solution variable after the RK time averaging. */
  template<class phiType> void
  clipPhi( const Patch* p, 
                     phiType& phi );


  // ---- ACCESS FUNCTIONS ----
  /** @brief Sets the time integrator. */ 
  inline void setTimeInt( ExplicitTimeInt* timeIntegrator ) {
  d_timeIntegrator = timeIntegrator; 
  }

  /** @brief return the list of models. */
  inline const vector<string> getModelsList(){
    return d_models; };

  /** @brief return the label for the source term */ 
  inline const VarLabel* getSourceLabel(){
    return d_sourceLabel; };

  /** @brief return a bool to tell if this equation is a weight.
   if false, it is understood that this eqn is a weighted 
   abscissa */
  inline bool weight(){
    return d_weight; };

  /** @brief Get the low clipping values */ 
  inline double getLowClip(){
    return d_lowClip; };

  /** @brief Sets this equation as a weight.
   this seems a little dangerous.  Is there a better way? */
  inline void setAsWeight(){
    d_weight = true; }; 

  /** @brief Set the quadrature node value */
  inline void setQuadNode(int node){
    d_quadNode = node; };

  /** @brief Get the quadrature node value */
  inline const int getQuadNode(){
    return d_quadNode; };

  inline const double getScalingConstant(){
    return d_scalingConstant; };
  
  inline const double getInitValue(){
    return d_initValue; }; 

private:

  const VarLabel* d_sourceLabel; //DQMOM Eqns only have ONE source term.  
  const VarLabel* d_icLabel; // This is the label that hold that actual IC value (not weighted IC)
  std::vector<string> d_models;  //This is the list of models for this internal coord. 

  bool d_weight; // if true then this is a weight (as opposed to a weighted abscissa)
  bool d_doClipping; 
  bool d_doLowClip;
  bool d_doHighClip;  

  int d_quadNode; // The quadrature node for this transport eqn. 

  double d_turbPrNo; 
  double d_lowClip; 
  double d_highClip; 
  double d_scalingConstant; 

}; // class DQMOMEqn
} // namespace Uintah

#endif


