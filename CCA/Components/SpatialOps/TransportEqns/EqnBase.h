#ifndef Uintah_Component_SpatialOps_TransportEquationBase_h
#define Uintah_Component_SpatialOps_TransportEquationBase_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>

//========================================================================

/** 
* @class TransportEquationBase
* @author Jeremy Thornock
* @date Oct 16, 2008
*
* @brief A base class for a transport equations.
*
*/

namespace Uintah {
class Fields;
class BoundaryCond; 
class ExplicitTimeInt;  
class EqnBase{

public:

  EqnBase( const Fields* fieldLabels, const VarLabel* transportVarLabel, string eqnName );

  virtual ~EqnBase();

  /** @brief Set any parameters from input file, initialize any constants, etc.. */
  virtual void problemSetup(const ProblemSpecP& inputdb) = 0;
  virtual void problemSetup(const ProblemSpecP& inputdb, int qn) = 0;

  /** @brief Creates instances of variables in the new_dw at the begining of the timestep 
             and copies old data into the new variable */
  virtual void sched_initializeVariables( const LevelP&, SchedulerP& sched ) = 0;
  
  /** @brief Schedule a transport equation to be built and solved */
  virtual void sched_evalTransportEqn( const LevelP&, 
                                       SchedulerP& sched, int timeSubStep ) = 0; 

  /** @brief Build the terms needed in the transport equation */
  virtual void sched_buildTransportEqn( const LevelP&, SchedulerP& sched ) = 0;

  /** @brief Solve the transport equation */
  virtual void sched_solveTransportEqn( const LevelP&, SchedulerP& sched ) = 0;

  /** @brief Compute the convective terms */ 
  template <class fT, class oldPhiT>  
  void computeConv( const Patch* patch, fT& Fdiff, 
                         oldPhiT& oldPhi );

  /** @brief Compute the diffusion terms */
  template <class fT, class oldPhiT, class lambdaT> 
  void computeDiff( const Patch* patch, fT& Fdiff, 
                    oldPhiT& oldPhi, lambdaT& lambda );

  /** @brief Method for cleaning up after a transport equation at the end of a timestep */
  virtual void sched_cleanUp( const LevelP&, SchedulerP& sched ) = 0; 

  /** @brief Apply boundary conditions */
  // probably want to make this is a template
  template <class phiType> void computeBCs( const Patch* patch, string varName, phiType& phi );

  // Access functions:
  inline void setBoundaryCond( BoundaryCond* boundaryCond ) {
  d_boundaryCond = boundaryCond; 
  }
  inline void setTimeInt( ExplicitTimeInt* timeIntegrator ) {
  d_timeIntegrator = timeIntegrator; 
  }
  inline const VarLabel* getTransportEqnLabel(){
    return d_transportVarLabel; };
  inline const std::string getEqnName(){
    return d_eqnName; };

//  /** @brief return a bool to tell if this equation is a weight.
//   if false, it is understood that this eqn is a weighted 
//   abscissa */
//  inline bool weight(){
//    return d_weight; };

//  /** @brief Sets this equation as a weight.
//   this seems a little dangerous.  Is there a better way? */
//  inline void setAsWeight(){
//    d_weight = true; }; 

//  /** @brief Set the quadrature node value */
//  inline void setQuadNode(int node){
//    d_quadNode = node; };
 

protected:

  template<class T> 
  struct FaceData {
    // 0 = e, 1=w, 2=n, 3=s, 4=t, 5=b
    //vector<T> values_[6];
    T p; 
    T e; 
    T w; 
    T n; 
    T s;
    T t;
    T b;
  };

  const Fields* d_fieldLabels;
  const VarLabel* d_transportVarLabel;
  std::string d_eqnName;  
  bool d_doConv, d_doDiff, d_addSources;

  const VarLabel* d_FdiffLabel;
  const VarLabel* d_FconvLabel; 
  const VarLabel* d_RHSLabel;

  int d_timeSubStep; 

  BoundaryCond* d_boundaryCond;
  ExplicitTimeInt* d_timeIntegrator; 

//  bool d_weight; // if true then this is a weight (as opposed to a weighted abscissa)

//  int d_quadNode; // The quadrature node for this transport eqn. 


private:



}; // end EqnBase
} // end namespace Uintah

#endif
