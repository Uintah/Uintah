#ifndef Uintah_Component_Arches_TransportEquationBase_h
#define Uintah_Component_Arches_TransportEquationBase_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <CCA/Components/Arches/ExplicitTimeInt.h>
#include <CCA/Components/Arches/TransportEqns/Discretization_new.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <Core/Parallel/Parallel.h>

#define YDIM
#define ZDIM

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
class ArchesLabel; 
class BoundaryCondition_new;
class Discretization_new; 
class ExplicitTimeInt;  
class EqnBase{

public:

  EqnBase( ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName );

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
  virtual void sched_buildTransportEqn( const LevelP&, SchedulerP& sched, int timeSubStep ) = 0;

  /** @brief Solve the transport equation */
  virtual void sched_solveTransportEqn( const LevelP&, SchedulerP& sched, int timeSubStep ) = 0;

  /** @brief Dummy init for MPMArches */ 
  virtual void sched_dummyInit( const LevelP&, SchedulerP& sched ) = 0; 

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

  /** @brief Set the initial value of the transported variable to some function */
  template <class phiType> void initializationFunction( const Patch* patch, phiType& phi ); 

  // Access functions:
  /** @brief Set the boundary condition object associated with this transport equation object */
  inline void setBoundaryCond( BoundaryCondition_new* boundaryCond ) {
    d_boundaryCond = boundaryCond; 
  }
  
  /** @brief Set the time integrator object associated with this transport equation object */
  inline void setTimeInt( ExplicitTimeInt* timeIntegrator ) {
    d_timeIntegrator = timeIntegrator; 
  }
  
  /** @brief Return VarLabel for the scalar transported by this equation object, pointing to NEW data warehouse */
  inline const VarLabel* getTransportEqnLabel(){
    return d_transportVarLabel; };
  
  /** @brief Return VarLabel for the scalar transported by this equation object, pointing to OLD data warehouse */
  inline const VarLabel* getoldTransportEqnLabel(){
    return d_oldtransportVarLabel; };
  
  /** @brief Return a string containing the human-readable label for this equation object */
  inline const std::string getEqnName(){
    return d_eqnName; };
  
  /** @brief Return a string containing the name of the initialization function being used (e.g. "constant") */ 
  inline const string getInitFcn(){
    return d_initFunction; }; 

  /** @brief Return the scaling constant for the given equation. */
  inline const double getScalingConstant(){
    return d_scalingConstant; };

  /** @brief Compute the boundary conditions for this transport equation object */
  template<class phiType> void
  computeBCsSpecial( const Patch* patch, 
                       string varName,
                       phiType& phi )
  {
    d_boundaryCond->setScalarValueBC( 0, patch, phi, varName ); 
  }

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

  ArchesLabel* d_fieldLabels; 
  const VarLabel* d_transportVarLabel;    ///< Label for scalar being transported, in NEW data warehouse
  const VarLabel* d_oldtransportVarLabel; ///< Label for scalar being transported, in OLD data warehouse
  std::string d_eqnName;                  ///< Human-readable label for this equation
  bool d_doConv;                          ///< Boolean: do convection for this equation object?
  bool d_doDiff;                          ///< Boolean: do diffusion for this equation object?
  bool d_addSources;                      ///< Boolean: add a right-hand side (i.e. convection, diffusion, source terms) to this equation object?

  const VarLabel* d_FdiffLabel;           ///< Label for diffusion term of this equation object
  const VarLabel* d_FconvLabel;           ///< Label for convection term of this equation object
  const VarLabel* d_RHSLabel;             ///< Label for RHS of this equation object

  std::string d_convScheme;               ///< Convection scheme (superbee, upwind, etc.)

  BoundaryCondition_new* d_boundaryCond;  ///< Boundary condition object associated with equation object
  ExplicitTimeInt* d_timeIntegrator;      ///< Time integrator object associated with equation object
  Discretization_new* d_disc;             ///< Discretization object associated with equation object

  std::string d_initFunction;             ///< A functional form for initial value.

  // constant initialization function:
  double d_constant_init;           ///< constant value for initialization

  // step initialization function:
  std::string d_step_dir;           ///< For a step initialization function, direction in which step should occur
  bool b_stepUsesPhysicalLocation;  ///< Boolean: is step function's physical location specified?
  bool b_stepUsesCellLocation;      ///< Boolean: is step function's cell location specified?
  double d_step_start;              ///< Physical location of step function start
  double d_step_end;                ///< Physical location of step function end
  double d_step_cellstart;          ///< Cell location of step function start
  double d_step_cellend;            ///< Cell location of step function end
  double d_step_value;              ///< Step function steps from 0 to d_step_value

  // Clipping:
  bool d_doClipping;                ///< Boolean: are values clipped?
  bool d_doLowClip;                 ///< Boolean: are low values clipped?
  bool d_doHighClip;                ///< Boolean: are high values clipped?

  double d_lowClip;                 ///< Value of low clipping
  double d_smallClip;               ///< Value of small clipping (used if scalar is a divisor, e.g. with DQMOM weights)
  double d_highClip;                ///< Value of high clipping

  // Other:
  double d_turbPrNo;                ///< Turbulent Prandtl number (used for scalar diffusion)
  double d_scalingConstant;         ///< Value by which to scale values 

private:


}; // end EqnBase


//---------------------------------------------------------------------------
// Method: Phi initialization using a function 
//---------------------------------------------------------------------------
template <class phiType>  
void EqnBase::initializationFunction( const Patch* patch, phiType& phi ) 
{
  proc0cout << "initializing scalar equation " << d_eqnName << endl;
  if( d_initFunction == "constant" || d_initFunction == "env_constant" ) {
    // don't do anything
  } else if( d_initFunction == "step" ) {

    if( d_step_dir == "y" ) {
#ifndef YDIM
      proc0cout << "WARNING: YDIM not turned on (compiled) with this version of the code, " << endl;
      proc0cout << "but you specified a step function that steps in the y-direction. " << endl;
      proc0cout << "To get this to work, made sure YDIM is defined in ScalarEqn.h" << endl;
      proc0cout << "Cannot initialize your scalar in y-dim with step function" << endl;
      proc0cout << endl;
#endif
      // otherwise do nothing

    } else if( d_step_dir == "z" ) {
#ifndef ZDIM
      proc0cout << "WARNING: ZDIM not turned on (compiled) with this version of the code, " << endl;
      proc0cout << "but you specified a step function that steps in the z-direction. " << endl;
      proc0cout << "To get this to work, made sure ZDIM is defined in ScalarEqn.h" << endl;
      proc0cout << "Cannot initialize your scalar in y-dim with step function" << endl;
      proc0cout << endl;
#endif
      // otherwise do nothing
    }
  } else {
    proc0cout << "WARNING: Your initialization function wasn't found." << endl;
  }

  for (CellIterator iter=patch->getCellIterator__New(0); !iter.done(); iter++){
    IntVector c = *iter; 
    Vector Dx = patch->dCell(); 

    double x,y,z; 
    double cellx, celly, cellz;

    if( b_stepUsesCellLocation ) {
      cellx = c[0];
#ifdef YDIM
      celly = c[1];
#endif
#ifdef ZDIM
      cellz = c[2];
#endif
    } else if ( b_stepUsesPhysicalLocation ) {
      x = c[0]*Dx.x() + Dx.x()/2.; // the +Dx/2 is because variable is cell-centered
#ifdef YDIM
      y = c[1]*Dx.y() + Dx.y()/2.; 
#endif
#ifdef ZDIM
      z = c[2]*Dx.z() + Dx.z()/2.;
#endif 
    }

    if ( d_initFunction == "constant" || d_initFunction == "env_constant" ) {
      // ========== CONSTANT VALUE INITIALIZATION ============
      phi[c] = d_constant_init/d_scalingConstant; 

    } else if (d_initFunction == "step" || d_initFunction == "env_step" ) {
      // =========== STEP FUNCTION INITIALIZATION =============
      
      if (d_step_dir == "x") {
        if (  (b_stepUsesPhysicalLocation && x > d_step_start && x < d_step_end)
           || (b_stepUsesCellLocation && cellx > d_step_cellstart && x < d_step_cellend) ) {
          phi[c] = d_step_value/d_scalingConstant; 
        } else {
          phi[c] = 0.0;
        }
      
      } else if (d_step_dir == "y") {
#ifdef YDIM
        if (  (b_stepUsesPhysicalLocation && y > d_step_start && y < d_step_end)
           || (b_stepUsesCellLocation && celly > d_step_cellstart && celly < d_step_cellend) ) {
          phi[c] = d_step_value/d_scalingConstant; 
        } else { 
          phi[c] = 0.0;
        }
#else
        phi[c] = 0.0;
#endif

      } else if (d_step_dir == "z") {
#ifdef ZDIM
        if (  (b_stepUsesPhysicalLocation && z > d_step_start && z < d_step_end)
           || (b_stepUsesCellLocation && cellz > d_step_cellstart && z < d_step_cellend) ) {
          phi[c] = d_step_value/d_scalingConstant; 
        } else {
          phi[c] = 0.0;
        }
#else
        phi[c] = 0.0;
#endif
      }
    }//end d_initFunction types

    // ======= add other initialization functions here ======

  }
}

} // end namespace Uintah

#endif
