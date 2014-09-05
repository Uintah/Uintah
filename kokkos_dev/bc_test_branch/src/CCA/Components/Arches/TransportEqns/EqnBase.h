#ifndef Uintah_Component_Arches_TransportEquationBase_h
#define Uintah_Component_Arches_TransportEquationBase_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Grid/Box.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <CCA/Components/Arches/ExplicitTimeInt.h>
#include <CCA/Components/Arches/TransportEqns/Discretization_new.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/IntrusionBC.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <CCA/Components/Arches/Directives.h>

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
class MixingRxnModel; 
class EqnBase{

public:

  EqnBase( ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName );

  virtual ~EqnBase();

  /** @brief Set any parameters from input file, initialize any constants, etc.. */
  virtual void problemSetup(const ProblemSpecP& inputdb) = 0;
  virtual void problemSetup(const ProblemSpecP& inputdb, int qn) = 0;

  /** @brief Setup any extra information that may need to occur later (like after the table is setup) **/ 
  void extraProblemSetup( ProblemSpecP& db ); 

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

  /** @brief Time averaging */ 
  virtual void sched_timeAve( const LevelP&, SchedulerP& sched, int timeSubStep ) = 0; 

  /** @brief Checks that boundary conditions for this variable are set for every 
   * face for every child */ 
  void sched_checkBCs( const LevelP&, SchedulerP& sched ); 
  void checkBCs( const ProcessorGroup* pc, 
                 const PatchSubset* patches, 
                 const MaterialSubset* matls, 
                 DataWarehouse* old_dw, 
                 DataWarehouse* new_dw ); 

  /** @brief Compute the convective terms */ 
  template <class fT, class oldPhiT>  
  void computeConv( const Patch* patch, fT& Fdiff, 
                         oldPhiT& oldPhi );

  /** @brief Method for cleaning up after a transport equation at the end of a timestep */
  virtual void sched_cleanUp( const LevelP&, SchedulerP& sched ) = 0; 

  /** @brief Apply boundary conditions */
  // probably want to make this is a template
  template <class phiType> void computeBCs( const Patch* patch, string varName, phiType& phi );

  /** @brief Set the initial value of the transported variable to some function */
  template <class phiType> void initializationFunction( const Patch* patch, phiType& phi, constCCVariable<double>& eps_v ); 
  
  /** @brief Set the initial value of the DQMOM transported variable to some function */
  template <class phiType, class constPhiType>  
  void initializationFunction( const Patch* patch, phiType& phi, constPhiType& weight, constCCVariable<double>& eps_v  );

  /** @brief Initializes the scalar to a value from the table as a function of the dependent variables **/
  void sched_tableInitialization( const LevelP&, SchedulerP& sched );
  void tableInitialization(const ProcessorGroup* pc, 
                           const PatchSubset* patches, 
                           const MaterialSubset* matls, 
                           DataWarehouse* old_dw, 
                           DataWarehouse* new_dw );

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
  inline double getScalingConstant(){
    return d_scalingConstant; };

  /** @brief Return the density guess boolean */
  inline bool getDensityGuessBool(){
    return d_use_density_guess; 
  };

  /** @brief Set the density guess bool **/ 
  inline void setDensityGuessBool( bool set_point ){ 
    d_use_density_guess = set_point; 
  };

  /** @brief Return a list of all sources associated with this transport equation */ 
  inline const vector<string> getSourcesList(){
    return d_sources; };

  /** @brief Compute the boundary conditions for this transport equation object */
  template<class phiType> void
  computeBCsSpecial( const Patch   * patch, 
                     const string  & varName,
                           phiType & phi )
  {
    d_boundaryCond->setScalarValueBC( 0, patch, phi, varName ); 
  };

  /** @brief Set the intrusion machinery **/ 
  inline void set_intrusion( IntrusionBC* intrusions ){ 
    _intrusions = intrusions; 
  };  

  /** @brief Set boolean for new intrusions **/ 
  inline void set_intrusion_bool( bool using_new_intrusions ){ 
    _using_new_intrusion = using_new_intrusions; 
  };

  /** @brief Set a reference to the mix/rxn table **/ 
  inline void set_table( MixingRxnModel* table ){
  	_table = table; 	
  }; 
  
  inline bool does_table_initialization(){ 
  	return _table_init; 
  };

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
  BoundaryCondition_new* d_boundaryCond;  ///< Boundary condition object associated with equation object
  ExplicitTimeInt* d_timeIntegrator;      ///< Time integrator object associated with equation object
  Discretization_new* d_disc;             ///< Discretization object associated with equation object
  IntrusionBC* _intrusions;               ///< Intrusions for boundary conditions. 
  MixingRxnModel* _table; 								///< Reference to the table for lookup 

  const VarLabel* d_transportVarLabel;    ///< Label for scalar being transported, in NEW data warehouse
  const VarLabel* d_oldtransportVarLabel; ///< Label for scalar being transported, in OLD data warehouse
  const VarLabel* d_FdiffLabel;           ///< Label for diffusion term of this equation object
  const VarLabel* d_FconvLabel;           ///< Label for convection term of this equation object
  const VarLabel* d_RHSLabel;             ///< Label for RHS of this equation object
  const VarLabel* d_mol_D_label;          ///< Molecular diffusivity label (computed elsewhere)

  bool d_doConv;                          ///< Boolean: do convection for this equation object?
  bool d_doDiff;                          ///< Boolean: do diffusion for this equation object?
  bool d_addSources;                      ///< Boolean: add a right-hand side (i.e. convection, diffusion, source terms) to this equation object?
  bool _using_new_intrusion;              ///< Indicates if new intrusions are being used. 

  std::string d_eqnName;                  ///< Human-readable label for this equation
  std::string d_convScheme;               ///< Convection scheme (superbee, upwind, etc.)
  std::string d_initFunction;             ///< A functional form for initial value.
  std::string d_mol_D_label_name;         ///< Name of the molecular diffusivity label. 
  std::string d_init_dp_varname; 					///< The name of a table dependent variable which could be used to initialize the transported variable
	

  // Clipping:
  bool d_doClipping;                ///< Boolean: are values clipped?
  bool d_doLowClip;                 ///< Boolean: are low values clipped?
  bool d_doHighClip;                ///< Boolean: are high values clipped?

  double d_lowClip;                 ///< Value of low clipping
  double d_highClip;                ///< Value of high clipping
  double d_smallClip;               ///< Value of small clipping (used if scalar is a divisor, e.g. with DQMOM weights)

  // Initialization:
  bool b_stepUsesCellLocation;      ///< Boolean: is step function's cell location specified?
  bool b_stepUsesPhysicalLocation;  ///< Boolean: is step function's physical location specified?

  // constant initialization function:
  double d_constant_init;           ///< constant value for initialization

  // Vector of geometry pieces for initialization 
  std::vector<GeometryPieceP> d_initGeom; 

  // step initialization function:
  std::string d_step_dir;           ///< For a step initialization function, direction in which step should occur
  double d_step_start;              ///< Physical location of step function start
  double d_step_end;                ///< Physical location of step function end
  int d_step_cellstart;             ///< Cell location of step function start
  int d_step_cellend;               ///< Cell location of step function end
  double d_step_value;              ///< Step function steps from 0 to d_step_value

  // gaussian initialization function: 
  double d_a_gauss;                 ///< constant a, height in gaussian function 
  double d_b_gauss;                 ///< constant b, position of gaussian function 
  double d_c_gauss;                 ///< constant c, width of gaussian function 
  double d_shift_gauss;             ///< shifts the gaussian function up or down 
  double d_dir_gauss;               ///< direction of the gaussian (0,1,2) == [x,y,z]

  // Other:
  double d_turbPrNo;                ///< Turbulent Prandtl number (used for scalar diffusion)
  double d_scalingConstant;         ///< Value by which to scale values 
  double curr_ssp_time;             ///< Current value of time considering ssp averaging  
  double curr_time;                 ///< "old" time (t)
  bool d_use_density_guess;         ///< Tells the solver to use the guessed density rather than the new density from the table
                                    ///<  Also, if true, the the equation is solved BEFORE the properties are computed. 
  vector<std::string> d_sources;    ///< List of source terms for this eqn
  double d_mol_diff;                ///< Molecular Diffusivity
  bool d_use_constant_D;            ///< Switch for using constant D or not. 
  bool _table_init;                 ///< Requires a table lookup for initialization 

private:


}; // end EqnBase


//---------------------------------------------------------------------------
// Method: Phi initialization using a function 
// DQMOM Weighted Abscissa
//---------------------------------------------------------------------------
template <class phiType, class constPhiType>  
void EqnBase::initializationFunction( const Patch* patch, phiType& phi, constPhiType& weight, constCCVariable<double>& eps_v  ) 
{
  proc0cout << "initializing scalar equation " << d_eqnName << endl;

  // Initialization function bullet proofing 
  if( d_initFunction == "step" || d_initFunction == "env_step" ) {
    if( d_step_dir == "y" ) {
#ifndef YDIM
      proc0cout << "WARNING: YDIM not turned on (compiled) with this version of the code, " << endl;
      proc0cout << "but you specified a step function that steps in the y-direction. " << endl;
      proc0cout << "To get this to work, made sure YDIM is defined in ScalarEqn.h" << endl;
      proc0cout << "Cannot initialize your scalar in y-dim with step function" << endl;
      throw InvalidValue("Exiting...", __FILE__, __LINE__);
#endif
      // otherwise do nothing

    } else if( d_step_dir == "z" ) {
#ifndef ZDIM
      proc0cout << "WARNING: ZDIM not turned on (compiled) with this version of the code, " << endl;
      proc0cout << "but you specified a step function that steps in the z-direction. " << endl;
      proc0cout << "To get this to work, made sure ZDIM is defined in ScalarEqn.h" << endl;
      proc0cout << "Cannot initialize your scalar in y-dim with step function" << endl;
      throw InvalidValue("Exiting...", __FILE__, __LINE__);
#endif
      // otherwise do nothing
    }
  }

  double pi = acos(-1.0); 

  for (CellIterator iter=patch->getCellIterator(0); !iter.done(); iter++){
    IntVector c = *iter; 
    Vector Dx = patch->dCell(); 
    Point  P  = patch->getCellPosition(c); 

    double x=0.0,y=0.0,z=0.0; 
    int cellx=0, celly=0, cellz=0;

    cellx = c[0];
    celly = c[1];
    cellz = c[2];

    x = P.x(); // the +Dx/2 is because variable is cell-centered
    y = P.y(); 
    z = P.z();

    if ( d_initFunction == "constant" || d_initFunction == "env_constant" ) {
      // ========== CONSTANT VALUE INITIALIZATION ============
      phi[c] = d_constant_init * weight[c];

    } else if (d_initFunction == "step" || d_initFunction == "env_step" ) {
      // =========== STEP FUNCTION INITIALIZATION =============
      if (d_step_dir == "x") {
        if (  (b_stepUsesPhysicalLocation && x >= d_step_start && x <= d_step_end)
           || (b_stepUsesCellLocation && cellx >= d_step_cellstart && x <= d_step_cellend) ) {
          phi[c] = d_step_value * weight[c];
        } else {
          phi[c] = 0.0;
        }
      
      } else if (d_step_dir == "y") {
        if (  (b_stepUsesPhysicalLocation && y >= d_step_start && y <= d_step_end)
           || (b_stepUsesCellLocation && celly >= d_step_cellstart && celly <= d_step_cellend) ) {
          phi[c] = d_step_value * weight[c];
        } else { 
          phi[c] = 0.0;
        }
      } else if (d_step_dir == "z") {
        if (  (b_stepUsesPhysicalLocation && z >= d_step_start && z <= d_step_end)
           || (b_stepUsesCellLocation && cellz >= d_step_cellstart && cellz <= d_step_cellend) ) {
          phi[c] = d_step_value * weight[c];
        } else {
          phi[c] = 0.0;
        }
      }
    } else if ( d_initFunction == "mms1" ) {
      //======= an MMS with the function phi = sin(2*pi*x)cos(2*pi*y) ======
      phi[c] = sin(2.0 * pi * x)*cos(2.0 * pi * y)* weight[c];

    } else if ( d_initFunction == "sine-x" ) { 
      //======= sin function in x ======

      phi[c] = sin( 2.0 * pi * x ); 

    } else if ( d_initFunction == "sine-y" ) { 
      //======= sin function in y ======

      phi[c] = sin( 2.0 * pi * y ); 

    } else if ( d_initFunction == "sine-z" ) { 
      //======= sin function in z ======

      phi[c] = sin( 2.0 * pi * z ); 

    // ======= add other initialization functions below here ======
    } else {

      throw InvalidValue("Error!: Your initialization function for equation "+d_eqnName+" wasn't found.", __FILE__, __LINE__);

    }//end d_initFunction types

    phi[c] *= eps_v[c];

  }
}
//---------------------------------------------------------------------------
// Method: Phi initialization using a function 
// Standard Scalar, DQMOM Weight
//---------------------------------------------------------------------------
template <class phiType>  
void EqnBase::initializationFunction( const Patch* patch, phiType& phi, constCCVariable<double>& eps_v ) 
{
  proc0cout << "initializing scalar equation " << d_eqnName << endl;

  // Initialization function bullet proofing 
  if( d_initFunction == "step" || d_initFunction == "env_step" ) {
    if( d_step_dir == "y" ) {
#ifndef YDIM
      proc0cout << "WARNING: YDIM not turned on (compiled) with this version of the code, " << endl;
      proc0cout << "but you specified a step function that steps in the y-direction. " << endl;
      proc0cout << "To get this to work, made sure YDIM is defined in ScalarEqn.h" << endl;
      proc0cout << "Cannot initialize your scalar in y-dim with step function" << endl;
      throw InvalidValue("Exiting...", __FILE__, __LINE__);
#endif
      // otherwise do nothing

    } else if( d_step_dir == "z" ) {
#ifndef ZDIM
      proc0cout << "WARNING: ZDIM not turned on (compiled) with this version of the code, " << endl;
      proc0cout << "but you specified a step function that steps in the z-direction. " << endl;
      proc0cout << "To get this to work, made sure ZDIM is defined in ScalarEqn.h" << endl;
      proc0cout << "Cannot initialize your scalar in y-dim with step function" << endl;
      throw InvalidValue("Exiting...", __FILE__, __LINE__);
#endif
      // otherwise do nothing
    }
  }

  double pi = acos(-1.0); 

  Box patchInteriorBox = patch->getBox(); 

  for (CellIterator iter=patch->getCellIterator(0); !iter.done(); iter++){
    IntVector c = *iter; 
    Vector Dx = patch->dCell(); 
    Point  P  = patch->getCellPosition(c);

    double x=0.0,y=0.0,z=0.0; 
    int cellx=0, celly=0, cellz=0;

    cellx = c[0];
    celly = c[1];
    cellz = c[2];

    x = P.x(); 
    y = P.y(); 
    z = P.z(); 

    if ( d_initFunction == "constant" || d_initFunction == "env_constant" ) {
      // ========== CONSTANT VALUE INITIALIZATION ============
      phi[c] = d_constant_init;

    } else if (d_initFunction == "step" || d_initFunction == "env_step" ) {
      // =========== STEP FUNCTION INITIALIZATION =============
      if (d_step_dir == "x") {
        if (  (b_stepUsesPhysicalLocation && x >= d_step_start && x <= d_step_end)
           || (b_stepUsesCellLocation && cellx >= d_step_cellstart && x <= d_step_cellend) ) {
          phi[c] = d_step_value;
        } else {
          phi[c] = 0.0;
        }
      
      } else if (d_step_dir == "y") {
        if (  (b_stepUsesPhysicalLocation && y >= d_step_start && y <= d_step_end)
           || (b_stepUsesCellLocation && celly >= d_step_cellstart && celly <= d_step_cellend) ) {
          phi[c] = d_step_value;
        } else { 
          phi[c] = 0.0;
        }
      } else if (d_step_dir == "z") {
        if (  (b_stepUsesPhysicalLocation && z >= d_step_start && z <= d_step_end)
           || (b_stepUsesCellLocation && cellz >= d_step_cellstart && cellz <= d_step_cellend) ) {
          phi[c] = d_step_value;
        } else {
          phi[c] = 0.0;
        }
      }
    } else if ( d_initFunction == "mms1" ) {
      //======= an MMS with the function phi = sin(2*pi*x)cos(2*pi*y) ======
      phi[c] = sin(2.0 * pi * x)*cos(2.0 * pi * y);

    } else if ( d_initFunction == "sine-x" ) { 
      //======= sin function in x ======

      phi[c] = sin( 2.0 * pi * x ); 

    } else if ( d_initFunction == "sine-y" ) { 
      //======= sin function in y ======

      phi[c] = sin( 2.0 * pi * y ); 

    } else if ( d_initFunction == "sine-z" ) { 
      //======= sin function in z ======

      phi[c] = sin( 2.0 * pi * z ); 

    } else if ( d_initFunction == "gaussian" ) { 

      //======= Gaussian ========

      if ( d_dir_gauss == 0 ){ 

        phi[c] = d_a_gauss * exp( -1.0*std::pow(x-d_b_gauss,2.0)/(2.0*std::pow(d_c_gauss,2.0))) + d_shift_gauss;

      } else if ( d_dir_gauss == 1 ){ 

        phi[c] = d_a_gauss * exp( -1.0*std::pow(y-d_b_gauss,2.0)/(2.0*std::pow(d_c_gauss,2.0))) + d_shift_gauss;

      } else { 

        phi[c] = d_a_gauss * exp( -1.0*std::pow(z-d_b_gauss,2.0)/(2.0*std::pow(d_c_gauss,2.0))) + d_shift_gauss;

      } 

    } else if (d_initFunction == "geometry_fill") {
      //======= Fills a geometry piece with the value of d_constant_init ======
      for (std::vector<GeometryPieceP>::iterator giter = d_initGeom.begin(); giter != d_initGeom.end(); giter++){

        GeometryPieceP g_piece = *giter; 
        Box geomBox = g_piece->getBoundingBox(); 
        Box intersectedBox = geomBox.intersect(patchInteriorBox); 

        if (!(intersectedBox.degenerate())){

          Point P = patch->cellPosition(*iter); 

          if ( g_piece->inside(P) )
            phi[c] = d_constant_init; 
          else 
            phi[c] = 0.0;

        } else {
          phi[c] = 0.0;
        }
      }

    } else if ( d_initFunction == "tabulated" ){ 

      //will do the actual initialization later on
      //but need this here to keep the InvalidValue from throwing..

    // ======= add other initialization functions below here ======
    } else {

      throw InvalidValue("Error!: Your initialization function for equation "+d_eqnName+" wasn't found.", __FILE__, __LINE__);

    }//end d_initFunction types

    phi[c] *= eps_v[c]; 

  }
}

} // end namespace Uintah

#endif
