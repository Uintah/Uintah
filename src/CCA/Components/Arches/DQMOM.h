//------------------------ DQMOM.h -----------------------------------

#ifndef Uintah_Components_Arches_DQMOM_h
#define Uintah_Components_Arches_DQMOM_h

#include <sci_defs/petsc_defs.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Patch.h>

#include <map>
#include <vector>
#include <string>


#ifdef HAVE_PETSC
extern "C" {
#include "petscksp.h"
}
#endif

// Uncomment for DQMOM verification:
//#define VERIFY_DQMOM 1

namespace Uintah {

//-------------------------------------------------------

/** 
  * @class    DQMOM
  * @author   Charles Reid (charlesreid1@gmail.com)
  * @date     March 2009
  *
  * @brief    This class constructs and solves the AX=B linear system for DQMOM scalars.
  *
  * @details  Using the direct quadrature method of moments (DQMOM), the NDF transport equation is represented
  *           by a set of delta functions times weights.  The moment transform of this NDF transport equation
  *           yields a set of (closed) moment transport equations. These equations can be expressed in terms 
  *           of the weights and abscissas of the quadrature approximation, and re-cast as a linear system,
  *           \f$ \mathbf{AX} = \mathbf{B} \f$.  This class solves the linear system to yield the source terms
  *           for the weight and weighted abscissa transport equations (the variables contained in \f$\mathbf{X}\f$).
  *
  */

//-------------------------------------------------------
class ArchesLabel; 
class DQMOMEqn; 
class ModelBase; 
class LU;
class DQMOM {

public:

  DQMOM( ArchesLabel* fieldLabels );

  ~DQMOM();

  typedef std::vector<int> MomentVector;

  /** @brief Obtain parameters from input file and process them, whatever that means 
   */
  void problemSetup( const ProblemSpecP& params );

  /** @brief              Populate the map containing labels for each moment 
      @param allMoments   Vector containing all moment indexes specified by user in <Moment> blocks within <DQMOM> block */
  void populateMomentsMap( vector<MomentVector> allMoments );

  /** @brief Schedule creation of linear solver object, creation of AX=B system, and solution of linear system. 
  */
  void sched_solveLinearSystem( const LevelP & level,
                                SchedulerP   & sched,
                                int            timeSubStep );
    
  /** @brief Create linear solver object, create linear system AX=B, and solve the linear system. 
  */
  void solveLinearSystem( const ProcessorGroup *,
                          const PatchSubset    * patches,
                          const MaterialSubset *,
                          DataWarehouse        * old_dw,
                          DataWarehouse        * new_dw );
  
  /** @brief Schedule calculation of all moments
  */
  void sched_calculateMoments( const LevelP & level,
                               SchedulerP   & sched,
                               int            timeSubStep );

  /** @brief Calculate the value of the moments */
  void calculateMoments( const ProcessorGroup *,
                         const PatchSubset    * patches,
                         const MaterialSubset *,
                         DataWarehouse        * old_dw,
                         DataWarehouse        * new_dw );



    /** @brief Destroy A, X, B, and solver object  
    */
    void destroyLinearSystem();

    /** @brief Access function for boolean, whether to calculate/save moments */
    bool getSaveMoments() {
      return b_save_moments; }

private:

  vector<string> InternalCoordinateEqnNames;

  // moment indexes
  vector<MomentVector> momentIndexes;

  // weights and weighted abscissa labels, IN SAME ORDER AS GIVEN IN INPUT FILE
  std::vector<DQMOMEqn* > weightEqns;
  std::vector<DQMOMEqn* > weightedAbscissaEqns;

  vector< vector<ModelBase> > weightedAbscissaModels;

  unsigned int N_xi;  // # of internal coordinates
  unsigned int N_;    // # of quadrature nodes

  ArchesLabel* d_fieldLabels; // this is no longer const because a modifiable instance of ArchesLabel is
                              // required to populate (i.e. modify) the moments map contained in the ArchesLabel
                              // class! (otherwise the compiler says "discards qualifiers"...)
  
  int d_timeSubStep;
  bool b_save_moments; // boolean - calculate & save moments?

  double d_solver_tolerance;
  double d_w_small;
  double d_weight_scaling_constant;
  vector<double> d_weighted_abscissa_scaling_constants;

  const VarLabel* d_normBLabel; 
  const VarLabel* d_normXLabel; 
  const VarLabel* d_normResLabel;
  const VarLabel* d_normResNormalizedLabel;
  const VarLabel* d_determinantLabel;

  double d_small_B; 

}; // end class DQMOM

} // end namespace Uintah

#endif
