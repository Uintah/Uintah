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

namespace Uintah {

//-------------------------------------------------------

/** 
  * @class    DQMOM
  * @author   Charles Reid (charlesreid1@gmail.com)
  * @date     March 16, 2009
  *
  * @brief    This class constructs and solves the AX=B linear system for DQMOM scalars.
  *
  * @details  Using the direct quadrature method of moments (DQMOM), the NDF transport equation is represented
  *           by a set of delta functions times weights.  The moment transform of this NDF transport equation
  *           yields a set of (closed) moment transport equations. These equations can be expressed in terms 
  *           of the weights and abscissas of the quadrature approximation, and re-cast as a linear system,
  *           \f$ \mathbf{AX} = \mathbf{B} \f$.  This class solves the linear system to yield the source terms
  *           for the weight and weighted abscissa transport equations (the variables contained in \f$\mathbf{X}\f$.
  *
  */

//-------------------------------------------------------
class ArchesLabel; 
class DQMOMEqn; 
class ModelBase; 
class LU;
class DQMOM {

public:

  DQMOM( const ArchesLabel* fieldLabels );

  ~DQMOM();

  /** @brief Obtain parameters from input file and process them, whatever that means 
   */
  void problemSetup( const ProblemSpecP& params );

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
  
  /** @brief Use iterative refinement to improve the solution to AX=B. 
    * @param A    A matrix - (decomposed) coefficients matrix (remains constant)
    * @param B    B matrix - right-hand side vector (remains constant)
    * @param X    X matrix - solution vector from initial factorization       */
  void iterativeSolutionMethod(LU* A, vector<double>* B, vector<double>* X);

    /** @brief Destroy A, X, B, and solver object  
    */
    void destroyLinearSystem();

private:

  vector<string> InternalCoordinateEqnNames;

  // moment indexes
  typedef vector<int> MomentVector;
  vector<MomentVector> momentIndexes;

  // weights and weighted abscissa labels, IN SAME ORDER AS GIVEN IN INPUT FILE
  std::vector<DQMOMEqn* > weightEqns;
  std::vector<DQMOMEqn* > weightedAbscissaEqns;

  vector< vector<ModelBase> > weightedAbscissaModels;

  unsigned int N_xi;  // # of internal coordinates
  unsigned int N_;    // # of quadrature nodes

  const ArchesLabel* d_fieldLabels;
  int d_timeSubStep;

  double d_solver_tolerance;

  const VarLabel* d_normBLabel; 
  const VarLabel* d_normXLabel; 
  const VarLabel* d_normResLabel;
  const VarLabel* d_normResNormalizedLabel;
  const VarLabel* d_conditionEstimateLabel;

}; // end class DQMOM

} // end namespace Uintah

#endif
