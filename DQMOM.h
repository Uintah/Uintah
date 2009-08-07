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
* @class DQMOM
* @author Charles Reid (charlesreid1@gmail.com)
* @date March 16, 2009
*
* @brief This class constructs and solves the AX=B linear system for DQMOM scalars.
*
*/

//-------------------------------------------------------
class ArchesLabel; 
class DQMOMEqn; 
class ModelBase; 
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

    /** @brief Destroy A, X, B, and solver object  
    */
    void destroyLinearSystem();

private:

  vector<string> InternalCoordinateEqnNames;

  // moment indexes
  typedef vector<int> MomentVector;
  vector<MomentVector> momentIndexes;

  // weights and weighted abscissa labels, IN SAME ORDER AS GIVEN IN INPUT FILE
  //    vector weightEqns[1] = weight, quad node 1
  //    vector weightEqns[2] = weight, quad node 2
  //    vector weightedAbscissaEqns[(1-1)N + 1] = weighted abscissa 1, quad node 1
  //    vector weightedAbscissaEqns[(1-1)N + 2] = weighted abscissa 1, quad node 2
  //    vector weightedAbscissaEqns[(2-1)N + 1] = weighted abscissa 2, quad node 1
  std::vector<DQMOMEqn* > weightEqns;
  std::vector<DQMOMEqn* > weightedAbscissaEqns;

  vector< vector<ModelBase> > weightedAbscissaModels;

  // # of internal coordinates
  unsigned int N_xi;
  // # of quadrature nodes
  unsigned int N_;

  const ArchesLabel* d_fieldLabels;
  int d_timeSubStep;

  double d_solver_tolerance;

  const VarLabel* d_normBLabel; 
  const VarLabel* d_normXLabel; 
  const VarLabel* d_normResLabel;

}; // end class DQMOM

} // end namespace Uintah

#endif
