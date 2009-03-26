//------------------------ DQMOM.h -----------------------------------

#ifndef Uintah_Components_Arches_DQMOMLinearSolver_h
#define Uintah_Components_Arches_DQMOMLinearSolver_h

#include <sci_defs/petsc_defs.h>

#include <CCA/Components/SpatialOps/SpatialOps.h>
#include <CCA/Components/SpatialOps/Fields.h>
#include <CCA/Components/SpatialOps/TransportEqns/EqnFactory.h>
#include <CCA/Components/SpatialOps/TransportEqns/EqnBase.h>
#include <CCA/Components/SpatialOps/TransportEqns/ScalarEqn.h>
#include <CCA/Components/SpatialOps/TransportEqns/DQMOMEqn.h>

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

class DQMOM {

public:

    DQMOM( const Fields* fieldLabels );

    ~DQMOM();

    /** @brief Obtain parameters from input file and process them, whatever that means 
    */
    void DQMOM::problemSetup( const ProblemSpecP& params, 
                              DQMOMEqnFactory& eqn_factory );

    /** @brief Schedule creation of linear solver object, creation of AX=B system, and solution of linear system.
    */
    void DQMOM::sched_solveLinearSystem( const LevelP& level,
                                          SchedulerP& sched );
    
    /** @brief Create linear solver object, create linear system AX=B, and solve the linear system.
    */
    void DQMOM::solveLinearSystem( const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset*,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw );

    /** @brief Destroy A, X, B, and solver object 
    */
    void DQMOM::destroyLinearSystem();

    //-------------------------------------------------------------------

protected:

private:

  vector<string> InternalCoordinateEqnNames;

  // moment indexes
  typedef vector<int> MomentVector;
  vector<MomentVector> momentIndexes;

  // weights and weighted abscissa labels, IN SAME ORDER AS GIVEN IN INPUT FILE
  //    vector weightEqns[1] = weight, quad node 1
  //    vector weightEqns[2] = weight, quad node 2
  //    vector weightedAbscissaEqns[1][1] = weighted abscissa 1, quad node 1
  //    vector weightedAbscissaEqns[1][2] = weighted abscissa 1, quad node 2
  //    vector weightedAbscissaEqns[2][1] = weighted abscissa 2, quad node 1
  //vector<DQMOMEqn&> weightEqns;
  //vector< vector<DQMOMEqn&> > weightedAbscissaEqns;
  vector<DQMOMEqn> weightEqns;
  vector<DQMOMEqn> weightedAbscissaEqns;

  // number of internal coordinates
  int N_xi;

  // number of quadrature nodes or environments
  int N;

  const Fields* d_fieldLabels;

}; // end class DQMOM

} // end namespace Uintah

#endif
