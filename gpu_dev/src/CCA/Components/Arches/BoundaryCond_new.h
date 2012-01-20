#ifndef Uintah_Components_Arches_BoundaryCondition_new_h
#define Uintah_Components_Arches_BoundaryCondition_new_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <CCA/Components/Arches/Directives.h>

//===========================================================================

/**
*   @class BoundaryCondition 
*   @author Jeremy Thornock
*   @brief This class sets the boundary conditions for scalars. 
*
*/

namespace Uintah {

class ArchesLabel; 
class BoundaryCondition_new {

public: 

  //** WARNING: This needs to be duplicated in BoundaryCondition.h for now until BoundaryCondition goes away **//
  enum BC_TYPE { VELOCITY_INLET, MASSFLOW_INLET, VELOCITY_FILE, MASSFLOW_FILE, PRESSURE, OUTLET, WALL, MMWALL, INTRUSION, SWIRL }; 

  typedef std::map<IntVector, double> CellToValueMap; 
  typedef std::map<Patch*, vector<CellToValueMap> > PatchToBCValueMap; 
  typedef std::map<std::string, CellToValueMap> ScalarToBCValueMap; 

  BoundaryCondition_new(const ArchesLabel* fieldLabels);

  ~BoundaryCondition_new();
  /** @brief Interface for the input file and set constants */ 
  void  problemSetup( ProblemSpecP& db, std::string eqn_name );

  /** @brief This method sets the boundary value of a scalar to 
             a value such that the interpolated value on the face results
             in the actual boundary condition. */   
  void setScalarValueBC(const ProcessorGroup*,
                        const Patch* patch,
                        CCVariable<double>& scalar, 
                        string varname );
  /** @brief This method set the boundary values of a vector to a 
   * value such that the interpolation or gradient computed between the 
   * interior cell and boundary cell match the boundary condition. */ 
  void setVectorValueBC( const ProcessorGroup*,
    const Patch* patch,
    CCVariable<Vector>& vec, 
    string varname );
  /** @brief This method set the boundary values of a vector to a 
   * value such that the interpolation or gradient computed between the 
   * interior cell and boundary cell match the boundary condition. This is 
   * a specialized case where the boundary value comes from some other vector */
  void setVectorValueBC( const ProcessorGroup*,
    const Patch* patch,
    CCVariable<Vector>& vec, constCCVariable<Vector>& const_vec, 
    string varname );

  /** @brief Sets the area fraction for each minus face according to the boundaries */
  void setAreaFraction( 
    const Patch* patch,
    CCVariable<Vector>& areaFraction, 
    CCVariable<double>& volFraction, 
    constCCVariable<int>& pcell, 
    const int wallType, 
    const int flowType );

  void sched_computeBCArea( SchedulerP& sched, 
                            const PatchSet* patches, 
                            const MaterialSet* matls );

  /** @brief Read in a file for boundary conditions **/ 
  std::map<IntVector, double> readInputFile( std::string file_name ); 


  typedef std::map< std::string, const VarLabel* > LabelMap; 

private: 
 
  //variables
  const ArchesLabel* d_fieldLabels;

  LabelMap areaMap; 
  ScalarToBCValueMap scalar_bc_from_file; 

  void computeBCArea( const ProcessorGroup*, 
                      const PatchSubset* patches, 
                      const MaterialSubset*, 
                      DataWarehouse*, 
                      DataWarehouse* new_dw );



}; // class BoundaryCondition_new
} // namespace Uintah

#endif 
