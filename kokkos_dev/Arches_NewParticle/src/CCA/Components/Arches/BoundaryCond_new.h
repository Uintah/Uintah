#ifndef Uintah_Components_Arches_BoundaryCondition_new_h
#define Uintah_Components_Arches_BoundaryCondition_new_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <CCA/Components/Arches/Directives.h>

namespace Uintah {

//===========================================================================

/**
  @class  BoundaryCondition 
  @date   
  @author Jeremy Thornock

  @brief This class sets the boundary conditions for scalars. 

*/

class ArchesLabel; 
class BoundaryCondition_new {

public: 

  BoundaryCondition_new(const ArchesLabel* fieldLabels);

  ~BoundaryCondition_new();

  /** @brief Interface for the input file and set constants */ 
  void problemSetup();

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
  void setAreaFraction( const Patch* patch,
                        CCVariable<Vector>& areaFraction, 
                        CCVariable<double>& volFraction, 
                        constCCVariable<int>& pcell, 
                        const int wallType, 
                        const int flowType );

  // The stuff below needs better commenting when I have this figured out. 
  /* --------------------------------------------------------------------- 
  Function~  getIteratorBCValueBCKind--
  Purpose~   does the actual work
  ---------------------------------------------------------------------  */
  template <class T>
  bool getIteratorBCValueBCKind( const Patch* patch, 
                                 const Patch::FaceType face,
                                 const int child,
                                 const string& desc,
                                 const int mat_id,
                                 T& bc_value,
                                 Iterator& bound_ptr,
                                 string& bc_kind)
  {
    //__________________________________
    //  find the iterator, BC value and BC kind
    Iterator nu;  // not used

    const BoundCondBase* bc = patch->getArrayBCValues(face,mat_id,
		                                          		    desc, bound_ptr,
                                                      nu, child);
    const BoundCond<T> *new_bcs =  dynamic_cast<const BoundCond<T> *>(bc);

    bc_value=T(-9);
    bc_kind="NotSet";
    if (new_bcs != 0) {      // non-symmetric
      bc_value = new_bcs->getValue();
      bc_kind =  new_bcs->getBCType__NEW();
    }        
    delete bc;

    // Did I find an iterator
    if( bc_kind == "NotSet" ){
      return false;
    }else{
      return true;
    }
  }

  void sched_computeBCArea( SchedulerP& sched, 
                            const PatchSet* patches, 
                            const MaterialSet* matls );


  typedef std::map< std::string, const VarLabel* > LabelMap; 

private: 
 
  //variables
  const ArchesLabel* d_fieldLabels;

  LabelMap areaMap; 

  void computeBCArea( const ProcessorGroup*, 
                      const PatchSubset* patches, 
                      const MaterialSubset*, 
                      DataWarehouse*, 
                      DataWarehouse* new_dw );



}; // class BoundaryCondition_new
} // namespace Uintah

#endif 
