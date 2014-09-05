#ifndef Uintah_Components_Arches_BoundaryCondition_new_h
#define Uintah_Components_Arches_BoundaryCondition_new_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>

#define XDIM
//#define YDIM
//#define ZDIM

//===========================================================================

/**
*   @class Boundary Condition 
*   @author Jeremy Thornock
*   @brief This class sets the boundary conditions for scalars. 
*
*/

namespace Uintah {

class ArchesLabel; 
class BoundaryCondition_new {

public: 

  BoundaryCondition_new(const ArchesLabel* fieldLabels);

  ~BoundaryCondition_new();
  /** @brief Interface for the input file and set constants */ 
  void problemSetup();
  /** @brief Schedular for setting the boundary condition */ 
  int scheduleSetBC( const LevelP& level,
			                SchedulerP& sched );
  /** @brief This method sets the boundary value of a scalar to 
             a value such that the interpolated value on the face results
             in the actual boundary condition. */   
  void setScalarValueBC(const ProcessorGroup*,
                        const Patch* patch,
                        CCVariable<double>& scalar, 
                        string varname );
  /** @brief Actually set the boundary conditions.  I think this won't be used */ 
  void setBC( const ProcessorGroup* ,
		          const PatchSubset* patches,
		          const MaterialSubset*,
              DataWarehouse* old_dw,
  	          DataWarehouse* new_dw );

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

private: 
 
  //variables
  const ArchesLabel* d_fieldLabels;



}; // class BoundaryCondition_new
} // namespace Uintah

#endif 
