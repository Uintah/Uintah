/**************************************
CLASS
   BoundaryCondition
   
   Class BoundaryCondition applies boundary conditions
   at physical boundaries. For boundary cell types it
   modifies stencil coefficients and source terms.

GENERAL INFORMATION
   BoundaryCondition.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class BoundaryCondition applies boundary conditions
   at physical boundaries. For boundary cell types it
   modifies stencil coefficients and source terms. 



WARNING
none
****************************************/
#ifndef included_BoundaryCondition
#define included_BoundaryCondition


class BoundaryCondition : 
{
public:
  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  //
  // Construct an instance of a BoundaryCondition.
  //
  // PRECONDITIONS
  //
  //
  // POSTCONDITIONS
  //
  // Default constructor.
 
   BoundaryCondition();


  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Destructor
   ~BoundaryCondition();
   
   // GROUP:  Methods
   ////////////////////////////////////////////////////////////////////////
   // Set boundary conditions terms. 
   
   void setPressureBoundaryCondition();
   void setMomentumBoundaryCondition(int index);
   void setScalarBoundaryCondition(int index);
   //uses TurbulenceModel to calculate wall bc's
   computeWallBC();
   // Set inlet velocity bc's, we need to do it because of staggered grid
   // need to pass velocity
   ComputeInletVelocityBC();
   // used for pressure boundary type
   ComputePressureBC();

 private:
   Discretization* d_discrete;
   Source* d_source;
   // used for calculating wall boundary conditions
   TurbulenceModel* d_turb_model;
   //stores the cell type
   CCVariable<vector>* d_cellType;


};
#endif  
  
