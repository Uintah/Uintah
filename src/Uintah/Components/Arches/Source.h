/**************************************
CLASS
   Source
   
   Class Source computes source terms for 
   N-S equations.  

GENERAL INFORMATION
   Source.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 1998

KEYWORDS


DESCRIPTION
   Class Source computes source terms for 
   N-S equations.  



WARNING
none
****************************************/
#ifndef included_Source
#define included_Source


class Source : 
{
public:
  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  //
  // Construct an instance of a Source.
  //
  // PRECONDITIONS
  //
  //
  // POSTCONDITIONS
  //
  // Default constructor.
 
   Source();


  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Destructor
   ~Source();
   // GROUP:  Access functions:
   ////////////////////////////////////////////////////////////////////////
   // Returns a pointer to the su and sp componenets of the source 
   // index identifies the velocity component
   CCVariable<vector>& getPressureLinearSource();
   CCVariable<vector>& getPressureNonlinearSource();
    
   FCVariable<vector>& getMomentumLinearSource(int index);
   FCVariable<vector>& getMomentumNonlinearSource(int index);
   
   CCVariable<vector>& getScalarLinearSource(int index);
   CCVariable<vector>& getScalarNonlinearSource(int index);


   
   // GROUP:  Methods
   ////////////////////////////////////////////////////////////////////////
   // Set source terms. Will need more parameters...like velocity and
   // scalars
   
   void setPressureSource();
   void setMomentumSource(int index);
   void setScalarSource(int index);
   // Modify source after underrelaxation
   virtual void modifySource() = 0;

 private:
   
   //nonlinear components of the source term
   CCVariable<vector>* d_press_su;
   // 3D vector
   FCVariable<vector>* d_mom_su; 
   // n-d vector where n is the numner of scalars solved
   CCVariable<vector>* d_scalar_su;
   //linearized componenet of the source term
   CCVariable<vector>* d_press_sp;
   FCVariable<vector>* d_mom_sp;
   CCVariable<vector>* d_scalar_sp;

};
#endif  
  
