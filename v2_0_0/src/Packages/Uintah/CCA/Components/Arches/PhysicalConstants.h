//----- PhysicalConstants.h --------------------------------------------------

#ifndef Uintah_Component_Arches_PhysicalConstants_h
#define Uintah_Component_Arches_PhysicalConstants_h

/***************************************************************************
CLASS
    PhysicalConstants
       Sets up the Physical Constants 
       
GENERAL INFORMATION
    PhysicalConstants.h - Declaration of PhysicalConstants class

    Author: Rajesh Rawat (rawat@crsim.utah.edu)
    
    Creation Date : 05-30-2000

    C-SAFE
    
    Copyright U of U 2000

KEYWORDS
    
DESCRIPTION

PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
    None
***************************************************************************/

#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {
  using namespace SCIRun;
class PhysicalConstants {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      // Constructor taking
      //   [in] 
      PhysicalConstants();

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      // Destructor
      ~PhysicalConstants();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Set up the problem specification database
      void problemSetup(const ProblemSpecP& params);

      // GROUP: Get Methods :
      ///////////////////////////////////////////////////////////////////////
      // Get Gravity Vector
      const Vector& getGravity(){ return d_gravity; }

      ///////////////////////////////////////////////////////////////////////
      // Get one component of Gravity Vector
      double getGravity(int index){
	if (index == 1) return d_gravity.x();
	else if (index == 2) return d_gravity.y();
	else return d_gravity.z();
      }

      ///////////////////////////////////////////////////////////////////////
      // Get molecular viscosity (of air)
      double getMolecularViscosity() { return d_viscosity; }

      ///////////////////////////////////////////////////////////////////////
      // Get absolute pressure (of 1 atmosphere)
      double getabsPressure() { return d_absPressure; }
    
protected :

private:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      // Copy Constructor (never instantiated)
      //   [in] 
      //        const PhysicalConstants&   
      PhysicalConstants(const PhysicalConstants&);

      // GROUP: Operators:
      ///////////////////////////////////////////////////////////////////////
      // Assignment Operator (never instantiated)
      //   [in] 
      //        const PhysicalConstants&   
      PhysicalConstants& operator=(const PhysicalConstants&);

private:

    Vector d_gravity;
    double d_viscosity;
    double d_absPressure;
    
}; // end class PhysicalConstants
} // End namespace Uintah


#endif


