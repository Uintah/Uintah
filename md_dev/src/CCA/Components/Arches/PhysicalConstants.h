/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


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

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)
    
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
#include <Core/Geometry/IntVector.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

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
  // Get RefPoint Location
  const IntVector& getRefPoint(){ return d_ref_point; }

  ///////////////////////////////////////////////////////////////////////
  // Get one component of Gravity Vector
  double getGravity(int index){
    if (index == 1){
      return d_gravity.x();
    }else if (index == 2){
      return d_gravity.y();
    }
    else{
      return d_gravity.z();
    }
  }

  ///////////////////////////////////////////////////////////////////////
  // Get molecular viscosity (of air)
  double getMolecularViscosity() { return d_viscosity; }

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
  IntVector d_ref_point;
  double d_viscosity;
    
}; // end class PhysicalConstants
} // End namespace Uintah


#endif


