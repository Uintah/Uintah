//----- MixingModel.h --------------------------------------------------

#ifndef Uintah_Component_Arches_MixingModel_h
#define Uintah_Component_Arches_MixingModel_h

/***************************************************************************
CLASS
    MixingModel
       Sets up the MixingModel ????
       
GENERAL INFORMATION
    MixingModel.h - Declaration of MixingModel class

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

#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

#include <vector>

namespace Uintah {
  class InletStream;
class MixingModel {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructor taking
      //   [in] 
      //
      MixingModel();

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Virtual destructor for mixing model
      //
      virtual ~MixingModel();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      //
      // Set up the problem specification database
      //
      virtual void problemSetup(const ProblemSpecP& params) = 0;

      // GROUP: Compute properties 
      // GROUP: Actual Action Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Carry out actual computation of properties
      //
      virtual void computeProps(const InletStream& inStream,
			       Stream& outStream) = 0;

      // GROUP: Get Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Get the number of mixing variables
      //
      virtual int getNumMixVars() const = 0;
      virtual bool isAdiabatic() const = 0;
      virtual int getNumMixStatVars() const = 0;
      virtual int getNumRxnVars() const = 0;



protected :

private:

}; // end class MixingModel

} // end namespace Uintah

#endif

