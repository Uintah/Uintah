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

#include <Uintah/Components/Arches/ArchesLabel.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/CFDInterface.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/VarLabel.h>
#include <SCICore/Geometry/IntVector.h>

#include <vector>

namespace Uintah {
namespace ArchesSpace {

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
      ///////////////////////////////////////////////////////////////////////
      //
      // Compute properties for inlet/outlet streams
      // pass stream that will contain computed density, enthalpy
      // and species values
      virtual void computeInletProperties(const std::vector<double>&
					  mixfractionStream,
					  Stream& inletStream) = 0;
      // GROUP: Actual Action Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Carry out actual computation of properties
      //
      vitual void computeProps(const ProcessorGroup*,
			       InletStream& inStream,
			       Stream& outStream) = 0;

      // GROUP: Get Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Get the number of mixing variables
      //
      virtual int getNumMixVars() const = 0;


protected :

private:

}; // end class MixingModel

} // end namespace ArchesSpace
} // end namespace Uintah

#endif

//
// $Log$
// Revision 1.1  2000/12/18 17:53:10  rawat
// adding mixing model for reacting flows
//
//
