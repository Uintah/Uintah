//----- ColdflowMixingModel.h --------------------------------------------------

#ifndef Uintah_Component_Arches_ColdflowMixingModel_h
#define Uintah_Component_Arches_ColdflowMixingModel_h

/***************************************************************************
CLASS
    ColdflowMixingModel
       Sets up the ColdflowMixingModel ????
       
GENERAL INFORMATION
    ColdflowMixingModel.h - Declaration of ColdflowMixingModel class

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
#include <Uintah/Components/Arches/MixingModel.h>
#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/CFDInterface.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/VarLabel.h>
#include <SCICore/Geometry/IntVector.h>

#include <vector>

namespace Uintah {
namespace ArchesSpace {

class ColdflowMixingModel: public MixingModel {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructor taking
      //   [in] 
      //
      ColdflowMixingModel(const ArchesLabel* label);

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Destructor
      //
      virtual ~ColdflowMixingModel();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      //
      // Set up the problem specification database
      //
      virtual void problemSetup(const ProblemSpecP& params);

      // GROUP: Actual Action Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Carry out actual computation of properties
      //
      virtual void computeProps(InletStream& inStream,
				Stream& outStream);

      // GROUP: Get Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Get the number of mixing variables
      //
      inline int getNumMixVars() const{ 
	return d_numMixingVars; 
      }


protected :

private:

      ///////////////////////////////////////////////////////////////////////
      //
      // Copy Constructor (never instantiated)
      //   [in] 
      //        const ColdflowMixingModel&   
      //
      ColdflowMixingModel(const ColdflowMixingModel&);

      // GROUP: Operators Not Instantiated:
      ///////////////////////////////////////////////////////////////////////
      //
      // Assignment Operator (never instantiated)
      //   [in] 
      //        const ColdflowMixingModel&   
      //
      ColdflowMixingModel& operator=(const ColdflowMixingModel&);

private:

      int d_numMixingVars;
      double d_denUnderrelax;
      IntVector d_denRef;
      std::vector<Stream> d_streams; 

}; // end class ColdflowMixingModel

} // end namespace ArchesSpace
} // end namespace Uintah

#endif

//
// $Log$
// Revision 1.1  2001/01/15 23:38:21  rawat
// added some more classes for implementing mixing model
//
// Revision 1.1  2000/12/18 17:53:10  rawat
// adding mixing model for reacting flows
//
//
