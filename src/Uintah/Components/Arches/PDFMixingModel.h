//----- PDFMixingModel.h --------------------------------------------------

#ifndef Uintah_Component_Arches_PDFMixingModel_h
#define Uintah_Component_Arches_PDFMixingModel_h

/***************************************************************************
CLASS
    PDFMixingModel
       Sets up the PDFMixingModel ????
       
GENERAL INFORMATION
    PDFMixingModel.h - Declaration of PDFMixingModel class

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

class PDFMixingModel: public MixingModel {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructor taking
      //   [in] 
      //
      PDFMixingModel(const ArchesLabel* label);

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Destructor
      //
      virtual ~PDFMixingModel();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      //
      // Set up the problem specification database
      //
      virtual void problemSetup(const ProblemSpecP& params);

      // GROUP: Compute properties 
      ///////////////////////////////////////////////////////////////////////
      //
      // Compute properties for inlet/outlet streams
      //
      virtual void computeInletProperties(const std::vector<double>&
					  mixfractionStream,
					  Stream& inletStream);

      // GROUP: Actual Action Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Carry out actual computation of properties
      //
      virtual void computeProps(const ProcessorGroup*,
				InletStream& inStream,
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
      //        const PDFMixingModel&   
      //
      PDFMixingModel(const PDFMixingModel&);

      // GROUP: Operators Not Instantiated:
      ///////////////////////////////////////////////////////////////////////
      //
      // Assignment Operator (never instantiated)
      //   [in] 
      //        const PDFMixingModel&   
      //
      PDFMixingModel& operator=(const PDFMixingModel&);

private:

      int d_numMixingVars;
      double d_denUnderrelax;
      IntVector d_denRef;
      std::vector<Stream> d_streams; 

}; // end class PDFMixingModel

} // end namespace ArchesSpace
} // end namespace Uintah

#endif

//
// $Log$
// Revision 1.1  2000/12/18 17:53:10  rawat
// adding mixing model for reacting flows
//
//
