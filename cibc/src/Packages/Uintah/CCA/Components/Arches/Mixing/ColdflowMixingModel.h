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

#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixingModel.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

class ColdflowMixingModel: public MixingModel {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructor taking
      //   [in] 
      //
      ColdflowMixingModel(bool calcReactingScalar,
                          bool calcEnthalpy,
                          bool calcVariance);

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
      virtual void computeProps(const InletStream& inStream,
				Stream& outStream);
      // GROUP: Get Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Get the number of mixing variables
      //
      inline bool getCOOutput() const{
	return 0;
      }
      inline bool getSulfurChem() const{
	return 0;
      }
      inline bool getSootPrecursors() const{
        return 0; 
      }

      inline double getAdiabaticAirEnthalpy() const{
	return 0.0;
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

      bool d_calcReactingScalar, d_calcEnthalpy, d_calcVariance;
      int d_numMixingVars;
      std::vector<Stream> d_streams; 

}; // end class ColdflowMixingModel

} // end namespace Uintah

#endif
