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
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Integrator.h>

#include <vector>

namespace Uintah {

class ColdflowMixingModel: public MixingModel {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructor taking
      //   [in] 
      //
      ColdflowMixingModel();

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
      inline Stream speciesStateSpace(const std::vector<double>& mixVar) {
	Stream noStream;
	return noStream;
      }

      // GROUP: Get Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Get the number of mixing variables
      //
      inline int getNumMixVars() const{ 
	return d_numMixingVars; 
      }
      inline bool isAdiabatic() const{ 
	return true;
      }
      inline int getNumMixStatVars() const{
	return 0;
      }
      inline int getNumRxnVars() const{
	return 0;
      }
      inline int getTableDimension() const{
	return 0;
      }
      //***warning** compute totalvars from number of species and dependent vars
      inline int getTotalVars() const {
	return 0;
      }
      inline ReactionModel* getRxnModel() const {
      }
      inline Integrator* getIntegrator() const {
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
      std::vector<Stream> d_streams; 

}; // end class ColdflowMixingModel

} // end namespace Uintah

#endif

//
// $Log$
// Revision 1.8  2001/11/28 23:51:03  spinti
// Return stream class in inline function speciesStateSpace.
//
<<<<<<< ColdflowMixingModel.h
=======
// Revision 1.7  2001/11/28 23:43:10  jas
// Commented out the return 0 in the speciesStateSpace since there is no
// conversion from int to Stream.  This needs to be fixed by someone who
// knows the proper fix.  The arches code will now compile.
//
// Revision 1.6  2001/11/27 23:29:01  spinti
// Added "return 0" to function speciesStateSpace.
//
>>>>>>> 1.7
// Revision 1.5  2001/11/08 19:13:44  spinti
// 1. Corrected minor problems in ILDMReactionModel.cc
// 2. Added tabulation capability to StanjanEquilibriumReactionModel.cc. Now,
//    a reaction table is created dynamically. The limits and spacing in the
//    table are specified in the *.ups file.
// 3. Corrected the mixture temperature computation in Stream::addStream. It
//    now is computed using a Newton search.
// 4. Made other minor corrections to various reaction model files.
//
// Revision 1.2  2001/08/25 07:32:45  skumar
// Incorporated Jennifer's beta-PDF mixing model code with some
// corrections to the equilibrium code.
// Added computation of scalar variance for use in PDF model.
// Properties::computeInletProperties now uses speciesStateSpace
// instead of computeProps from d_mixingModel.
//
// Revision 1.1  2001/01/31 16:35:30  rawat
// Implemented mixing and reaction models for fire.
//
// Revision 1.1  2001/01/15 23:38:21  rawat
// added some more classes for implementing mixing model
//
// Revision 1.1  2000/12/18 17:53:10  rawat
// adding mixing model for reacting flows
//
//
