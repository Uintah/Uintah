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

#include <Packages/Uintah/CCA/Components/Arches/Mixing/InletStream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/ReactionModel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  class Integrator;

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
 
      // GROUP: Actual Action Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Carry out actual computation of properties
      //
      virtual void computeProps(const InletStream& inStream,
				Stream& outStream) = 0;

      /////////////////////////////////////////////////////////////////////////
      // speciesStateSpace returns the state space (dependent) variables,
      // including species composition, given a set of mixture fractions. The 
      // species composition of each stream must be known.
      // The state space variables returned are: density, temperature, heat 
      // capacity, molecular weight, enthalpy, mass fractions 
      // All variables are in SI units with pressure in Pascals.
      // Parameters:
      // [in] mixVar is an array of independent variables
      virtual Stream speciesStateSpace(const std::vector<double>& mixVar) = 0;

      // GROUP: Get Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      //
      virtual bool isAdiabatic() const = 0;
      virtual int getNumMixVars() const = 0; // Get number of mixing variables
      virtual int getNumMixStatVars() const = 0;
      virtual int getNumRxnVars() const = 0;
      virtual int getTableDimension() const = 0;
      virtual std::string getMixTableType() const = 0;
      virtual int getTotalVars() const = 0;
      virtual ReactionModel* getRxnModel() const = 0;
      virtual Integrator* getIntegrator() const = 0;


protected :

private:

}; // end class MixingModel

} // end namespace Uintah

#endif

