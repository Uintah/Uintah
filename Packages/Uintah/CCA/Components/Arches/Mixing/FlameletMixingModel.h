//----- FlameletMixingModel.h --------------------------------------------------

#ifndef Uintah_Component_Arches_FlameletMixingModel_h
#define Uintah_Component_Arches_FlameletMixingModel_h

/***************************************************************************
CLASS
    FlameletMixingModel
       Sets up the FlameletMixingModel ????
       
GENERAL INFORMATION
    FlameletMixingModel.h - Declaration of FlameletMixingModel class

    Author: Rajesh Rawat (rawat@crsim.utah.edu)
    Revised: Jennifer Spinti (spinti@crsim.utah.edu)
    
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
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/DynamicTable.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  class InletStream;
  
class FlameletMixingModel: public MixingModel{

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructs an instance of FlameletMixingModel
      //   [in] 
      //
      FlameletMixingModel();

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Destructor
      //
      virtual ~FlameletMixingModel();

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

      /////////////////////////////////////////////////////////////////////////
      // speciesStateSpace returns the state space (dependent) variables,
      // including species composition, given a set of mixture fractions. The 
      // species composition of each stream must be known.
      // The state space variables returned are: density, temperature, heat 
      // capacity, molecular weight, enthalpy, mass fractions 
      // All variables are in SI units with pressure in Pascals.
      // Parameters:
      // [in] mixVar is an array of independent variables
      virtual Stream speciesStateSpace(const std::vector<double>& mixVar) {
	Stream noStream;
	return noStream;
      }

      // GROUP: Get Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Get the number of mixing variables
      //
      inline bool isAdiabatic() const{ 
	return d_adiabatic; 
      }
      inline int getNumMixVars() const{ 
	return d_numMixingVars; 
      }
      inline int getNumMixStatVars() const{
	return d_numMixStatVars;
      }
      inline int getNumRxnVars() const{
	return d_numRxnVars;
      }
      inline int getTableDimension() const{
	return 0;
      }
      inline std::string getMixTableType() const{
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
      //        const FlameletMixingModel&   
      //
      FlameletMixingModel(const FlameletMixingModel&);

      // GROUP: Operators Not Instantiated:
      ///////////////////////////////////////////////////////////////////////
      //
      // Assignment Operator (never instantiated)
      //   [in] 
      //        const FlameletMixingModel&   
      //
      FlameletMixingModel& operator=(const FlameletMixingModel&);

private:
      // Looks for needed entry in table and returns that entry. If entry 
      // does not exist and table is dynamic, it calls integrator to compute
      // entry before returning it. If table is static and entry is non-existent,
      // it exits program.
      void tableLookUp(double mixfrac, double mixfracVars, int axial_loc,
		       Stream& outStream);
      void readFlamelet();
      int d_numMixingVars;
      int d_numMixStatVars;
      int d_numRxnVars;
      int d_depStateSpaceVars;
      bool d_adiabatic;
      int d_tableDimension;
      std::vector <std::vector <double> > table;
      std::vector <std::string> tags;
      int d_numVars, d_numMixfrac, d_numMixvars, d_numAxialLocs;
      std::vector<double> meanMix;
      std::vector<double> meanVars;
      std::vector<int> meanAxialLocs;
}; // end class FlameletMixingModel

} // end namespace Uintah

#endif








