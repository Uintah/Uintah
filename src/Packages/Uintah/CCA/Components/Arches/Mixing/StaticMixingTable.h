//----- StaticMixingTable.h --------------------------------------------------

#ifndef Uintah_Component_Arches_StaticMixingTable_h
#define Uintah_Component_Arches_StaticMixingTable_h

/***************************************************************************
CLASS
    StaticMixingTable 
       
GENERAL INFORMATION
    StaticMixingTable.h - Declaration of StaticMixingTable class

    Author: Padmabhushana R Desam (desam@crsim.utah.edu)
    
    Creation Date : 08-14-2003

    C-SAFE
    
    Copyright U of U 2000

KEYWORDS
    Mixing Table 
DESCRIPTION
      Reads the mixing reaction tables created by standalone programs. The current implementation
      is to read specific tables. Future revisons will focus on making this more generic and
      different mixing tables will be standarized with an interface. 
    
PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
    Making it more generic to accomodate different mixing table formats
 
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
  
class StaticMixingTable: public MixingModel{

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructs an instance of StaticMixingTable
      //
      StaticMixingTable();

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Destructor
      //
      virtual ~StaticMixingTable();

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
	return 0;
      }
      //***warning** compute totalvars from number of species and dependent vars
      inline int getTotalVars() const {
	return 0;
      }
      inline ReactionModel* getRxnModel() const {
	return 0;
      }      
      inline Integrator* getIntegrator() const {
	return 0;
      }

protected :

private:

      ///////////////////////////////////////////////////////////////////////
      //
      // Copy Constructor (never instantiated)
      //   [in] 
      //        const StaticMixingTable&   
      //
      StaticMixingTable(const StaticMixingTable&);

      // GROUP: Operators Not Instantiated:
      ///////////////////////////////////////////////////////////////////////
      //
      // Assignment Operator (never instantiated)
      //   [in] 
      //        const StaticMixingTable&   
      //
      StaticMixingTable& operator=(const StaticMixingTable&);

private:
      // Looks for needed entry in table and returns that entry. If entry 
      // does not exist and table is dynamic, it calls integrator to compute
      // entry before returning it. If table is static and entry is non-existent,
      // it exits program.
      double tableLookUp(double mixfrac, double mixfracVars, double heat_loss, int var_index); 
      void readMixingTable(std::string inputfile);
      int d_numMixingVars;
      int d_numMixStatVars;
      int d_numRxnVars;
      int d_depStateSpaceVars;
      bool d_adiabatic;
      int d_tableDimension;
      std::vector <std::vector <double> > table;
      std::vector <std::string> tags;
      int d_enthalpycount, d_mixfraccount, d_mixvarcount, d_speciescount,d_varcount, mixfrac_size;
      int co2_index, h2o_index;
      std::vector <std::vector<double> > meanMix;
      std::vector<double> meanVars;
      std::vector<double> enthalpyLoss;
      std::vector<std::string> species_list;
      bool d_nonadiabatic_table;
}; // end class StaticMixingTable

} // end namespace Uintah

#endif








