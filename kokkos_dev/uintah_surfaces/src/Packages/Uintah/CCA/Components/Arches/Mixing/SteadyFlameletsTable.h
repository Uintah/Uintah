//----- SteadyFlameletsTable.h --------------------------------------------------

#ifndef Uintah_Component_Arches_SteadyFlameletsTable_h
#define Uintah_Component_Arches_SteadyFlameletsTable_h

/***************************************************************************
CLASS
    SteadyFlameletsTable 
       
GENERAL INFORMATION
    StadyFlameletsTable.h - Declaration of SteadyFlameletsTable class

    Author: Padmabhushana R Desam (desam@crsim.utah.edu)
    
    Creation Date : 09-16-2003

    C-SAFE
    
    Copyright U of U 2000

KEYWORDS
    Mixing Table 
DESCRIPTION
      Reads the mixing tables with flamelets, created by a program written by James Sutherland. 
      The current implementation is to read specific tables. Future revisons will focus 
      on making this more generic and different mixing tables will be standarized with an 
      interface. 
    
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
  
class SteadyFlameletsTable: public MixingModel{

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructs an instance of SteadyFlamletsTable
      //
      SteadyFlameletsTable();
      // Constructor with thermal NOx flag
      SteadyFlameletsTable(bool d_thermalNOx);
      

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Destructor
      //
      virtual ~SteadyFlameletsTable();

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

      inline bool getCOOutput() const{
	return 0;
      }
      inline bool getSulfurChem() const{
	return 0;
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
      //        const SteadyFlameletsTable&   
      //
      SteadyFlameletsTable(const SteadyFlameletsTable&);

      // GROUP: Operators Not Instantiated:
      ///////////////////////////////////////////////////////////////////////
      //
      // Assignment Operator (never instantiated)
      //   [in] 
      //        const SteadyFlameletsTable&   
      //
      SteadyFlameletsTable& operator=(const SteadyFlameletsTable&);

private:
      // Looks for needed entry in table and returns that entry.  
      void tableLookUp(double mixfrac, double mixfracVars, double scalDisp, Stream& outStream); 
      double chitableLookUp(double mixfrac, double mixfracVars);
      void readMixingTable(std::string inputfile);
      void readChiTable();
      int d_numMixingVars;
      int d_numMixStatVars;
      int d_numRxnVars;
      int d_depStateSpaceVars;
      bool d_adiabatic;
      int d_tableDimension;
      std::vector <std::vector <double> > table;
      std::vector <double> chitable;
      std::vector <std::string> tags;
      int d_scaldispcount, d_mixfraccount, d_mixvarcount,d_varcount;
      // For chi table
      int dc_mixfraccount, dc_mixvarcount;
      double mixfrac_Div,mixvar_Div;
      int co2_index, h2o_index, c2h2_index, NO_index;
      std::vector<double> meanMix;
      std::vector<double> scalarDisp;
      std::vector<std::string> variables_list;
      bool d_calcthermalNOx; // Flag for the thermal NOx
}; // end class SteadyFlameletsTable

} // end namespace Uintah

#endif








