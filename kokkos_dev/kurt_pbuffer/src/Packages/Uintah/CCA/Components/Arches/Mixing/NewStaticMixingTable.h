//----- NewStaticMixingTable.h --------------------------------------------------

#ifndef Uintah_Component_Arches_NewStaticMixingTable_h
#define Uintah_Component_Arches_NewStaticMixingTable_h

/***************************************************************************
CLASS
    NewStaticMixingTable 
       
GENERAL INFORMATION
    NewStaticMixingTable.h - Declaration of NewStaticMixingTable class

    Author: Padmabhushana R Desam (desam@crsim.utah.edu)
    
    Creation Date : 08-14-2003

    C-SAFE
    
    Copyright U of U 2000

KEYWORDS
    Mixing Table 
DESCRIPTION
      Reads and interpolates the mixing reaction tables created to standard format specifications 
      Currently supports non-adiabatic equilibrium tables(3-Dimensions). 
    
PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
	Read and interpolate 4-D tables e.g., non-adiabatic flamelets/extent of reaction tables 
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
  
class NewStaticMixingTable: public MixingModel{

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructs an instance of NewStaticMixingTable
      //
      NewStaticMixingTable();

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Destructor
      //
      virtual ~NewStaticMixingTable();

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
	return d_co_output;
      }
      inline bool getSulfurChem() const{
	return d_sulfur_chem;
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
      //        const NewStaticMixingTable&   
      //
      NewStaticMixingTable(const NewStaticMixingTable&);

      // GROUP: Operators Not Instantiated:
      ///////////////////////////////////////////////////////////////////////
      //
      // Assignment Operator (never instantiated)
      //   [in] 
      //        const NewStaticMixingTable&   
      //
      NewStaticMixingTable& operator=(const NewStaticMixingTable&);

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
      int d_heatlosscount, d_mixfraccount, d_mixvarcount, d_varscount;
      int co2_index, h2o_index;

      int h2s_index, so2_index, so3_index, co_index;
      bool d_co_output;
      bool d_sulfur_chem;

      std::vector <std::vector<double> > meanMix;
      std::vector<double> heatLoss;
      std::vector<double> variance;
      std::vector<int> eachindepvarcount;
      std::vector<std::string> indepvars_names;
      std::vector<std::string> vars_names;
      std::vector<std::string> vars_units;
      int d_indepvarscount;
      int Hl_index, F_index, Fvar_index;
      int T_index, Rho_index, Cp_index, Enthalpy_index, Hs_index;
      double d_H_fuel, d_H_air;
      bool d_adiab_enth_inputs;
}; // end class NewStaticMixingTable

} // end namespace Uintah

#endif








