#ifndef Uintah_Component_Arches_ChemkinInterface_h
#define Uintah_Component_Arches_ChemkinInterface_h
/****************************************************************************
   CLASS
      ChemkinInterface
        This class provides the interface to all Chemkin subroutines. The
	thermodynamic and chemical kinetics files are read first and a binary 
	output file is created. This binary file is then read into the
	appropriate work arrays for interfacing with all Chemkin routines.
	
   GENERAL INFORMATION
      ChemkinInterface.h - Declaration of ChemkinInterface class

      Author: Jennifer Spinti (spinti@crsim.utah.edu) & Rajesh Rawat

      Creation Date: 19 October 1999
 
      C-SAFE

      Copyright U of U 1999

   KEYWORDS
      Chemkin, Reaction_Model

   DESCRIPTION
     The ChemkinInterface class provides the only access to Chemkin sub-
     routines, including subroutines which compute equilibrium (stanjan). 
     If a reaction model uses Chemkin to read in mechanism date, compute 
     reaction rates, and/or compute mixture properties, etc., it can only 
     access the appropriate Chemkin routines through this class. No direct 
     calls to Chemkin routines should be made in any other class.

     NOTE: If the total number of species or elements in a given mechanism
     exceeds 100, the dimensioning of several arrays in ChemkinHacks.f will
     have to be modified.

   PATTERNS
      None

   WARNINGS
      None

   POSSIBLE REVISIONS:

  ***************************************************************************/
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  using namespace std;
  const int NAMELENGTH = 20;
  // This constant is used to dimension char arrays which will be compared to
  // the data type "CharArray" returned by some Chemkin subroutines.
  const int CHARLENGTH = 16;
  class ChemkinInterface {

  public:
    // GROUP: Constructors:
    /////////////////////////////////////////////////////////////////////////
    //
    // Constructs an instance of ChemkinInterface. No variables are specified,
    // but one files is required input: chem.bin. This file is the output of
    // ckinterp, which must be run a priori
    // PRECONDITIONS
    //  None
    // POSTCONDITIONS
    //  This is a properly constructed instance of ChemkinInterface.
    ChemkinInterface();

    // GROUP: Destructor:
    /////////////////////////////////////////////////////////////////////////
    //
    // Destructor
    //
    ~ChemkinInterface();

    // GROUP: Access functions
    //////////////////////////////////////////////////////////////////////
    // getOutFile returns the Fortran unit number assigned to output file
    // 
    int getOutFile () const;
    //////////////////////////////////////////////////////////////////////
    // getNumElements returns the number of elements in reaction mechanism
    // 
    int getNumElements () const;
    //////////////////////////////////////////////////////////////////////
    // getNumSpecies returns the number of species in reaction mechanism
    // 
    int getNumSpecies () const;
    //////////////////////////////////////////////////////////////////////
    // getNumRxns returns the number of reactions in reaction mechanism
    // 
    int getNumRxns () const;
    //////////////////////////////////////////////////////////////////////
    // getNFit returns the number of coefficients in fits to thermodynamic 
    // data
    int getNFit () const;
    //////////////////////////////////////////////////////////////////////
    // getElementNames returns the pointer to the two-dimensional array 
    // containing the names of elements in reaction mechanism
    // 
    //void getElementNames(char** elementNames);
    //////////////////////////////////////////////////////////////////////
    // getSpeciesNames returns the pointer to the two-dimensional array 
    // containing the names of species in reaction mechanism
    // 				      
    //void getSpeciesNames(char** speciesNames);
    //////////////////////////////////////////////////////////////////////
    // getAtomicWeight returns the array containing the atomic weights of 
    // the elements
    //
    void getAtomicWeight(double *atomicWeight);
    //////////////////////////////////////////////////////////////////////
    // getMoleWeight returns the array containing the molecular weights of 
    // the species 
    //
    void getMoleWeight(double *moleWeight);
    //////////////////////////////////////////////////////////////////////
    // getLeniwk returns the length of the integer work array used 
    // internally by Chemkin.
    //int getLeniwk() const;
    //////////////////////////////////////////////////////////////////////
    // getLenrwk returns the length of the real work array used internally 
    // by Chemkin.
    //int getLenrwk() const;
    //////////////////////////////////////////////////////////////////////
    // getIntWorkArray returns the integer work array used internally by
    // Chemkin. This array is required for all calls to Chemkin subroutines
    //void getIntWorkArray(int *ickwrk) const;
    //////////////////////////////////////////////////////////////////////
    // getRealWorkArray returns the real work array used internally by
    // Chemkin. This array is required for all calls to Chemkin subroutines
    //void getRealWorkArray(double *rckwrk) const;

    // GROUP: Manipulators
    // Returns index of given species in species name list
    int getSpeciesIndex(char *name);
    // Returns index of given element in element name list
    int getElementIndex(char *name);
    /////////////////////////////////////////////////////////////////////////
    // getMixMoleWeight returns the mean molecular weight of a gas mixture
    // given the species mass fractions
    //
    double getMixMoleWeight(vector<double> Yvec);
    //////////////////////////////////////////////////////////////////////
    // getMixEnthalpy returns the mean enthalpy of a mixture in mass units 
    // (J/kg) given temperature(K) and species mass fractions
    //
    double getMixEnthalpy(double temp, vector<double> Yvec);
    //////////////////////////////////////////////////////////////////////
    // getMixSpecificHeat returns the mean specific heat at constant pressure
    // (J/kg*K) given temperature(K) and species mass fractions
    //
    double getMixSpecificHeat(double temp, vector<double> Yvec);
    //////////////////////////////////////////////////////////////////////
    // getMassDensity returns the mass density (kg/m^3)of a gas mixture
    // given the pressure(Pa), temperature(K), and species mass fractions
    //
    double getMassDensity(double press, double temp, 
			  vector<double> Yvec);
    //////////////////////////////////////////////////////////////////////
    // convertMolestoMass returns the array of species mass fractions
    // given the mole fractions
    //
    vector<double> convertMolestoMass(vector<double> Xvec); 
    //////////////////////////////////////////////////////////////////////
    // convertMasstoMoles returns the array of species mole fractions
    // given the mass fractions
    //
    vector<double> convertMasstoMoles(vector<double> Yvec);
    //////////////////////////////////////////////////////////////////////
    // getSpeciesEnthalpy returns the array of enthalpies for the species
    // in mass units (J/kg)
    //
    void getSpeciesEnthalpy(double temp, double *speciesEnthalpy);
    //////////////////////////////////////////////////////////////////////
    // getMolarRates returns the molar production rates (kmoles/(m^3*s)of 
    // the species given the temperature(K), pressure(Pa), and mass fractions
    //
    void getMolarRates(double press, double temp, vector<double> Yvec, 
		       double *wdot);


    //Variables needed by Stanjan equilibrium model
    // Array of element names
    char **d_elementNames;
    // Array of species names
    char **d_speciesNames;
    // Integer work array
    int *d_ickwrk;
    // Real work array
    double *d_rckwrk;

  private:
    // Unit number for Fortran output file
    int d_lout;
    // Total number of elements in mechanism
    int d_numElements;
    // Total number species in mechanism
    int d_numSpecies;
    // Total number of reactions in mechanism
    int d_numRxns;
    // Number of coefficients in fits to thermodynamic data
    int d_nfit;
    // Array of species names
    //char **d_speciesNames;
    // Array of element names
    //char **d_elementNames;
    // Atomic weights of the elements
    double *d_atomicWeight;
    // Molecular weights of the species
    double *d_moleWeight;
    // Length of real work array
    int d_lenrwk;
    // Length of integer work array
    int d_leniwk;
    // Length of character work array
    int d_lencwk;
    // Real work array
    //double *d_rckwrk;
    // Integer work array
    //int *d_ickwrk;
    // Character work array
    char **d_cckwrk;

  }; // End Class ChemkinInterface
} // end namespace uintah

#endif



