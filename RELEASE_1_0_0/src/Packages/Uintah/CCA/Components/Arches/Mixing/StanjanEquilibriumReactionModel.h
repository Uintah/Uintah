#ifndef Uintah_Component_Arches_StanjanEquilibriumReactionModel_h
#define Uintah_Component_Arches_StanjanEquilibriumReactionModel_h

/****************************************************************************
CLASS
    StanjanEquilibriumReactionModel
	The StanjanEquilibriumReactionModel class computes chemical equilibrium 
	for a system.

GENERAL INFORMATION
    StanjanEquilibriumReactionModel.h - Declaration of 
        StanjanEquilibriumReactionModel class

    Author: Diem-Phuong Nguyen (diem@crsim.utah.edu)
    Modified by: Jennifer Spinti (spinti@crsim.utah.edu)

    Creation Date: 30 Mar 2000
    Last Modified: 22 Dec 2000
 
    C-SAFE

    Copyright U of U 2000

KEYWORDS
    Reaction_Model, Equilibrium, Stanjan

DESCRIPTION
    The StanjanEquilibriumReactionModel class is derived from the ReactionModel 
    base class. The input required is a set of mixing variables and a normalized 
    heat loss.
  
    This class then computes chemical equilibrium for the given system based on  
    the Stanjan equilibrium package.  This class returns the values for the state 
    state space variables at chemical equilibrium.  These state space values 
    including temperature, pressure, density, mixture molecular weight, heat 
    capacity,and species mole fractions.

PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS:
   1. Increase flexibility by adding other equilibrium capabilities.  Add the 
   NASA equilibrium code.
   2. **Add in gas phase radiation properties, which are currently commented out


  ***************************************************************************/

#include <Packages/Uintah/CCA/Components/Arches/Mixing/ReactionModel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <vector>

using namespace std;
namespace Uintah {
  class KD_Tree;
  class InputState;
  class ChemkinInterface;
  class Stream;
  // Reference temperature defined to be the lower limit of integration in the
  // determination of the system sensible enthalpy
  const double TREF = 298.0;

  class StanjanEquilibriumReactionModel: public ReactionModel {
  public:
    // GROUP: Constructors:
    /////////////////////////////////////////////////////////////////////////
    //
    // Constructs an instance of StanjanEquilibriumReactionModel given the number of 
    // independent mixing variables and the identification of the system as 
    // either adiabatic or nonadiabatic. Additionally, the total number of state
    // space variables are specified. This constructor also calls the 
    // constructor of ReactionModel.
    // PRECONDITIONS
    //  numMixVar is positive.
    //  numStateSpaceVar is positive.
    // POSTCONDITIONS
    //  This is a properly constructed instance of 
    //  StanjanEquilibriumReactionModel.
    //
    // Constructor taking
    //   [in] adiabatic If =1, system is adiabatic; otherwise, its nonadiabatic.

    StanjanEquilibriumReactionModel(bool adiabatic);

    // GROUP: Destructor:
    /////////////////////////////////////////////////////////////////////////
    //
    // Destructor
    //
    ~StanjanEquilibriumReactionModel();

    // GROUP: Problem Setup :
    ///////////////////////////////////////////////////////////////////////
    //
    // problemSetup performs functions required to run the reaction model that
    // don't explicitly involve assigning memory
    //
    virtual void problemSetup(const ProblemSpecP& params);

    // GROUP: Access function
    //////////////////////////////////////////////////////////////////////
    // returns the pointer to chemkin interface class
    virtual ChemkinInterface* getChemkinInterface() {
      return d_reactionData;
    }


    // GROUP: Manipulators
    /////////////////////////////////////////////////////////////////////////
    //
    //computeEnthalpy returns the enthalpies (J/kg) used in the linearization 
    // of the local enthalpy(absH), given values for the mixing variables. The 
    // linearization can be written as:
    //     absH = adH + gamma*sensH 
    // where adH is the adiabatic enthalpy and sensH is the sensible enthalpy. 
    // Gamma is the normalized residual enthalpy (resH) and is one of the 
    // independent variables (mixRxnVar). It is defined as:
    //     resH = absH - adH
    //     gamma = resH/sensH
    // This formulation will not work when adH = 0 (e.g. mixtures containing 
    // only O2 and/or N2).
    // This routine computes the adiabatic enthalpy and adiabatic flame 
    // temperature (AFT)using stanjan.  It calculates the sensible enthalpy
    // based on the AFT using the Chemkin routine ckhbms.
    //
    virtual Stream computeRxnStateSpace(Stream& mixRxnVar);
#if 0
    virtual Stream computeEnthalpy(Stream& mixRxnVar);
#endif

  private:
    void computeEquilibrium(double tin, double initPress, int heatLoss,
			    double enthalpy, const vector<double>& initMassFract,
			    Stream& equilSoln);
    
    void computeRadiationProperties();
    // Class object that stores all the information about the reaction
    // mechanism read in through Chemkin including species, elements, reaction
    // rates, and thermodynamic information.
    ChemkinInterface* d_reactionData; 

    bool d_adiabatic;
    // Determines how mixing variables are defined. If = 1, mixing variables are
    // treated independently
    // int d_lMixInd;  
    // Optical path length (m?)- a characteristic length used to 
    // estimate radiation
    // gas absorption coefficients
    // double d_opl;
 
  }; // End Class StanjanEquilibriumReactionModel


  // Fortran subroutine for calculating turbulent gas adsorption coefficients
  extern "C" {void rpropi_(double *er, double *sm, double *tk, double *b0,
			   double *s2, double *rhop, int *idco2, int *idh2o,
			   int *ilc, double *d_opl,
			   double *pa, double *abkg, double *emb); }

} // end namespace Uintah

#endif





