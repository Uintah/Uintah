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
    base class. The input required is a set of mixing variables and a linearized 
    heat loss.
  
    This class then computes chemical equilibrium for the given system based on  
    the Stanjan equilibrium package.  This class returns the values for the state 
    state space variables at chemical equilibrium.  These state space values 
    including temperature, pressure, density, mixture molecular weight, heat 
    capacity,and species mole fractions.
 
    The local (absolute) enthalpy is linearized using the adiabatic and sensible 
    enthalpies (J/kg).  The sensible enthalpy is chosen as an arbitrary enthalpy 
    whereas the adiabatic enthalpy is chosen for the standardized constant 
    enthalpy. The linearization can be written as:
         absH = adH + gamma*sensH 
    where adH is the adiabatic enthalpy and sensH is the sensible enthalpy. 
    Gamma is the normalized residual enthalpy (resH) and is one of the 
    independent variables (mixRxnVar). It is defined as:
        resH = absH - adH
        gamma = resH/sensH
    This formulation will not work when adH = 0 (e.g. mixtures containing 
    only O2 and/or N2).

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
#include <Packages/Uintah/CCA/Components/Arches/Mixing/DynamicTable.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  class ChemkinInterface;
  class Stream;
  class MixingModel;
  class MixRxnTable;
  class MixRxnTableInfo;
  // Reference temperature defined to be the lower limit of integration in the
  // determination of the system sensible enthalpy
  const double TREF = 200.0;
  const double TLOW = 100.00;
  const double THIGH = 4000.0;
  const int MAXITER = 1000;
 

  class StanjanEquilibriumReactionModel: public ReactionModel, public DynamicTable{
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
    virtual void problemSetup(const ProblemSpecP& params, 
			      MixingModel* mixModel);

    // GROUP: Access function
    //////////////////////////////////////////////////////////////////////
    // returns the pointer to chemkin interface class
    inline ChemkinInterface* getChemkinInterface() const{
      return d_reactionData;
    }
    inline bool getSootBool() const{
      return d_lsoot;
    }
    inline int getTotalDepVars() const{
      return d_depStateSpaceVars;
    }

    // GROUP: Actual Action Methods :
    ///////////////////////////////////////////////////////////////////////
    //
    // Gets the state space (dependent) variables by  interpolation from a 
    // table using the values of the independent variables
    //
    virtual void getRxnStateSpace(const Stream& unreactedMixture, 
				  std::vector<double>& varsHFPi, 
				  Stream& reactedStream);
    // Computes the state space (dependent) variables using the Stanjan
    // equilibrium code given the unreacted stream information and values 
    // for the reaction variables
    virtual void computeRxnStateSpace(const Stream& unreactedMixture, 
				      const std::vector<double>& mixRxnVar, 
				      Stream& equilStateSpace);
    virtual double computeTemperature(const double absEnthalpy, 
				      const std::vector<double>& massFract, 
				      double initTemp);   

  private:
    // Looks for needed entry in KDTree and returns that entry. If entry 
    // does not exist, calls integrator to compute entry before returning it.
    void tableLookUp(int* tableKeyIndex, Stream& equilStateSpace);
    void convertIndextoValues(int* tableKeyIndex);
    void computeEquilibrium(double initTemp, double initPress,
			    const std::vector<double>& initMassFract, 
			    Stream& equilSoln);
    void computeRadiationProperties();
    // Class object that stores all the information about the reaction
    // mechanism read in through Chemkin including species, elements, reaction
    // rates, and thermodynamic information.
    ChemkinInterface* d_reactionData;
    MixingModel* d_mixModel;
    bool d_adiabatic;
    int d_rxnTableDimension;
    int d_depStateSpaceVars;
    bool d_lsoot;
    std::vector<double> d_indepVars;
    MixRxnTableInfo* d_rxnTableInfo;
    // Data structure class that stores the table entries for state-space
    // variables as a function of independent variables.
    // This could be implemented either as a k-d or a binary tree data structure.
    MixRxnTable* d_rxnTable;
 
  }; // End Class StanjanEquilibriumReactionModel


  // Fortran subroutine for calculating turbulent gas adsorption coefficients
  extern "C" {void rpropi_(double *er, double *sm, double *tk, double *b0,
			   double *s2, double *rhop, int *idco2, int *idh2o,
			   int *ilc, double *d_opl,
			   double *pa, double *abkg, double *emb); }

} // end namespace Uintah

#endif





