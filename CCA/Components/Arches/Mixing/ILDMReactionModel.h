#ifndef Uintah_Component_Arches_ILDMReactionModel_h
#define Uintah_Component_Arches_ILDMReactionModel_h

/****************************************************************************
CLASS
    ILDMReactionModel
	The ILDMReactionModel class uses a one-dimensional manifold to 
	determine the state of a system.

GENERAL INFORMATION
    ILDMReactionModel.h - Declaration of 
       ILDMReactionModel class

    Author: Jennifer Spinti (spinti@crsim.utah.edu)

    Creation Date: 4 June 2001
    Last Modified: 11 June 2001
 
    C-SAFE

    Copyright U of U 2001

KEYWORDS
    Reaction_Model, ILDM

DESCRIPTION
    The ILDMReactionModel class is derived from the ReactionModel 
    base class. The input required is a set of mixing variables, one normalized
    reaction variable, and a normalized heat loss.

    Data files being read in are set up as follows. The word ENTRY appear at the
    start of each new data set , i.e. each (h,f) pair. The first line after ENTRY
    contains four values: h, f, max value of parameter, min value of parameter.
    The rest of the lines are the state space vector at each value of the parameter.
    Note- This class assumes that NUM_DEP_VARS will alway be 7; if it changes, 
    modifications must be made to this class.
  
    This class then computes the chemical state of the system using tabulated data 
    generated using the ILDM technique. This class returns the values for the state 
    state space variables including temperature, pressure, density, mixture 
    molecular weight, heat capacity,species mole fractions, and the rate for the
    reaction variable.

PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS:
   1.


  ***************************************************************************/

#include <Packages/Uintah/CCA/Components/Arches/Mixing/ReactionModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/DynamicTable.h>

#include <vector>

using namespace std;
namespace Uintah {
  class KD_Tree;
  class PDFMixingModel; //Change to PDFMixingModel???
  class Stream;
 
  // Reference temperature defined to be the lower limit of integration in the
  // determination of the system sensible enthalpy

  //const double TREF = 298.0; //defined in StanjanEquilibriumReactionModel

  class ILDMReactionModel: public ReactionModel, public DynamicTable {

  public:

    // GROUP: Constructors:
    /////////////////////////////////////////////////////////////////////////
    //
    // Constructor taking
    //   [in] adiabatic If =1, system is adiabatic; otherwise, its nonadiabatic.
    ILDMReactionModel(bool adiabatic);

    // GROUP: Destructor:
    /////////////////////////////////////////////////////////////////////////
    //
    // Destructor
    //
    ~ILDMReactionModel();

    // GROUP: Problem Setup :
    ///////////////////////////////////////////////////////////////////////
    //
    // problemSetup performs functions required to run the reaction model that
    // don't explicitly involve assigning memory
    //
    virtual void problemSetup(const ProblemSpecP& params,
			      MixingModel* mixModel);

    // GROUP: Get Methods :
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

    // GROUP: Manipulators
    ///////////////////////////////////////////////////////////////////////
    //
    // Computes the state space (dependent) variables given the unreacted
    // stream information and values for the reaction variables
    //
    virtual Stream computeRxnStateSpace(Stream& unreactedMixture, 
					std::vector<double>& varsHFPi,
					bool adiabatic);  


  private:
    // Looks for needed entry in KDTree and returns that entry. If entry 
    // does not exist, calls integrator to compute entry before returning it.
    Stream tableLookUp(int* tableKeyIndex);

    // Class object that stores all the information about the reaction
    // mechanism read in through Chemkin including species, elements, reaction
    // rates, and thermodynamic information.
    ChemkinInterface* d_reactionData;
    bool d_adiabatic;
    bool d_lsoot;
    int d_numMixVars;
    int d_numRxnVars;
    int d_depStateSpaceVars;
    MixRxnTableInfo* d_rxnTableInfo;
    // Data structure class that stores the table entries for state-space
    // variables as a function of independent variables.
    // This could be implemented either as a k-d or a binary tree data structure.
    KD_Tree* d_rxnTable;


  }; // End Class ILDMReactionModel

} // end namespace Uintah

#endif





