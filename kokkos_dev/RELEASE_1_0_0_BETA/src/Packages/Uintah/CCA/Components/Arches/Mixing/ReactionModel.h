#ifndef Uintah_Component_Arches_ReactionModel_h
#define Uintah_Component_Arches_ReactionModel_h 
/****************************************************************************
   CLASS
     ReactionModel
       The ReactionModel Class is used for providing state space variables 
       including reaction-rate information to MixingModel.

   GENERAL INFORMATION
     ReactionModel.h - Declaration of ReactionModel Class

     Author: Rajesh Rawat (rawat@crsim.utah.edu) 
             Jennifer Spinti (spinti@crsim.utah.edu)

     Creation Date: 22 June 1999

     C-SAFE

     Copyright U of U 1999

   KEYWORDS
     Reaction_Model, Equilibrium, Manifold

   DESCRIPTION
     ReactionModel is an abstract class which provides a general interface to 
     a MixingModel for computing state space variables as a function of 
     independent mixing and reaction progress variables. Different reaction 
     models such as 1-2 step global chemistry, equilibrium, manifold method,
     and Dedonder mechanism will be derived from this base class. For details
     about the different reaction models please refer to [Rawat, 1997].
     
     Reference: Rawat,R., Modeling Finite-Rate Chemistry in Turbulent Reacting
     Flows, Ph.D. Dissertation, Department of Chemical & Fuels Engg, Univ of
     Utah, Salt Lake City, UT(1997).
                

   PATTERNS
     None

   WARNINGS
     None

   POSSIBLE REVISIONS:
   1. DeDonder mechanism (will be researched and implemented by Jennifer and Diem)


  **************************************************************************/
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <vector>

using namespace std;

namespace  Uintah {
  class ChemkinInterface;
  class Stream;

  const double SMALL_VALUE = 1.e-20;
 
  class ReactionModel {
  public:

    // GROUP: Static Variables:
    ////////////////////////////////////////////////////////////////////////
    //  
    // Constant close to machine's zero
    // 
    //static const double SMALL_VALUE;
 

    // GROUP: Constructors:
    ////////////////////////////////////////////////////////////////////////
    //
    // Constructs an instance of ReactionModel with the given number of 
    // independent mixing and reaction progress variables. In addition,
    // total number of state-space variables are specified.
    // PRECONDITIONS
    //  numMixVar is positive.
    //  numRxnVar is positive.
    //  numStateSpaceVar is positive.
    // POSTCONDITIONS
    //  This is a properly constructed instance of ReactionModel.
    //
    // Constructor taking
    //   [in] numMixVar Number of independent mixing variables.
    //   [in] numRxnVar Number of reaction progress variables.
    //   [in] numStateSpaceVar Total number of state space variables.
    //   [in] adiabatic If =1, system is adiabatic; otherwise, its nonadiabatic.
    //   [bool] createRxnTable If true, reaction table will be constructed and
    // desired state space variables will be interpolated from table. If false,
    // state space variables will be computed directly. 
    ReactionModel();

    // GROUP: Destructor:
    ///////////////////////////////////////////////////////////////////////
    //
    // Destructor 
    //
    virtual ~ReactionModel();

    // GROUP: Problem Setup :
    ///////////////////////////////////////////////////////////////////////
    //
    //
    virtual void problemSetup(const ProblemSpecP& params) = 0;

    // GROUP: Access functions
    //////////////////////////////////////////////////////////////////////
    // returns the pointer to chemkin interface class
    virtual ChemkinInterface* getChemkinInterface() = 0;
    //
    // [in] mixRxnVar Array of independent variables from which dependent
    // variables are calculated; array includes heat loss if system is nonadiabatic
    virtual Stream computeRxnStateSpace(Stream& mixRxnVar) = 0;
#if 0
    virtual Stream computeEnthalpy(Stream& mixRxnVar) = 0;
#endif
    //virtual vector<double> findRxnVariables(double* mixRxnVar, double* yArray) = 0;
  
  private:  
  }; // End Class ReactionModel


} // end namespace Uintah

#endif

//
// $Log$
// Revision 1.1  2001/01/31 16:35:30  rawat
// Implemented mixing and reaction models for fire.
//
// Revision 1.1.1.1 1999/06/03 14:40 Raj
// Initial New Public Checkin to CVS
//
//
