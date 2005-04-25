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

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace  Uintah {
  class ChemkinInterface;
  class Stream;
  class MixingModel;

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
    // Constructs an instance of ReactionModel 
    // PRECONDITIONS
    // POSTCONDITIONS
    //  This is a properly constructed instance of ReactionModel.
    // 
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
    virtual void problemSetup(const ProblemSpecP& params,
			      MixingModel* mixModel) = 0;

    //
    // GROUP: Actual Action Methods :
    ///////////////////////////////////////////////////////////////////////
    //
    // Computes the state space (dependent) variables given the unreacted
    // stream information and values for the independent variables 
    // (no variance)
    //
    virtual void getRxnStateSpace(const Stream& unreactedMixture,
				  std::vector<double>& mixRxnVar,
				  Stream& reactedStream) = 0;
    virtual void computeRxnStateSpace(const Stream& unreactedMixture, 
				      const std::vector<double>& mixRxnVar, 
				      Stream& equilStateSpace) = 0;
    virtual double computeTemperature(const double absEnthalpy, 
				      const std::vector<double>& massFract, 
				      double initTemp) = 0;   


      
    // GROUP: Get Methods :
    ///////////////////////////////////////////////////////////////////////
    //    
    // Returns the pointer to chemkin interface class
    //
    virtual ChemkinInterface* getChemkinInterface() const = 0;
    virtual bool getSootBool() const = 0;
    virtual int getTotalDepVars() const = 0;
  
  private:  
  }; // End Class ReactionModel


} // end namespace Uintah

#endif

//
// $Log$
// Revision 1.11  2003/08/07 00:48:14  sparker
// SGI 64 bit warnings rampage
//
// Revision 1.10  2003/01/22 00:43:04  spinti
// Added improved BetaPDF mixing model and capability to create a betaPDF table a priori. Cleaned up favre averaging and streamlined sections of code.
//
// Revision 1.9  2002/05/31 22:04:45  spinti
// *** empty log message ***
//
// Revision 1.6  2002/03/28 23:14:51  spinti
// 1. Added in capability to save mixing and reaction tables as KDTree or 2DVector
// 2. Tables can be declared either static or dynamic
// 3. Added capability to run using static clipped Gaussian MixingModel table
// 4. Removed mean values mixing model option from PDFMixingModel and made it
//    a separate mixing model, MeanMixingModel.
//
// Revision 1.5  2001/11/08 19:13:44  spinti
// 1. Corrected minor problems in ILDMReactionModel.cc
// 2. Added tabulation capability to StanjanEquilibriumReactionModel.cc. Now,
//    a reaction table is created dynamically. The limits and spacing in the
//    table are specified in the *.ups file.
// 3. Corrected the mixture temperature computation in Stream::addStream. It
//    now is computed using a Newton search.
// 4. Made other minor corrections to various reaction model files.
//
// Revision 1.2  2001/07/16 21:15:38  rawat
// added enthalpy solver and Jennifer's changes in Mixing and Reaction model required for ILDM and non-adiabatic cases
//
// Revision 1.1  2001/01/31 16:35:30  rawat
// Implemented mixing and reaction models for fire.
//
// Revision 1.1.1.1 1999/06/03 14:40 Raj
// Initial New Public Checkin to CVS
//
//
