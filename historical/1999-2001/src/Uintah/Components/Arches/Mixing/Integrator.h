//----- Integrator.h --------------------------------------------------

#ifndef Uintah_Component_Arches_Integrator_h
#define Uintah_Component_Arches_Integrator_h

/***************************************************************************
CLASS
    Integrator
       Sets up the Integrator ????
       
GENERAL INFORMATION
    Integrator.h - Declaration of Integrator class

    Author: Rajesh Rawat (rawat@crsim.utah.edu)
    
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

#include <Uintah/Components/Arches/ArchesLabel.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/MixingModel.h>
#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/CFDInterface.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/VarLabel.h>
#include <SCICore/Geometry/IntVector.h>

#include <vector>

namespace Uintah {
namespace ArchesSpace {
  // Function required by the integrator 
  double fnc(double *x);


class Integrator: {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructor taking
      //   [in] 
      //
      Integrator(int tableDimension, PDFMixingModel* mixModel);

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Destructor
      //
      ~Integrator();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      //
      // Set up the problem specification database
      //
      void problemSetup(const ProblemSpecP& params);

      // GROUP: Actual Action Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Carry out actual integration
      //
      Stream integrate(int* tableKeyIndex);
      
      void computeKeyValues(int* tableKeyIndex);
      double fun(double* x);

protected :

private:

      ///////////////////////////////////////////////////////////////////////
      //
      // Copy Constructor (never instantiated)
      //   [in] 
      //        const Integrator&   
      //
      Integrator(const Integrator&);

      // GROUP: Operators Not Instantiated:
      ///////////////////////////////////////////////////////////////////////
      //
      // Assignment Operator (never instantiated)
      //   [in] 
      //        const Integrator&   
      //
      Integrator& operator=(const Integrator&);

private:
      
      PDFMixingModel* d_PDFMixModel;
      PDFShape* d_mixingPDF
      int d_tableDimesnion;
      bool d_lfavre;
      int d_count;
      

}; // end class Integrator

} // end namespace ArchesSpace
} // end namespace Uintah

#endif

//
// $Log$
// Revision 1.1  2001/01/15 23:38:21  rawat
// added some more classes for implementing mixing model
//
// Revision 1.1  2000/12/18 17:53:10  rawat
// adding mixing model for reacting flows
//
//
