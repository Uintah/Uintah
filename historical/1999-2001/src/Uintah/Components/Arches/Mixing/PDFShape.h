//----- PDFShape.h --------------------------------------------------

#ifndef Uintah_Component_Arches_PDFShape_h
#define Uintah_Component_Arches_PDFShape_h

/***************************************************************************
CLASS
    PDFShape
       PDFShape class computes the shape of the PDF
       
GENERAL INFORMATION
    PDFShape.h - Declaration of PDFShape class

    Author: Rajesh Rawat (rawat@crsim.utah.edu)
    
    Creation Date : 05-30-2000

    C-SAFE
    
    Copyright U of U 2000

KEYWORDS
    
DESCRIPTION
     PDFShape class for a given mean and variance of the independent variable
     computes the shape of the PDF. This information is used by the PDF
     mixing model. PDFShape is a base class using which different PDFShape classes
     like, BetaPDF and ClippedGaussian are derived.                


PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
    None
***************************************************************************/

#include <Uintah/Components/Arches/ArchesLabel.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/CFDInterface.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/VarLabel.h>
#include <SCICore/Geometry/IntVector.h>

#include <vector>

namespace Uintah {
namespace ArchesSpace {

class PDFShape {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructor taking
      //   [in] 
      //
      PDFShape();

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Virtual destructor for mixing model
      //
      virtual ~PDFShape();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      //
      // Set up the problem specification database
      //
      virtual void problemSetup(const ProblemSpecP& params) = 0;

      // GROUP: Compute properties 
      // GROUP: Actual Action Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Carry out actual computation of properties
      //
      virtual void computePDFFunction(double* meanMixVars, double statVars) = 0;
      virtual double computeShapeFunction(REAL *var)= 0;
      virtual bool validIntegral() = 0;
protected :

private:

}; // end class PDFShape

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
