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
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

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
      virtual double computeShapeFunction(double *var)= 0;
      virtual bool validIntegral() = 0;
protected :

private:

}; // end class PDFShape

} // end namespace Uintah

#endif


