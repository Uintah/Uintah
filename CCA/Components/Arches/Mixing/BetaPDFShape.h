//----- BetaPDFShape.h --------------------------------------------------

#ifndef Uintah_Component_Arches_BetaPDFShape_h
#define Uintah_Component_Arches_BetaPDFShape_h

/***************************************************************************
CLASS
    BetaPDFShape
        BetaPDFShape class computes the beta PDF shape.
       
GENERAL INFORMATION
    BetaPDFShape.h - Declaration of BetaPDFShape class

    Author: Rajesh Rawat (rawat@crsim.utah.edu)
    
    Creation Date : 05-30-2000

    C-SAFE
    
    Copyright U of U 2000

KEYWORDS
    
DESCRIPTION
     BetaPDFShape class for a given mean and variance of the independent variable
     computes the Beta PDF shape. This class is derived from PDFShape class. This
     function is described in Grimaji[1992].


PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
    None
***************************************************************************/

#include <Packages/Uintah/CCA/Components/Arches/Mixing/PDFShape.h>
#include <vector>

namespace Uintah {

class BetaPDFShape: public PDFShape {

public:

  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  //
  // Constructs an instance of a BetaPDFShape with the given mean and statistics
  // of a given variable.
  // PRECONDITIONS
  //  meanMixVar varies from 0 to 1.
  //  statVar is positive and varies between 0 to meanMixVar*(1.0 - meanMixVar)
  // POSTCONDITIONS
  //  This is a properly constructed instance of a BetaPDFShape
  //
  // Constructor taking
  //   [in] meanMixVar Mean of independent variable for which PDFShape is
  //        constructed.
  //   [in] statVar Variance of the independent variable
  BetaPDFShape(int dimPDF);

  // GROUP: Destructors :
  ///////////////////////////////////////////////////////////////////////
  //
  // Destructor
  //
  virtual ~BetaPDFShape();

  // GROUP: Problem Setup :
  ///////////////////////////////////////////////////////////////////////
  //
  // Set up the problem specification database
  //
  virtual void problemSetup(const ProblemSpecP& params);

  // GROUP: Manipulators
  //////////////////////////////////////////////////////////////////////
  // computePDFFunction calculates the shape of the PDF function
  // Parameters: 
  // 
  virtual void computePDFFunction(double* meanMixVar,
				  double statVar);
  // GROUP: Manipulators
  //////////////////////////////////////////////////////////////////////
  // computeShapeFunction returns the value of PDF computed at var.
  // Parameters:
  // [in] var is the value of the independent variable at which value of
  // the shape function is computed. 
  // 
  virtual double computeShapeFunction(double *var);
  // Returns true if value of gamma fnc is greater than Small Variance
  bool validIntegral(){
    return d_validGammaValue;
  }
 protected:  
 private:
  double computeBetaPDFShape(double* meanMixVar,
			     double statVar);
  int d_dimPDF; // dimensionality of the PDF
  bool d_validGammaValue;
  double d_gammafnValue;
  std::vector<double> d_coef;
}; // end class BetaPDFShape
 const double SMALL_VARIANCE = 1e-06;
 const double SMALL = 1e-15;
 const double ZERO = 1e-40;

 extern "C" {double dgammaln_(double* x); }
} // end namespace Uintah

#endif

