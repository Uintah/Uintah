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

#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixingModel.h>

#include <vector>
double fnc(double *x);
 // Fortran subroutine for numerical integration
extern "C" {void dqagpe_(double fun(double* f),double *a, double *b, 
			int *i, double *points, double *epsabs, 
			double *epsrel, int *limit, double *result, 
			double *abserr, int *neval,
			int *ier,double *alist, double *blist, double *rlist,
			double *elist, double *pts,
			int *level, int *ndin, int *iord,int *last);  }


namespace Uintah {
  // Function required by the integrator 


class PDFShape;
class PDFMixingModel;
class MixRxnTableInfo;
class ReactionModel;
class Integrator{

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructor taking
      //   [in] 
      //
      Integrator(int tableDimension, PDFMixingModel* mixModel,
		 ReactionModel* rxnModel, MixRxnTableInfo* tableInfo);

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
      Stream computeMeanValues(int* tableKeyIndex);
      void convertKeytoMeanValues(int* tableKeyIndex);
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
      
      ReactionModel* d_rxnModel;
      MixRxnTableInfo* d_tableInfo;
      PDFShape* d_mixingPDF;
      std::vector<double> d_meanValues;
      std::vector<double> d_keyValues;
      //Vector of all independent variables, excluding variance
      std::vector<double> d_varsHFPi;
      int d_tableDimension;
      bool d_lfavre;
      int d_count;
      

}; // end class Integrator
// used as integration parameters
const int LIMITS = 100;
const int NPTS = 10; 
} // end namespace Uintah

#endif

