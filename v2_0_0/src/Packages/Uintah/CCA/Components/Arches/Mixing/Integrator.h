//----- Integrator.h --------------------------------------------------

#ifndef Uintah_Component_Arches_Integrator_h
#define Uintah_Component_Arches_Integrator_h

/***************************************************************************
CLASS
    Integrator
       The Integrator class is required by subgrid scale PDF mixing models.
       It performs integrations over a prescribed PDF shape. 
       
GENERAL INFORMATION
    Integrator.h - Declaration of Integrator class

    Author: Rajesh Rawat (rawat@crsim.utah.edu)
    Revised: Jennife Spinti (spinti@crsim.utah.edu)
    
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
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
 
#if !defined(_AIX)
#  define dqagpe dqagpe_
#  define dqagp dqagp_
#endif

// Function required by the integrator 
double fnc(double *x);
// Fortran subroutine for numerical integration
extern "C" {void dqagpe(double fun(double* f),double *a, double *b, 
			int *i, double *points, double *epsabs, 
			double *epsrel, int *limit, double *result, 
			double *abserr, int *neval,
			int *ier,double *alist, double *blist, double *rlist,
			double *elist, double *pts,
			int *level, int *ndin, int *iord,int *last);  }

extern "C" {void dqagp(double fun(double* f),double *a, double *b,
                        int *i, double *points, double *epsabs,
                        double *epsrel, double *result,
                        double *abserr, int *neval,
                        int *ier, int *lenw, int *leniw,
                        int *last, int *iwork, double* work);  }

namespace Uintah {

class PDFShape;
class PDFMixingModel;
class MixRxnTableInfo;
class ReactionModel;

// used as integration parameters
const int LIMITS = 500;
const int NPTS = 10; 

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
      void convertKeytoFloatValues(int tableKeyIndex[], std::vector<double>& indepVars);
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
      
      int d_tableDimension;
      ReactionModel* d_rxnModel;
      MixRxnTableInfo* d_tableInfo;
      PDFShape* d_mixingPDF;
      //std::vector<double> d_meanValues;
      std::vector<double> d_keyValues;
      //Vector of all independent variables, excluding variance
      std::vector<double> d_varsHFPi;
      int d_count;
      Stream d_unreactedStream;
      Stream d_meanSpaceVars;
      Stream d_resultStateVars;
      

}; // end class Integrator

} // end namespace Uintah

#endif

