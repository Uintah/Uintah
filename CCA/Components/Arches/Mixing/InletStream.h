//----- InletStream.h -----------------------------------------------

#ifndef Uintah_Components_Arches_InletStream_h
#define Uintah_Components_Arches_InletStream_h

/**************************************
CLASS
   InletStream
   
   Class InletStream creates and stores the mixing variables that are used in Arches

GENERAL INFORMATION
   InletStream.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   July 20, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS

DESCRIPTION

WARNING
   none

************************************************************************/
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
    class InletStream {
    public:
      InletStream();
      InletStream(int numMixVars, int numMixStatVars, int numRxnVars);
      ~InletStream();
      std::vector<double> d_mixVars;
      std::vector<double> d_mixVarVariance;
      double d_enthalpy;
      bool d_initEnthalpy;
      std::vector<double> d_rxnVars;
      int d_axialLoc;

    }; // End class InletStream

}  // End namespace Uintah

#endif

