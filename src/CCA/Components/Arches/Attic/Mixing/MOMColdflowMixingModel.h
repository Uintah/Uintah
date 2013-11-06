/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//----- MOMColdflowMixingModel.h --------------------------------------------------

#ifndef Uintah_Component_Arches_MOMColdflowMixingModel_h
#define Uintah_Component_Arches_MOMColdflowMixingModel_h

/***************************************************************************
CLASS
    MOMColdflowMixingModel
       Sets up the MOMColdflowMixingModel ????
       
GENERAL INFORMATION
    MOMColdflowMixingModel.h - Declaration of MOMColdflowMixingModel class

    Author: Rajesh Rawat (rawat@crsim.utah.edu)
    
    Creation Date : 05-30-2000

    C-SAFE
    

KEYWORDS
    
DESCRIPTION

PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
    None
***************************************************************************/

#include <CCA/Components/Arches/Mixing/MixingModel.h>

#include <vector>
#include <string>

namespace Uintah {
class ExtraScalarSolver; 
class MOMColdflowMixingModel: public MixingModel {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructor taking
      //   [in] 
      //
      MOMColdflowMixingModel(bool calcReactingScalar,
                             bool calcEnthalpy,
                             bool calcVariance);

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Destructor
      //
      virtual ~MOMColdflowMixingModel();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      //
      // Set up the problem specification database
      //
      virtual void problemSetup(const ProblemSpecP& params);

      // GROUP: Actual Action Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Carry out actual computation of properties
      //
      virtual void computeProps(const InletStream& inStream,
                                Stream& outStream);
      // GROUP: Get Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Get the number of mixing variables
      //
      inline bool getCOOutput() const{
        return 0;
      }
      inline bool getSulfurChem() const{
        return 0;
      }
      inline bool getSootPrecursors() const{
        return 0; 
      }
      inline bool getTabulatedSoot() const{
        return 0; 
      }

      inline double getAdiabaticAirEnthalpy() const{
        return 0.0;
      }

      inline double getFStoich() const{
        return 0.0;
      }

      inline double getCarbonFuel() const{
        return 0.0;
      }

      inline double getCarbonAir() const{
        return 0.0;
      }

      inline void setCalcExtraScalars(bool calcExtraScalars) {
        d_calcExtraScalars=calcExtraScalars;
      }

      inline void setExtraScalars(std::vector<ExtraScalarSolver*>* extraScalars) {
        d_extraScalars = extraScalars;
      }

protected :

private:

      ///////////////////////////////////////////////////////////////////////
      //
      // Copy Constructor (never instantiated)
      //   [in] 
      //        const MOMColdflowMixingModel&   
      //
      MOMColdflowMixingModel(const MOMColdflowMixingModel&);

      // GROUP: Operators Not Instantiated:
      ///////////////////////////////////////////////////////////////////////
      //
      // Assignment Operator (never instantiated)
      //   [in] 
      //        const MOMColdflowMixingModel&   
      //
      MOMColdflowMixingModel& operator=(const MOMColdflowMixingModel&);

private:

      bool d_calcReactingScalar, d_calcEnthalpy, d_calcVariance;
      int d_numMixingVars;
      std::vector<Stream> d_streams; 
      bool d_calcExtraScalars;
      std::vector<ExtraScalarSolver*>* d_extraScalars;


}; // end class MOMColdflowMixingModel

} // end namespace Uintah

#endif
