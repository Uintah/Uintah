/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

//----- MixingModel.h --------------------------------------------------

#ifndef Uintah_Component_Arches_MixingModel_h
#define Uintah_Component_Arches_MixingModel_h

/***************************************************************************
CLASS
    MixingModel
       Sets up the MixingModel ????
       
GENERAL INFORMATION
    MixingModel.h - Declaration of MixingModel class

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

#include <CCA/Components/Arches/Mixing/InletStream.h>
#include <CCA/Components/Arches/Mixing/Stream.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <vector>
#include <string>

namespace Uintah {
class MixingModel {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructor taking
      //   [in] 
      //
      MixingModel();

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Virtual destructor for mixing model
      //
      virtual ~MixingModel();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      //
      // Set up the problem specification database
      //
      virtual void problemSetup(const ProblemSpecP& params) = 0;
 
      // GROUP: Actual Action Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Carry out actual computation of properties
      //
      virtual void computeProps(const InletStream& inStream,
                                Stream& outStream) = 0;


      // GROUP: Get Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      //
      virtual double getAdiabaticAirEnthalpy() const = 0;
      virtual double getFStoich() const = 0;
      virtual double getCarbonFuel() const = 0;
      virtual double getCarbonAir() const = 0;
      virtual bool getCOOutput() const = 0;
      virtual bool getSulfurChem() const = 0;
      virtual bool getSootPrecursors() const = 0;
      virtual bool getTabulatedSoot() const = 0;

      inline void setNonAdiabPartBool( bool adiabGas_nonadiabPart ) {
        d_adiabGas_nonadiabPart = adiabGas_nonadiabPart; 
      }

protected :
  
      bool d_adiabGas_nonadiabPart; 

private:


}; // end class MixingModel

} // end namespace Uintah

#endif

