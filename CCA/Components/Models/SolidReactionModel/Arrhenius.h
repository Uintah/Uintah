/*
 
 The MIT License
 
 Copyright (c) 1997-2012 Center for the Simulation of Accidental Fires and 
 Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
 University of Utah.
 
 License for the specific language governing rights and limitations under
 Permission is hereby granted, free of charge, to any person obtaining a 
 copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation 
 the rights to use, copy, modify, merge, publish, distribute, sublicense, 
 and/or sell copies of the Software, and to permit persons to whom the 
 Software is furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included 
 in all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 DEALINGS IN THE SOFTWARE.
 
 */


#ifndef UINTAH_HOMEBREW_Arrhenius_H
#define UINTAH_HOMEBREW_Arrhenius_H

#include <CCA/Components/Models/SolidReactionModel/RateConstant.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {
    /**************************************
     
     CLASS
     Arrhenius
     
     Short description...
     
     GENERAL INFORMATION
     
     Arrhenius.h
     
     Joseph R. Peterson
     Department of Chemistry
     University of Utah
     
     Center for the Model of Accidental Fires and Explosions (C-SAFE)
     
     Copyright (C) 2012 SCI Group
     
     KEYWORDS
     Arrhenius
     
     DESCRIPTION
     Long description...
     
     WARNING
     
     ****************************************/
    class Arrhenius : public RateConstant {
    public:
        Arrhenius(ProblemSpecP& params);
     
        virtual void outputProblemSpec(ProblemSpecP& ps);
   
        /// @brief Gets a rate constant given a temperature
        /// @param T Temperature in Kelvin at which to get constant
        /// @return rate Rate at given temperature
        virtual double getConstant(double T);
        
    private:
        double d_A;     // Frequency factor,  s^-1
        double d_Ea;    // Activation energy, J/molK
    };
    
} // End namespace Uintah



#endif
