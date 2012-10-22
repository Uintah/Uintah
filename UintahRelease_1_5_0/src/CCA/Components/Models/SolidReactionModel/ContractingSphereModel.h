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

#ifndef UINTAH_HOMEBREW_ContractingSphereModel_H
#define UINTAH_HOMEBREW_ContractingSphereModel_H

#include <CCA/Components/Models/SolidReactionModel/RateModel.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {
    /**************************************
     
     CLASS
     ContractingSphereModel
     
     Short description...
     
     GENERAL INFORMATION
     
     ContractingSphereModel.h
     
     Joseph R. Peterson
     Department of Chemistry
     University of Utah
     
     Center for the Model of Accidental Fires and Explosions (C-SAFE)
     
          
     KEYWORDS
     ContractingSphereModel
     
     DESCRIPTION
     Long description...
     
     WARNING
     
     ****************************************/
    class ContractingSphereModel: public RateModel {
    public:
        ContractingSphereModel(ProblemSpecP &params);
     
        virtual void outputProblemSpec(ProblemSpecP& ps);
   
        /// @brief Get the contribution to the rate from the fraction reacted
        /// @param fractionReactant The fraction in the volume that is reactant, i.e. m_r/(m_r+m_p)
        /// @return a scalar for the extent of reaction
        virtual double getDifferentialFractionChange(double fractionReactant);
        
    private:
    };
    
} // End namespace Uintah



#endif
