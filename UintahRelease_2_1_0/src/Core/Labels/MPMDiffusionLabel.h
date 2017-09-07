/*
 * MPMDiffusionLabel.h
 *
 *  Created on: Jul 19, 2017
 *      Author: jbhooper
 *
 *
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#ifndef CORE_LABELS_MPMDIFFUSIONLABEL_H_
#define CORE_LABELS_MPMDIFFUSIONLABEL_H_

namespace Uintah {

  class VarLabel;

  class MPMDiffusionLabel {

    public:
      MPMDiffusionLabel();
     ~MPMDiffusionLabel();

      // Particle variables involved in diffusion calculations
     const VarLabel* pExternalScalarFlux;
     const VarLabel* pExternalScalarFlux_preReloc;

     const VarLabel* pFlux;
     const VarLabel* pFlux_preReloc;

     const VarLabel* pDiffusivity;
     const VarLabel* pDiffusivity_preReloc;

     const VarLabel* pConcentration;
     const VarLabel* pConcentration_preReloc;

     const VarLabel* pConcPrevious;
     const VarLabel* pConcPrevious_preReloc;

     const VarLabel* pGradConcentration;
     const VarLabel* pGradConcentration_preReloc;

     const VarLabel* pChemicalPotential;
     const VarLabel* pChemicalPotential_preReloc;

     const VarLabel* pArea;
     const VarLabel* pArea_preReloc;

     const VarLabel* pGradChemPotential;

     // Grid variables involved in diffusion calculations
     const VarLabel* gConcentration;
     const VarLabel* gConcentrationNoBC;
     const VarLabel* gConcentrationRate;
     const VarLabel* gConcentrationStar;
     const VarLabel* gExternalScalarFlux;
     const VarLabel* gHydrostaticStress;

     const VarLabel* gChemicalPotential;

     // Reduction variables associated with diffusion logic
     const VarLabel* rTotalConcentration;
     const VarLabel* rMaxConcentration;
     const VarLabel* rMinConcentration;
     const VarLabel* rAvgConcentration;
  };

}




#endif /* CORE_LABELS_MPMDIFFUSIONLABEL_H_ */
