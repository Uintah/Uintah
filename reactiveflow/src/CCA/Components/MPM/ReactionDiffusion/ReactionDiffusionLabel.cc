/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#include <CCA/Components/MPM/ReactionDiffusion/ReactionDiffusionLabel.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>

using namespace std;
using namespace Uintah;


ReactionDiffusionLabel::ReactionDiffusionLabel() {

	// Particle Variables
  pConcentrationLabel = VarLabel::create( "p.concentration",
      ParticleVariable<double>::getTypeDescription() );

  pConcPreviousLabel = VarLabel::create( "p.concPrevious",
      ParticleVariable<double>::getTypeDescription() );

  pdCdtLabel = VarLabel::create( "p.dCdt",
		  ParticleVariable<double>::getTypeDescription() );

  pConcentrationGradientLabel = VarLabel::create("p.concentrationGradient",
	    ParticleVariable<Vector>::getTypeDescription());

  pHydroStressLabel = VarLabel::create( "p.hydroStress",
      ParticleVariable<double>::getTypeDescription() );

  pConcentrationLabel_preReloc = VarLabel::create("p.concentration+",
			ParticleVariable<double>::getTypeDescription() );

  pConcPreviousLabel_preReloc = VarLabel::create( "p.concPrevious+",
      ParticleVariable<double>::getTypeDescription() );

  pdCdtLabel_preReloc = VarLabel::create( "p.dCdt+",
		  ParticleVariable<double>::getTypeDescription() );

	// Grid Variables
  gConcentrationLabel = VarLabel::create( "g.concentration",
			NCVariable<double>::getTypeDescription());

  gConcentrationCCLabel = VarLabel::create( "g.concentrationCC",
                        CCVariable<double>::getTypeDescription() );

  gConcentrationRateLabel = VarLabel::create("g.concentrationRate",
			NCVariable<double>::getTypeDescription());

  gConcentrationNoBCLabel = VarLabel::create("g.concentrationnobc",
			NCVariable<double>::getTypeDescription());

  gdCdtLabel = VarLabel::create("g.dCdt",
	    NCVariable<double>::getTypeDescription());

  gConcentrationStarLabel = VarLabel::create("g.concentrationStar",
	    NCVariable<double>::getTypeDescription());

  gHydrostaticStressLabel = VarLabel::create("g.hydrostaticStressRD",
	    NCVariable<double>::getTypeDescription());
}

ReactionDiffusionLabel::~ReactionDiffusionLabel() {

	// Particle Variables
  VarLabel::destroy(pConcentrationLabel);
  VarLabel::destroy(pConcPreviousLabel);
  VarLabel::destroy(pdCdtLabel);
  VarLabel::destroy(pConcentrationGradientLabel);
  VarLabel::destroy(pHydroStressLabel);

  VarLabel::destroy(pConcentrationLabel_preReloc);
  VarLabel::destroy(pConcPreviousLabel_preReloc);
  VarLabel::destroy(pdCdtLabel_preReloc);

	// Grid Variables
  VarLabel::destroy(gConcentrationLabel);
  VarLabel::destroy(gConcentrationCCLabel);
  VarLabel::destroy(gConcentrationRateLabel);
  VarLabel::destroy(gConcentrationNoBCLabel);
  VarLabel::destroy(gdCdtLabel);
  VarLabel::destroy(gConcentrationStarLabel);
  VarLabel::destroy(gHydrostaticStressLabel);
}
