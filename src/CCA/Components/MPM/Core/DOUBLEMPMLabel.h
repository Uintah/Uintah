/*
 * MPMDiffusionLabel.h
 *
 *  Created on: Oct 4, 2018
 *      Author: quocanh
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

#ifndef CORE_LABELS_DOUBLEMPMLABEL_H_
#define CORE_LABELS_DOUBLEPMLABEL_H_

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/ParticleInterpolator.h>

#include <string>
#include <vector>

namespace Uintah {

	class VarLabel;

	class DOUBLEMPMLabel {

	public:
		DOUBLEMPMLabel();
		~DOUBLEMPMLabel();

		// Particle variables involved in hydro-coupling calculation
		const VarLabel* pPorePressureLabel;
		const VarLabel* pPorePressureTensorLabel;
		const VarLabel* pPourosityLabel;
		const VarLabel* pPermeabilityLabel;
		const VarLabel* pVelocityLiquidLabel;


		// Grid variables involved in diffusion calculations
		const VarLabel* gAccelerationLiquidLabel;
		const VarLabel* gMassLiquidLabel;
		const VarLabel* gVelocityLiquidLabel;
		const VarLabel* gVelocityStarLiquidLabel;
		const VarLabel* gInternalForceLiquidLabel;
		const VarLabel* gPourosityLabel;
		const VarLabel* gDragForceLabel;

	};

}




#endif /* CORE_LABELS_DOUBLEMPMLABEL_H_ */
