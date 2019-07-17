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
		const VarLabel* pPorePressureLabel_preReloc;

		const VarLabel* pFreeSurfaceLabel;
		const VarLabel* pFreeSurfaceLabel_preReloc;

		const VarLabel* pStressFilterLabel;

		const VarLabel* pPorosityLabel;
		const VarLabel* pPorosityLabel_preReloc;

		const VarLabel* pPermeabilityLabel;
		const VarLabel* pPermeabilityLabel_preReloc;

		//VarLabel* pVelocitySolidLabel;
		const VarLabel* pVelocityLiquidLabel;
		const VarLabel* pVelocityLiquidLabel_preReloc;

		const VarLabel* pVelocityGradLiquidLabel;
		const VarLabel* pVelocityGradLiquidLabel_preReloc;

		const VarLabel* pMassSolidLabel;
		const VarLabel* pMassSolidLabel_preReloc;

		const VarLabel* pMassLiquidLabel;
		const VarLabel* pMassLiquidLabel_preReloc;

		const VarLabel* pBulkModulLiquidLabel;
		const VarLabel* pBulkModulLiquidLabel_preReloc;

		//VarLabel* pVolumeSolidLabel;
		//VarLabel* pVolumeLiquidLabel;

		// Grid variables involved in diffusion calculations
		const VarLabel* gAccelerationLiquidLabel;
		const VarLabel* gMassLiquidLabel;
		//const VarLabel* gVolumeLiquidLabel;
		const VarLabel* gVelocityLiquidLabel;
		const VarLabel* gVelocityStarLiquidLabel;
		const VarLabel* gInternalForceLiquidLabel;

		const VarLabel* gPorosityLabel;
		const VarLabel* gDragForceLabel;

		const VarLabel* gMassSolidLabel;
		//const VarLabel* gVolumeSolidLabel;
		//const VarLabel* gVeloctySolidLabel;

		const VarLabel* gPorePressureLabel;
		const VarLabel* gStressLabel;

		const VarLabel* gDraggingLabel;

		const VarLabel* gnodeSurfaceLabel;

		const VarLabel* VolumeRatioLabel;
	};

}
#endif /* CORE_LABELS_DOUBLEMPMLABEL_H_ */
