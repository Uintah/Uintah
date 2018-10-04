/*
 * MPMDiffusionLabels.cc
 *
 *  Created on: Oct 04, 2018
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

#include <CCA/Components/MPM/Core/DOUBLEMPMLabel.h>

#include <Core/Geometry/Vector.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using namespace Uintah;

// For solid particles, the variables are from MPMLabel

DOUBLEMPMLabel::DOUBLEMPMLabel()
{
	// Liquid particles variables
	pPorePressureLabel			= VarLabel::create("p.PorePressure", ParticleVariable<double>::getTypeDescription());

	pPorePressureTensorLabel	= VarLabel::create("p.PorePressureTensor", ParticleVariable<Matrix3>::getTypeDescription());

	pPourosityLabel				= VarLabel::create("p.Pourosity", ParticleVariable<double>::getTypeDescription());

	pPermeabilityLabel			= VarLabel::create("p.Permeability", ParticleVariable<double>::getTypeDescription());

	pVelocityLiquidLabel		= VarLabel::create("p.VelocityLiquid", ParticleVariable<Vector>::getTypeDescription());

	// Grid liquid variables
	gAccelerationLiquidLabel	= VarLabel::create("g.accelerationLiquid", NCVariable<Vector>::getTypeDescription());

	gMassLiquidLabel			= VarLabel::create("g.massLiquid",	ParticleVariable<double>::getTypeDescription());

	gVelocityLiquidLabel		= VarLabel::create("g.velocityLiquid", NCVariable<Vector>::getTypeDescription());

	gVelocityStarLiquidLabel	= VarLabel::create("g.velocityLiquid_star", NCVariable<Vector>::getTypeDescription());

	gInternalForceLiquidLabel	= VarLabel::create("g.internalforceLiquid", NCVariable<Vector>::getTypeDescription());

	gPourosityLabel				= VarLabel::create("g.Pourosity", NCVariable<double>::getTypeDescription());

	gDragForceLabel				= VarLabel::create("g.DragForce", NCVariable<Vector>::getTypeDescription());



}

DOUBLEMPMLabel::~DOUBLEMPMLabel()
{
	// Liquid particles variables
	VarLabel::destroy(pPorePressureLabel);
	VarLabel::destroy(pPorePressureTensorLabel);
	VarLabel::destroy(pPourosityLabel);
	VarLabel::destroy(pPermeabilityLabel);
	VarLabel::destroy(pVelocityLiquidLabel);

	// Grid liquid variables
	VarLabel::destroy(gAccelerationLiquidLabel);
	VarLabel::destroy(gMassLiquidLabel);
	VarLabel::destroy(gVelocityLiquidLabel);
	VarLabel::destroy(gVelocityStarLiquidLabel);
	VarLabel::destroy(gInternalForceLiquidLabel);
	VarLabel::destroy(gPourosityLabel);
	VarLabel::destroy(gDragForceLabel);
}


