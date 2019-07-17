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
	// particles variables
	//pXLiquidLabel				= VarLabel::create("p.xLiquid",ParticleVariable<Point>::getTypeDescription(),
	//							IntVector(0, 0, 0), VarLabel::PositionVariable);

	pPorePressureLabel			= VarLabel::create("p.PorePressure", ParticleVariable<double>::getTypeDescription());
	pPorePressureLabel_preReloc = VarLabel::create("p.PorePressure+", ParticleVariable<double>::getTypeDescription());

	pFreeSurfaceLabel			= VarLabel::create("p.FreeSurface", ParticleVariable<double>::getTypeDescription());
	pFreeSurfaceLabel_preReloc  = VarLabel::create("p.FreeSurface+", ParticleVariable<double>::getTypeDescription());


	pStressFilterLabel			= VarLabel::create("p.StressFilter", ParticleVariable<Matrix3>::getTypeDescription());

	pPorosityLabel				= VarLabel::create("p.Porosity", ParticleVariable<double>::getTypeDescription());
	pPorosityLabel_preReloc		= VarLabel::create("p.Porosity+", ParticleVariable<double>::getTypeDescription());

	pPermeabilityLabel			= VarLabel::create("p.Permeability", ParticleVariable<double>::getTypeDescription());
	pPermeabilityLabel_preReloc = VarLabel::create("p.Permeability+", ParticleVariable<double>::getTypeDescription());

	//pVelocitySolidLabel			= VarLabel::create("p.VelocitySolid", ParticleVariable<Vector>::getTypeDescription());

	pVelocityLiquidLabel		= VarLabel::create("p.VelocityLiquid", ParticleVariable<Vector>::getTypeDescription());
	pVelocityLiquidLabel_preReloc = VarLabel::create("p.VelocityLiquid+", ParticleVariable<Vector>::getTypeDescription());

	pVelocityGradLiquidLabel	= VarLabel::create("p.VelocityGradLiquid", ParticleVariable<Vector>::getTypeDescription());
	pVelocityGradLiquidLabel_preReloc = VarLabel::create("p.VelocityGradLiquid+", ParticleVariable<Vector>::getTypeDescription());

	pMassSolidLabel				= VarLabel::create("p.massSolid",	ParticleVariable<double>::getTypeDescription());
	pMassSolidLabel_preReloc	= VarLabel::create("p.massSolid+", ParticleVariable<double>::getTypeDescription());

	pMassLiquidLabel			= VarLabel::create("p.massLiquid",	ParticleVariable<double>::getTypeDescription());
	pMassLiquidLabel_preReloc	= VarLabel::create("p.massLiquid+", ParticleVariable<double>::getTypeDescription());

	//pVolumeSolidLabel			= VarLabel::create("p.VolumeSolid", ParticleVariable<double>::getTypeDescription());

	//pVolumeLiquidLabel			= VarLabel::create("p.VolumeLiquid", ParticleVariable<double>::getTypeDescription());

	pBulkModulLiquidLabel	  = VarLabel::create("p.BulkModulLiquid", ParticleVariable<double>::getTypeDescription());
	pBulkModulLiquidLabel_preReloc = VarLabel::create("p.BulkModulLiquid+", ParticleVariable<double>::getTypeDescription());


	// Grid liquid variables
	gAccelerationLiquidLabel	= VarLabel::create("g.accelerationLiquid", NCVariable<Vector>::getTypeDescription());

	gMassLiquidLabel			= VarLabel::create("g.massLiquid", NCVariable<double>::getTypeDescription());

	//gVolumeLiquidLabel			= VarLabel::create("g.gVolumeLiquid", NCVariable<double>::getTypeDescription());

	gVelocityLiquidLabel		= VarLabel::create("g.velocityLiquid", NCVariable<Vector>::getTypeDescription());

	gVelocityStarLiquidLabel	= VarLabel::create("g.velocityStarLiquid", NCVariable<Vector>::getTypeDescription());

	gInternalForceLiquidLabel	= VarLabel::create("g.internalforceLiquid", NCVariable<Vector>::getTypeDescription());

	gPorosityLabel				= VarLabel::create("g.Porosity", NCVariable<double>::getTypeDescription());

	gDragForceLabel				= VarLabel::create("g.DragForce", NCVariable<Vector>::getTypeDescription());

	gMassSolidLabel				= VarLabel::create("g.massSolid", NCVariable<double>::getTypeDescription());

	//gVolumeSolidLabel			= VarLabel::create("g.gVolumeSolid", NCVariable<double>::getTypeDescription());
	//gVeloctySolidLabel		= VarLabel::create("g.gVeloctySolid", NCVariable<Vector>::getTypeDescription());
	gPorePressureLabel			= VarLabel::create("g.PorePressure", NCVariable<double>::getTypeDescription());

	gStressLabel				= VarLabel::create("g.Stress", NCVariable<Matrix3>::getTypeDescription());


	gDraggingLabel				= VarLabel::create("g.Dragging", NCVariable<double>::getTypeDescription());

	gnodeSurfaceLabel			= VarLabel::create("g.nodeSurface", NCVariable<double>::getTypeDescription());
	
	// Cell variables
	VolumeRatioLabel			= VarLabel::create("c.VolumeRatio",CCVariable<double>::getTypeDescription());
}

DOUBLEMPMLabel::~DOUBLEMPMLabel()
{
	// particles variables
	VarLabel::destroy(pPorePressureLabel);
	VarLabel::destroy(pPorePressureLabel_preReloc);

	VarLabel::destroy(pFreeSurfaceLabel);
	VarLabel::destroy(pFreeSurfaceLabel_preReloc);

	VarLabel::destroy(pStressFilterLabel);

	VarLabel::destroy(pPorosityLabel);
	VarLabel::destroy(pPorosityLabel_preReloc);

	VarLabel::destroy(pPermeabilityLabel);
	VarLabel::destroy(pPermeabilityLabel_preReloc);

	//VarLabel::destroy(pVelocitySolidLabel);
	VarLabel::destroy(pVelocityLiquidLabel);
	VarLabel::destroy(pVelocityLiquidLabel_preReloc);
	
	VarLabel::destroy(pVelocityGradLiquidLabel);
	VarLabel::destroy(pVelocityGradLiquidLabel_preReloc);

	VarLabel::destroy(pMassSolidLabel);
	VarLabel::destroy(pMassSolidLabel_preReloc);

	VarLabel::destroy(pMassLiquidLabel);
	VarLabel::destroy(pMassLiquidLabel_preReloc);

	VarLabel::destroy(pBulkModulLiquidLabel);
	VarLabel::destroy(pBulkModulLiquidLabel_preReloc);

	//VarLabel::destroy(pVolumeSolidLabel);
	//VarLabel::destroy(pVolumeLiquidLabel);

	// Grid liquid variables
	VarLabel::destroy(gAccelerationLiquidLabel);
	VarLabel::destroy(gMassLiquidLabel);
	//VarLabel::destroy(gVolumeLiquidLabel);
	VarLabel::destroy(gVelocityLiquidLabel);
	VarLabel::destroy(gVelocityStarLiquidLabel);

	VarLabel::destroy(gInternalForceLiquidLabel);

	VarLabel::destroy(gPorosityLabel);
	VarLabel::destroy(gDragForceLabel);

	VarLabel::destroy(gMassSolidLabel);
	//VarLabel::destroy(gVolumeSolidLabel);
	//VarLabel::destroy(gVeloctySolidLabel);

	VarLabel::destroy(gPorePressureLabel);

	VarLabel::destroy(gStressLabel);

	VarLabel::destroy(gDraggingLabel);

	VarLabel::destroy(gnodeSurfaceLabel);

	// Cell variables
	VarLabel::destroy(VolumeRatioLabel);
}


