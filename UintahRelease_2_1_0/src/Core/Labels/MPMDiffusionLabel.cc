/*
 * MPMDiffusionLabels.cc
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

#include <Core/Labels/MPMDiffusionLabel.h>

#include <Core/Geometry/Vector.h>

#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>

using namespace Uintah;

MPMDiffusionLabel::MPMDiffusionLabel()
{
  // Particle variables involved in diffusion calculations
  pExternalScalarFlux = VarLabel::create("p.externalScalarFlux", ParticleVariable<double>::getTypeDescription());
  pExternalScalarFlux_preReloc = VarLabel::create("p.externalScalarFlux+", ParticleVariable<double>::getTypeDescription());

  pFlux = VarLabel::create("p.flux", ParticleVariable<Vector>::getTypeDescription());
  pFlux_preReloc = VarLabel::create("p.flux+", ParticleVariable<Vector>::getTypeDescription());

  pDiffusivity = VarLabel::create("p.diffusivity", ParticleVariable<double>::getTypeDescription());
  pDiffusivity_preReloc = VarLabel::create("p.diffusivity+", ParticleVariable<double>::getTypeDescription());

  pConcentration = VarLabel::create("p.concentration", ParticleVariable<double>::getTypeDescription());
  pConcentration_preReloc = VarLabel::create("p.concentration+", ParticleVariable<double>::getTypeDescription());

  pGradConcentration = VarLabel::create("p.concentrationGradient", ParticleVariable<Vector>::getTypeDescription());
  pGradConcentration_preReloc = VarLabel::create("p.concentrationGradient+", ParticleVariable<Vector>::getTypeDescription());

  pConcPrevious = VarLabel::create("p.concPrevious", ParticleVariable<double>::getTypeDescription());
  pConcPrevious_preReloc = VarLabel::create("p.concPrevious+", ParticleVariable<double>::getTypeDescription());

  pChemicalPotential = VarLabel::create("p.mu", ParticleVariable<double>::getTypeDescription());
  pChemicalPotential_preReloc = VarLabel::create("p.mu+", ParticleVariable<double>::getTypeDescription());

  pArea = VarLabel::create( "p.area", ParticleVariable<Vector>::getTypeDescription());
  pArea_preReloc = VarLabel::create( "p.area+", ParticleVariable<Vector>::getTypeDescription());

  pGradChemPotential = VarLabel::create("p.muGradient", ParticleVariable<Vector>::getTypeDescription());



  // Nodal variables involved in diffusion calculations
  gConcentration = VarLabel::create("g.concentration", NCVariable<double>::getTypeDescription());
  gConcentrationNoBC = VarLabel::create("g.concentrationnobc", NCVariable<double>::getTypeDescription());
  gConcentrationRate = VarLabel::create("g.concentrationRate", NCVariable<double>::getTypeDescription());
  gConcentrationStar = VarLabel::create("g.concentrationStar", NCVariable<double>::getTypeDescription());
  gExternalScalarFlux = VarLabel::create("g.externalscalarflux", NCVariable<double>::getTypeDescription());
  gHydrostaticStress = VarLabel::create("g.hydrostaticStressRD", NCVariable<double>::getTypeDescription());
  gChemicalPotential = VarLabel::create("g.mu", NCVariable<double>::getTypeDescription());

  // Reduction variables involved in diffusion calculations
  rTotalConcentration = VarLabel::create("TotalConc", sum_vartype::getTypeDescription());
  rAvgConcentration = VarLabel::create("p.avgConc", sum_vartype::getTypeDescription());

  rMinConcentration = VarLabel::create("MinConcLabel", min_vartype::getTypeDescription());
  rMaxConcentration = VarLabel::create("MaxConcLabel", max_vartype::getTypeDescription());

}

MPMDiffusionLabel::~MPMDiffusionLabel()
{
  // Particle variables involved in diffusion calculations
  VarLabel::destroy(pExternalScalarFlux);
  VarLabel::destroy(pExternalScalarFlux_preReloc);
  VarLabel::destroy(pFlux);
  VarLabel::destroy(pFlux_preReloc);
  VarLabel::destroy(pDiffusivity);
  VarLabel::destroy(pDiffusivity_preReloc);
  VarLabel::destroy(pConcentration);
  VarLabel::destroy(pConcentration_preReloc);
  VarLabel::destroy(pGradConcentration);
  VarLabel::destroy(pGradConcentration_preReloc);
  VarLabel::destroy(pConcPrevious);
  VarLabel::destroy(pConcPrevious_preReloc);
  VarLabel::destroy(pChemicalPotential);
  VarLabel::destroy(pChemicalPotential_preReloc);
  VarLabel::destroy(pArea);
  VarLabel::destroy(pArea_preReloc);
  VarLabel::destroy(pGradChemPotential);

  // Nodal vairables involved in diffusion calculations
  VarLabel::destroy(gConcentration);
  VarLabel::destroy(gConcentrationNoBC);
  VarLabel::destroy(gConcentrationRate);
  VarLabel::destroy(gConcentrationStar);
  VarLabel::destroy(gExternalScalarFlux);
  VarLabel::destroy(gHydrostaticStress);
  VarLabel::destroy(gChemicalPotential);

  // Reduction variables involved in diffusion calculations
  VarLabel::destroy(rTotalConcentration);
  VarLabel::destroy(rAvgConcentration);
  VarLabel::destroy(rMinConcentration);
  VarLabel::destroy(rMaxConcentration);
}



