/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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
#include <CCA/Components/MPM/Core/ShellMPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using namespace Uintah;
using namespace std;

ShellMPMLabel::ShellMPMLabel()
{
  // Labels for shell materials
  pThickTopLabel = VarLabel::create("p.thickTop",
                     ParticleVariable<double>::getTypeDescription());
  pInitialThickTopLabel = VarLabel::create("p.thickTop0",
                     ParticleVariable<double>::getTypeDescription());
  pThickBotLabel = VarLabel::create("p.thickBot",
                     ParticleVariable<double>::getTypeDescription());
  pInitialThickBotLabel = VarLabel::create("p.thickBot0",
                     ParticleVariable<double>::getTypeDescription());
  pNormalLabel = VarLabel::create("p.normal",
                     ParticleVariable<Vector>::getTypeDescription());
  pInitialNormalLabel = VarLabel::create("p.normal0",
                     ParticleVariable<Vector>::getTypeDescription());

  pThickTopLabel_preReloc = VarLabel::create("p.thickTop+",
                     ParticleVariable<double>::getTypeDescription());
  pInitialThickTopLabel_preReloc = VarLabel::create("p.thickTop0+",
                     ParticleVariable<double>::getTypeDescription());
  pThickBotLabel_preReloc = VarLabel::create("p.thickBot+",
                     ParticleVariable<double>::getTypeDescription());
  pInitialThickBotLabel_preReloc = VarLabel::create("p.thickBot0+",
                     ParticleVariable<double>::getTypeDescription());
  pNormalLabel_preReloc = VarLabel::create("p.normal+",
                     ParticleVariable<Vector>::getTypeDescription());
  pInitialNormalLabel_preReloc = VarLabel::create("p.normal0+",
                     ParticleVariable<Vector>::getTypeDescription());

  pTypeLabel = VarLabel::create("p.type",
                     ParticleVariable<int>::getTypeDescription());
  pTypeLabel_preReloc = VarLabel::create("p.type+",
                     ParticleVariable<int>::getTypeDescription());

  gNormalRotRateLabel = VarLabel::create( "g.normalRotRate",
                     NCVariable<Vector>::getTypeDescription());
  gNormalRotMomentLabel = VarLabel::create( "g.normalRotMoment",
                     NCVariable<Vector>::getTypeDescription());

  gNormalRotMassLabel = VarLabel::create( "g.normalRotMass",
                     NCVariable<double>::getTypeDescription());
  gNormalRotAccLabel = VarLabel::create( "g.normalRotAcc",
                     NCVariable<Vector>::getTypeDescription());

} 

ShellMPMLabel::~ShellMPMLabel()
{
  // Destroy Labels for shell materials
  VarLabel::destroy(pThickTopLabel);
  VarLabel::destroy(pInitialThickTopLabel);
  VarLabel::destroy(pThickBotLabel);
  VarLabel::destroy(pInitialThickBotLabel);
  VarLabel::destroy(pNormalLabel);
  VarLabel::destroy(pInitialNormalLabel);

  VarLabel::destroy(pThickTopLabel_preReloc);
  VarLabel::destroy(pInitialThickTopLabel_preReloc);
  VarLabel::destroy(pThickBotLabel_preReloc);
  VarLabel::destroy(pInitialThickBotLabel_preReloc);
  VarLabel::destroy(pNormalLabel_preReloc);
  VarLabel::destroy(pInitialNormalLabel_preReloc);

  VarLabel::destroy(pTypeLabel);
  VarLabel::destroy(pTypeLabel_preReloc);

  VarLabel::destroy(gNormalRotRateLabel); 
  VarLabel::destroy(gNormalRotMomentLabel); 
  VarLabel::destroy(gNormalRotMassLabel); 
  VarLabel::destroy(gNormalRotAccLabel); 
}
