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
#include <CCA/Components/MPM/Core/CZLabel.h>
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

CZLabel::CZLabel()
{
  // For Cohesive Zones
  czCountLabel = VarLabel::create("czCount",
                                   sumlong_vartype::getTypeDescription());

  czAreaLabel = VarLabel::create("cz.area",
                     ParticleVariable<double>::getTypeDescription());
  czAreaLabel_preReloc = VarLabel::create("cz.area+",
                     ParticleVariable<double>::getTypeDescription());

  czNormLabel = VarLabel::create("cz.norm",
                     ParticleVariable<Vector>::getTypeDescription());
  czNormLabel_preReloc = VarLabel::create("cz.norm+",
                     ParticleVariable<Vector>::getTypeDescription());

  czTangLabel = VarLabel::create("cz.tang",
                     ParticleVariable<Vector>::getTypeDescription());
  czTangLabel_preReloc = VarLabel::create("cz.tang+",
                     ParticleVariable<Vector>::getTypeDescription());

  czDispTopLabel = VarLabel::create("cz.disptop",
                     ParticleVariable<Vector>::getTypeDescription());
  czDispTopLabel_preReloc = VarLabel::create("cz.disptop+",
                     ParticleVariable<Vector>::getTypeDescription());

  czDispBottomLabel = VarLabel::create("cz.dispbottom",
                     ParticleVariable<Vector>::getTypeDescription());
  czDispBottomLabel_preReloc = VarLabel::create("cz.dispbottom+",
                     ParticleVariable<Vector>::getTypeDescription());

  czSeparationLabel = VarLabel::create("cz.separation",
                     ParticleVariable<Vector>::getTypeDescription());
  czSeparationLabel_preReloc = VarLabel::create("cz.separation+",
                     ParticleVariable<Vector>::getTypeDescription());

  czForceLabel = VarLabel::create("cz.force",
                     ParticleVariable<Vector>::getTypeDescription());
  czForceLabel_preReloc = VarLabel::create("cz.force+",
                     ParticleVariable<Vector>::getTypeDescription());

  czTopMatLabel = VarLabel::create("cz.topmat",
                     ParticleVariable<int>::getTypeDescription());
  czTopMatLabel_preReloc = VarLabel::create("cz.topmat+",
                     ParticleVariable<int>::getTypeDescription());

  czBotMatLabel = VarLabel::create("cz.botmat",
                     ParticleVariable<int>::getTypeDescription());
  czBotMatLabel_preReloc = VarLabel::create("cz.botmat+",
                     ParticleVariable<int>::getTypeDescription());

  czFailedLabel = VarLabel::create("cz.failed",
                     ParticleVariable<int>::getTypeDescription());
  czFailedLabel_preReloc = VarLabel::create("cz.failed+",
                     ParticleVariable<int>::getTypeDescription());

  czIDLabel = VarLabel::create("cz.CZID",
                        ParticleVariable<long64>::getTypeDescription() );

  czIDLabel_preReloc = VarLabel::create("cz.CZID+",
                        ParticleVariable<long64>::getTypeDescription() );

  pCellNACZIDLabel =
    VarLabel::create("cellNACZID", CCVariable<short int>::getTypeDescription());

} 

CZLabel::~CZLabel()
{
  // For Cohesive Zones
  VarLabel::destroy(czCountLabel);
  VarLabel::destroy(czAreaLabel);
  VarLabel::destroy(czAreaLabel_preReloc);
  VarLabel::destroy(czNormLabel);
  VarLabel::destroy(czNormLabel_preReloc);
  VarLabel::destroy(czTangLabel);
  VarLabel::destroy(czTangLabel_preReloc);
  VarLabel::destroy(czDispTopLabel);
  VarLabel::destroy(czDispTopLabel_preReloc);
  VarLabel::destroy(czDispBottomLabel);
  VarLabel::destroy(czDispBottomLabel_preReloc);
  VarLabel::destroy(czSeparationLabel);
  VarLabel::destroy(czSeparationLabel_preReloc);
  VarLabel::destroy(czForceLabel);
  VarLabel::destroy(czForceLabel_preReloc);
  VarLabel::destroy(czTopMatLabel);
  VarLabel::destroy(czTopMatLabel_preReloc);
  VarLabel::destroy(czBotMatLabel);
  VarLabel::destroy(czBotMatLabel_preReloc);
  VarLabel::destroy(czFailedLabel);
  VarLabel::destroy(czFailedLabel_preReloc);
  VarLabel::destroy(czIDLabel);
  VarLabel::destroy(czIDLabel_preReloc);
  VarLabel::destroy(pCellNACZIDLabel);
}
