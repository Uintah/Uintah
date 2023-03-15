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
#include <CCA/Components/MPM/Core/TracerLabel.h>
#include <Core/Math/Matrix3.h>
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

TracerLabel::TracerLabel()
{
  // Reduction variables
  tracerCountLabel = VarLabel::create("tracerCount",
                                   sumlong_vartype::getTypeDescription());

  tracerIDLabel = VarLabel::create("t.tracerID",
			ParticleVariable<long64>::getTypeDescription() );

  tracerIDLabel_preReloc = VarLabel::create("t.tracerID+",
			ParticleVariable<long64>::getTypeDescription() );

  tracerCemVecLabel = VarLabel::create("t.CemVec",
			ParticleVariable<Vector>::getTypeDescription() );

  tracerCemVecLabel_preReloc = VarLabel::create("t.CemVec+",
			ParticleVariable<Vector>::getTypeDescription() );

  pCellNATracerIDLabel =
    VarLabel::create("cellNATracerID", 
                                   CCVariable<short int>::getTypeDescription());
}

TracerLabel::~TracerLabel()
{
  VarLabel::destroy(tracerIDLabel);
  VarLabel::destroy(tracerIDLabel_preReloc);
  VarLabel::destroy(tracerCountLabel);
  VarLabel::destroy(tracerCemVecLabel);
  VarLabel::destroy(tracerCemVecLabel_preReloc);
  VarLabel::destroy(pCellNATracerIDLabel);
}
