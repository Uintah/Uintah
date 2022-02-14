/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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
#include <CCA/Components/MPM/Core/TriangleLabel.h>
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

TriangleLabel::TriangleLabel()
{
  // Reduction variable
  triangleCountLabel = VarLabel::create("triangleCount",
                                   sumlong_vartype::getTypeDescription());

  triangleIDLabel = VarLabel::create("tri.triangleID",
			ParticleVariable<long64>::getTypeDescription() );

  triangleIDLabel_preReloc = VarLabel::create("tri.triangleID+",
			ParticleVariable<long64>::getTypeDescription() );

  triMidToN0VectorLabel = VarLabel::create("tri.MidToN0Vector",
			ParticleVariable<Vector>::getTypeDescription() );

  triMidToN0VectorLabel_preReloc = VarLabel::create("tri.MidToN0Vector+",
			ParticleVariable<Vector>::getTypeDescription() );

  triMidToN1VectorLabel = VarLabel::create("tri.MidToN1Vector",
			ParticleVariable<Vector>::getTypeDescription() );

  triMidToN1VectorLabel_preReloc = VarLabel::create("tri.MidToN1Vector+",
			ParticleVariable<Vector>::getTypeDescription() );

  triMidToN2VectorLabel = VarLabel::create("tri.MidToN2Vector",
			ParticleVariable<Vector>::getTypeDescription() );

  triMidToN2VectorLabel_preReloc = VarLabel::create("tri.MidToN2Vector+",
			ParticleVariable<Vector>::getTypeDescription() );

  triCemVecN0VectorLabel = VarLabel::create("tri.CemVecN0",
			ParticleVariable<Vector>::getTypeDescription() );

  triCemVecN0VectorLabel_preReloc = VarLabel::create("tri.CemVecN0+",
			ParticleVariable<Vector>::getTypeDescription() );

  triCemVecN1VectorLabel = VarLabel::create("tri.CemVecN1",
			ParticleVariable<Vector>::getTypeDescription() );

  triCemVecN1VectorLabel_preReloc = VarLabel::create("tri.CemVecN1+",
			ParticleVariable<Vector>::getTypeDescription() );

  triCemVecN2VectorLabel = VarLabel::create("tri.CemVecN2",
			ParticleVariable<Vector>::getTypeDescription() );

  triCemVecN2VectorLabel_preReloc = VarLabel::create("tri.CemVecN2+",
			ParticleVariable<Vector>::getTypeDescription() );

  triUseInPenaltyLabel = VarLabel::create("tri.UseInPenalty",
			ParticleVariable<IntVector>::getTypeDescription() );

  triUseInPenaltyLabel_preReloc = VarLabel::create("tri.UseInPenalty+",
			ParticleVariable<IntVector>::getTypeDescription() );

  triAreaLabel        = VarLabel::create("tri.Area",
			ParticleVariable<double>::getTypeDescription() );

  triAreaLabel_preReloc = VarLabel::create("tri.Area+",
			ParticleVariable<double>::getTypeDescription() );

  triClayLabel        = VarLabel::create("tri.Clay",
			ParticleVariable<double>::getTypeDescription() );

  triClayLabel_preReloc = VarLabel::create("tri.Clay+",
			ParticleVariable<double>::getTypeDescription() );

  triAreaAtNodesLabel = VarLabel::create("tri.AreaAtNodes",
			ParticleVariable<Vector>::getTypeDescription() );

  triAreaAtNodesLabel_preReloc = VarLabel::create("tri.AreaAtNodes+",
			ParticleVariable<Vector>::getTypeDescription() );

  triNormalLabel = VarLabel::create("tri.normal",
			ParticleVariable<Vector>::getTypeDescription() );

  triNormalLabel_preReloc = VarLabel::create("tri.normal+",
			ParticleVariable<Vector>::getTypeDescription() );
#if 0
  triNode0TriangleIDsLabel = VarLabel::create("tri.Node0TriangleIDs",
			ParticleVariable<Stencil7>::getTypeDescription() );

  triNode1TriangleIDsLabel = VarLabel::create("tri.Node1TriangleIDs",
			ParticleVariable<Stencil7>::getTypeDescription() );

  triNode2TriangleIDsLabel = VarLabel::create("tri.Node2TriangleIDs",
			ParticleVariable<Stencil7>::getTypeDescription() );

  triNode0TriangleIDsLabel_preReloc = VarLabel::create("tri.Node0TriangleIDs+",
			ParticleVariable<Stencil7>::getTypeDescription() );

  triNode1TriangleIDsLabel_preReloc = VarLabel::create("tri.Node1TriangleIDs+",
			ParticleVariable<Stencil7>::getTypeDescription() );

  triNode2TriangleIDsLabel_preReloc = VarLabel::create("tri.Node2TriangleIDs+",
			ParticleVariable<Stencil7>::getTypeDescription() );

  triInContactLabel = VarLabel::create("tri.InContact",
			ParticleVariable<int>::getTypeDescription() );
#endif
}

TriangleLabel::~TriangleLabel()
{
  VarLabel::destroy(triangleCountLabel);
  VarLabel::destroy(triangleIDLabel);
  VarLabel::destroy(triangleIDLabel_preReloc);
  VarLabel::destroy(triMidToN0VectorLabel);
  VarLabel::destroy(triMidToN0VectorLabel_preReloc);
  VarLabel::destroy(triMidToN1VectorLabel);
  VarLabel::destroy(triMidToN1VectorLabel_preReloc);
  VarLabel::destroy(triMidToN2VectorLabel);
  VarLabel::destroy(triMidToN2VectorLabel_preReloc);
  VarLabel::destroy(triCemVecN0VectorLabel);
  VarLabel::destroy(triCemVecN0VectorLabel_preReloc);
  VarLabel::destroy(triCemVecN1VectorLabel);
  VarLabel::destroy(triCemVecN1VectorLabel_preReloc);
  VarLabel::destroy(triCemVecN2VectorLabel);
  VarLabel::destroy(triCemVecN2VectorLabel_preReloc);
  VarLabel::destroy(triUseInPenaltyLabel);
  VarLabel::destroy(triUseInPenaltyLabel_preReloc);
  VarLabel::destroy(triAreaLabel);
  VarLabel::destroy(triAreaLabel_preReloc);
  VarLabel::destroy(triAreaAtNodesLabel);
  VarLabel::destroy(triAreaAtNodesLabel_preReloc);
  VarLabel::destroy(triNormalLabel);
  VarLabel::destroy(triNormalLabel_preReloc);
  VarLabel::destroy(triClayLabel);
  VarLabel::destroy(triClayLabel_preReloc);
#if 0
  VarLabel::destroy(triNode0TriangleIDsLabel);
  VarLabel::destroy(triNode1TriangleIDsLabel);
  VarLabel::destroy(triNode2TriangleIDsLabel);
  VarLabel::destroy(triNode0TriangleIDsLabel_preReloc);
  VarLabel::destroy(triNode1TriangleIDsLabel_preReloc);
  VarLabel::destroy(triNode2TriangleIDsLabel_preReloc);
  VarLabel::destroy(triInContactLabel);
#endif
}
