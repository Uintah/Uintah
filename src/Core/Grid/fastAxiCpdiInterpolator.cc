/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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
#include <Core/Grid/fastAxiCpdiInterpolator.h>
#include <Core/Grid/cpdiInterpolator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>

using namespace Uintah;

fastAxiCpdiInterpolator::fastAxiCpdiInterpolator()
{
  d_size = 64;
  d_patch = 0;
}

fastAxiCpdiInterpolator::fastAxiCpdiInterpolator(const Patch* patch):fastCpdiInterpolator(patch)
{
  d_size = 64;
  d_patch = patch;
}

fastAxiCpdiInterpolator::~fastAxiCpdiInterpolator()
{
}

fastAxiCpdiInterpolator* fastAxiCpdiInterpolator::clone(const Patch* patch)
{
  return scinew fastAxiCpdiInterpolator(patch);
}

int fastAxiCpdiInterpolator::findCellAndWeights(const Point& pos,
                                            vector<IntVector>& ni,
                                            vector<double>& S,
                                            const Matrix3& size)
{
  fastCpdiInterpolator::findCellAndWeights(pos,ni,S,size);
  return 27;
}

int fastAxiCpdiInterpolator::findCellAndShapeDerivatives(const Point& pos,
                                                     vector<IntVector>& ni,
                                                     vector<Vector>& d_S,
                                                     const Matrix3& size)
{
  fastCpdiInterpolator::findCellAndShapeDerivatives(pos,ni,d_S,size);
  return 27;
}

int fastAxiCpdiInterpolator::findCellAndWeightsAndShapeDerivatives(const Point& pos,
                                                          vector<IntVector>& ni,
                                                          vector<double>& S,
                                                          vector<Vector>& d_S,
                                                          const Matrix3& size)
{
  fastCpdiInterpolator::findCellAndWeightsAndShapeDerivatives(pos,ni,S,d_S,size);
  return 27;
}

int fastAxiCpdiInterpolator::size()
{
  return d_size;
}
