/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include <Core/Grid/axiCpdiInterpolator.h>
#include <Core/Grid/cpdiInterpolator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>

using namespace SCIRun;
using namespace Uintah;

axiCpdiInterpolator::axiCpdiInterpolator()
{
  d_size = 64;
  d_patch = 0;
}

axiCpdiInterpolator::axiCpdiInterpolator(const Patch* patch):cpdiInterpolator(patch)
{
  d_size = 64;
  d_patch = patch;
}

axiCpdiInterpolator::~axiCpdiInterpolator()
{
}

axiCpdiInterpolator* axiCpdiInterpolator::clone(const Patch* patch)
{
  return scinew axiCpdiInterpolator(patch);
}

void axiCpdiInterpolator::findCellAndWeights(const Point& pos,
					    vector<IntVector>& ni,
					    vector<double>& S,
					    const Matrix3& size,
                                            const Matrix3& defgrad)
{
  Matrix3 defgrad1=Matrix3(defgrad(0,0),defgrad(0,1),defgrad(0,2),
                           defgrad(1,0),defgrad(1,1),defgrad(1,2),
                           defgrad(2,0),defgrad(2,1),1);
  cpdiInterpolator::findCellAndWeights(pos,ni,S,size,defgrad1);
}

void axiCpdiInterpolator::findCellAndShapeDerivatives(const Point& pos,
						     vector<IntVector>& ni,
						     vector<Vector>& d_S,
						     const Matrix3& size,
                                                     const Matrix3& defgrad)
{
  Matrix3 defgrad1=Matrix3(defgrad(0,0),defgrad(0,1),defgrad(0,2),
                           defgrad(1,0),defgrad(1,1),defgrad(1,2),
                           defgrad(2,0),defgrad(2,1),1);
  cpdiInterpolator::findCellAndShapeDerivatives(pos,ni,d_S,size,defgrad1);
}

void axiCpdiInterpolator::findCellAndWeightsAndShapeDerivatives(const Point& pos,
							  vector<IntVector>& ni,
							  vector<double>& S,
							  vector<Vector>& d_S,
							  const Matrix3& size,
                                                          const Matrix3& defgrad)
{
  Matrix3 defgrad1=Matrix3(defgrad(0,0),defgrad(0,1),defgrad(0,2),
                           defgrad(1,0),defgrad(1,1),defgrad(1,2),
                           defgrad(2,0),defgrad(2,1),1);
  cpdiInterpolator::findCellAndWeightsAndShapeDerivatives(pos,ni,S,d_S,
                                                          size,defgrad1);
}

int axiCpdiInterpolator::size()
{
  return d_size;
}
