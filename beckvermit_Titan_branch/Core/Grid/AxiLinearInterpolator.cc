/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <Core/Grid/AxiLinearInterpolator.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;
    
AxiLinearInterpolator::AxiLinearInterpolator()
{
  d_size = 8;
  d_patch = 0;
}

AxiLinearInterpolator::AxiLinearInterpolator(const Patch* patch)
{
  d_size = 8;
  d_patch = patch;
}

AxiLinearInterpolator::~AxiLinearInterpolator()
{
}

AxiLinearInterpolator* AxiLinearInterpolator::clone(const Patch* patch)
{
  return scinew AxiLinearInterpolator(patch);
 }
    
//__________________________________
void AxiLinearInterpolator::findCellAndWeights(const Point& pos,
                                           vector<IntVector>& ni, 
                                           vector<double>& S,
                                           const Matrix3& size,
                                           const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(pos );
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  ni[0] = IntVector(ix, iy, 0);
  ni[1] = IntVector(ix, iy, 1);
  ni[2] = IntVector(ix, iy+1, 0);
  ni[3] = IntVector(ix, iy+1, 1);
  ni[4] = IntVector(ix+1, iy, 0);
  ni[5] = IntVector(ix+1, iy, 1);
  ni[6] = IntVector(ix+1, iy+1, 0);
  ni[7] = IntVector(ix+1, iy+1, 1);
  double fx = cellpos.x() - ix;
  double fy = cellpos.y() - iy;
  double fx1 = 1-fx;
  double fy1 = 1-fy;
  S[0] = fx1 * fy1 * 0.5;
  S[1] = S[0];
  S[2] = fx1 * fy  * 0.5;
  S[3] = S[2];
  S[4] = fx  * fy1 * 0.5;
  S[5] = S[4];
  S[6] = fx  * fy  * 0.5;
  S[7] = S[6];
}

//______________________________________________________________________
// 
void AxiLinearInterpolator::findCellAndShapeDerivatives(const Point& pos,
                                                        vector<IntVector>& ni,
                                                        vector<Vector>& d_S,
                                                        const Matrix3& size,
                                                        const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(pos);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  ni[0] = IntVector(ix, iy, 0);
  ni[1] = IntVector(ix, iy, 1);
  ni[2] = IntVector(ix, iy+1, 0);
  ni[3] = IntVector(ix, iy+1, 1);
  ni[4] = IntVector(ix+1, iy, 0);
  ni[5] = IntVector(ix+1, iy, 1);
  ni[6] = IntVector(ix+1, iy+1, 0);
  ni[7] = IntVector(ix+1, iy+1, 1);
  double fx = cellpos.x() - ix;
  double fy = cellpos.y() - iy;
  double fx1 = 1-fx;
  double fy1 = 1-fy;
  d_S[0] = Vector(-fy1 * 0.5, -fx1 * 0.5, 0.);
  d_S[1] = d_S[0];
  d_S[2] = Vector(-fy  * 0.5,  fx1 * 0.5, 0.);
  d_S[3] = d_S[2];
  d_S[4] = Vector( fy1 * 0.5, -fx  * 0.5, 0.);
  d_S[5] = d_S[4];
  d_S[6] = Vector( fy  * 0.5,  fx  * 0.5, 0.);
  d_S[7] = d_S[6];
}

void 
AxiLinearInterpolator::findCellAndWeightsAndShapeDerivatives(const Point& pos,
                                                         vector<IntVector>& ni,
                                                         vector<double>& S,
                                                         vector<Vector>& d_S,
                                                         const Matrix3& size,
                                                         const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(pos);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  ni[0] = IntVector(ix, iy, 0);
  ni[1] = IntVector(ix, iy, 1);
  ni[2] = IntVector(ix, iy+1, 0);
  ni[3] = IntVector(ix, iy+1, 1);
  ni[4] = IntVector(ix+1, iy, 0);
  ni[5] = IntVector(ix+1, iy, 1);
  ni[6] = IntVector(ix+1, iy+1, 0);
  ni[7] = IntVector(ix+1, iy+1, 1);
  double fx = cellpos.x() - ix;
  double fy = cellpos.y() - iy;
  double fx1 = 1-fx;
  double fy1 = 1-fy;
  S[0] = fx1 * fy1 * 0.5;
  S[1] = S[0];
  S[2] = fx1 * fy  * 0.5;
  S[3] = S[2];
  S[4] = fx  * fy1 * 0.5;
  S[5] = S[4];
  S[6] = fx  * fy  * 0.5;
  S[7] = S[6];
  double r = pos.x();
  d_S[0] = Vector(-fy1 *0.5, -fx1 *0.5, S[0]/r);
  d_S[1] = d_S[0];
  d_S[2] = Vector(-fy  *0.5,  fx1 *0.5, S[2]/r);
  d_S[3] = d_S[2];
  d_S[4] = Vector( fy1 *0.5, -fx  *0.5, S[4]/r);
  d_S[5] = d_S[4];
  d_S[6] = Vector( fy  *0.5,  fx  *0.5, S[6]/r);
  d_S[7] = d_S[6];
}

int AxiLinearInterpolator::size()
{
  return d_size;
}
