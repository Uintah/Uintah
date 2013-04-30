/*
 * The MIT License
 *
 * Copyright (c) 1997-2013 The University of Utah
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

#include <CCA/Components/MD/MDSystem.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <iostream>

#include <sci_values.h>

using namespace Uintah;

MDSystem::MDSystem()
{

}

MDSystem::~MDSystem()
{

}

MDSystem::MDSystem(ProblemSpecP& ps)
{
  ProblemSpecP mdsystem_ps = ps->findBlock("MDSystem");
  mdsystem_ps->get("pressure", d_pressure);
  mdsystem_ps->get("temperature", d_temperature);
  mdsystem_ps->get("orthorhombic", d_orthorhombic);
  mdsystem_ps->get("ghostcells", d_numGhostCells);
  mdsystem_ps->get("maxiterations", d_maxPolarizableIterations);
  if (d_orthorhombic) {
    Vector InputBox;
    mdsystem_ps->get("boxSize", InputBox);
    for (size_t row = 0; row < 3; ++row) {
      for (size_t col = 0; col < 3; ++col) {
        d_unitCell(row, col) = 0.0;
      }
    }
    d_unitCell(0, 0) = InputBox[0];
    d_unitCell(1, 1) = InputBox[1];
    d_unitCell(2, 2) = InputBox[2];
  } else {
    // Read in non orthorhombic unit cell
  }
  this->calcCellVolume();
  d_inverseCell = d_unitCell.Inverse();
}

void MDSystem::calcCellVolume()
{
  Vector A, B, C;
  A[0] = d_unitCell(0, 0);
  A[1] = d_unitCell(0, 1);
  A[2] = d_unitCell(0, 2);
  B[0] = d_unitCell(1, 0);
  B[1] = d_unitCell(1, 1);
  B[2] = d_unitCell(1, 2);
  C[0] = d_unitCell(2, 0);
  C[1] = d_unitCell(2, 1);
  C[2] = d_unitCell(2, 2);

  d_cellVolume = Dot(Cross(A, B), C);
}
