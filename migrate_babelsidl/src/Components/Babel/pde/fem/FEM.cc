/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  License for the specific language governing rights and limitations under
  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  DEALINGS IN THE SOFTWARE.
*/


/*
 *  FEM.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#include "FEM.h"

#include <stdlib.h>
#include <iostream>

using namespace SCIRun;

FEMgenerator::FEMgenerator(sidl::array<int>& dirichletNodes,
                           sidl::array<double>& dirichletValues)
{
  this->dirichletNodes = dirichletNodes; 
  this->dirichletValues = dirichletValues;
}

// Computes the 1st order differentiation arrays for linear triangles.
// x, y 3 x 1 arrays
void FEMgenerator::diffTriangle(double b[3],
                                double c[3],
                                double &area,
                                const double x[3],
                                const double y[3])
{
  double A2 = (x[1] * y[2] - x[2] * y[1]) -
              (x[0] * y[2] - x[2] * y[0]) +
              (x[0] * y[1] - x[1] * y[0]);
  area = A2 / 2;
  for (int i = 0; i < 3; i++) {
    int i1 = (i + 1) % 3;
    int i2 = (i + 2) % 3;
    b[i] = (y[i1] - y[i2]) / A2;
    c[i] = (x[i2] - x[i1]) / A2;
  }
}

void FEMgenerator::localMatrices(double A[3][3], double f[3],
                                 const double x[3], const double y[3])
{
  double b[3], c[3], area;
  diffTriangle(b, c, area, x, y);

  if (fabs(area) < 1e-10) {
    std::cerr << "\n Bad triangle: area=0 x y=" << std::endl;
    for (int i = 0; i < 3; i++) {
      std::cerr << x[i] << " " << y[i] << ";";
    }
  }
  for (int i = 0; i < 3; i++) {
    int i1 = (i + 1) % 3;
    int i2 = (i + 2) % 3;
    f[i] = area * (2 * source(i) + source(i1) + source(i2)) / 12;
    for (int j = 0; j < 3; j++) {
      A[i][j] = area * (b[i] * b[j] + c[i] * c[j]);
    }
  }
}

// Create the global matrices from the local matrices
void FEMgenerator::globalMatrices(sidl::array<double>& nodes1d,
                                  sidl::array<int>& tmesh1d)
{
  int N = nodes1d.length() / 2;

  Ag = sidl::array<double>::create2dRow(N, N);
  fg = sidl::array<double>::create1d(N);

  for (int i = 0; i < N; i++) {
    fg.set(i,0);
    for (int j = 0; j < N; j++) {
      Ag.set(i,j,0);
    }
  }

  //get number of triangles
  int Ntri = tmesh1d.length() / 3;

  for (int i = 0; i < Ntri; i++) {
    double x[3], y[3];
    for (int j = 0; j < 3; j++) {
      x[j] = nodes1d.get(tmesh1d.get(i * 3 + j) * 2);
      y[j] = nodes1d.get(tmesh1d.get(i * 3 + j) * 2 + 1);
    }
    double A[3][3], f[3];
    localMatrices(A, f, x, y);
    for(int row = 0; row < 3; row++) {
      int grow = tmesh1d.get(i * 3 + row);
      if (! isConst(grow)) {
        fg.set(grow, fg.get(grow) + f[row]);
        for (int col = 0; col < 3; col++) {
          int gcol = tmesh1d.get(i * 3 + col);
          if (! isConst(gcol)) {
            Ag.set(grow, gcol, Ag.get(grow,gcol) + A[row][col]);
          } else {
            //u(gcol) is the constant boundary value
            fg.set(grow, fg.get(grow) - (A[row][col] * boundary(gcol)));
          }
        }
      }
    }
  }
  //put into global function
  for(int grow=0; grow<N; grow++){
    if(isConst(grow)){
      for(int k=0; k<N; k++){
        Ag.set(grow,k,0);
        Ag.set(k,grow,0);
      }
      Ag.set(grow,grow,1);
      fg.set(grow,boundary(grow));
    }
  }

  //this->fg=fg;
}

double FEMgenerator::source(int index)
{
  return 0;
}

double FEMgenerator::boundary(int index)
{
  for (unsigned int i = 0; i < dirichletNodes.length(); i++){
    if (index == dirichletNodes[i]) {
      return dirichletValues[i];
    }
  }
  return 0;
}

bool FEMgenerator::isConst(int index)
{
  for(unsigned int i=0; i<dirichletNodes.length(); i++){
    if (index == dirichletNodes[i]) {
      return true;
    }
  }
  return false;
}

