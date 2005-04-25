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
 *  Matrix.cc
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#include <CCA/Components/LinSolver/LinSolver.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "Matrix.h"

Matrix::Matrix()
{
  nRow=nCol=0;
  data=0;
}

Matrix::Matrix(int nRow, int nCol)
{
  this->nRow=nRow;
  this->nCol=nCol;
  data=new double[nRow*nCol];
}  

Matrix::~Matrix()
{
  if(data!=0) delete []data;
}

void Matrix::copy(const Matrix &m)
{
  if(data!=0) delete data;
  nRow=m.nRow;
  nCol=m.nCol;
  data=new double[nRow*nCol];
  memcpy(data, m.data, nRow*nCol*sizeof(double));
}

Matrix & Matrix::operator=(const Matrix &m)
{
  copy(m);
  return *this;
}

double Matrix::getElement(int row, int col)
{
  return data[row*nCol+col];
}

void Matrix::setElement(int row, int col, double val)
{
  data[row*nCol+col]=val;
}


int Matrix::numOfRows()
{
  return nRow;
}

int Matrix::numOfCols()
{
  return nCol;
}

 


