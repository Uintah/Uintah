/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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

 


