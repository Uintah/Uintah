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
 *  Matrix.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#ifndef Matrix_h
#define Matrix_h

#include <Core/CCA/spec/cca_sidl.h>

class Matrix: public sci::cca::Matrix{
public:
  Matrix();
  Matrix(int nRow, int nCol);
  ~Matrix();
  double getElement(int row, int col);
  void setElement(int row, int col, double val);
  int numOfRows();
  int numOfCols();
  Matrix & operator=(const Matrix &m);
private:
  void copy(const Matrix &m);
  int nRow;
  int nCol;
  double *data;
};

#endif

