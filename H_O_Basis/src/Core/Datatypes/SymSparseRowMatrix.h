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
 *  SymSparseRowMatrix.h:  Symmetric Sparse Row Matrices
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_SymSparseRowMatrix_h
#define SCI_project_SymSparseRowMatrix_h 1

#include <Core/share/share.h>

#include <Core/Datatypes/Matrix.h>
#include <Core/Containers/Array1.h>

namespace SCIRun {

class AddMatrices;
class SCICORESHARE SymSparseRowMatrix : public Matrix {
    int nnrows;
    int nncols;
    double dummy;
    double minVal;
    double maxVal;
protected:
public:
    int* rows;
    int* columns;
    int nnz;
    double* a;
    int* upper_columns;
    int* upper_rows;
    double* upper_a;
    int upper_nnz;
    void compute_upper();
    virtual double* get_val(){return a;}
    virtual int* get_row(){return rows;}
    virtual int* get_col(){return columns;}
  
    SymSparseRowMatrix();
    SymSparseRowMatrix(int, int, Array1<int>&, Array1<int>&);
    SymSparseRowMatrix(int, int, int*, int*, int);
    virtual ~SymSparseRowMatrix();
    SymSparseRowMatrix(const SymSparseRowMatrix&);
    SymSparseRowMatrix& operator=(const SymSparseRowMatrix&);
    virtual double& get(int, int);
    int getIdx(int, int);
    virtual void put(int, int, const double&);
    virtual int nrows() const;
    virtual int ncols() const;
    virtual double minValue();
    virtual double maxValue();
    double density();
    virtual void getRowNonzeros(int r, Array1<int>& idx, Array1<double>& val);
    virtual void solve(ColumnMatrix&);
    virtual void zero();
    virtual void mult(const ColumnMatrix& x, ColumnMatrix& b,
		      int& flops, int& memrefs, int beg=-1, int end=-1,
		      int spVec=0) const;
    virtual void mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
				int& flops, int& memrefs, int beg=-1, int end=-1, int spVec=0);
    virtual void print() const;
    virtual void print(std::ostream &) const;

    MatrixRow operator[](int r);
    friend SCICORESHARE class AddMatrices;

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace SCIRun

#endif


