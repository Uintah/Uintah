
/*
 *  Matrix.h: Matrix definitions
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Matrix_h
#define SCI_project_Matrix_h 1

#include <SCICore/CoreDatatypes/Datatype.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Containers/String.h>

namespace SCICore {
namespace CoreDatatypes {

using SCICore::Containers::LockingHandle;
using SCICore::Containers::Array1;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class SymSparseRowMatrix;
class SparseRowMatrix;
class DenseMatrix;
class ColumnMatrix;
class Matrix;
class MatrixRow;
typedef LockingHandle<Matrix> MatrixHandle;

class SCICORESHARE Matrix : public Datatype {
protected:
    enum Sym {
        non_symmetric,
        symmetric
    };
    enum Representation {
	sparse,
	symsparse,
	dense,
	tridiagonal,
	other
    };

    Sym sym;
    Matrix(Sym symmetric, Representation dense);
    int extremaCurrent;
private:
    Representation rep;
public:

  virtual double* get_val(){return NULL;}
  virtual int* get_row(){return NULL;}
  virtual int* get_col() {return NULL;}

  
    clString getType();
    SymSparseRowMatrix* getSymSparseRow();
    SparseRowMatrix* getSparseRow();
    DenseMatrix* getDense();
    int is_symmetric();
    void is_symmetric(int symm);
    virtual ~Matrix();
    virtual Matrix* clone();
    virtual double& get(int, int)=0;
    inline MatrixRow operator[](int r);

    virtual void zero()=0;
    virtual int nrows() const=0;
    virtual int ncols() const=0;
    virtual void getRowNonzeros(int r, Array1<int>& idx, Array1<double>& v)=0;
    virtual double minValue()=0;
    virtual double maxValue()=0;
    virtual void mult(const ColumnMatrix& x, ColumnMatrix& b,
		      int& flops, int& memrefs, int beg=-1, int end=-1) const=0;
    virtual void mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
				int& flops, int& memrefs, int beg=-1, int end=-1)=0;

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class SCICORESHARE MatrixRow {
    Matrix* matrix;
    int row;
public:
    inline MatrixRow(Matrix* matrix, int row) : matrix(matrix), row(row) {}
    inline ~MatrixRow() {}

    inline double& operator[](int col) {return matrix->get(row, col);}
};

inline MatrixRow Matrix::operator[](int row)
{
    return MatrixRow(this, row);
}

void Mult(ColumnMatrix&, const Matrix&, const ColumnMatrix&);

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:47  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:23  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:48  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:40  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:09  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

#endif
