
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

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Containers/String.h>

namespace SCICore {
namespace Datatypes {

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

    int separate_raw;
    clString raw_filename;

    Sym sym;
    Matrix(Sym symmetric, Representation dense);
    int extremaCurrent;
private:
    Representation rep;
public:

  virtual double* get_val(){return 0;}
  virtual int* get_row(){return 0;}
  virtual int* get_col() {return 0;}

  
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
		      int& flops, int& memrefs, int beg=-1, int end=-1, int spVec=0) const=0;
    virtual void mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
				int& flops, int& memrefs, int beg=-1, int end=-1, int spVec=0)=0;

    // separate raw files
    void set_raw(int v) { separate_raw = v; }
    int get_raw() { return separate_raw; }
    void set_raw_filename( clString &f ) { raw_filename = f; separate_raw = 1;}
    clString &get_raw_filename() { return raw_filename; }

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

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.5  2000/07/12 15:45:08  dmw
// Added Yarden's raw output thing to matrices, added neighborhood accessors to meshes, added ScalarFieldRGushort
//
// Revision 1.4  1999/09/08 02:26:48  sparker
// Various #include cleanups
//
// Revision 1.3  1999/08/25 03:48:35  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
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
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

#endif
