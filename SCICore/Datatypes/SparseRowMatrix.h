
/*
 *  SparseRowMatrix.h:  Sparse Row Matrices
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_SparseRowMatrix_h
#define SCI_project_SparseRowMatrix_h 1

#include <SCICore/Datatypes/Matrix.h>
#include <SCICore/Containers/Array1.h>

namespace SCICore {
namespace Datatypes {

using SCICore::Containers::Array1;

class AddMatrices;
class SCICORESHARE SparseRowMatrix : public Matrix {
    int nnrows;
    int nncols;
    double dummy;
    double minVal;
    double maxVal;
protected:
public:
    int* columns;
    int* rows;
    int nnz;
    double* a;
    SparseRowMatrix();
    SparseRowMatrix(int, int, Array1<int>&, Array1<int>&);
    SparseRowMatrix(int, int, int*, int*, int, double*);
    SparseRowMatrix(int, int, int*, int*, int);
    virtual ~SparseRowMatrix();
    SparseRowMatrix(const SparseRowMatrix&);
    SparseRowMatrix& operator=(const SparseRowMatrix&);

    void transpose( SparseRowMatrix &);
    virtual double& get(int, int);
    int getIdx(int, int);
    virtual void put(int, int, const double&);
    virtual void add(int, int, const double&);
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
				int& flops, int& memrefs, int beg=-1, 
				int end=-1, int spVec=0);
    virtual void print();
    MatrixRow operator[](int r);
    friend class AddMatrices;
    virtual double* get_val(){return a;}
    virtual int* get_row(){return rows;}
    virtual int* get_col(){return columns;}
  int get_nnz() { return nnz; }
    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.6  2000/10/29 04:46:17  dmw
// changed private/public status, added a flag for whether datavalues were associate with elements or nodes
//
// Revision 1.5  2000/07/12 15:45:10  dmw
// Added Yarden's raw output thing to matrices, added neighborhood accessors to meshes, added ScalarFieldRGushort
//
// Revision 1.4  2000/03/04 00:18:30  dmw
// added new Mesh BC and fixed sparserowmatrix bug
//
// Revision 1.3  1999/08/25 03:48:41  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:54  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:29  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:56  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:46  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:18  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

#endif
