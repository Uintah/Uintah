#ifndef SCI_project_ColumnMatrix_h
#define SCI_project_ColumnMatrix_h 1

/*
 *  ColumnMatrix.h: for RHS and LHS
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <CoreDatatypes/Datatype.h>
#include <Containers/LockingHandle.h>

#ifdef KCC
#include <iosfwd.h>  // Forward declarations for KCC C++ I/O routines
#else
class ostream;
#endif

namespace SCICore {
namespace CoreDatatypes {

using SCICore::Containers::LockingHandle;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class ColumnMatrix;
typedef LockingHandle<ColumnMatrix> ColumnMatrixHandle;

class ColumnMatrix : public Datatype {
    int rows;
    double* data;
public:

  double* get_rhs() const {return data;}
  void put_lhs(double* lhs) {data = lhs;}
  
    ColumnMatrix(int rows=0);
    ~ColumnMatrix();
    ColumnMatrix(const ColumnMatrix&);
    virtual ColumnMatrix* clone() const;
    ColumnMatrix& operator=(const ColumnMatrix&);
    int nrows() const;
    inline double& operator[](int) const;

    double vector_norm();
    double vector_norm(int& flops, int& memrefs);
    double vector_norm(int& flops, int& memrefs, int beg, int end);

    friend void Mult(ColumnMatrix&, const ColumnMatrix&, double s);
    friend void Mult(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&);
    friend void Mult(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&,
		     int& flops, int& memrefs);
    friend void Mult(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&,
		     int& flops, int& memrefs, int beg, int end);
    friend void Sub(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&);
    friend void Sub(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&,
		    int& flops, int& memrefs);
    friend double Dot(const ColumnMatrix&, const ColumnMatrix&);
    friend double Dot(const ColumnMatrix&, const ColumnMatrix&,
		      int& flops, int& memrefs);
    friend double Dot(const ColumnMatrix&, const ColumnMatrix&,
		      int& flops, int& memrefs, int beg, int end);
    friend void ScMult_Add(ColumnMatrix&, double s, const ColumnMatrix&,
			   const ColumnMatrix&);
    friend void ScMult_Add(ColumnMatrix&, double s, const ColumnMatrix&,
			   const ColumnMatrix&, int& flops, int& memrefs);
    friend void ScMult_Add(ColumnMatrix&, double s, const ColumnMatrix&,
			   const ColumnMatrix&, int& flops, int& memrefs,
			   int beg, int end);

    friend void Copy(ColumnMatrix&, const ColumnMatrix&);
    friend void Copy(ColumnMatrix&, const ColumnMatrix&, int& flops, int& refs,
		     int beg, int end);
    friend void AddScMult(ColumnMatrix&, const ColumnMatrix&, double s, const ColumnMatrix&);
    friend void Add(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&);
    friend void Add(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&);

    void zero();
    void print(ostream&);
    void resize(int);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#include <Util/Assert.h>

inline double& ColumnMatrix::operator[](int i) const
{
    ASSERTRANGE(i, 0, rows);
    return data[i];
}

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:20  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:46  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:36  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:05  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:51  dav
// Import sources
//
//

#endif

