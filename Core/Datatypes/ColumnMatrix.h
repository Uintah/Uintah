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

#ifndef SCI_project_ColumnMatrix_h
#define SCI_project_ColumnMatrix_h 1

#include <SCICore/share/share.h>

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Util/FancyAssert.h>

#include <iosfwd>  // Forward declarations for KCC C++ I/O routines

namespace SCICore {
namespace Datatypes {

using SCICore::Containers::LockingHandle;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class ColumnMatrix;
typedef LockingHandle<ColumnMatrix> ColumnMatrixHandle;

class SCICORESHARE ColumnMatrix : public Datatype {
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

    friend SCICORESHARE void Mult(ColumnMatrix&, const ColumnMatrix&, double s);
    friend SCICORESHARE void Mult(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&);
    friend SCICORESHARE void Mult(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&,
		     int& flops, int& memrefs);
    friend SCICORESHARE void Mult(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&,
		     int& flops, int& memrefs, int beg, int end);
    friend SCICORESHARE void Sub(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&);
    friend SCICORESHARE void Sub(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&,
		    int& flops, int& memrefs);
    friend SCICORESHARE double Dot(const ColumnMatrix&, const ColumnMatrix&);
    friend SCICORESHARE double Dot(const ColumnMatrix&, const ColumnMatrix&,
		      int& flops, int& memrefs);
    friend SCICORESHARE double Dot(const ColumnMatrix&, const ColumnMatrix&,
		      int& flops, int& memrefs, int beg, int end);
    friend SCICORESHARE void ScMult_Add(ColumnMatrix&, double s, const ColumnMatrix&,
			   const ColumnMatrix&);
    friend SCICORESHARE void ScMult_Add(ColumnMatrix&, double s, const ColumnMatrix&,
			   const ColumnMatrix&, int& flops, int& memrefs);
    friend SCICORESHARE void ScMult_Add(ColumnMatrix&, double s, const ColumnMatrix&,
			   const ColumnMatrix&, int& flops, int& memrefs,
			   int beg, int end);

    friend SCICORESHARE void Copy(ColumnMatrix&, const ColumnMatrix&);
    friend SCICORESHARE void Copy(ColumnMatrix&, const ColumnMatrix&, int& flops, int& refs,
		     int beg, int end);
    friend SCICORESHARE void AddScMult(ColumnMatrix&, const ColumnMatrix&, double s, const ColumnMatrix&);
    friend SCICORESHARE void Add(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&);
    friend SCICORESHARE void Add(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&);

    void zero();
    void print(std::ostream&);
    void resize(int);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#include <SCICore/Util/Assert.h>

inline double& ColumnMatrix::operator[](int i) const
{
    ASSERTRANGE(i, 0, rows);
    return data[i];
}

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.5  2000/03/23 10:29:19  sparker
// Use new exceptions/ASSERT macros
// Fixed compiler warnings
//
// Revision 1.4  1999/10/07 02:07:30  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/08/25 03:48:31  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:44  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
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
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:51  dav
// Import sources
//
//

#endif

