#ifdef __GNUG__

#include "Matrix.h"
#include "Matrix.cc"
#include "SymmetricMatrix.h"
#include "SymmetricMatrix.cc"
#include "../boundedarray/BoundedArray.h"
#include "../boundedarray/BoundedArray.cc"

// Instantiate the Matrix class for doubles 

template class Matrix<double>;
template class Matrix<float>;
template Matrix<double> operator*(Matrix<double> const &, Matrix<double> const &);
template Matrix<float> operator-(Matrix<float> const &, Matrix<float> const &);

template class BoundedArray<double>;
template class BoundedArray<float>;
template BoundedArray<double> operator*(Matrix<double> const &, BoundedArray<double> const &);


template class SymmetricMatrix<double>;
template SymmetricMatrix<double> operator*(double const &, SymmetricMatrix<double> const &);
template Matrix<double> operator*(SymmetricMatrix<double> const &, SymmetricMatrix<double> const &);

#endif

// $Log$
// Revision 1.1  2000/02/24 06:11:59  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:54  sparker
// Stuff may actually work someday...
//
// Revision 1.3  1999/01/26 16:06:09  guilkey
// Removed ident capability due to compile time conflicts.
//
// Revision 1.2  1999/01/26 00:07:26  campbell
// Added ident and logging capabilities
//
