#ifdef __GNUG__

#include "Matrix.h"
#include "Matrix.cc"
#include "SymmetricMatrix.h"
#include "SymmetricMatrix.cc"
#include "BoundedArray.h"
#include "BoundedArray.cc"

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

