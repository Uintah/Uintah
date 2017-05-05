#ifndef SolveLinearSystem_h
#define SolveLinearSystem_h

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/multi_array.hpp>

#include <CCA/Components/Wasatch/Coal/CharOxidation/CharData.h>


 // Solves x = (A^-1)b with LU-factorization
bool SolveLinearSystem( const CHAR::Array2D& A,
                        const CHAR::Vec&     b,
                              CHAR::Vec&     x )
{
// typedef permutation_matrix<std::size_t> PermMat;
 const size_t n = A.shape()[0];

 // see if the A is square
 if( n != A.shape()[1] ){
   std::cout<<"\n\nInputed matrix must be square!!!\n\n";
 }

 // check whether A, b and x have consistent dimensions
 if( n != x.size() || n != b.size() ){
   std::cout<<"\n\nInputed matrix and vectors do not have consistent dimensions !!!\n\n";
 }

 boost::numeric::ublas::matrix<double> A_(n,n);
 boost::numeric::ublas::vector<double> x_(n);

 for( size_t i = 0; i < n; ++i ){
   x_(i) = b[i];

   for( size_t j = 0; j < n; ++j ){
     A_(i,j) = A[i][j];
   }
 }

 boost::numeric::ublas::permutation_matrix<std::size_t> pm(n);

 // do LU-factorization
 int res = lu_factorize(A_, pm);
 if( res != 0 ) return false;

 // back-substitute to get the solution
 boost::numeric::ublas::lu_substitute(A_, pm, x_);

 for( size_t i = 0; i < n; ++i ){
   x[i] = x_(i);
 }

 return true;
}

#endif
