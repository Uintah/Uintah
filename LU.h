#ifndef Uintah_Components_Arches_LU_h
#define Uintah_Components_Arches_LU_h

#include <Core/Grid/Variables/VarTypes.h>
#include <assert.h>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

/**
 *  @class LU
 *  @author James Sutherland and Charles Reid
 *  @date   November 2005 and April 2009
 
 *  @brief Performs LU decomposition of a matrix system using Crout's method with partial pivoting (see "Numerical Recipes" by Press et al).
 */

namespace Uintah {

class LU{
 public:
  LU( const int dim, const int bandwidth );

  LU( LU &CopyThis );
  
  ~LU();

  inline double& operator ()(const int row, const int col){
    assert( row <= dim_ );
    assert( col <= dim_ );
    isReady_ = false;
    return AA_(row,col);
  };

  /** @brief Performs the LU decomposition/factorization using Crout's method with partial pivoting. 
  */
  void decompose();

  /** @brief Performs back-substitution of the LU system given a right-hand side, and overwrites RHS with solution vector.
   *  @param rhs Pointer to right-hand side vector.
   */
  void back_subs( double* rhs );

  /** @brief Dumps the contents of the matrix A to std::out.
  */
  void dump();

  /** @brief Returns dimension of matrix A.
  */
  const int getDimension() {
    return dim_; };

 private:

  LU();  // no default constructor.

  class SparseMatrix{
  public:
    SparseMatrix( const int dim, const int bandwidth );
    ~SparseMatrix();
    inline double& operator ()(const int row, const int col){
      return AA_[row][col];
    };
    void dump();
  private:
    SparseMatrix();
    double **AA_;
    const int dim_, band_;
  };

  const int dim_;
  bool isReady_;
  bool isSingular_;
  SparseMatrix AA_;

  std::vector<int> indx;

};

}
#endif
