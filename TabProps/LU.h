#ifndef LU_h
#define LU_h

#include <assert.h>

/**
 *  @class LU
 *  @author James C. Sutherland
 *  @date   November, 2005
 *
 *  @brief Supports LU-decompositon for a matrix.
 */
class LU{
 public:
  LU( const int dim, const int bandwidth );
  ~LU();

  inline double& operator ()(const int row, const int col){
    assert( row <= dim_ );
    assert( col <= dim_ );
    isReady_ = false;
    return AA_(row,col);
  };

  // perform the LU-factorization
  void decompose();

  // perform back-substitution given the rhs.
  // Over-writes the rhs with the solution vector.
  void back_subs( double* rhs );

  void dump();

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
  SparseMatrix AA_;

};

#endif
