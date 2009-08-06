#ifndef Uintah_Components_Arches_LU_h
#define Uintah_Components_Arches_LU_h

#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/Parallel.h>
#include <assert.h>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

/**
  * @class    LU
  * 
  * @brief    Performs LU decomposition of a matrix system using Crout's method with partial pivoting;
  *           additionally, the class can optionally implement iterative refinement of X=A\B.
  * @details  This class may be extended in the future to also use QR decomposition.
  *
  * @author   James Sutherland (major modifications by Charles Reid)
  * @date     November 2005 (modified April 2009)
  */

namespace Uintah {

class LU{
 public:
  
  //=========================================================
  // Constructors/destructors

  /** @brief            Default constructor
    *
    * @param dim        Dimension of matrix (square matrix)
    * @param bandwidth  Number of diagonals - not used (matrices are stored densely) */
  LU( const int dim, const int bandwidth );

  /** @brief            Copy constructor
    * @param CopyThis   The LU object to be copied */
  LU( LU &CopyThis );
  
  ~LU();



  // =======================================================
  // Solution functions

  /** @brief      Solves the matrix system AX=B; does not overwrite RHS vector
    *
    * @param rhs  Right-hand side vector (the B of AX=B)
    * @param soln Solution vector (the X of AX=B)
    *
    * @returns    Solution vector is returned in soln */
  void solve( double* rhs, double* soln, bool doIterativeSolve );





  //=========================================================
  // Iterative refinement (public methods)

  /** @brief      Performs iterative refinement 
    *
    * @param rhs  Right-hand side vector (B of AX=B)
    * @param Xsingle    Single working precision (i.e. double) solution vector (X of AX=B)
    *
    * @returns    Iterated solution X(i), where i = iteration where any of the stopping criteria were met  */
  vector<long double> iterative_refinement( double* rhs, double* Xsingle );
  // SHOULD THIS BE LONG DOUBLE??? OR IS IT JUST THE SUBTRACTION THAT NEEDS TO BE DONE IN DOUBLE PREC?
 
  /** @brief      Access function for final relative norm (error estimate)
    * @returns    final_relative_norm variable, which contains the relative norm of 
    *             \f$ \Vert dx \Vert_{\infty} / \Vert x \Vert_{\infty} \f$ for the last step 
    *             of iterative refinement   */
  double getConvergenceRate() {
    assert(isRefined_);
    return rho_max; }


  //========================================================
  // Utility functions

  /** @brief      Multiply the LU matrix by a vector: A*Y = Z (template class)
    *
    * @param Y    The vector that the LU matrix is being multiplied by
    * @param Z    The vector that holds the result of the multiplication
    *
    * @returns    Result is returned in vector Z */
  template <typename T1, typename T2> 
  void 
  multiply( T1* Y, T2* Z )
  {
    int i,j;
    T2 rowsum;
    // loop over all rows
    for (i = 0; i < dim_; ++i) {
      rowsum = 0;
      // loop over all columns
      for (j = 0; j < dim_; ++j) {
        rowsum += AA_(i,j)*Y[j];
      }
      Z[i] = rowsum;
    }
  }

  /** @brief      Calculates the residual of AX-B (template class)
    *
    * @param soln Solution vector (the X of AX-B)
    * @param rhs  Right-hand side vector (the B of AX-B) 
    * @param Res  Vector of arbitrary type (float, double, etc.) that holds the residual values
    *
    * @returns    Residual is returned in residual vector */
  template <typename T1, typename T2, typename T3> 
  void
  getResidual( T1* soln, T2* rhs, T3* res)
  {
    // multiply A by X and put the result in Res
    multiply( soln, res );
  
    // sutract B from Res (from A*X)
    for (int z = 0; z < dim_; ++z) {
      res[z] = res[z] - rhs[z];
    }
  }
 
  /** @brief      Calculate the norm of a function
    *
    * @param a    Vector whose norm is being calculated
    * @param type Type of norm to calculate:  \n
    *             0 = L_infty norm            \n
    *             1 = L1 norm                 \n
    *             2 = L2 norm                 \n
    *             ...                         \n
    *             p = Lp norm                 */
  template <typename T>
  T getNorm( T* a, unsigned int type )
  {
    double TheNorm = 0;
	 
    if (type > 5) {
      type = 0;
    }
    if (type == 0) {
    // L_infty norm
      for (int z = 0; z < dim_; ++z) {
    	  if (fabs(a[z]) > TheNorm) {
    	    TheNorm = fabs(a[z]);
    	  }
    	}
    } else if (type <= 5) {
      // L_p norm
      // || x ||_p   [=]   ( sum( |x_i|^p ) )^(1/p)
      for (int z = 0; z < dim_; ++z) {
        TheNorm += pow(fabs( (double)a[z] ),(int)type);
      }
      TheNorm = pow( TheNorm, 1.0/type );  
    } 
    return TheNorm;
  }

  /** @brief      Dumps the contents of the LU matrix to std::out. */
  void dump();

  /** @brief      Returns dimension of the LU object
    * 
    * @returns    The dimension of the LU object */
  const int getDimension() {
    return dim_; };

  /** @brief            Define (i,j) operator to access row i and column j 
    *
    * @param row        Row in LU object to access
    * @param col        Column in LU object to access */
  inline double& operator ()(const int row, const int col){
    assert( row <= dim_ );
    assert( col <= dim_ );
    return AA_(row,col);
  };


 private:

  LU();  // no default constructor.


  // ===============================================
  // LU Decomposition & Back-Substitution using Crout's Method

  /** @brief      Performs the LU decomposition/factorization using Crout's method with partial pivoting. */
  void decompose();

  /** @brief      (Overloaded) Performs back-substitution of the LU system given a right-hand side; overwrites RHS with solution vector.
    * @param rhs  Pointer to right-hand side vector. */
  void back_subs( double* rhs );

  /** @brief      (Overloaded) Performs back-substitution method for LU system; does not overwrite RHS vector
    * @param rhs  Pointer to right-hand side vector
    * @param soln Pointer to solution vector */
  void back_subs( double* rhs, double* soln );



  // ===============================================
  // Iterative Refinement 

  /** @brief      Update the state of X and the maximum error estimate 
    *
    * @param normXi       Norm (Linfty) of X(i)
    * @param normdXi      Norm (Linfty) of dX(i) (prior iteration)
    * @param normdXip1    Norm (Linfty) of dX(i+1) (current iteration)
    * @param finRelNorm   Norm (Linfty) of final relative error:
    *                     \f[ \text{FinalRelativeNorm} = \Vert dx^{(i+1)} \Vert / \Vert x^{(i)} \Vert \f]
    */
  void update_xstate( double norm_X_i, 
                      double norm_dX_i, 
                      double norm_dX_ip1,
                      int* x_state );


  // ===============================================
  // Sparse Matrix class

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

 

  // ===============================================
  // Private members

  const int dim_;                 /// Dimension of LU object
  SparseMatrix AA_;               /// Private instance of SparseMatrix (this needs a better name - this isn't really a sparse matrix, it's jut a matrix)
  
  bool isDecomposed_;             /// Flag set when LU object has been decomposed (is ready for back-substitution)
  bool isSingular_;               /// Flag set if a row of the LU object contains all zeros
  bool isRefined_;                /// Flag set when iterative refinement has been run

  std::vector<int> indx;

  // Iterative refinement members
  double rho_thresh;              /// Threshold convergence rate;           \n
                                  /// this value measures the "similarity"
                                  /// of dX for step (i) and step (i+1)
                                  /// (1 is totally similar, etc...).       \n
                                  /// A value of 0.5 is recommended,
                                  /// and a value of 0.9 is aggressive (meaning 
                                  /// it can find solutions to extremely
                                  /// ill-conditioned matrices, but it converges
                                  /// more slowly and has a less accurate condition
                                  /// number estimate).
                                  /// ***This value is set by the user.
  double rho_max;                 /// Maimum convergence rate obtained during the 
                                  /// iterative refinement procedure.  This value 
                                  /// provides an estimate of the condition number:
                                  /// \f[ \rho^{(i)} = \epsilon*c_n*g*\text{cond}(A) \f]
                                  /// (not sure what c_n or g are...)
  double relnorm_dX_i;            /// Relative L_infty norm of dX at previous iteration
  double relnorm_dX_ip1;          /// Relative L_infty norm of dX at current iteration
  double relnorm_X_i;             /// Relative L_infty norm of X 
  double final_rel_norm;          /// Relative L_infty norm of dX at last iteration




  bool increase_precision;        /// Boolean switch: increase precision for X?

  int imax;                       /// Maximum number of iterations.
                                  /// ***This value is set by the user.

};

}
#endif
