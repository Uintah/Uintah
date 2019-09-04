#ifndef Uintah_Components_Arches_LU_h
#define Uintah_Components_Arches_LU_h

#include <Core/Grid/Variables/VarTypes.h>
#include <iostream>
#include <iomanip>

/**
  * @class    LU
  * 
  * @brief    Performs LU decomposition of a matrix system using Crout's method with partial pivoting;
  *           additionally, the class can optionally implement iterative refinement of \f$X=A\\B\f$.
  * @details  This class may be extended in the future to also use QR decomposition.
  *
  * @author   Charles Reid and James Sutherland
  * @date     November 2005   Crout's Method
  *           April 2009      Crout's Method with Partial Pivoting
  *           July 2009       Iterative Refinement, templated matrix operations
  */

namespace Uintah {

class LU{
  public:
    
    //=========================================================
    // Constructors/destructors
  
    /** @brief            Default constructor
      *
      * @param dim        Dimension of matrix (square matrix) */ 
    LU( const int dim);
  
    /** @brief            Copy constructor
      * @param CopyThis   The LU object to be copied */
    LU( LU &CopyThis );
    
    ~LU();
  
  
  


    // ===============================================
    // LU Decomposition & Back-Substitution using Crout's Method
  
    /** @brief      Performs the LU decomposition/factorization using Crout's method with partial pivoting. */
    void decompose();
  
    /** @brief      (Overloaded) Performs back-substitution method for LU system; does not overwrite RHS vector
      * @param rhs  Pointer to right-hand side vector
      * @param soln Pointer to solution vector */
    void back_subs( double* rhs,      double* soln );
    void back_subs( long double* rhs, double* soln );
  
  

    //=========================================================
    // Iterative refinement (public methods)

    /** @brief                Perform iterative refinement
      *
      * @param rhs            Right-hand side vector B of AX=B
      * @param soln           Solution vector X of AX=B (single working precision, i.e. double precision)
      * @param refined_soln   Refined solution (double working precision, i.e. quadruple precision) */
    void iterative_refinement( LU Aoriginal, double* rhs, double* soln, long double* refined_soln );

    /** @brief      Access function for final relative norm (error estimate)
      * @returns    final_relative_norm variable, which contains the relative norm of 
      *             \f$ \Vert dx \Vert_{\infty} / \Vert x \Vert_{\infty} \f$ 
      *             for the last step of iterative refinement   */
    double getConvergenceRate() {
      if(isRefined_)
        return condition_estimate;
      else
        return -999999;
    }

    /** @brief      Access function for the determinant
      * @returns    The determinant of dense matrix A */
    double getDeterminant() {
      ASSERT(isDecomposed_);
      return determinant;
    }
  
  
  
    //========================================================
    // Utility functions

    /** @brief      Check if matrix A is singular  */
    bool isSingular() {
      ASSERT(isDecomposed_);
      return isSingular_;
    }

    /** @brief      Check if matrix A is decomposed */
    bool isDecomposed() {
      return isDecomposed_;
    }
  
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
    getResidual( T1* rhs, T2* soln, T3* res)
    {
        // multiply A by X and put the result in Res
        multiply( soln, res );
      
        // sutract B from Res (from A*X)
        for (int z = 0; z < dim_; ++z) {
            res[z] = res[z] - rhs[z];
        }
    }

    /** @brief      Calculate the norm of a function
      * @param a    Vector whose nom is being calculated
      * @param type Tpe of norm to calculate:   \n
      *             0 = L_infinity norm         \n
      *             1 = L1 norm                 \n
      *             2 = L2 norm                 \n
      *             */
    double getNorm( double* a, int type );
    double getNorm( long double* a, int type );
 


    /** @brief      Dumps the contents of the LU matrix to std::out. */
    void dump();

  
  
    /** @brief      Returns dimension of the LU object
      * 
      * @returns    The dimension of the LU object */
    unsigned int getDimension() {
      return dim_; };
  
  
  
    /** @brief            Define (i,j) operator to access row i and column j 
      *
      * @param row        Row in LU object to access
      * @param col        Column in LU object to access */
    inline double& operator ()(const int row, const int col){
      ASSERT( row < dim_ && row >= 0 );
      ASSERT( col < dim_ && col >= 0 );
      return AA_(row,col);
    };
  
  
  private:
  
    LU();  // no default constructor.
  
  

    // ===============================================
    // Iterative Refinement (Private Methods)
  
   
  
    /** @brief      Update the state of X and the maximum error estimate 
      *
      * @param normXi       Norm (Linfty) of X(i)
      * @param normdXi      Norm (Linfty) of dX(i) (prior iteration)
      * @param normdXip1    Norm (Linfty) of dX(i+1) (current iteration)
      * @param finRelNorm   Norm (Linfty) of final relative error:
      *                     $ f [ \text{FinalRelativeNorm} = \Vert dx^{(i+1)} \Vert / \Vert x^{(i)} \Vert f] $
      */
    void update_xstate( double norm_X_i, 
                        double norm_dX_i, 
                        double norm_dX_ip1,
                        int* x_state );



    // ===============================================
    // Sparse Matrix class
  
    class DenseMatrix{
    public:
      DenseMatrix( const int dim);
      //DenseMatrix( const DenseMatrix copyThisMatrix );
      ~DenseMatrix();
      inline double& operator ()(const int row, const int col){
        return AA_[row][col];
      };
      void dump();
      int getDimension() {
        return dim_; 
      };
      double calculateDeterminant();
    private:
      DenseMatrix();
      double **AA_;
      const int dim_;
    };
  
   
  
    // ===============================================
    // Private members
  
    // Press:
    double d;
    std::vector<int> indx;
    
    // Determinant:
    double determinant;
    
    const int dim_;                 /// Dimension of LU object
    DenseMatrix AA_;               /// Private instance of DenseMatrix 
    
    bool isDecomposed_;             /// Flag set when LU object has been decomposed (is ready for back-substitution)
    bool isSingular_;               /// Flag set if a row of the LU object contains all zeros
    bool isRefined_;                /// Flag set when iterative refinement has been run
  
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

    double rho_max;                 /// Maximum convergence rate obtained during the 
                                    /// iterative refinement procedure.  This value 
                                    /// provides an estimate of the condition number:
                                    /// \f$\rho^{(i)} = \epsilon*c_n*g*\text{cond}(A)\f$
                                    /// (not sure what c_n or g are...)
    
    double condition_estimate;      /// Estimate of condition number

    double norm_dX_i;               /// Relative L_infty norm of dX at previous iteration
    double norm_dX_ip1;             /// Relative L_infty norm of dX at current iteration
    double norm_dX_final;           /// Relative L_infty norm of dX at last iteration
    double norm_X_i;                /// Relative L_infty norm of X 

    bool increase_precision;        /// Boolean switch: increase precision for X?
  
    int imax;                       /// Maximum number of iterations.
                                    /// ***This value is set by the user.

};

}
#endif
