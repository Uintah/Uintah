#ifndef Uintah_Components_Arches_DQMOM_h
#define Uintah_Components_Arches_DQMOM_h

#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Patch.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <CCA/Components/Arches/Directives.h>
#include <numeric>

namespace Uintah {

//-------------------------------------------------------

/** 
  * @class    DQMOM
  * @author   Charles Reid, Julien Pedel
  * @date     March 2009      "Initial" Arches version
  *           July 2009       Iterative Refinement
  *           November 2009   LAPACK (via DenseMatrix in Uintah framework)
  * `         March 2010      Optimized moment solver
  *
  * @brief    This class constructs and solves the AX=B linear system for DQMOM scalars.
  *
  * @details  
  * Using the direct quadrature method of moments (DQMOM), the NDF transport equation is represented
  * by a set of delta functions times weights.  The moment transform of this NDF transport equation
  * yields a set of (closed) moment transport equations. These equations can be expressed in terms 
  * of the weights and abscissas of the quadrature approximation, and re-cast as a linear system,
  * \f$ \mathbf{AX} = \mathbf{B} \f$.  This class solves the linear system to yield the source terms
  * for the weight and weighted abscissa transport equations (the variables contained in \f$\mathbf{X}\f$).
  *
  * The optimized moment solver uses the algorithm described in Fox (2009), "Optimal moment sets for
  * multivariate DQMOM" (Ind. Eng. Chem. Res. 2009, 48, 9686-9696)
  *
  * @todo
  * Fix the getModelList() function call - it will be changed to return a vector of VarLabels, not a vector of strings
  */

//-------------------------------------------------------

class ArchesLabel; 
class DQMOMEqn; 
class ModelBase; 
class LU;
class DQMOM {

public:

  DQMOM( ArchesLabel* fieldLabels );

  ~DQMOM();

  typedef std::vector<int> MomentVector;

  /** @brief Obtain parameters from input file and process them, whatever that means 
   */
  void problemSetup( const ProblemSpecP& params );

  /** @brief              Populate the map containing labels for each moment 
      @param allMoments   Vector containing all moment indices specified by user in <Moment> blocks within <DQMOM> block */
  void populateMomentsMap( vector<MomentVector> allMoments );

  /** @brief Schedule creation of linear solver object, creation of AX=B system, and solution of linear system. 
  */
  void sched_solveLinearSystem( const LevelP & level,
                                SchedulerP   & sched,
                                int            timeSubStep );
    
  /** @brief Create linear solver object, create linear system AX=B, and solve the linear system. 
  */
  void solveLinearSystem( const ProcessorGroup *,
                          const PatchSubset    * patches,
                          const MaterialSubset *,
                          DataWarehouse        * old_dw,
                          DataWarehouse        * new_dw );
  
  /** @brief Schedule calculation of all moments
  */
  void sched_calculateMoments( const LevelP & level,
                               SchedulerP   & sched,
                               int            timeSubStep );

  /** @brief Calculate the value of the moments */
  void calculateMoments( const ProcessorGroup *,
                         const PatchSubset    * patches,
                         const MaterialSubset *,
                         DataWarehouse        * old_dw,
                         DataWarehouse        * new_dw );



    /** @brief Destroy A, X, B, and solver object  
    */
    void destroyLinearSystem();

    /** @brief Access function for boolean, whether to calculate/save moments */
    bool getSaveMoments() {
      return b_save_moments; }
  
    const VarLabel* d_normBLabel; 

private:

  /** @brief    Construct the DQMOM linear system AX=B for LU and C++ vector A and B, which use Crout's method (slower) */
  void constructLinearSystem( LU              &A,
                              vector<double>  &B,
                              vector<double>  &weights,
                              vector<double>  &weightedAbscissas,
                              vector<double>  &models,
                              int              verbosity=0);

  /** @brief    Construct the DQMOM linear system AX=B for ColumnMatrix and DenseMatrix A and B, which use Lapack (faster) */
  void constructLinearSystem( DenseMatrix*    &AA,
                              ColumnMatrix*   &BB,
                              vector<double>  &weights,
                              vector<double>  &weightedAbscissas,
                              vector<double>  &models,
                              int              verbosity=0);

  /** @brief    Construct the A matrix (DenseMatrix) for the optimized DQMOM linear system */
  void constructAopt( DenseMatrix*   &AA,
                      vector<double> &Abscissas);

  /** @brief    Construct the RHS (ColumnMatrix) vector for the optimized DQMOM linear system */
  void constructBopt( ColumnMatrix*  &BB,
                      vector<double> &weights,
                      vector<double> &Abscissas,
                      vector<double> &models);

  /** @brief    Do a quick calculation of the powers of optimal abscissas (possible because optimal abscissas have values of -1, 0, or 1)
      @details
      This is a much faster "power" function for the optimal DQMOM abscissas.
      Since the optimal abscissas are KNOWN to be -1, 0, or 1, 
      calculating the powers of the optimal abscissas is very easy and doesn't need the actual
      power function pow().
      */
  inline int my_pow( int abscissa, int power )
  {
    int return_var = 0;
    switch( abscissa ) {
      case -1:
        if( power%2 == 0 ) {
          return_var = 1;
        } else if( abs(power%2) == 1 ) {
          return_var = -1;
        }
        break;
      case 0:
        if( power == 0 ) {
          return_var = 1;
        } else { 
          return_var = 0;
        }
        break;
      case 1:
        return_var = abscissa;
        break;
    }
    return return_var;
  }

  vector<MomentVector> momentIndexes;           ///< Vector containing all moment indices
  vector<DQMOMEqn* > weightEqns;           ///< Weight equation labels, IN SAME ORDER AS GIVEN IN INPUT FILE
  vector<DQMOMEqn* > weightedAbscissaEqns; ///< Weighted abscissa equation labels, IN SAME ORDER AS GIVEN IN INPUT FILE
  vector<string> InternalCoordinateEqnNames;
  vector< vector<ModelBase> > weightedAbscissaModels;

  typedef map<const MomentVector, const VarLabel*> MomentMap;
  MomentMap DQMOMMoments;     ///< DQMOM moment values
  MomentMap DQMOMMomentsMean; ///< DQMOM moment values, for moments about the mean

  ArchesLabel* d_fieldLabels; 
  
  unsigned int N_xi;  ///< Number of internal coordinates
  unsigned int N_;    ///< Number of quadrature nodes
  
  int d_timeSubStep;
  bool b_save_moments; ///< boolean - calculate & save moments?

  double d_solver_tolerance;
  double d_maxConditionNumber;
  double d_w_small;
  double d_weight_scaling_constant;
  vector<double> d_weighted_abscissa_scaling_constants;
  vector<double> d_opt_abscissas;
  DenseMatrix* AAopt;

  const VarLabel* d_normXLabel; 
  const VarLabel* d_normResLabel;
  const VarLabel* d_normResNormalizedLabelB;
  const VarLabel* d_normResNormalizedLabelX;
  const VarLabel* d_conditionNumberLabel;

  double d_small_normalizer; ///< When X (or B) is smaller than this, don't normalize the residual by it
  bool b_useLapack;
  bool b_calcConditionNumber;
  bool b_optimize;
  string d_solverType;


#if defined(VERIFY_LINEAR_SOLVER)
  /** @brief  Get an A and B matrix from a file, then solve the linear system
              AX=B and compare the solution to the pre-determined solution.
              This method verifies the AX=B solution procedure and linear solver.
  */
  void verifyLinearSolver();

  string vls_file_A;      ///< Name of file containing A (matrix)
  string vls_file_X;      ///< Name of file containing X (solution)
  string vls_file_B;      ///< Name of file containing B (RHS)
  string vls_file_R;      ///< Name of file containing R (residual)
  string vls_file_normR;  ///< Name of file containing normR (residual normalized by B)
  string vls_file_norms;  ///< Name of file containing norms (X, residuals)

  int vls_dimension;      ///< Dimension of problem
  double vls_tol;         ///< Tolerance for comparisons
  bool b_have_vls_matrices_been_printed;

#endif

#if defined(VERIFY_AB_CONSTRUCTION)
  /** @brief  Construct A and B using weights and weighted abscissas found in a file,
              then compare the constructed A and B to the real A and B. 
              This method verifies the A and B construction procedure.
  */
  void verifyABConstruction();

  string vab_file_A;      ///< Name of file containing A
  string vab_file_B;      ///< Name of file containing B
  string vab_file_inputs; ///< Name of file contianing weight and weighted abscissa inputs
  string vab_file_moments;///< Name of file containing moment indices

  int vab_dimension;      ///< Dimension of the problem
  int vab_N, vab_N_xi;    ///< Number of environments, internal coordinates of the problem
  double vab_tol;         ///< Tolerance for comparisons

  bool b_have_vab_matrices_been_printed;
#endif

#if defined(VERIFY_LINEAR_SOLVER) || defined(VERIFY_AB_CONSTRUCTION)
  /** @brief  Compares the elements of two vectors; if elements are not within tolerance,
              prints a message. */
  void compare(vector<double> vector1, vector<double> vector2, double tolerance);
  void compare(ColumnMatrix* &vector1, ColumnMatrix* &vector2, int dimension, double tolerance);

  /** @brief  Compares the elements of two matrices; if elements are not within tolerance,
              prints a message. */
  void compare(LU matrix1, LU matrix2, double tolerance);
  void compare(DenseMatrix* &matrix1, DenseMatrix* &matrix2, int dimension, double tolerance);

  /** @brief  Compares two scalars; if elements are not within tolerance,
              prints a message. */
  void compare(double x1, double x2, double tolerance);

  /** @brief  Take input divided up by white space, tokenize it, and put it into a vector of strings. */
  void tokenizeInput( const string& str,
                      vector<string>& tokens,
                      const string& delimiters = " " );

  /** @brief  Read a matrix from a file */
  void getMatrixFromFile( LU& matrix, string filename );
  void getMatrixFromFile( DenseMatrix* &matrix, int dimension, string filename );

  /** @brief  Read a vector from a file */
  void getVectorFromFile( vector<double>& vec, string filename );
  void getVectorFromFile( ColumnMatrix* &vec, int dimension, string filename );

  /** @brief   Read a vector from an already-open filestream */
  void getVectorFromFile( vector<double>& vec, ifstream& filestream );
  
  /** @brief  Read moment indices from a file */
  void getMomentsFromFile( vector<MomentVector>& moments, string filename );

#endif

#if defined(DEBUG_MATRICES)
  bool b_isFirstTimeStep;
#endif

}; // end class DQMOM


/** @brief    Sort a vector of moment vectors in lexicographic order
    @details
    This puts the moment vector in lexicographic order, meaning moments are ordered by

    1) Their global index (sum of all moment indices)

    2) Ascending moment index order, from left to right

    So, for example, the first few moments are ordered as:

    [0,0,0]  \n 
    [1,0,0]  \n
    [0,1,0]  \n
    [0,0,1]  \n
    [2,0,0]  \n
    [0,2,0]  \n
    [0,0,2]  \n
    etc...

    The primary purpose is to accelerate the construction of the B matrix.
    The entries of B for the zeroth and first moments are much easier to calculate,
    so ordering the moments allows the indices of these moments to be known.
    */
inline bool vector_lexicographic_sort( vector<int> a, vector<int> b ) 
{ 
  bool a_lt_b = false;

  int sum_a = accumulate( a.begin(), a.end(), 0 );
  int sum_b = accumulate( b.begin(), b.end(), 0 );

  if( sum_a == sum_b ) {
    vector<int>::iterator ia = a.begin(); 
    vector<int>::iterator ib = b.begin();
    for( ; ia < a.end(); ++ia, ++ib ) {
      if( (*ia) != (*ib) ) {
        a_lt_b = (*ia) < (*ib);
      }
    }
  } else {
    a_lt_b = (sum_a < sum_b);
  }

  return a_lt_b;

}



} // end namespace Uintah

#endif
