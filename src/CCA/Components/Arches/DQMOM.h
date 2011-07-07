//------------------------ DQMOM.h -----------------------------------

#ifndef Uintah_Components_Arches_DQMOM_h
#define Uintah_Components_Arches_DQMOM_h

#include <CCA/Components/Arches/Directives.h>

#include <CCA/Ports/DataWarehouse.h>

#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <map>

namespace Uintah {

//-------------------------------------------------------

/** 
  * @class    DQMOM
  * @author   Charles Reid (charlesreid1@gmail.com)
  * @date     March 2009      "Initial" Arches version
  *           July 2009       Iterative Refinement
  *           November 2009   LAPACK (via DenseMatrix in Uintah framework)
  *
  * @brief    This class constructs and solves the AX=B linear system for DQMOM scalars.
  *
  * @details  Using the direct quadrature method of moments (DQMOM), the NDF transport equation is represented
  *           by a set of delta functions times weights.  The moment transform of this NDF transport equation
  *           yields a set of (closed) moment transport equations. These equations can be expressed in terms 
  *           of the weights and abscissas of the quadrature approximation, and re-cast as a linear system,
  *           \f$ \mathbf{AX} = \mathbf{B} \f$.  This class solves the linear system to yield the source terms
  *           for the weight and weighted abscissa transport equations (the variables contained in \f$\mathbf{X}\f$).
  *
  */

//-------------------------------------------------------

class ArchesLabel; 
class DQMOMEqn; 
class ModelBase; 
class LU;

class DQMOM {

public:

  DQMOM( ArchesLabel* fieldLabels, std::string which_dqmom );

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

private:

  void constructLinearSystem( LU              &A,
                              vector<double>  &B,
                              vector<double>  &weights,
                              vector<double>  &weightedAbscissas,
                              vector<double>  &models,
                              int              verbosity=0);

  void constructLinearSystem( DenseMatrix*    &AA,
                              ColumnMatrix*   &BB,
                              vector<double>  &weights,
                              vector<double>  &weightedAbscissas,
                              vector<double>  &models,
                              int              verbosity=0);

  void constructAopt( DenseMatrix*   &AA,
                      vector<double> &Abscissas);

  void constructAopt_unw( DenseMatrix*   &AA,
                          vector<double> &Abscissas);

  void constructBopt( ColumnMatrix*  &BB,
                      vector<double> &weights,
                      vector<double> &Abscissas,
                      vector<double> &models);

  void constructBopt_unw( ColumnMatrix*  &BB,
                          vector<double> &Abscissas,
                          vector<double> &models);


  vector<string> InternalCoordinateEqnNames;
  
  vector<MomentVector> momentIndexes; ///< Vector containing all moment indices

  std::vector<DQMOMEqn* > weightEqns;           ///< Weight equation labels, IN SAME ORDER AS GIVEN IN INPUT FILE
  std::vector<DQMOMEqn* > weightedAbscissaEqns; ///< Weighted abscissa equation labels, IN SAME ORDER AS GIVEN IN INPUT FILE

  vector< vector<ModelBase*> > weightedAbscissaModels;

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

  const VarLabel* d_normBLabel; 
  const VarLabel* d_normXLabel; 
  const VarLabel* d_normResLabel;
  const VarLabel* d_normResNormalizedLabelB;
  const VarLabel* d_normResNormalizedLabelX;
  const VarLabel* d_conditionNumberLabel;

  double d_small_normalizer; ///< When X (or B) is smaller than this, don't normalize the residual by it
  bool b_useLapack;
  bool b_calcConditionNumber;
  bool b_optimize;
  bool d_unweighted;
  std::string d_which_dqmom; 
  std::string d_solverType;

  struct constCCVarWrapper {
    constCCVariable<double> data;
  };

  typedef constCCVarWrapper constCCVarWrapperTypeDef;

  struct constCCVarWrapper_withModels {
    constCCVariable<double> data;
    vector<constCCVarWrapperTypeDef> models;
  };

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

} // end namespace Uintah

#endif
