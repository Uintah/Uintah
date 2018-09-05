/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//------------------------ DQMOM.h -----------------------------------

#ifndef Uintah_Components_Arches_DQMOM_h
#define Uintah_Components_Arches_DQMOM_h

#include <CCA/Components/Arches/Directives.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <map>

#include <sci_defs/cuda_defs.h>

#ifdef HAVE_CUDA
#ifdef __cplusplus
extern "C" {
#endif
void launchConstructLinearSystemKernel(double* weightsArray,
                                       double* weightedAbscissasArray,
                                       double* modelsArray,
                                       int*    momentIndicesArray,
                                       double* AAArray,
                                       double* BBArray,
                                       int     num_cells);
#ifdef __cplusplus
}
#endif
#endif

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
  void populateMomentsMap( std::vector<MomentVector> allMoments );

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
                          DataWarehouse        * new_dw,
                          const int timeSubStep );

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
      return m_save_moments; }

private:

  void constructLinearSystem( LU              &A,
                              std::vector<double>  &B,
                              std::vector<double>  &weights,
                              std::vector<double>  &weightedAbscissas,
                              std::vector<double>  &models,
                              int              verbosity=0);

  void constructLinearSystem( DenseMatrix*    &AA,
                              ColumnMatrix*   &BB,
                              std::vector<double>  &weights,
                              std::vector<double>  &weightedAbscissas,
                              std::vector<double>  &models,
                              int              verbosity=0);

  void constructAopt( DenseMatrix*   &AA,
                      std::vector<double> &Abscissas);

  void constructAopt_unw( DenseMatrix*   &AA,
                          std::vector<double> &Abscissas);

  void constructBopt( ColumnMatrix*  &BB,
                      std::vector<double> &weights,
                      std::vector<double> &Abscissas,
                      std::vector<double> &models);

  void constructBopt_unw( ColumnMatrix*  &BB,
                          std::vector<double> &Abscissas,
                          std::vector<double> &models);

  void construct_Simplest_XX( ColumnMatrix*  &XX,
                           std::vector<double> &models);


  std::vector<std::string> InternalCoordinateEqnNames;

  std::vector<MomentVector> momentIndexes; ///< Vector containing all moment indices

  std::vector<DQMOMEqn* > weightEqns;           ///< Weight equation labels, IN SAME ORDER AS GIVEN IN INPUT FILE
  std::vector<DQMOMEqn* > weightedAbscissaEqns; ///< Weighted abscissa equation labels, IN SAME ORDER AS GIVEN IN INPUT FILE

  std::vector< std::vector<ModelBase*> > weightedAbscissaModels;

  ArchesLabel* m_fieldLabels;

  unsigned int m_N_xi;  ///< Number of internal coordinates
  unsigned int m_N_;    ///< Number of quadrature nodes

  bool m_save_moments; ///< boolean - calculate & save moments?

  double m_solver_tolerance;
  double m_maxConditionNumber;
  double m_w_small;
  //double d_weight_scaling_constant;
  //std::vector<double> d_weighted_abscissa_scaling_constants;
  std::vector<double> m_opt_abscissas;
  DenseMatrix* m_AAopt;

  const VarLabel* m_normBLabel;
  const VarLabel* m_normXLabel;
  const VarLabel* m_normResLabel;
  const VarLabel* m_normRedNormalizedLabelB;
  const VarLabel* m_normRedNormalizedLabelX;
  const VarLabel* m_conditionNumberLabel;

  double m_small_normalizer; ///< When X (or B) is smaller than this, don't normalize the residual by it
  bool m_useLapack;
  bool m_calcConditionNumber;
  bool m_optimize;
  bool m_unmweighted;
  std::string m_which_dqmom;
  std::string m_solverType;

  struct constCCVarWrapper {
    constCCVariable<double> data;
  };

  typedef constCCVarWrapper constCCVarWrapperTypeDef;

  struct constCCVarWrapper_withModels {
    constCCVariable<double> data;
    std::vector<constCCVarWrapperTypeDef> models;
  };

#if defined(VERIFY_LINEAR_SOLVER)
  /** @brief  Get an A and B matrix from a file, then solve the linear system
              AX=B and compare the solution to the pre-determined solution.
              This method verifies the AX=B solution procedure and linear solver.
  */
  void verifyLinearSolver();

  std::string vls_file_A;      ///< Name of file containing A (matrix)
  std::string vls_file_X;      ///< Name of file containing X (solution)
  std::string vls_file_B;      ///< Name of file containing B (RHS)
  std::string vls_file_R;      ///< Name of file containing R (residual)
  std::string vls_file_normR;  ///< Name of file containing normR (residual normalized by B)
  std::string vls_file_norms;  ///< Name of file containing norms (X, residuals)

  int vls_dimension;      ///< Dimension of problem
  double vls_tol;         ///< Tolerance for comparisons
  bool m_vls_mat_print;

#endif

#if defined(VERIFY_AB_CONSTRUCTION)
  /** @brief  Construct A and B using weights and weighted abscissas found in a file,
              then compare the constructed A and B to the real A and B.
              This method verifies the A and B construction procedure.
  */
  void verifyABConstruction();

  std::string vab_file_A;      ///< Name of file containing A
  std::string vab_file_B;      ///< Name of file containing B
  std::string vab_file_inputs; ///< Name of file contianing weight and weighted abscissa inputs
  std::string vab_file_moments;///< Name of file containing moment indices

  int vab_dimension;      ///< Dimension of the problem
  int vab_N, vab_m_N_xi;    ///< Number of environments, internal coordinates of the problem
  double vab_tol;         ///< Tolerance for comparisons

  bool m_vab_mat_print;
#endif

#if defined(VERIFY_LINEAR_SOLVER) || defined(VERIFY_AB_CONSTRUCTION)
  /** @brief  Compares the elements of two vectors; if elements are not within tolerance,
              prints a message. */
  void compare(std::vector<double> vector1, std::vector<double> vector2, double tolerance);
  void compare(ColumnMatrix* &vector1, ColumnMatrix* &vector2, int dimension, double tolerance);

  /** @brief  Compares the elements of two matrices; if elements are not within tolerance,
              prints a message. */
  void compare(LU matrix1, LU matrix2, double tolerance);
  void compare(DenseMatrix* &matrix1, DenseMatrix* &matrix2, int dimension, double tolerance);

  /** @brief  Compares two scalars; if elements are not within tolerance,
              prints a message. */
  void compare(double x1, double x2, double tolerance);

  /** @brief  Take input divided up by white space, tokenize it, and put it into a vector of strings. */
  void tokenizeInput( const std::string& str,
                      std::vector<std::string>& tokens,
                      const std::string& delimiters = " " );

  /** @brief  Read a matrix from a file */
  void getMatrixFromFile( LU& matrix, std::string filename );
  void getMatrixFromFile( DenseMatrix* &matrix, int dimension, std::string filename );

  /** @brief  Read a vector from a file */
  void getVectorFromFile( std::vector<double>& vec, std::string filename );
  void getVectorFromFile( ColumnMatrix* &vec, int dimension, std::string filename );

  /** @brief   Read a vector from an already-open filestream */
  void getVectorFromFile( std::vector<double>& vec, ifstream& filestream );

  /** @brief  Read moment indices from a file */
  void getMomentsFromFile( std::vector<MomentVector>& moments, std::string filename );

#endif

#if defined(DEBUG_MATRICES)
  bool m_isFirstTimestep;
#endif

}; // end class DQMOM

} // end namespace Uintah

#endif
