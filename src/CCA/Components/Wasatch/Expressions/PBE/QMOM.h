/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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

#ifndef QMOM_Expr_h
#define QMOM_Expr_h

#include <expression/Expression.h>
#include <cmath>

#include <Core/Util/DebugStream.h>

#include <sci_defs/uintah_defs.h>

//set up a debug stream for qmom without ifdef compiler commands
static Uintah::DebugStream dbgqmom("WASATCH_QMOM_DBG", false);  //qmom debugging
#define dbg_qmom_on dbgqmom.active() && Uintah::Parallel::getMPIRank() == 0
#define dbg_qmom  if( dbg_qmom_on ) dbgqmom

#define DSYEV FIX_NAME(dsyev)

// declare lapack eigenvalue solver
extern "C"{
  void DSYEV( char* jobz, char* uplo, int* n, double* a, int* lda,
	      double* w, double* work, int* lwork, int* info );
}


/**
 *  \class QMOM
 *  \author Tony Saad
 *  \todo add documentation
 *  \todo add till support
 *  \brief This class takes in the list of the moments of the system, 
 *  and uses product difference algorithm to solve for the weights and
 *  abscissae fir this set of moments.
 *  If the system returns non-physical values or a singular matrix, then
 *  an optional flag can be specfied in the input file (RealizableQMOM) 
 *  to reduce  the number of moments used in the algorithm by 2, 
 *  and return 0 and 1 for the n-th weight and abscissa
 */
template<typename FieldT>
class QMOM : public Expr::Expression<FieldT>
{
//  typedef std::vector<const FieldT*> FieldTVec;
//  FieldTVec knownMoments_;
  const Expr::TagList knownMomentsTagList_;
  DECLARE_VECTOR_OF_FIELDS(FieldT, knownMoments_)

  const int nMoments_;

  std::vector< std::vector<double> > pmatrix_;
  std::vector<double> a_, b_, alpha_, jMatrix_, eigenValues_, weights_, work_;
  const bool realizable_;
  int nonRealizablePoints_;

  QMOM( const Expr::TagList knownMomentsTagList, const bool realizable );
  
  bool product_difference(const std::vector<typename FieldT::const_iterator>& knownMomentsIterator);


public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::TagList& result,
             const Expr::TagList& knownMomentsTagList,
             const bool realizable )
    : ExpressionBuilder(result),
      knownMomentsTagList_( knownMomentsTagList ),
      realizable_ (realizable)
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const
    {
      return new QMOM<FieldT>( knownMomentsTagList_, realizable_ );
    }

  private:
    const Expr::TagList knownMomentsTagList_;
    const bool realizable_;
  };

  ~QMOM();

  void evaluate();
};


// ###################################################################
//
//                          Implementation
//
// ###################################################################


template<typename FieldT>
QMOM<FieldT>::
QMOM( const Expr::TagList knownMomentsTaglist, const bool realizable)
  : Expr::Expression<FieldT>(),
    knownMomentsTagList_(knownMomentsTaglist),
    nMoments_( knownMomentsTaglist.size() ),
    realizable_ (realizable)
{
  this->template create_field_vector_request<FieldT>(knownMomentsTaglist, knownMoments_);
  pmatrix_.resize(nMoments_);
  for (int i = 0; i<nMoments_; i++)
    pmatrix_[i].resize(nMoments_ + 1);

  const int nEnvironments = nMoments_/2;
  alpha_.resize(nMoments_);
  a_.resize( nEnvironments );
  b_.resize( nEnvironments );
  jMatrix_.resize( nEnvironments*nEnvironments ); // note that jMatrix is stored as a vector for LAPACK
  eigenValues_.resize( nEnvironments );
  weights_.resize(nEnvironments);
  nonRealizablePoints_=0;
}

//--------------------------------------------------------------------

template<typename FieldT>
QMOM<FieldT>::
~QMOM()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
QMOM<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  // the results vector is organized as follows:
  // (w0, a0, w1, a1, et...)
  typedef typename Expr::Expression<FieldT>::ValVec ResultsVec;
  ResultsVec& results = this->get_value_vec();

  const int nEnvironments = nMoments_/2;
  nonRealizablePoints_=0;
  
  // loop over interior points in the patch. To do this, we grab a sample
  // iterator from any of the exisiting fields, for example, m0.
  const FieldT& sampleField = knownMoments_[0]->field_ref();
  typename FieldT::const_iterator sampleIterator = sampleField.interior_begin();

  //
  // create a vector of iterators for the known moments and for the results
  //
  std::vector<typename FieldT::const_iterator> knownMomentsIterators;
  std::vector<typename FieldT::iterator> resultsIterators;
  for (int i=0; i<nMoments_; ++i) {
    typename FieldT::const_iterator thisIterator = knownMoments_[i]->field_ref().interior_begin();
    knownMomentsIterators.push_back(thisIterator);

    typename FieldT::iterator thisResultsIterator = results[i]->interior_begin();
    resultsIterators.push_back(thisResultsIterator);
  }

  //
  // now loop over the interior points, construct the matrix, and solve for the weights & abscissae
  //
  while (sampleIterator != sampleField.interior_end()) {

    if (  *knownMomentsIterators[0] == 0 ) {
      // in case m_0 is zero, set the weights to zero and abscissae to 1
      // helps with numerical staility
      for (int i=0; i<nEnvironments; ++i) {
        int matLoc = 2*i;
        *resultsIterators[matLoc] = 0.0;     // weight
        *resultsIterators[matLoc + 1] = 1.0; // abscissa
      }
      //increment iterators
      ++sampleIterator;
      for (int i=0; i<nMoments_; ++i) {
        knownMomentsIterators[i] += 1;
        resultsIterators[i] += 1;
      }
      continue; // break the while loop
    }

    //__________
    // call the product_difference algorithm. This is where all the magic happens.
    const bool status = product_difference(knownMomentsIterators);
    
    if (!status) { 
      std::ostringstream errorMsg;
      errorMsg << std::endl << "ERROR QMOM: Product-Difference algorithm failed eigenvalue solve." << std::endl;
      throw std::runtime_error( errorMsg.str() );
    }
    
    // put the weights and abscissae into the results of this expression
    for (int i=0; i<nEnvironments; ++i) {
      const int matLoc = 2*i;
      *resultsIterators[matLoc] = weights_[i];
      if (eigenValues_[i] != 0.0 ) {
        *resultsIterators[matLoc + 1] = eigenValues_[i];
      } else {
        *resultsIterators[matLoc + 1] = 1.0;  //prevent div by 0 in other functions
      }
    }
    
    //__________
    // increment iterators
    ++sampleIterator;
    for (int i=0; i<nMoments_; ++i) {
      knownMomentsIterators[i] += 1;
      resultsIterators[i] += 1;
    }
  } // end interior points loop
//
  if(nonRealizablePoints_ > 0) dbg_qmom << "WARNING QMOM: I found " << nonRealizablePoints_ << " nonrealizable points." << std::endl;
}

//--------------------------------------------------------------------

template< typename FieldT >
bool
QMOM<FieldT>::
product_difference(const std::vector<typename FieldT::const_iterator>& knownMomentsIterator) {
  int nEnvironments = nMoments_/2;  
  // initialize the p matrix
  for (int i=0; i<nMoments_; ++i) {
    for (int j=0; j<nMoments_+1; ++j) {
      pmatrix_[i][j]=0.0;
    }
  }
  pmatrix_[0][0] = 1.0;

  // start by putting together the p matrix. this is documented in the wasatch
  // pdf documentation.
  for (int iRow=0; iRow<=nMoments_-2; iRow += 2) {
    // get the the iRow moment for this point
    pmatrix_[iRow][1] = *knownMomentsIterator[iRow];
    // get the the (iRow+1) moment for all points
    pmatrix_[iRow+1][1] = -*knownMomentsIterator[iRow+1];
  }
  // keep filling the p matrix. this gets easier now
  for (int jCol=2; jCol<=nMoments_; ++jCol) {
    for (int iRow=0; iRow <= nMoments_ - jCol; ++iRow) {
      pmatrix_[iRow][jCol] = pmatrix_[0][jCol-1]*pmatrix_[iRow+1][jCol-2]
      - pmatrix_[0][jCol-2]*pmatrix_[iRow+1][jCol-1];
    }
  }
  
  alpha_[0]=0.0;
  for (int jCol=1; jCol<nMoments_; ++jCol)
    alpha_[jCol] = pmatrix_[0][jCol+1]/(pmatrix_[0][jCol]*pmatrix_[0][jCol-1]);
  
  //_________________________
  // construct a and b arrays
  bool needsReduction = false;
  int reduceBy = 0; // reduce the number of environments by this much in case of negative b
  int nReducedMoments = 0; // this is the new number of moments to use
  for (int jCol=0; jCol < nEnvironments-1; ++jCol) {
    const int twojcol = 2*jCol;
    a_[jCol] = alpha_[twojcol+1] + alpha_[twojcol];
    const double rhsB = alpha_[twojcol+1]*alpha_[twojcol+2];
    //
    if (rhsB < 0) {
      needsReduction = true;
      reduceBy = nEnvironments - jCol -1;
      nReducedMoments = nMoments_ - 2*reduceBy;
      if ( !realizable_ ) {
        std::cout << knownMomentsTagList_ << std::endl; //if there is an error display which set of equations failed (in case there are multiple polymorphs)
        for (int i = 0; i<nMoments_; i++) printf("M[%i] = %.16f \n", i, *knownMomentsIterator[i]);
        std::ostringstream errorMsg;
        errorMsg << std::endl << "ERROR QMOM: Negative number detected while constructing the b auxiliary vector while processing the QMOM expression." << std::endl << "Value: b["<<jCol<<"] = "<<rhsB << std::endl;
        throw std::runtime_error( errorMsg.str() );
      } else {
        nonRealizablePoints_++;

        //display warning if moment reduction occurs
 //       dbg_qmom << "WARNING QMOM: Negative number detected in b auxiliary vector, decreasing number of moments from " << nMoments_ << " to " << nReducedMoments <<" and recalculating." << std::endl;

        if (nReducedMoments < 1) {
          std::ostringstream errorMsg;
          errorMsg << std::endl << "ERROR QMOM: Cannot reduce the number of moments to 0. Existing..." << std::endl;
          throw std::runtime_error( errorMsg.str() );
        }
        break;
      }
    }
    b_[jCol] = -std::sqrt(rhsB);
  }
  // fill in the last entry for a
  a_[nEnvironments-1] = alpha_[2*(nEnvironments-1) + 1] + alpha_[2*(nEnvironments-1)];
  
  //___________________
  //initialize the JMatrix and the eigenvalues and eigenvectors to 0
  std::fill(jMatrix_.begin(), jMatrix_.end(),0.0);
  std::fill(eigenValues_.begin(), eigenValues_.end(),0.0);
  std::fill(weights_.begin(), weights_.end(),0.0);
  
  // if there is a need for realizability, reduce the number of environments
  if(realizable_ && needsReduction) nEnvironments -= reduceBy;
  
  //___________________
  // construct J matrix. only fill in the values that are needed after moment reduction
  // The rest of the matrix remains zero.
  for (int iRow=0; iRow<nEnvironments - 1; ++iRow) {
    jMatrix_[iRow+iRow*nEnvironments] = a_[iRow];
    jMatrix_[iRow+(iRow+1)*nEnvironments] = b_[iRow];
    jMatrix_[iRow + 1 + iRow*nEnvironments] = b_[iRow];
  }
  jMatrix_[nEnvironments*nEnvironments-1] = a_[nEnvironments - 1];

  //__________
  // Eigenvalue solve
  /* Query and allocate the optimal workspace */
  int n = nEnvironments, lda = nEnvironments, info, lwork;
  double wkopt;
  lwork = -1;
  char jobz='V';
  char matType = 'U';
  DSYEV( &jobz, &matType, &n, &jMatrix_[0], &lda, &eigenValues_[0], &wkopt, &lwork, &info );
  lwork = (int)wkopt;
  work_.resize(lwork);
  // Solve eigenproblem. eigenvectors are stored in the jMatrix, columnwise
  DSYEV( &jobz, &matType, &n, &jMatrix_[0], &lda, &eigenValues_[0], &work_[0], &lwork, &info );
  
  bool status = ( info>0 || info<0 )? false : true;

  //__________
  // calculate the weights. The abscissae are stored in eigenValues_
  double m0 = *knownMomentsIterator[0];
  for (int i=0; i<nEnvironments; ++i) {
    const int matLoc = i*nEnvironments;
    weights_[i] = jMatrix_[matLoc]*jMatrix_[matLoc]*m0;
  }
  
  //isolate the debug stream statements to only print when an error occurs
  if (!status) {
    //p matrix
    for (int iRow=0; iRow<nMoments_; iRow++) {
      for (int jCol=0; jCol<nMoments_+1; jCol++) {
        dbg_qmom << "p[" << iRow << "][" << jCol << "] = " << pmatrix_[iRow][jCol] << "  ";
      }
      dbg_qmom << std::endl;
    }
    
    //display the alpha vector
    for (int iRow=0; iRow<nMoments_; iRow++) {
      dbg_qmom << "alpha[" << iRow << "] = " << alpha_[iRow] << "  ";
    }
    dbg_qmom << std::endl;
    
    //display the a & b vectors
    for (int iRow=0; iRow<nEnvironments; iRow++) {
      dbg_qmom << "a[" << iRow << "] = " << a_[iRow] << "  ";
    }
    dbg_qmom << std::endl;
    for (int iRow=0; iRow<nEnvironments; iRow++) {
      dbg_qmom << "b[" << iRow << "] = " << b_[iRow] << "  "; 
    }
    dbg_qmom << std::endl;
    
    //display the J vector
    for (int iRow=0; iRow<nEnvironments*nEnvironments; iRow++) {
      dbg_qmom << "J[" << iRow << "] = " << jMatrix_[iRow] << "  ";
    }
    dbg_qmom << std::endl;
    
    //display the eigenvaleus after solve
    for (int iRow=0; iRow<nEnvironments; iRow++) {
      dbg_qmom << "Eigenvalue[" << iRow << "] = " << eigenValues_[iRow] << "  ";
    }
    dbg_qmom << std::endl;
    
    if (realizable_)
      dbg_qmom << "WARNING QMOM: Negative number detected in b auxiliary vector, decreasing number of moments from " << nMoments_ << " to " << nReducedMoments <<" and recalculating." << std::endl;
  }
  
  return status;
}

#endif // QMOM_Expr_h
