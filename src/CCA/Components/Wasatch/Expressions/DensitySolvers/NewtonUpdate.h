/*
 * The MIT License
 *
 * Copyright (c) 2015 The University of Utah
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

#ifndef Wasatch_NewtonUpdate_Expr_h
#define Wasatch_NewtonUpdate_Expr_h

#include <expression/Expression.h>
#include <expression/matrix-assembly/Compounds.h>
#include <expression/matrix-assembly/MatrixExpression.h>
#include <expression/matrix-assembly/MapUtilities.h>
#include <expression/matrix-assembly/DenseSubMatrix.h>
#include <CCA/Components/Wasatch/Expressions/DensitySolvers/HelperFunctions.h>
namespace WasatchCore{
  /**
   *  \class NewtonUpdate
   *  \author Josh McConnell
   *  \date May 2018
   *
   *  \brief performs a single Newton iteration
   */

  template< typename FieldT >
  class NewtonUpdate
  : public Expr::Expression<FieldT>
  {
    DECLARE_VECTOR_OF_FIELDS( FieldT, residual_ )
    DECLARE_VECTOR_OF_FIELDS( FieldT, jacobian_ )
    DECLARE_VECTOR_OF_FIELDS( FieldT, phiOld_ )

    const unsigned int numVariables_;
    
    NewtonUpdate( const Expr::TagList& residualTags,
                  const Expr::TagList& jacobianTags,
                  const Expr::TagList& phioldtags );

  public:

    class Builder : public Expr::ExpressionBuilder
    {
    public:

      Builder( const Expr::TagList& resultTags,
              const Expr::TagList& residualTags,
              const Expr::TagList& jacobianTags,
              const Expr::TagList& phioldtags,
              const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
        : ExpressionBuilder( resultTags, nghost ),
          residualTags_( residualTags ),
          jacobianTags_( jacobianTags ),
          phiOldTags_  ( phioldtags   )
      {}

      Expr::ExpressionBase* build() const{
        return new NewtonUpdate<FieldT>(residualTags_, jacobianTags_, phiOldTags_);
      }

    private:
      const Expr::TagList residualTags_, jacobianTags_, phiOldTags_;

    };

    ~NewtonUpdate(){};
    void evaluate();
  };



  // ###################################################################
  //
  //                          Implementation
  //
  // ###################################################################



  template< typename FieldT >
  NewtonUpdate<FieldT>::
  NewtonUpdate( const Expr::TagList& residualTags,
                const Expr::TagList& jacobianTags,
                const Expr::TagList& phiOldTags )
    : Expr::Expression<FieldT>(),
      numVariables_( residualTags.size() )
  {
    this->set_gpu_runnable(true);

    this->template create_field_vector_request<FieldT>( residualTags, residual_ );
    this->template create_field_vector_request<FieldT>( jacobianTags, jacobian_ );
    this->template create_field_vector_request<FieldT>( phiOldTags  , phiOld_   );
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  void
  NewtonUpdate<FieldT>::
  evaluate()
  {
    using namespace SpatialOps;
    typename Expr::Expression<FieldT>::ValVec&  resultVec = this->get_value_vec();

    // obtain memory for the jacobian and residual
    std::vector<SpatialOps::SpatFldPtr<FieldT> > resPtrs, jacPtrs;
    for( int i=0; i<numVariables_; ++i ){
      resPtrs.push_back( SpatialOps::SpatialFieldStore::get<FieldT>( *resultVec[i] ) );

      for( int j=0; j<numVariables_; ++j ){
        jacPtrs.push_back( SpatialOps::SpatialFieldStore::get<FieldT>( *resultVec[j] ) );
      }
    }
    FieldVector<FieldT> residualVec( resPtrs );
    FieldMatrix<FieldT> jacobian   ( jacPtrs );

    /*
    * set residuals
    */
    for(int i=0; i<numVariables_; ++i) residualVec(i) <<= residual_[i]->field_ref();

    /*
    * set jacobian elements
    */
    for( int i = 0; i<numVariables_; ++i ){
      for( int j = 0; j<numVariables_; ++j ){
        jacobian(i,j) <<= jacobian_[square_ij_to_flat(numVariables_, i, j)]->field_ref();
      }
    }

    // perform linear solve, and update guesses for species mass fractions and temperature
    residualVec = jacobian.solve( residualVec );

    for( size_t i=0; i<numVariables_; ++i ){
      FieldT& phi = *resultVec[i];
      phi <<= phiOld_[i]->field_ref() - residualVec(i);
    }
  }

  //--------------------------------------------------------------------

}

#endif // Wasatch_NewtonUpdate_Expr_h
