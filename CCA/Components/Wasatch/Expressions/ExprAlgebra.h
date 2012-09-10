/*
 * Copyright (c) 2012 The University of Utah
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

#ifndef ExprAlgebra_Expr_h
#define ExprAlgebra_Expr_h

#include <expression/Expression.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>
#include <spatialops/OperatorDatabase.h>


#include <stdexcept>

/**
 *  \class ExprAlgebra
 *  \authors Tony Saad, Amir Biglari
 *  \date April, 2012
 *  \brief Implements simple algebraic operations between expressions. This useful
 *         for initializing data using existing initialized expressions. E.g. we
 *         must initialize the solution variable in scalar transport equation and
 *         momentum transport equation in some cases by multiplying initialized
 *         density and primary variable (e.g. velocity), then this expression
 *         comes to help.
 */
template< typename FieldT,
          typename SrcT1,
          typename SrcT2 >
class ExprAlgebra
 : public Expr::Expression<FieldT>
{
public:
   enum OperationType{
     SUM,
     DIFFERENCE,
     PRODUCT
   };

public:
   class Builder : public Expr::ExpressionBuilder
   {
   public:
     /**
      *  @brief Build a ExprAlgebra expression.  Note that this should
      *   only be used for initialization or post-processing - not in
      *   performance critical operations.
      *
      *  @param resultTag the tag for the value that this expression computes
      *
      *  @param src1Tag the tag to hold the value of the first source field
      *
      *  @param src2Tag the tag to hold the value of the second source field
      *
      *  @param algebraicOperation selects the operation to apply
      */
     Builder( const Expr::Tag& resultTag,
              const Expr::Tag& src1Tag,
              const Expr::Tag& src2Tag,
              const OperationType algebraicOperation );

     Expr::ExpressionBase* build() const;

   private:
     const Expr::Tag src1Tag_, src2Tag_;
     const OperationType algebraicOperation_;
   };

   ~ExprAlgebra();
   void advertise_dependents( Expr::ExprDeps& exprDeps );
   void bind_fields( const Expr::FieldManagerList& fml );
   void bind_operators( const SpatialOps::OperatorDatabase& opDB );
   void evaluate();

private:
   const Expr::Tag src1Tag_, src2Tag_;
   const SrcT1* src1_;
   const SrcT2* src2_;
   const OperationType algebraicOperation_;

   typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, SrcT1, FieldT >::type  Src1InterpT;
   typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, SrcT2, FieldT >::type  Src2InterpT;

   const Src1InterpT* src1InterpOp_; ///< Interpolate source 1 to the FieldT where we are building the result
   const Src2InterpT* src2InterpOp_; ///< Interpolate source 21 to the FieldT where we are building the result

  ExprAlgebra( const Expr::Tag& src1Tag,
               const Expr::Tag& src2Tag,
               const OperationType algebraicOperation );
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT, typename SrcT1, typename SrcT2 >
ExprAlgebra<FieldT,SrcT1,SrcT2>::
ExprAlgebra( const Expr::Tag& src1Tag,
             const Expr::Tag& src2Tag,
             const OperationType algebraicOperation )
  : Expr::Expression<FieldT>(),
    src1Tag_( src1Tag ),
    src2Tag_( src2Tag ),
    algebraicOperation_( algebraicOperation )
{}

//--------------------------------------------------------------------

template< typename FieldT, typename SrcT1, typename SrcT2 >
ExprAlgebra<FieldT,SrcT1,SrcT2>::
~ExprAlgebra()
{}

//--------------------------------------------------------------------

template< typename FieldT, typename SrcT1, typename SrcT2 >
void
ExprAlgebra<FieldT,SrcT1,SrcT2>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( src1Tag_ );
  exprDeps.requires_expression( src2Tag_ );
}

//--------------------------------------------------------------------

template< typename FieldT, typename SrcT1, typename SrcT2 >
void
ExprAlgebra<FieldT,SrcT1,SrcT2>::
bind_fields( const Expr::FieldManagerList& fml )
{
  src1_ = &fml.template field_manager< SrcT1 >().field_ref( src1Tag_ );
  src2_ = &fml.template field_manager< SrcT2 >().field_ref( src2Tag_ );
}

//--------------------------------------------------------------------

template< typename FieldT, typename SrcT1, typename SrcT2 >
void
ExprAlgebra<FieldT,SrcT1,SrcT2>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  src1InterpOp_ = opDB.retrieve_operator<Src1InterpT>();
  src2InterpOp_ = opDB.retrieve_operator<Src2InterpT>();
}

//--------------------------------------------------------------------

template< typename FieldT, typename SrcT1, typename SrcT2 >
void
ExprAlgebra<FieldT,SrcT1,SrcT2>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();

  SpatialOps::SpatFldPtr<FieldT> tmp1 = SpatialOps::SpatialFieldStore::get<FieldT>( result );
  SpatialOps::SpatFldPtr<FieldT> tmp2 = SpatialOps::SpatialFieldStore::get<FieldT>( result );

  src1InterpOp_->apply_to_field( *src1_, *tmp1 );
  src2InterpOp_->apply_to_field( *src2_, *tmp2 );

  switch( algebraicOperation_ ){
  case( SUM       ) : result <<= *tmp1 + *tmp2;  break;
  case( DIFFERENCE) : result <<= *tmp1 - *tmp2;  break;
  case( PRODUCT   ) : result <<= *tmp1 * *tmp2;  break;
  }
}

//--------------------------------------------------------------------

template< typename FieldT, typename SrcT1, typename SrcT2 >
ExprAlgebra<FieldT,SrcT1,SrcT2>::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::Tag& src1Tag,
                  const Expr::Tag& src2Tag,
                  const OperationType algebraicOperation )
  : ExpressionBuilder( resultTag ),
    src1Tag_( src1Tag ),
    src2Tag_( src2Tag ),
    algebraicOperation_( algebraicOperation )
{}

//--------------------------------------------------------------------

template< typename FieldT, typename SrcT1, typename SrcT2 >
Expr::ExpressionBase*
ExprAlgebra<FieldT,SrcT1,SrcT2>::
Builder::build() const
{
  return new ExprAlgebra<FieldT,SrcT1,SrcT2>( src1Tag_,src2Tag_,algebraicOperation_ );
}

//====================================================================

#endif // ExprAlgebra_Expr_h
