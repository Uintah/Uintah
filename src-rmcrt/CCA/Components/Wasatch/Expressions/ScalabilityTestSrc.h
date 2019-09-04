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

#ifndef Scalability_Test_Src
#define Scalability_Test_Src

#include <expression/Expression.h>

/**
 *  \ingroup 	Expressions
 *  \class 	ScalabilityTestSrc
 *  \date 	March, 2015
 *  \author 	Tony Saad
 *
 *  \brief Creates a nonlinear, uncoupled, source term for use in
 *         scalability tests.
 *
 */
template< typename FieldT >
class ScalabilityTestSrcUncoupled : public Expr::Expression<FieldT>
{
  DECLARE_FIELD(FieldT, f_)
  
  ScalabilityTestSrcUncoupled( const Expr::Tag& var );
  
  ~ScalabilityTestSrcUncoupled();
  
public:
  
  void evaluate();
  
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const;
    Builder( const Expr::Tag& result,
            const Expr::Tag& var );
    ~Builder(){}
  private:
    const Expr::Tag tag_;
  };
  
};

/**
 *  \ingroup 	Expressions
 *  \class 	ScalabilityTestSrc
 *  \date 	April, 2011
 *  \author 	Tony Saad
 *
 *  \brief Creates an all-to-all strongly coupled source term for use in
 *         scalability tests.
 *
 */
template< typename FieldT >
class ScalabilityTestSrc : public Expr::Expression<FieldT>
{
  DECLARE_VECTOR_OF_FIELDS(FieldT, phi_)
  const int nvar_;

  std::vector<double> tmpVec_;

  ScalabilityTestSrc( const Expr::Tag& var,
                      const int nvar );

  ~ScalabilityTestSrc();

public:

  void evaluate();

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const;
    Builder( const Expr::Tag& result,
             const Expr::Tag& var,
             const int nvar );
    ~Builder(){}
  private:
    const Expr::Tag tag_;
    const int nvar_;
  };

};

//====================================================================

#endif // Scalability_Test_Src
