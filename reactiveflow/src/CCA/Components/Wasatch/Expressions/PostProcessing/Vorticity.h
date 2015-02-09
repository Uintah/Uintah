/*
 * The MIT License
 *
 * Copyright (c) 2012-2015 The University of Utah
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

#ifndef Vorticity_Expr_h
#define Vorticity_Expr_h

#include <expression/Expression.h>
#include <spatialops/structured/FVStaggered.h>

// helper struct to identify which faces or directions we take derivatives on
template< typename Vel1T, typename Vel2T> struct VorticityFaceSelector;

// for omega_x = dw/dy - dv/dz
template<> struct VorticityFaceSelector<SpatialOps::ZVolField, SpatialOps::YVolField>
{
private:
  typedef SpatialOps::ZVolField Vel1T;
  typedef SpatialOps::YVolField Vel2T;
public:
  typedef SpatialOps::FaceTypes<Vel1T>::YFace Vel1FaceT;
  typedef SpatialOps::FaceTypes<Vel2T>::ZFace Vel2FaceT;
};

// for omega_y = du/dz - dw/dx
template<> struct VorticityFaceSelector<SpatialOps::XVolField, SpatialOps::ZVolField>
{
private:
  typedef SpatialOps::XVolField Vel1T;
  typedef SpatialOps::ZVolField Vel2T;
public:
  typedef SpatialOps::FaceTypes<Vel1T>::ZFace Vel1FaceT;
  typedef SpatialOps::FaceTypes<Vel2T>::XFace Vel2FaceT;
};

// for omega_z = dv/dx - du/dy
template<> struct VorticityFaceSelector<SpatialOps::YVolField, SpatialOps::XVolField>
{
private:
  typedef SpatialOps::YVolField Vel1T;
  typedef SpatialOps::XVolField Vel2T;
public:
  typedef SpatialOps::FaceTypes<Vel1T>::XFace Vel1FaceT;
  typedef SpatialOps::FaceTypes<Vel2T>::YFace Vel2FaceT;
};

/**
 *  \class   Vorticity
 *  \author  Tony Saad
 *  \date    February, 2012
 *  \ingroup Expressions
 *
 *  \brief Calculates a cell centered vorticity. Note that the order of the
 *         velocity tags is important as it dictates fieldtypes.
 *
 */
template< typename FieldT,
typename Vel1T,
typename Vel2T >
class Vorticity
: public Expr::Expression<FieldT>
{
  const Expr::Tag vel1t_, vel2t_;

  typedef typename VorticityFaceSelector<Vel1T, Vel2T>::Vel1FaceT Vel1FaceT;
  typedef typename VorticityFaceSelector<Vel1T, Vel2T>::Vel2FaceT Vel2FaceT;

  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, Vel1T, Vel1FaceT >::type Vel1GradT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, Vel2T, Vel2FaceT >::type Vel2GradT;

  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, Vel1FaceT, FieldT >::type InterpVel1FaceT2FieldT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, Vel2FaceT, FieldT >::type InterpVel2FaceT2FieldT;

  const Vel1T* vel1_;
  const Vel2T* vel2_;

  const Vel1GradT* vel1GradTOp_;
  const Vel2GradT* vel2GradTOp_;

  const InterpVel1FaceT2FieldT* interpVel1FaceT2FieldTOp_;
  const InterpVel2FaceT2FieldT* interpVel2FaceT2FieldTOp_;

  Vorticity( const Expr::Tag& vel1tag,
             const Expr::Tag& vel2tag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:

    /**
     *  \param vel1tag the velocity corresponding to the Vel1T template parameter
     *  \param vel2tag the velocity corresponding to the Vel2T template parameter
     */
    Builder( const Expr::Tag& result,
            const Expr::Tag& vel1tag,
            const Expr::Tag& vel2tag);
    ~Builder(){}
    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag v1t_, v2t_;
  };

  ~Vorticity();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

#endif // Vorticity_Expr_h
