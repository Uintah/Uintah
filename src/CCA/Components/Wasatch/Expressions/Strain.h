/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

#ifndef Strain_Expr_h
#define Strain_Expr_h

#include <expression/Expression.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>

//==================================================================

// note that the ordering of Vel1T and Vel2T are very important, and
// must be consistent with the order of the velocity tags passed
// into the Strain constructor.
template< typename FaceT > struct StrainHelper
{
  // Default to collocated velocity field
  typedef SVolField Vel1T;
  typedef SVolField Vel2T;
};
// nomenclature: XSurfXField - first letter is volume type: S, X, Y, Z
// then it is followed by the field type
template<> struct StrainHelper<SpatialOps::XSurfXField>
{
  // XSurfXField - XVol-XSurf
  // tau_xx
  typedef XVolField Vel1T;
  typedef XVolField Vel2T;
};
template<> struct StrainHelper<SpatialOps::XSurfYField>
{
  // XSurfYField - XVol-YSurf
  // tau_yx (tau on a y face in the x direction)
  typedef XVolField Vel1T;
  typedef YVolField Vel2T;
};
template<> struct StrainHelper<SpatialOps::XSurfZField>
{
  // XSurfZField - XVol-ZSurf
  // tau_zx (tau on a z face in the x direction)
  typedef XVolField Vel1T;
  typedef ZVolField Vel2T;
};

template<> struct StrainHelper<SpatialOps::YSurfXField>
{
  // tau_xy
  typedef YVolField Vel1T;
  typedef XVolField Vel2T;
};
template<> struct StrainHelper<SpatialOps::YSurfYField>
{
  // tau_yy
  typedef YVolField Vel1T;
  typedef YVolField Vel2T;
};
template<> struct StrainHelper<SpatialOps::YSurfZField>
{
  // tau_zy
  typedef YVolField Vel1T;
  typedef ZVolField Vel2T;
};

template<> struct StrainHelper<SpatialOps::ZSurfXField>
{
  // tau_xz
  typedef ZVolField Vel1T;
  typedef XVolField Vel2T;
};
template<> struct StrainHelper<SpatialOps::ZSurfYField>
{
  // tau_yz
  typedef ZVolField Vel1T;
  typedef YVolField Vel2T;
};
template<> struct StrainHelper<SpatialOps::ZSurfZField>
{
  // tau_zz
  typedef ZVolField Vel1T;
  typedef ZVolField Vel2T;
};

//==================================================================

// note that the ordering of Vel1T and Vel2T are very important, and
// must be consistent with the order of the velocity tags passed
// into the Strain constructor.
template< typename StrainT, typename MomDirT > struct CollocatedStrainHelper
{
  typedef SVolField Vel1InterpT;
};
// nomenclature: XSurfXField - first letter is volume type: S, X, Y, Z
// then it is followed by the field type

// tau_yx (x-momentum)
template<> struct CollocatedStrainHelper<SpatialOps::SSurfYField, SpatialOps::XDIR>
{
  // dv/dx + du/dy (vel1 = v, vel2 = u)
  typedef SpatialOps::XSurfYField Vel1InterpT; // interpolate v to the XSurfY (corners of cell) and then compute dvdx
};

// tau_zx (x-momentum)
template<> struct CollocatedStrainHelper<SpatialOps::SSurfZField, SpatialOps::XDIR>
{
  // dw/dx + du/dz (vel1 = w, vel2 = u)
  typedef SpatialOps::XSurfZField Vel1InterpT; // interpolate w to the XSurfZ (corners of cell) and then compute dwdx
};

// tau_xy (y-momentum)
template<> struct CollocatedStrainHelper<SpatialOps::SSurfXField, SpatialOps::YDIR>
{
  // du/dy + dv/dx (vel1 = u, vel2 = v)
  typedef SpatialOps::YSurfXField Vel1InterpT; // interpolate u to the YSurfX (corners of cell) and then compute dudy
};

// tau_zy (y-momentum)
template<> struct CollocatedStrainHelper<SpatialOps::SSurfZField, SpatialOps::YDIR>
{
  // dw/dy + dv/dz (vel1 = w, vel2 = v)
  typedef SpatialOps::YSurfZField Vel1InterpT; // interpolate w to the YSurfZ (corners of cell) and then compute dwdy
};


// tau_xz (z-momentum)
template<> struct CollocatedStrainHelper<SpatialOps::SSurfXField, SpatialOps::ZDIR>
{
  // du/dz + dw/dx (vel1 = u, vel2 = w)
  typedef SpatialOps::ZSurfXField Vel1InterpT; // interpolate u to the ZSurfX (corners of cell) and then compute dudz
};

// tau_yz (y-momentum)
template<> struct CollocatedStrainHelper<SpatialOps::SSurfYField, SpatialOps::ZDIR>
{
  // dv/dz + dw/dy (vel1 = v, vel2 = w)
  typedef SpatialOps::ZSurfYField Vel1InterpT; // interpolate v to the ZSurfY (corners of cell) and then compute dvdz
};

/**
 *  \class 	Strain
 *  \author 	James C. Sutherland, Tony Saad
 *  \date 	 June 2012, (Originally created: December, 2010).
 *  \ingroup	Expressions
 *
 *  \brief Calculates a component of the Strain tensor.
 *
 *  The Strain tensor is given as
 *  \f[ S_{ij} = - \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) + \frac{2}{3} \delta_{ij} \frac{\partial u_k}{\partial x_k} \f]
 *
 *  \tparam StrainT The type of field for this Strain component.
 *  \tparam Vel1T   The type of field for the first velocity component.
 *  \tparam Vel2T   The type of field for the second velocity component.
 */
template< typename StrainT,
typename Vel1T,
typename Vel2T >
class Strain
: public Expr::Expression<StrainT>
{
  
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, StrainT >::type  SVol2StrainInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient,    Vel1T,     StrainT >::type  Vel1GradT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient,    Vel2T,     StrainT >::type  Vel2GradT;
  
  const Vel1GradT*          vel1GradOp_;   ///< Calculate the velocity gradient dui/dxj at the Strain face
  const Vel2GradT*          vel2GradOp_;   ///< Calculate the velocity gradient duj/dxi at the Strain face
  
  DECLARE_FIELD( Vel1T, u1_ )
  DECLARE_FIELD( Vel2T, u2_ )
  
  Strain( const Expr::Tag& vel1Tag,
          const Expr::Tag& vel2Tag );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \param result the Strain component being calculated
     *  \param vel1Tag the first velocity component
     *  \param vel2Tag the second velocity component
     *
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& vel1Tag,
             const Expr::Tag& vel2Tag);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::Tag vel1t_, vel2t_;
  };
  
  ~Strain();
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
};


/**
 *  \brief specialized version of the Strain tensor for normal Strain.
 */
template< typename StrainT,
typename VelT >
class Strain< StrainT, VelT, VelT >
: public Expr::Expression<StrainT>
{
 
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, StrainT >::type  SVol2StrainInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient,    VelT,      StrainT >::type  VelGradT;
  
  const VelGradT*           velGradOp_;    ///< Calculate the velocity gradient dui/dxj at the Strain face
  
  DECLARE_FIELD( VelT,      u_   )
  
  Strain( const Expr::Tag& velTag );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \param result the Strain component calculated here
     *  \param vel1Tag the first velocity component
     *  \param vel2Tag the second velocity component
     *
     *  note that in this case, the second velocity component will be
     *  ignored.  It is kept for consistency with the off-diagonal
     *  Strain builder.
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& vel1Tag,
             const Expr::Tag& vel2Tag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::Tag velt_;
  };
  
  ~Strain();
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

/**
 *  \class 	Strain
 *  \author 	Tony Saad
 *  \date 	 November 2015
 *  \ingroup	Expressions
 *
 *  \brief Calculates a component of the Strain tensor for a collocated grid arrangement.
 *
 *  The Strain tensor is given as
 *  \f[ S_{ij} = - \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) + \frac{2}{3} \delta_{ij} \frac{\partial u_k}{\partial x_k} \f]
 *
 *  \tparam StrainT The type of field for this Strain component.
 *  \tparam MomDirT The momentum direction.
 */
template< typename StrainT,
typename MomDirT >
class CollocatedStrain
: public Expr::Expression<StrainT>
{
  typedef typename CollocatedStrainHelper<StrainT,MomDirT>::Vel1InterpT VelInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, VelInterpT >::type  VelInterpOpT;
  
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient,    VelInterpT, StrainT >::type  Vel1GradT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient,    SVolField,  StrainT >::type  Vel2GradT;
  
  const VelInterpOpT*       velInterpOp_;
  const Vel1GradT*          vel1GradOp_;   ///< Calculate the velocity gradient dui/dxj at the Strain face
  const Vel2GradT*          vel2GradOp_;   ///< Calculate the velocity gradient duj/dxi at the Strain face
  
  DECLARE_FIELDS( SVolField, u1_, u2_ )
  
  CollocatedStrain( const Expr::Tag& vel1Tag,
                    const Expr::Tag& vel2Tag );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \param result the Strain component being calculated
     *  \param vel1Tag the first velocity component
     *  \param vel2Tag the second velocity component
     *
     */
    Builder( const Expr::Tag& result,
            const Expr::Tag& vel1Tag,
            const Expr::Tag& vel2Tag );

    /**
     *  \brief calculates the normal strain components
     *  \param result the Strain component being calculated
     *  \param vel1Tag the first velocity component
     *
     */
    Builder( const Expr::Tag& result,
            const Expr::Tag& velTag );
    
    ~Builder(){}
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::Tag vel1t_, vel2t_;
  };
  
  ~CollocatedStrain();
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};


#endif // Strain_Expr_h
