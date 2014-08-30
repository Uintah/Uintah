/*
 * Copyright (c) 2014 The University of Utah
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

#ifndef NEBO_STENCIL_BUILDER_H
#define NEBO_STENCIL_BUILDER_H

//To define new stencils use the following four macros.
//NEBO_FIRST_POINT and NEBO_FIRST_IJK define a new Nebo stencil point collection.
//NEBO_ADD_POINT and NEBO_ADD_IJK add a stencil point to an existing Nebo stencil point collection.
//
//NEBO_FIRST_POINT and NEBO_ADD_POINT are designed to work together.
//Likewise, NEBO_FIRST_IJK and NEBO_ADD_IJK are designed to work together.
//They can be mixed, but it is not recommended.
//
//NEBO_FIRST_POINT and NEBO_ADD_POINT are designed to work with stencil points based off of field types.
//For examples, look at Stencil2Collection, Stencil4Collection, and FDStencilCollection definitions in this file.
//
//NEBO_FIRST_IJK and NEBO_ADD_IJK are designed to work on constant stencil points, whose shape does not change depending on field type.
//For examples, look at NullStencilCollection and the seven BoxFilter*StencilCollection definitions in this file.



//Define a new stencil from an IndexTriplet
#define NEBO_FIRST_POINT_WO_TN(POINT) NeboStencilPointCollection< POINT, NeboNil >
#define NEBO_FIRST_POINT(POINT) typename NEBO_FIRST_POINT_WO_TN(POINT)

//Add a point (IndexTriplet) to an existing stencil
#define NEBO_ADD_POINT(POINT) template AddPoint< POINT >::Result

//Define a new stencil from three constant integers
#define NEBO_FIRST_IJK(X, Y, Z) NeboStencilPointCollection< IndexTriplet<X,Y,Z>, NeboNil >

//Add a point (three constant integers) to an existing stencil
#define NEBO_ADD_IJK(X, Y, Z) AddPoint< IndexTriplet<X,Y,Z> >::Result

namespace SpatialOps {

  /**
   *  \struct Subtract
   *  \brief Perform compile-time subtraction over a list of IndexTriplet types
   */
  template< typename List, typename IT >
    struct ListSubtract
    {
      typedef NeboStencilPointCollection< typename Subtract< typename List::Point, IT >::result,
                                          typename ListSubtract< typename List::Collection, IT >::result >
              result;
    };

  template< typename Point, typename IT >
    struct ListSubtract< NeboStencilPointCollection< Point, NeboNil >, IT >
    {
      typedef NeboStencilPointCollection< typename Subtract< Point, IT >::result, NeboNil >
              result;
    };

  /**
   * \struct NeboStencilBuilder
   * \brief Supports definition of new Nebo stencils.
   *
   * \tparam OperatorType the type of operator, e.g. SpatialOps::Gradient, SpatialOps::Interpolant, etc.
   * \tparam PntCltnT     defines the stencil points
   * \tparam SrcFieldT    the type of field that this operator acts on
   * \tparam DestFieldT   the type of field that this operator produces
   */
    template<typename OperatorType, typename PntCltnT, typename SrcFieldT, typename DestFieldT>
    struct NeboStencilBuilder {
    public:
        typedef OperatorType  type;                 ///< operator type (Interpolant, Gradient, Divergence)
        typedef PntCltnT      PointCollectionType;  ///< collection of stencil points
        typedef SrcFieldT     SrcFieldType;         ///< source field type
        typedef DestFieldT    DestFieldType;        ///< destination field type
        typedef NeboStencilCoefCollection<PointCollectionType::length>
                              CoefCollection;       ///< collection of coefficients

        // typedefs for when argument is a Nebo expression
        template<typename Arg>
        struct WithArg {
            typedef NeboStencil<Initial, PointCollectionType, Arg, DestFieldType> Stencil;
            typedef NeboExpression<Stencil, DestFieldType> Result;
        };

        // typedefs for when argument is a field
        typedef NeboConstField<Initial, SrcFieldType> FieldArg;
        typedef NeboStencil<Initial, PointCollectionType, FieldArg, DestFieldType> FieldStencil;
        typedef NeboExpression<FieldStencil, DestFieldType> FieldResult;

        /**
         *  \brief construct a stencil with the specified coefficients
         */
        NeboStencilBuilder(const CoefCollection & coefs)
         : coefCollection_(coefs)
        {}

        ~NeboStencilBuilder() {}

        /**
         * \brief Return coefficient collection
         */
        CoefCollection const & coefs(void) const { return coefCollection_; };

        /**
         * \brief Apply this operator to the supplied source field to produce the supplied destination field
         * \param src the field that the operator is applied to
         * \param dest the resulting field.
         */
        void apply_to_field( const SrcFieldType & src, DestFieldType & dest ) const {
            dest <<= operator()(src);
        }

        /**
         * \brief Nebo's inline operator for field values
         * \param src the field to which the operator is applied
         */
        inline FieldResult operator ()( const SrcFieldType & src ) const {
            return FieldResult(FieldStencil(FieldArg(src), coefs()));
        }

        /**
         * \brief Nebo's inline operator for Nebo expressions
         * \param src the Nebo expression to which the operator is applied
         */
        template<typename Arg>
        inline typename WithArg<Arg>::Result
        operator ()( const NeboExpression<Arg, SrcFieldType> & src ) const {
            typedef typename WithArg<Arg>::Stencil Stencil;
            typedef typename WithArg<Arg>::Result Result;
            return Result(Stencil(src.expr(), coefs()));
        }

    private:
        const CoefCollection coefCollection_;
    };

    template<typename OperatorType, typename SrcFieldType, typename DestFieldType>
    struct Stencil2Collection {
        // source field offset
        typedef typename SrcFieldType::Location::Offset                      SrcOffset;
        // destination field offset
        typedef typename DestFieldType::Location::Offset                     DestOffset;
        // low (first) stencil point location (relative to the destination point)
        typedef typename GreaterThan<SrcOffset, DestOffset>::result::Negate  LowStPt;
        // high (second) stencil point location (relative to the destination point)
        typedef typename LessThan<SrcOffset, DestOffset>::result             HighStPt;
        // collection of all stencil points in this stencil
        typedef NEBO_FIRST_POINT(LowStPt)::NEBO_ADD_POINT(HighStPt)          StPtCollection;
    };

    template<typename OperatorType, typename SrcFieldType, typename DestFieldType>
    struct Stencil4Collection {
        template<bool Boolean, typename True, typename False>
        struct TemplateIf;

        template<typename True, typename False>
        struct TemplateIf<true, True, False> { True typedef result; };

        template<typename True, typename False>
        struct TemplateIf<false, True, False> { False typedef result; };

        // source field offset
        typedef typename SrcFieldType::Location::Offset  SrcOffset;
        // destination field offset
        typedef typename DestFieldType::Location::Offset DestOffset;
        // unit vectors
        typedef IndexTriplet<1, 0, 0>   XUnit;
        typedef IndexTriplet<0, 1, 0>   YUnit;
        typedef IndexTriplet<0, 0, 1>   ZUnit;
        // first direction (unit vector)
        typedef typename TemplateIf<((int)(SrcOffset::X) != (int)(DestOffset::X)),
                                    XUnit,
                                    YUnit>::result                                  FirstDir;
        // second direction (unit vector)
        typedef typename TemplateIf<((int)(SrcOffset::Z) != (int)(DestOffset::Z)),
                                    ZUnit,
                                    YUnit>::result                                  SecondDir;
        // source offset in the first direction
        typedef typename Multiply<SrcOffset, FirstDir>::result          SrcInFirstDir;
        // source offset in the second direction
        typedef typename Multiply<SrcOffset, SecondDir>::result         SrcInSecondDir;
        // destination offset in the first direction
        typedef typename Multiply<DestOffset, FirstDir>::result         DestInFirstDir;
        // destination offset in the second direction
        typedef typename Multiply<DestOffset, SecondDir>::result        DestInSecondDir;
        // low value in the first direction
        typedef typename GreaterThan<SrcInFirstDir,
                                                 DestInFirstDir>::result::Negate    LoValInFirstDir;
        // high value in the first direction
        typedef typename LessThan<SrcInFirstDir,
                                              DestInFirstDir>::result               HiValInFirstDir;
        // low value in the second direction
        typedef typename GreaterThan<SrcInSecondDir,
                                            DestInSecondDir>::result::Negate        LoValInSecondDir;
        // high value in the second direction
        typedef typename LessThan<SrcInSecondDir,
                                              DestInSecondDir>::result              HiValInSecondDir;
        // stencil point locations (relative to the destination point)
        typedef typename Add<LoValInFirstDir, LoValInSecondDir>::result StPt1;
        typedef typename Add<HiValInFirstDir, LoValInSecondDir>::result StPt2;
        typedef typename Add<LoValInFirstDir, HiValInSecondDir>::result StPt3;
        typedef typename Add<HiValInFirstDir, HiValInSecondDir>::result StPt4;
        // collection of all stencil points in this stencil
        typedef NEBO_FIRST_POINT(StPt1)::NEBO_ADD_POINT(StPt2)
                ::NEBO_ADD_POINT(StPt3)::NEBO_ADD_POINT(StPt4)                      StPtCollection;
    };

    template<typename OperatorType, typename SrcFieldType, typename DestFieldType>
    struct FDStencilCollection {
        typedef typename OperatorType::DirT                          DirT;
        typedef typename UnitTriplet<DirT>::type                     DirVec;
        typedef typename DirVec::Negate                              LowStPt;
        typedef DirVec                                               HighStPt;
        typedef NEBO_FIRST_POINT(LowStPt)::NEBO_ADD_POINT(HighStPt)  StPtCollection;
    };

  /**
   * \struct NeboEdgelessStencilBuilder
   * \brief Supports definition of new Nebo stencils that do NOT invalidate ghost cells.
   *
   * \tparam OperatorType the type of operator, e.g. SpatialOps::Gradient, SpatialOps::Interpolant, etc.
   * \tparam PntCltnT     defines the stencil points
   * \tparam SrcFieldT    the type of field that this operator acts on
   * \tparam DestFieldT   the type of field that this operator produces
   */
    template<typename OperatorType, typename PntCltnT, typename SrcFieldT, typename DestFieldT>
    struct NeboEdgelessStencilBuilder {
    public:
        typedef OperatorType  type;                 ///< operator type (Interpolant, Gradient, Divergence)
        typedef PntCltnT      PointCollectionType;  ///< collection of stencil points
        typedef SrcFieldT     SrcFieldType;         ///< source field type
        typedef DestFieldT    DestFieldType;        ///< destination field type
        typedef NeboStencilCoefCollection<PointCollectionType::length>
                              CoefCollection;       ///< collection of coefficients

        // typedefs for when argument is a Nebo expression
        template<typename Arg>
        struct WithArg {
            typedef NeboEdgelessStencil<Initial, PointCollectionType, Arg, DestFieldType> Stencil;
            typedef NeboExpression<Stencil, DestFieldType> Result;
        };

        // typedefs for when argument is a field
        typedef NeboConstField<Initial, SrcFieldType> FieldArg;
        typedef NeboEdgelessStencil<Initial, PointCollectionType, FieldArg, DestFieldType> FieldStencil;
        typedef NeboExpression<FieldStencil, DestFieldType> FieldResult;

        /**
         *  \brief construct a stencil with the specified coefficients
         */
        NeboEdgelessStencilBuilder(const CoefCollection & coefs)
         : coefCollection_(coefs)
        {}

        ~NeboEdgelessStencilBuilder() {}

        /**
         * \brief Return coefficient collection
         */
        CoefCollection const & coefs(void) const { return coefCollection_; };

        /**
         * \brief Apply this operator to the supplied source field to produce the supplied destination field
         * \param src the field that the operator is applied to
         * \param dest the resulting field.
         */
        void apply_to_field( const SrcFieldType & src, DestFieldType & dest ) const {
            dest <<= operator()(src);
        }

        /**
         * \brief Nebo's inline operator for field values
         * \param src the field to which the operator is applied
         */
        inline FieldResult operator ()( const SrcFieldType & src ) const {
            return FieldResult(FieldStencil(FieldArg(src), coefs()));
        }

        /**
         * \brief Nebo's inline operator for Nebo expressions
         * \param src the Nebo expression to which the operator is applied
         */
        template<typename Arg>
        inline typename WithArg<Arg>::Result
        operator ()( const NeboExpression<Arg, SrcFieldType> & src ) const {
            typedef typename WithArg<Arg>::Stencil Stencil;
            typedef typename WithArg<Arg>::Result Result;
            return Result(Stencil(src.expr(), coefs()));
        }

    private:
        const CoefCollection coefCollection_;
    };

  /**
   * \struct NeboSumStencilBuilder
   * \brief Supports definition of new Nebo sum stencils, which sums given stencil points WITHOUT coefficients.
   *
   * \tparam PntCltnT     defines the stencil points
   * \tparam SrcFieldT    the type of field that this operator acts on
   * \tparam DestFieldT   the type of field that this operator produces
   */
    template<typename PntCltnT, typename SrcFieldT, typename DestFieldT>
    struct NeboSumStencilBuilder {
    public:
        typedef PntCltnT      PointCollectionType;  ///< collection of stencil points
        typedef SrcFieldT     SrcFieldType;         ///< source field type
        typedef DestFieldT    DestFieldType;        ///< destination field type

        // typedefs for when argument is a Nebo expression
        template<typename Arg>
        struct WithArg {
            typedef NeboSumStencil<Initial, PointCollectionType, Arg, DestFieldType> Stencil;
            typedef NeboExpression<Stencil, DestFieldType> Result;
        };

        // typedefs for when argument is a field
        typedef NeboConstField<Initial, SrcFieldType> FieldArg;
        typedef NeboSumStencil<Initial, PointCollectionType, FieldArg, DestFieldType> FieldStencil;
        typedef NeboExpression<FieldStencil, DestFieldType> FieldResult;

        /**
         *  \brief construct a stencil
         */
        NeboSumStencilBuilder()
        {}

        ~NeboSumStencilBuilder() {}

        /**
         * \brief Apply this operator to the supplied source field to produce the supplied destination field
         * \param src the field that the operator is applied to
         * \param dest the resulting field.
         */
        void apply_to_field( const SrcFieldType & src, DestFieldType & dest ) const {
            dest <<= operator()(src);
        }

        /**
         * \brief Nebo's inline operator for field values
         * \param src the field to which the operator is applied
         */
        inline FieldResult operator ()( const SrcFieldType & src ) const {
            return FieldResult(FieldStencil(FieldArg(src)));
        }

        /**
         * \brief Nebo's inline operator for Nebo expressions
         * \param src the Nebo expression to which the operator is applied
         */
        template<typename Arg>
        inline typename WithArg<Arg>::Result
        operator ()( const NeboExpression<Arg, SrcFieldType> & src ) const {
            typedef typename WithArg<Arg>::Stencil Stencil;
            typedef typename WithArg<Arg>::Result Result;
            return Result(Stencil(src.expr()));
        }
    };

    struct NullStencilCollection {
        typedef NEBO_FIRST_IJK(0, 0, 0) StPtCollection;
    };

  /**
   * \struct NeboAverageStencilBuilder
   * \brief Supports definition of new Nebo average stencils, which automatically averages given stencil points.
   *
   * \tparam PntCltnT     defines the stencil points
   * \tparam SrcFieldT    the type of field that this operator acts on
   * \tparam DestFieldT   the type of field that this operator produces
   */
    template<typename PntCltnT, typename SrcFieldT, typename DestFieldT>
    struct NeboAverageStencilBuilder {
    public:
        typedef PntCltnT      PointCollectionType;  ///< collection of stencil points
        typedef SrcFieldT     SrcFieldType;         ///< source field type
        typedef DestFieldT    DestFieldType;        ///< destination field type

        // typedefs for when argument is a Nebo expression
        template<typename Arg>
        struct WithArg {
            typedef NeboSumStencil<Initial, PointCollectionType, Arg, DestFieldType> Stencil;
            typedef NeboScalar<Initial, double> Scalar;
            typedef ProdOp<Initial, Stencil, Scalar> Average;
            typedef NeboExpression<Average, DestFieldType> Result;
        };

        // typedefs for when argument is a field
        typedef NeboConstField<Initial, SrcFieldType> FieldArg;
        typedef NeboScalar<Initial, double> FieldScalar;
        typedef NeboSumStencil<Initial, PointCollectionType, FieldArg, DestFieldType> FieldStencil;
        typedef ProdOp<Initial, FieldStencil, FieldScalar> FieldAverage;
        typedef NeboExpression<FieldAverage, DestFieldType> FieldResult;

        /**
         *  \brief construct a stencil
         */
        NeboAverageStencilBuilder()
        {}

        ~NeboAverageStencilBuilder() {}

        /**
         * \brief Apply this operator to the supplied source field to produce the supplied destination field
         * \param src the field that the operator is applied to
         * \param dest the resulting field.
         */
        void apply_to_field( const SrcFieldType & src, DestFieldType & dest ) const {
            dest <<= operator()(src);
        }

        /**
         * \brief Nebo's inline operator for field values
         * \param src the field to which the operator is applied
         */
        inline FieldResult operator ()( const SrcFieldType & src ) const {
            return FieldResult(FieldAverage(FieldStencil(FieldArg(src)),
                                            FieldScalar(1.0 / double(PointCollectionType::length))));
        }

        /**
         * \brief Nebo's inline operator for Nebo expressions
         * \param src the Nebo expression to which the operator is applied
         */
        template<typename Arg>
        inline typename WithArg<Arg>::Result
        operator ()( const NeboExpression<Arg, SrcFieldType> & src ) const {
            typedef typename WithArg<Arg>::Stencil Stencil;
            typedef typename WithArg<Arg>::Scalar Scalar;
            typedef typename WithArg<Arg>::Average Average;
            typedef typename WithArg<Arg>::Result Result;
            return Result(Average(Stencil(src.expr()),
                                  Scalar(1.0 / double(PointCollectionType::length))));
        }
    };

    struct BoxFilter3DStencilCollection {
      typedef NEBO_FIRST_IJK(-1,-1,-1)::NEBO_ADD_IJK( 0,-1,-1)::NEBO_ADD_IJK( 1,-1,-1)
              ::NEBO_ADD_IJK(-1, 0,-1)::NEBO_ADD_IJK( 0, 0,-1)::NEBO_ADD_IJK( 1, 0,-1)
              ::NEBO_ADD_IJK(-1, 1,-1)::NEBO_ADD_IJK( 0, 1,-1)::NEBO_ADD_IJK( 1, 1,-1)
              ::NEBO_ADD_IJK(-1,-1, 0)::NEBO_ADD_IJK( 0,-1, 0)::NEBO_ADD_IJK( 1,-1, 0)
              ::NEBO_ADD_IJK(-1, 0, 0)::NEBO_ADD_IJK( 0, 0, 0)::NEBO_ADD_IJK( 1, 0, 0)
              ::NEBO_ADD_IJK(-1, 1, 0)::NEBO_ADD_IJK( 0, 1, 0)::NEBO_ADD_IJK( 1, 1, 0)
              ::NEBO_ADD_IJK(-1,-1, 1)::NEBO_ADD_IJK( 0,-1, 1)::NEBO_ADD_IJK( 1,-1, 1)
              ::NEBO_ADD_IJK(-1, 0, 1)::NEBO_ADD_IJK( 0, 0, 1)::NEBO_ADD_IJK( 1, 0, 1)
              ::NEBO_ADD_IJK(-1, 1, 1)::NEBO_ADD_IJK( 0, 1, 1)::NEBO_ADD_IJK( 1, 1, 1)
          StPtCollection;
    };

    struct BoxFilter2DXYStencilCollection {
      typedef NEBO_FIRST_IJK(-1,-1, 0)::NEBO_ADD_IJK( 0,-1, 0)::NEBO_ADD_IJK( 1,-1, 0)
              ::NEBO_ADD_IJK(-1, 0, 0)::NEBO_ADD_IJK( 0, 0, 0)::NEBO_ADD_IJK( 1, 0, 0)
              ::NEBO_ADD_IJK(-1, 1, 0)::NEBO_ADD_IJK( 0, 1, 0)::NEBO_ADD_IJK( 1, 1, 0)
          StPtCollection;
    };

    struct BoxFilter2DXZStencilCollection {
      typedef NEBO_FIRST_IJK(-1, 0,-1)::NEBO_ADD_IJK( 0, 0,-1)::NEBO_ADD_IJK( 1, 0,-1)
              ::NEBO_ADD_IJK(-1, 0, 0)::NEBO_ADD_IJK( 0, 0, 0)::NEBO_ADD_IJK( 1, 0, 0)
              ::NEBO_ADD_IJK(-1, 0, 1)::NEBO_ADD_IJK( 0, 0, 1)::NEBO_ADD_IJK( 1, 0, 1)
          StPtCollection;
    };

    struct BoxFilter2DYZStencilCollection {
      typedef NEBO_FIRST_IJK( 0,-1,-1)::NEBO_ADD_IJK( 0, 0,-1)::NEBO_ADD_IJK( 0, 1,-1)
              ::NEBO_ADD_IJK( 0,-1, 0)::NEBO_ADD_IJK( 0, 0, 0)::NEBO_ADD_IJK( 0, 1, 0)
              ::NEBO_ADD_IJK( 0,-1, 1)::NEBO_ADD_IJK( 0, 0, 1)::NEBO_ADD_IJK( 0, 1, 1)
          StPtCollection;
    };

    struct BoxFilter1DXStencilCollection {
      typedef NEBO_FIRST_IJK(-1, 0, 0)::NEBO_ADD_IJK( 0, 0, 0)::NEBO_ADD_IJK( 1, 0, 0)
          StPtCollection;
    };

    struct BoxFilter1DYStencilCollection {
      typedef NEBO_FIRST_IJK( 0,-1, 0)::NEBO_ADD_IJK( 0, 0, 0)::NEBO_ADD_IJK( 0, 1, 0)
          StPtCollection;
    };

    struct BoxFilter1DZStencilCollection {
      typedef NEBO_FIRST_IJK( 0, 0,-1)::NEBO_ADD_IJK( 0, 0, 0)::NEBO_ADD_IJK( 0, 0, 1)
          StPtCollection;
    };

    template<typename OperatorType, typename SrcFieldType, typename DestFieldType>
    struct MaskShiftPoints {
      // source field offset
      typedef typename SrcFieldType::Location::Offset                      SrcOffset;
      // destination field offset
      typedef typename DestFieldType::Location::Offset                     DestOffset;
      // minus-side stencil point location
      typedef typename LessThan<SrcOffset, DestOffset>::result MinusPoint;
      // plus-side stencil point location
      typedef typename GreaterThan<SrcOffset,
                                               DestOffset>::result::Negate PlusPoint;
    };

    //must be a finite difference (FD) stencil, pull direction and points from operator:
    template<typename OperatorType, typename FieldType>
    struct MaskShiftPoints<OperatorType, FieldType, FieldType> {
      typedef typename OperatorType::type                    OpT;
      // FDStencilCollection:
      typedef FDStencilCollection<OpT, FieldType, FieldType> FDStencil;
      // minus-side stencil point location
      typedef typename FDStencil::DirVec                     MinusPoint;
      // plus-side stencil point location
      typedef typename FDStencil::DirVec::Negate             PlusPoint;
    };

  /**
   * \struct NeboMaskShiftBuilder
   * \brief Supports definition of new Nebo mask shift stencils, which converts a mask of one field type into another field type.
   *
   * \tparam OperatorT    the type of the operator
   * \tparam SrcFieldT    the type of field that this operator acts on
   * \tparam DestFieldT   the type of field that this operator produces
   */
    template<typename OperatorT, typename SrcFieldT, typename DestFieldT>
      struct NeboMaskShiftBuilder {
      public:
        typedef OperatorT  OperatorType;  ///< operator type
        typedef SrcFieldT  SrcFieldType;  ///< source field type
        typedef DestFieldT DestFieldType; ///< destination field type

        typedef SpatialMask<SrcFieldType>  SrcMask;  ///< source mask type
        typedef SpatialMask<DestFieldType> DestMask; ///< destination mask type

        typedef typename MaskShiftPoints<OperatorType, SrcFieldType, DestFieldType>::MinusPoint MinusPoint; ///< negative face shift for mask
        typedef typename MaskShiftPoints<OperatorType, SrcFieldType, DestFieldType>::PlusPoint  PlusPoint;  ///< positive face shift for mask

        typedef NeboMask<Initial, SrcFieldType> Mask; ///< Nebo mask type

        typedef NeboMaskShift<Initial, MinusPoint, Mask, DestFieldType> MinusShift; ///< shift type for negative shift
        typedef NeboMaskShift<Initial, PlusPoint,  Mask, DestFieldType> PlusShift;  ///< shift type for positive shift

        typedef NeboBooleanExpression<MinusShift, DestFieldType> MinusResult; ///< result type for negative shift
        typedef NeboBooleanExpression<PlusShift,  DestFieldType> PlusResult;  ///< result type for positive shift

        /**
         *  \brief construct a stencil
         */
        NeboMaskShiftBuilder()
        {}

        ~NeboMaskShiftBuilder() {}

        /**
         * \brief Compute the minus side shift
         * \param src the mask to which the operator is applied
         */
        inline MinusResult minus( const SrcMask & src ) const {
          return MinusResult(MinusShift(Mask(src)));
        }

        /**
         * \brief Compute the plus side shift
         * \param src the mask to which the operator is applied
         */
        inline PlusResult plus( const SrcMask & src ) const {
          return PlusResult(PlusShift(Mask(src)));
        }
    };

    /**
     * \struct NeboBoundaryConditionBuilder
     * \brief Supports definition of new Nebo boundary condition.
     *
     * \tparam OperatorT   Operator to invert
     *
     * Note that Gamma is assumed to be the origin, for the stencil points and mask points.
     */
    template<typename OperatorT>
    struct NeboBoundaryConditionBuilder {
    public:
      typedef OperatorT                            OperatorType;   ///< type of operator to invert
      typedef typename OperatorType::SrcFieldType  PhiFieldType;   ///< field type of phi, which this operator modifies
      typedef typename OperatorType::DestFieldType GammaFieldType; ///< type of fields in gamma, which this operator reads

      typedef typename OperatorType::PointCollectionType PointCollection; ///< stencil point collection

      typedef typename PointCollection::Last        LowPoint;      ///< stencil offset for low point in phi, assumes gamma is origin
      typedef typename PointCollection::AllButLast  NonLowPoints;  ///< stencil offsets for all but low point in phi, assumes gamma is origin
      typedef typename PointCollection::First       HighPoint;     ///< stencil offset for high point in phi, assumes gamma is origin
      typedef typename PointCollection::AllButFirst NonHighPoints; ///< stencil offsets for all but high point in phi, assumes gamma is origin

      typedef typename Subtract<IndexTriplet<0,0,0>, LowPoint>:: result LowGammaPoint;
      typedef typename Subtract<IndexTriplet<0,0,0>, HighPoint>::result HighGammaPoint;
      typedef typename ListSubtract<NonLowPoints,  LowPoint>:: result                           NonLowSrcPoints;
      typedef typename ListSubtract<NonHighPoints, HighPoint>::result                           NonHighSrcPoints;
      typedef NeboMaskShiftBuilder<OperatorType, GammaFieldType, PhiFieldType>                  Shift;

      typedef NeboStencilCoefCollection<PointCollection::length> CoefCollection; ///< collection of coefficients

      typedef NeboEdgelessStencilBuilder<NeboNil, NEBO_FIRST_POINT_WO_TN(LowGammaPoint),  GammaFieldType, PhiFieldType> MinusGammaType;
      typedef NeboEdgelessStencilBuilder<NeboNil, NEBO_FIRST_POINT_WO_TN(HighGammaPoint), GammaFieldType, PhiFieldType> PlusGammaType;
      typedef NeboEdgelessStencilBuilder<NeboNil, NonLowSrcPoints,                        PhiFieldType,   PhiFieldType> MinusPhiType;
      typedef NeboEdgelessStencilBuilder<NeboNil, NonHighSrcPoints,                       PhiFieldType,   PhiFieldType> PlusPhiType;

      typedef std::vector<IntVec> Points;
      typedef Points::const_iterator PointIterator;

      /**
       *  \brief construct a boundary condition
       */
      NeboBoundaryConditionBuilder(OperatorType const & op)
      : lowCoef_(op.coefs().last()),
        highCoef_(op.coefs().coef()),
        minusGamma_(1.0),
        plusGamma_(1.0),
        minusPhi_(op.coefs().all_but_last()),
        plusPhi_(op.coefs().others()),
        shift_()
      {}

      ~NeboBoundaryConditionBuilder() {}

      /**
       * \brief Apply boundary condition with gamma as an expression
       * \param points the mask of points where boundary condition applies
       * \param shift
       * \param shiftGamma
       * \param shiftPhi
       * \param phi the field to modify
       * \param gamma the Nebo expression to read
       * \param coef
       */
      template<typename ExprType, typename ShiftGamma, typename ShiftPhi>
      inline void cpu_apply( const Points & points,
                             const IntVec & shift,
                             const ShiftGamma & shiftGamma,
                             const ShiftPhi & shiftPhi,
                             PhiFieldType & phi,
                             const NeboExpression<ExprType, GammaFieldType> & gamma,
                             const double coef ) const
      {
        typedef NeboField<Initial, PhiFieldType> LhsTypeInit;
        typedef typename LhsTypeInit::SeqWalkType LhsType;
        LhsTypeInit lhsInit(phi);
        LhsType lhs = lhsInit.init();

        typedef typename ShiftGamma::template WithArg<ExprType>::Stencil GammaType;
        typedef typename ShiftPhi::FieldStencil PhiType;
        typedef DiffOp<Initial, GammaType, PhiType> DiffType;
        typedef NeboScalar<Initial, double> NumType;
        typedef DivOp<Initial, DiffType, NumType> RhsTypeInit;
        typedef typename RhsTypeInit::SeqWalkType RhsType;
        //arguments to init() call are meaningless, but they are not used at all...
        RhsType rhs = RhsTypeInit(DiffType(shiftGamma(gamma).expr(),
                                           shiftPhi(phi).expr()),
                                  NumType(coef)).init(IntVec(0,0,0),
                                                      GhostData(0),
                                                      IntVec(0,0,0));

        PointIterator       ip = points.begin();
        PointIterator const ep = points.end();
        for(; ip != ep; ip++) {
          const int x = (*ip)[0] - shift[0];
          const int y = (*ip)[1] - shift[1];
          const int z = (*ip)[2] - shift[2];
          lhs.ref(x, y, z) = rhs.eval(x, y, z);
        }
      }

      /**
       * \brief Apply boundary condition with gamma as an expression
       * \param mask the mask of points where boundary condition applies
       * \param phi the field to modify
       * \param gamma the Nebo expression to read
       * \param minus boolean flag, true if operating on negative face
       */
      template<typename ExprType>
      inline void operator()(SpatialMask<GammaFieldType> mask,
                             PhiFieldType & phi,
                             const NeboExpression<ExprType, GammaFieldType> & gamma,
                             bool minus) const {
        if(phi.active_device_index() == CPU_INDEX) {
          if(minus)
            cpu_apply<ExprType, MinusGammaType, MinusPhiType>(mask.points(),
                                                              Shift::MinusPoint::int_vec(),
                                                              minusGamma_,
                                                              minusPhi_,
                                                              phi,
                                                              gamma,
                                                              lowCoef_);
          else
            cpu_apply<ExprType, PlusGammaType, PlusPhiType>(mask.points(),
                                                            Shift::PlusPoint::int_vec(),
                                                            plusGamma_,
                                                            plusPhi_,
                                                            phi,
                                                            gamma,
                                                            highCoef_);
        }
        else {
          if(minus)
            phi <<= cond(shift_.minus(mask), (minusGamma_(gamma) - minusPhi_(phi)) / lowCoef_)
                        (phi);
          else
            phi <<= cond(shift_.plus(mask), (plusGamma_(gamma) - plusPhi_(phi)) / highCoef_)
                        (phi);
        }
      }

      /**
       * \brief Apply boundary condition with gamma as a field
       * \param mask the mask of points where boundary condition applies
       * \param phi the field to modify
       * \param gamma the field to read
       * \param minus
       */
      inline void operator()(SpatialMask<GammaFieldType> mask,
                             PhiFieldType & phi,
                             const GammaFieldType & gamma,
                             bool minus) const
      {
        typedef NeboConstField<Initial, GammaFieldType> GammaField;
        typedef NeboExpression<GammaField, GammaFieldType> GammaExpr;
        (*this)(mask, phi, GammaExpr(GammaField(gamma)), minus);
      }

      /**
       * \brief Apply boundary condition with gamma as a scalar
       * \param mask the mask of points where boundary condition applies
       * \param phi the field to modify
       * \param gamma the scalar to read
       * \param minus
       */
      inline void operator()(SpatialMask<GammaFieldType> mask,
                             PhiFieldType & phi,
                             const double gamma,
                             bool minus) const
      {
        typedef NeboScalar<Initial, double> GammaScalar;
        typedef NeboExpression<GammaScalar, GammaFieldType> GammaExpr;
        (*this)(mask, phi, GammaExpr(GammaScalar(gamma)), minus);
      }

    private:
      const double lowCoef_;
      const double highCoef_;
      const MinusGammaType minusGamma_;
      const PlusGammaType plusGamma_;
      const MinusPhiType minusPhi_;
      const PlusPhiType plusPhi_;
      const Shift shift_;
    };

} // namespace SpatialOps
#endif // NEBO_STENCIL_BUILDER_H
