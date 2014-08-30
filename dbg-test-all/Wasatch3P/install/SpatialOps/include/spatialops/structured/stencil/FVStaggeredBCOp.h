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

#ifndef SpatialOps_FVStaggeredStencilBCOp_h
#define SpatialOps_FVStaggeredStencilBCOp_h

#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/IndexTriplet.h>
#include <spatialops/structured/MemoryWindow.h>
#include <spatialops/Nebo.h>

namespace SpatialOps{

  namespace bmpl = boost::mpl;

  /**
   * \enum BCSide
   * \brief Allows identification of whether we are setting the BC
   *        on the right or left side when using an operator.
   */
  enum BCSide{
    MINUS_SIDE,  ///< Minus side
    PLUS_SIDE,   ///< Plus side
    NO_SIDE      ///< for wide stencils where we set on a point rather than a face
  };

  namespace detail{
    template<typename OpT, typename T2 > struct OpDirHelper{
      typedef typename T2::DirT type;
      typedef IndexTriplet<0,0,0> S1Extra;
      typedef IndexTriplet<0,0,0> S2Extra;
    };
    template<typename T2> struct OpDirHelper< GradientX,    T2 >{ typedef typename GradientX   ::DirT type;  typedef IndexTriplet<-1,0,0> S1Extra; typedef IndexTriplet<1,0,0> S2Extra; };
    template<typename T2> struct OpDirHelper< GradientY,    T2 >{ typedef typename GradientY   ::DirT type;  typedef IndexTriplet<0,-1,0> S1Extra; typedef IndexTriplet<0,1,0> S2Extra; };
    template<typename T2> struct OpDirHelper< GradientZ,    T2 >{ typedef typename GradientZ   ::DirT type;  typedef IndexTriplet<0,0,-1> S1Extra; typedef IndexTriplet<0,0,1> S2Extra; };
    template<typename T2> struct OpDirHelper< InterpolantX, T2 >{ typedef typename InterpolantX::DirT type;  typedef IndexTriplet<-1,0,0> S1Extra; typedef IndexTriplet<1,0,0> S2Extra; };
    template<typename T2> struct OpDirHelper< InterpolantY, T2 >{ typedef typename InterpolantY::DirT type;  typedef IndexTriplet<0,-1,0> S1Extra; typedef IndexTriplet<0,1,0> S2Extra; };
    template<typename T2> struct OpDirHelper< InterpolantZ, T2 >{ typedef typename InterpolantZ::DirT type;  typedef IndexTriplet<0,0,-1> S1Extra; typedef IndexTriplet<0,0,1> S2Extra; };
  }

  /**
   * \class BoundaryConditionOp
   * \brief Provides a simple interface to set a boundary condition via an operator.
   * \tparam OpT the operator type for use in setting a BC
   * \tparam BCEval a functor for obtaining the BC value
   *
   *  NOTE: The BoundaryConditionOp class should only be used with
   *        operators that involve the scalar volume.
   */
  template< typename OpT,
            typename BCEval >
  class BoundaryConditionOp
  {
    typedef typename OpT::SrcFieldType                           SrcFieldT;
    typedef typename OpT::DestFieldType                          DestFieldT;
    typedef typename OpT::PointCollectionType::Collection::Point S1Shift;
    typedef typename OpT::PointCollectionType::Point             S2Shift;

    const BCEval bcEval_;  ///< functor to set the value of the BC
    const IntVec apoint_;  ///< the index for the value in the source field we will set
    const IntVec bpoint_;  ///< the index for the value in the source field we use to obtain the value we want to set.
    double ca_, cb_;       ///< high and low coefficients for the operator
    const bool singlePointBC_;
    std::vector<int> flatGhostPoints_;
    std::vector<int> flatInteriorPoints_;

    BoundaryConditionOp& operator=( const BoundaryConditionOp& ); // no assignment
    BoundaryConditionOp();                                        // no default constructor

  public:

    typedef BCEval BCEvalT;  ///< Expose the BCEval type.

    /**
     *  \param destIndex The (i,j,k) location at which we want to specify
     *         the boundary condition.  This is indexed 0-based on
     *         the interior (neglecting ghost cells), and refers to
     *         the index in the "destination" field of the operator.
     *
     *  \param side The side of the cell (MINUS_SIDE or PLUS_SIDE) that
     *         this BC is to be applied on.
     *
     *  \param bceval The evaluator to obtain the bc value at this point.
     *
     *  \param opdb The database for spatial operators. An operator of
     *         type OpT will be extracted from this database.
     */
    BoundaryConditionOp( const IntVec& destIndex,
                         const BCSide side,
                         const BCEval bceval,
                         const OperatorDatabase& opdb );

    /**
     *  @param window The memory window of the field on which the BC is
     *         is being applied.
     *
     *  @param destIndices A vector of IJK indices designating the list of
     *         points on which the BC is to be applied.
     *
     *  \param side The side of the cell (MINUS_SIDE or PLUS_SIDE) that
     *         this BC is to be applied on.
     *
     *  \param bceval The evaluator to obtain the bc value at this point.
     *
     *  \param opdb The database for spatial operators. An operator of
     *         type OpT will be extracted from this database.
     *
     *  @par Design Considerations
     *  \li We may need to change the way BCEval works since in the current
     *      model, the SAME bceval is applied at all points. This will not work
     *      with spatially varying bcs.
     */
    BoundaryConditionOp( const MemoryWindow& window,
                         const std::vector<IntVec>& destIndices,
                         const BCSide side,
                         const BCEval bceval,
                         const OperatorDatabase& opdb );

    ~BoundaryConditionOp(){}

    double getGhostCoef() const{ return ca_; }
    double getInteriorCoef() const{ return cb_; }
    const std::vector<int>& getFlatGhostPoints() const{ return flatGhostPoints_; }
    const std::vector<int>& getFlatInteriorPoints() const{ return flatInteriorPoints_; }

    /**
     *  \brief Impose the boundary condition on the supplied field.
     */
    void operator()( SrcFieldT& f ) const;

    /**
     *  \brief Impose the boundary condition on the supplied fields.
     */
    void operator()( std::vector<SrcFieldT*>& f ) const;

    static bool is_gpu_runnable(){ return false; }

  }; // class BoundaryConditionOp


  // ================================================================
  //
  //                            Implementation
  //
  // ================================================================

  template< typename OpT, typename BCEval >
  BoundaryConditionOp<OpT,BCEval>::
  BoundaryConditionOp( const IntVec& destPoint,
                       const BCSide side,
                       const BCEval bceval,
                       const OperatorDatabase& soDatabase )
      : bcEval_( bceval ),
        apoint_( destPoint + ( (side==MINUS_SIDE || side==NO_SIDE) ? S1Shift::int_vec() : S2Shift::int_vec() ) ),
        bpoint_( destPoint + ( (side==MINUS_SIDE || side==NO_SIDE) ? S2Shift::int_vec() : S1Shift::int_vec() ) ),
        singlePointBC_(true)
  {
    // let phi_a be the ghost value, phi_b be the internal value, and phi_bc be the boundary condition,
    //   phi_bc = a*phi_a + b*phi_b
    // then
    //   phi_a = (phi_bc - b*phi_b) / a
    //
    const OpT* const op = soDatabase.retrieve_operator<OpT>();
    ca_ = (side==MINUS_SIDE || side==NO_SIDE ? op->coefs().get_coef(0) : op->coefs().get_coef(1) );
    cb_ = (side==MINUS_SIDE || side==NO_SIDE ? op->coefs().get_coef(1) : op->coefs().get_coef(0) );
  }

  //------------------------------------------------------------------

  template< typename OpT, typename BCEval >
  BoundaryConditionOp<OpT,BCEval>::
  BoundaryConditionOp( const SpatialOps::MemoryWindow& window,
                       const std::vector<IntVec>& destIJKPoints,
                       const BCSide side,
                       const BCEval bceval,
                       const OperatorDatabase& soDatabase )
  : bcEval_( bceval ),
    singlePointBC_(false)
  {
    // let phi_a be the ghost value, phi_b be the internal value, and phi_bc be the boundary condition,
    //   phi_bc = a*phi_a + b*phi_b
    // then
    //   phi_a = (phi_bc - b*phi_b) / a
    //
    const OpT* const op = soDatabase.retrieve_operator<OpT>();
    ca_ = (side==MINUS_SIDE ? op->coefs().get_coef(0) : op->coefs().get_coef(1) );
    cb_ = (side==MINUS_SIDE ? op->coefs().get_coef(1) : op->coefs().get_coef(0) );
    //
    std::vector<IntVec>::const_iterator destPointsIter = destIJKPoints.begin();
    for( ; destPointsIter != destIJKPoints.end(); ++destPointsIter ) {
      flatGhostPoints_.push_back(window.flat_index(*destPointsIter + ( (side==MINUS_SIDE) ? S1Shift::int_vec() : S2Shift::int_vec() )));    // a_point
      flatInteriorPoints_.push_back(window.flat_index(*destPointsIter + ( (side==MINUS_SIDE) ? S2Shift::int_vec() : S1Shift::int_vec() ))); // b_point
    }
  }

  //------------------------------------------------------------------

  template< typename OpT, typename BCEval >
  void
  BoundaryConditionOp<OpT,BCEval>::
  operator()( SrcFieldT& f ) const
  {
    // jcs: this is not very efficient, but I am not sure that we can do any better at this point.

    const MemoryWindow& w = f.window_without_ghost();

    if (singlePointBC_) {
      const MemoryWindow wa( w.glob_dim(), w.offset()+apoint_, IntVec(1,1,1) );
      const MemoryWindow wb( w.glob_dim(), w.offset()+bpoint_, IntVec(1,1,1) );
      SrcFieldT fa( wa, f );
      SrcFieldT fb( wb, f );
      fa <<= ( bcEval_() - cb_ * fb ) / ca_;
    }
    else {
      // jcs note that we could speed this up a bit by saving off the index triplets rather than the flat indices (which require us to convert back here).
      std::vector<int>::const_iterator ia = flatGhostPoints_.begin(); // ia is the ghost flat index
      std::vector<int>::const_iterator ib = flatInteriorPoints_.begin(); // ib is the interior flat index
      for( ; ia != flatGhostPoints_.end(); ++ia, ++ib ){
        const MemoryWindow wa( w.glob_dim(), w.offset()+w.ijk_index_from_local(*ia), IntVec(1,1,1) );
        const MemoryWindow wb( w.glob_dim(), w.offset()+w.ijk_index_from_local(*ib), IntVec(1,1,1) );
        SrcFieldT fa( wa, f );
        SrcFieldT fb( wb, f );
        fa <<= ( bcEval_() - cb_ * fb ) / ca_;
      }
    }
  }

  //------------------------------------------------------------------

} // namespace SpatialOps

#endif // SpatialOps_FVStaggeredStencilBCOp_h
