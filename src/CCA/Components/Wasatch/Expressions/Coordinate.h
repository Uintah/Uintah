#ifndef Coordinate_Expr_h
#define Coordinate_Expr_h

#include <CCA/Components/Wasatch/CoordHelper.h>

#include <expression/PlaceHolderExpr.h>

namespace Wasatch{

  // note that this is in the Wasatch namespace since it is tied to
  // Uintah through the CoordHelper class.

  /**
   *  \class 	Coordinate
   *  \author 	James C. Sutherland
   *  \ingroup	Expressions
   *
   *  \brief shell expression to cause coordinates to be set.  Useful
   *         for initialization, MMS, etc.
   *
   *  \tparam FieldT the type of field to set for this coordinate
   */
  template< typename FieldT >
  class Coordinate
    : public Expr::PlaceHolder<FieldT>
  {
    Coordinate( CoordHelper& coordHelper,
                const Direction dir )
      : Expr::PlaceHolder<FieldT>()
    {
      coordHelper.requires_coordinate<FieldT>( dir );
    }

  public:
    class Builder : public Expr::ExpressionBuilder
    {
    public:
      /**
       *  \brief Build a Coordinate expression.
       *  \param result the coordinate calculated by this expression
       *  \param coordHelper - the CoordHelper object.
       *  \param dir - the Direction to set for this coordinate (e.g. x, y, z)
       */
      Builder( const Expr::Tag& result, CoordHelper& coordHelper, const Direction dir )
        : ExpressionBuilder(result),
          coordHelper_( coordHelper ),
          dir_( dir )
      {}
      Expr::ExpressionBase* build() const{ return new Coordinate<FieldT>(coordHelper_,dir_); }
    private:
      CoordHelper& coordHelper_;
      const Direction dir_;
    };
    ~Coordinate(){}
  };

} // namespace Wasatch

#endif // Coordinate_Expr_h
