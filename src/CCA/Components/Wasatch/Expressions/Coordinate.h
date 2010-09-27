#ifndef Coordinate_Expr_h
#define Coordinate_Expr_h

#include <CCA/Components/Wasatch/CoordHelper.h>

#include <expression/PlaceHolderExpr.h>

namespace Wasatch{

  /**
   *  \class Coordinate
   *  \author James C. Sutherland
   *  \brief shell expression to cause coordinates to be set.  Useful
   *         for initialization, MMS, etc.
   */
  template< typename FieldT >
  class Coordinate
    : public Expr::PlaceHolder<FieldT>
  {
    Coordinate( CoordHelper& coordHelper,
                const Direction dir,
                const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg )
      : Expr::PlaceHolder<FieldT>(id,reg)
    {
      coordHelper.requires_coordinate<FieldT>( dir );
    }

  public:
    class Builder : public Expr::ExpressionBuilder
    {
    public:
      Builder( CoordHelper& coordHelper, const Direction dir )
        : coordHelper_( coordHelper ),
          dir_( dir )
      {}
      Expr::ExpressionBase*
      build( const Expr::ExpressionID& id,
             const Expr::ExpressionRegistry& reg ) const
      {
        return new Coordinate<FieldT>(coordHelper_,dir_,id,reg);
      }
    private:
      CoordHelper& coordHelper_;
      const Direction dir_;
    };
    ~Coordinate(){}
  };

} // namespace Wasatch

#endif // Coordinate_Expr_h
