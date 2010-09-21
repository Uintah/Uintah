#ifndef Coordinate_Expr_h
#define Coordinate_Expr_h

#include <CCA/Components/Wasatch/Wasatch.h>

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
    Coordinate( Wasatch* const wasatch,
                const Direction dir,
                const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg )
      : Expr::PlaceHolder<FieldT>(id,reg)
    {
      wasatch->requires_coordinate<FieldT>( dir );
    }

  public:
    class Builder : public Expr::ExpressionBuilder
    {
    public:
      Builder( Wasatch* const wasatch, const Direction dir )
        : wasatch_( wasatch_ ),
          dir_( dir )
      {}
      Expr::ExpressionBase*
      build( const Expr::ExpressionID& id,
             const Expr::ExpressionRegistry& reg ) const
      {
        return new Coordinate<FieldT>(wasatch_,dir_,id,reg);
      }
    private:
      Wasatch* const wasatch_;
      const Direction dir_;
    };
    ~Coordinate(){}
  };

} // namespace Wasatch

#endif // Coordinate_Expr_h
