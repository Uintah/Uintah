#include <expression/DefaultFieldManager.h>
#include <spatialops/structured/FVStaggered.h>
#include <spatialops/structured/FieldComparisons.h>

#include "../TestHelper.h"

typedef SpatialOps::SVolField FieldT;

int main()
{
  try{
    Expr::DefaultFieldManager<FieldT> fm;

    Expr::Tag t1n("test1",Expr::STATE_N   ), t1cf("test1",Expr::CARRY_FORWARD);
    Expr::Tag t2n("test2",Expr::STATE_NONE), t2cf("test2",Expr::CARRY_FORWARD);

    fm.register_field( t1n  );
    fm.register_field( t1cf );

    fm.register_field( t2n  );
    fm.register_field( t2cf );

    Expr::FieldManagerBase* fm2 = new Expr::DefaultFieldManager<FieldT>();
    fm.register_field( Expr::Tag("test3",Expr::STATE_NONE) );

    const Expr::FieldAllocInfo info( SpatialOps::IntVec(5,6,7),
                                     0, 0,
                                     false, false, false );

    fm.allocate_fields( boost::cref(info) );

    TestHelper status(true);

    {
      FieldT& f1 = fm.field_ref( t1n );
      using SpatialOps::operator<<=;
      f1 <<= 1.23;
      fm.copy_field_forward( t1cf, fm.field_ref(t1cf) );
      const FieldT& f2 = fm.field_ref( t1cf );
      status( field_equal(f1, f2), "1" );
    }

    {
      FieldT& f1 = fm.field_ref( t2n );
      using SpatialOps::operator<<=;
      f1 <<= 1.23;
      fm.copy_field_forward( t2n, fm.field_ref(t2cf) );
      const FieldT& f2 = fm.field_ref( t2cf );
      status( field_equal(f1, f2), "2" );
    }

    {
      FieldT& f1 = fm.field_ref( t1n );
      FieldT& f2 = fm.field_ref( Expr::Tag("test3",Expr::STATE_NONE) );
      using SpatialOps::operator<<=;
      f2 <<= f1;
      status( field_equal(f1, f2), "3" );
    }

    if( status.ok() ) return 0;
  }
  catch( std::exception& e ){
    std::cout << e.what() << std::endl
        << "FAIL" << std::endl;
  }
  return -1;
}
