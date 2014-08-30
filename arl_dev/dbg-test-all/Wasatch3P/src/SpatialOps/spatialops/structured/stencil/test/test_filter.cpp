#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/Grid.h>
#include <spatialops/Nebo.h>

#include <test/TestHelper.h>

using namespace SpatialOps;

template<typename DirT>
bool run_test( const IntVec dim )
{
  TestHelper status(false);

  const bool bc[3] = {false,false,false};
  const DoubleVec length(1,1,1);

  const GhostData ghost(1);
  const BoundaryCellInfo bcinfo = BoundaryCellInfo::build<SVolField>(bc[0],bc[1],bc[2]);
  const MemoryWindow mw = get_window_with_ghost( dim, ghost, bcinfo );

  SVolField    f( mw, bcinfo, ghost, NULL );
  SVolField fbar( mw, bcinfo, ghost, NULL );

  const Grid grid(dim,length);
  grid.set_coord<DirT>(f);

  if( dim[0]>1 && dim[1]>1 && dim[2]>1 ) {
      const NeboAverageStencilBuilder<BoxFilter3DStencilCollection::StPtCollection, SVolField, SVolField> filter;
      filter.apply_to_field( f, fbar );
  } else if( dim[0]>1 && dim[1]>1 ) {
      const NeboAverageStencilBuilder<BoxFilter2DXYStencilCollection::StPtCollection, SVolField, SVolField> filter;
      filter.apply_to_field( f, fbar );
  } else if( dim[0]>1 && dim[2]>1 ) {
      const NeboAverageStencilBuilder<BoxFilter2DXZStencilCollection::StPtCollection, SVolField, SVolField> filter;
      filter.apply_to_field( f, fbar );
  } else if( dim[1]>1 && dim[2]>1 ) {
      const NeboAverageStencilBuilder<BoxFilter2DYZStencilCollection::StPtCollection, SVolField, SVolField> filter;
      filter.apply_to_field( f, fbar );
  } else if( dim[0]>1 ) {
      const NeboAverageStencilBuilder<BoxFilter1DXStencilCollection::StPtCollection, SVolField, SVolField> filter;
      filter.apply_to_field( f, fbar );
  } else if( dim[1]>1 ) {
      const NeboAverageStencilBuilder<BoxFilter1DYStencilCollection::StPtCollection, SVolField, SVolField> filter;
      filter.apply_to_field( f, fbar );
  } else if( dim[2]>1 ) {
      const NeboAverageStencilBuilder<BoxFilter1DZStencilCollection::StPtCollection, SVolField, SVolField> filter;
      filter.apply_to_field( f, fbar );
  }

  SVolField::const_iterator i1=f.interior_begin();
  SVolField::const_iterator i1e=f.interior_end();
  SVolField::const_iterator i2=fbar.interior_begin();
  for( ; i1!=i1e; ++i1, ++i2 ){
    const double err = std::abs( *i1-*i2 ) / *i1;
    status( err<1e-15 );
//    std::cout << *i1 << " : " << *i2 << ", err=" << err << std::endl;
  }

  return status.ok();
}

int main()
{
  try{
    TestHelper status(true);

    status( run_test<XDIR>( IntVec(30, 1, 1) ), "[30, 1, 1], X" );
    status( run_test<XDIR>( IntVec(30,30, 1) ), "[30,30, 1], X" );
    status( run_test<XDIR>( IntVec(30, 1,30) ), "[30, 1,30], X" );
    status( run_test<XDIR>( IntVec(30,30,30) ), "[30,30,30], X" );

    std::cout << std::endl;

    status( run_test<YDIR>( IntVec( 1,30, 1) ), "[ 1,30, 1], Y" );
    status( run_test<YDIR>( IntVec(30,30, 1) ), "[30,30, 1], Y" );
    status( run_test<YDIR>( IntVec( 1,30,30) ), "[ 1,30,30], Y" );
    status( run_test<YDIR>( IntVec(30,30,30) ), "[30,30,30], Y" );

    std::cout << std::endl;

    status( run_test<ZDIR>( IntVec( 1, 1,30) ), "[ 1, 1,30], Z" );
    status( run_test<ZDIR>( IntVec(30, 1,30) ), "[30, 1,30], Z" );
    status( run_test<ZDIR>( IntVec( 1,30,30) ), "[ 1,30,30], Z" );
    status( run_test<ZDIR>( IntVec(30,30,30) ), "[30,30,30], Z" );

    std::cout << std::endl;

    if( status.ok() ){
      std::cout << "PASS" << std::endl;
      return 0;
    }
  }
  catch( std::exception& err ){
    std::cout << err.what() << std::endl;
  }
  std::cout << "FAIL" << std::endl;
  return -1;
}

