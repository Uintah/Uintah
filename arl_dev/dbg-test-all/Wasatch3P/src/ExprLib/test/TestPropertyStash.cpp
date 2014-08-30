#include <expression/PropertyStash.h>
#include <spatialops/structured/IntVec.h>
#include <iostream>

int main()
{
  bool isOkay = true;

  Expr::PropertyStash props;

  props.set( "a", 1.23 );
  props.set( "b", std::string("hi") );
  props.set( "intvec", SpatialOps::IntVec(1,2,3) );

  try{
    const std::string bb = props.get<std::string>("b");
    const double aa = props.get<double>("a");
    const SpatialOps::IntVec iv = props.get<SpatialOps::IntVec>("intvec");
  }
  catch( std::exception& err ){
    std::cout << err.what() << std::endl;
    isOkay = false;
  }

  try{
    const int bb = props.get<int>("b"); // should fail
    isOkay = false;
  }
  catch( std::exception& err ){
    // ok - fail expected
  }
  if( isOkay ){
    std::cout << "PASS" << std::endl;
    return 0;
  }
  std::cout << "FAIL" << std::endl;
  return -1;
}
