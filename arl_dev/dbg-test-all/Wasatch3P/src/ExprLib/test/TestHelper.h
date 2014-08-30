#ifndef HLA_TestHelper_h
#define HLA_TestHelper_h

#include <iostream>

class TestHelper
{
  bool isokay;
  bool report;
public:
  TestHelper( const bool reportResults=true )
    : report( reportResults )
  {
    isokay=true;
  }

  void operator()( const bool result )
  {
    if( isokay )
      isokay = result;
    if( report ){
      if( result )
        std::cout << "  pass " << std::endl;
      else
        std::cout << "  fail " << std::endl;
    }
  }

  template<typename T>
  void operator()( const bool result, const T t )
  {
    if( isokay )
      isokay = result;
    if( report ){
      if( result )
        std::cout << "  pass " << t << std::endl;
      else
        std::cout << "  fail " << t << std::endl;
    }
  }

  bool ok() const{ return isokay; }

  bool isfailed() const{ return !isokay; }

};

#endif // HLA_TestHelper_h
