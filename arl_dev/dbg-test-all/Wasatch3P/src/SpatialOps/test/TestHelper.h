#ifndef AME_TestHelper_h
#define AME_TestHelper_h

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

  inline bool ok() const{ return isokay; }

  inline bool isfailed() const{ return !isokay; }

  inline void report_status( const bool r ){ report=r; }

};

#endif // AME_TestHelper_h
