#ifndef UINTAH_HOMEBREW_PROC0COUT_H
#define UINTAH_HOMEBREW_PROC0COUT_H

//
// Class: Proc0Cout
//
//    Just a wrapper, more or less, for cout that only actually uses
//    cout if the code is being executed on processor 0.  Helps to clean
//    up the code and to avoid unnecessary debug spew.
//
// Note: This is a singleton class, don't try to instantiate one.
//    Just use "proc0cout << output;"
//

#include <string>

#include <Core/Geometry/IntVector.h>

#include <Packages/Uintah/Core/Grid/Patch.h>

namespace Uintah {

class Proc0Cout {

public:

  static inline Proc0Cout & proc0cout() {
    if( Proc0Cout::singleton == NULL ) {
      singleton = new Proc0Cout();
    }
    return *Proc0Cout::singleton;
  }

  Proc0Cout & operator<<( const std::string & s );
  Proc0Cout & operator<<( const char * str );
  Proc0Cout & operator<<( int i );
  Proc0Cout & operator<<( double d  );
  Proc0Cout & operator<<( const IntVector & iv  );
  Proc0Cout & operator<<( const Patch & p  );

private:

  Proc0Cout();
  ~Proc0Cout();

  static Proc0Cout * singleton;
};

#define proc0cout Proc0Cout::proc0cout()

} // end namespace Uintah

#endif
