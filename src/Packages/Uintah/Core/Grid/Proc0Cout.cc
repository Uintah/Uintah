
#include <Packages/Uintah/Core/Grid/Proc0Cout.h>

#include <Packages/Uintah/Core/Parallel/Parallel.h>

#include <iostream>

using namespace Uintah;
using namespace std;

Proc0Cout * Proc0Cout::singleton = NULL;

Proc0Cout::Proc0Cout() {}
Proc0Cout::~Proc0Cout() {}

Proc0Cout &
Proc0Cout::operator<<( const string & s ) 
{
  if( Parallel::getMPIRank() == 0 ) {
    cout << s;
  }
  return *this;
}

Proc0Cout &
Proc0Cout::operator<<( const char * str ) 
{
  if( Parallel::getMPIRank() == 0 ) {
    cout << str;
  }
  return *this;
}

Proc0Cout &
Proc0Cout::operator<<( int i ) 
{
  if( Parallel::getMPIRank() == 0 ) {
    cout << i;
  }
  return *this;
}

Proc0Cout &
Proc0Cout::operator<<( double d ) 
{
  if( Parallel::getMPIRank() == 0 ) {
    cout << d;
  }
  return *this;
}

Proc0Cout &
Proc0Cout::operator<<( const Patch & patch )
{
  if( Parallel::getMPIRank() == 0 ) {
    cout << patch;
  }
  return *this;
}

Proc0Cout &
Proc0Cout::operator<<( const IntVector & iv )
{
  if( Parallel::getMPIRank() == 0 ) {
    cout << iv;
  }
  return *this;
}

