
#ifndef UINTAH_HOMEBREW_Primes_H
#define UINTAH_HOMEBREW_Primes_H

#include <Core/Math/TntJama/tnt.h>
#include <Core/Math/uintahshare.h>
namespace Uintah {
  class UINTAHSHARE Primes {
  public:
    static const int MaxFactors = 64;
    typedef unsigned long FactorType[MaxFactors];
    static int factorize(unsigned long n, FactorType);

  private:
    bool havePrimes;
    
  };
}

#endif
