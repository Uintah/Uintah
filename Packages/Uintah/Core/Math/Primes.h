
#ifndef UINTAH_HOMEBREW_Primes_H
#define UINTAH_HOMEBREW_Primes_H

namespace Uintah {
  class Primes {
  public:
    static const int MaxFactors = 64;
    typedef unsigned long FactorType[MaxFactors];
    static int factorize(unsigned long n, FactorType);

  private:
    bool havePrimes;
    
  };
}

#endif
