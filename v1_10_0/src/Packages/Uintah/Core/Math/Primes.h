
#ifndef UINTAH_HOMEBREW_Primes_H
#define UINTAH_HOMEBREW_Primes_H


class Primes {
public:
  static const int MaxFactors; 
  typedef unsigned long FactorType[/*MaxFactors*/64];
  static int factorize(unsigned long n, FactorType);

private:
  bool havePrimes;
    
};

#endif
