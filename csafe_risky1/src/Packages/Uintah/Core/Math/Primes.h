
#ifndef UINTAH_HOMEBREW_Primes_H
#define UINTAH_HOMEBREW_Primes_H


class Primes {
public:
    static const int MaxFactors = 64; // Enough for any 64 bit number
    typedef unsigned long FactorType[MaxFactors];
    static int factorize(unsigned long n, FactorType);

private:
    bool havePrimes;
    
};

#endif
