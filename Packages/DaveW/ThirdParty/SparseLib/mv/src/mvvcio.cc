//
//      ostream_complex.cc      Add I/O routine for printing out complex
//                              numbers.
//


#ifdef COMPLEX_OSTREAM
#include <complex.h>

// AT&T Cfront does not provide for cout << complex ...
//
ostream& operator<<(ostream &s, COMPLEX z)
{
    s << (double) real(z) << " "  << (double) imag(z) ;
    return s;
}

#endif
