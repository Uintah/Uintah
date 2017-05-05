/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/*
 *  Expon.h: Interface to Exponentiation functions...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 */

#ifndef SCI_Math_Expon_h
#define SCI_Math_Expon_h 1
#include <Core/Util/Assert.h>
#include <Core/Util/FancyAssert.h>
#include <map>
#include <cmath>
#include <iostream>

namespace Uintah {

  inline double Pow(double d, double p)
  {
    return pow(d,p);
  }


  inline double Pow(double x, unsigned int p)
  {
    double result=1;
    while(p){
      if(p&1)
        result*=x;
      x*=x;
      p>>=1;
    }
    return result;
  }

  inline double Pow(double x, int p)
  {
    if(p < 0){
      p=-p;
      double result=1;
      while(p){
        if(p&1)
          result*=x;
        x*=x;
        p>>=1;
      }
      return 1./result;
    } else { 
      double result=1;
      while(p){
        if(p&1)
          result*=x;
        x*=x;
        p>>=1;
      }
      return result;
    }
  }

  inline int Sqrt(int i)
  {
    return (int)sqrt((double)i);
  }

  inline long double Sqrt(long double d)
  {
    return sqrtl(d);
  }

  inline double Sqrt(double d)
  {
    return sqrt(d);
  }

  inline float Sqrt(float d)
  {
    return sqrtf(d);
  }

#if defined(linux)
  inline double Cbrt(double d)
  {
    return pow(d, 1./3.);
  }

  inline long double Cbrt(long double d)
  {
    return powl(d, 1./3.);
  }
#else
  inline double Cbrt(double d)
  {
    return cbrt(d);
  }

  inline long double Cbrt(long double d)
  {
    return cbrtl(d);
  }
#endif

  inline double Sqr(double x)
  {
    return x*x;
  }

  inline double Exp(double x)
  {
    return exp(x);
  }

  inline float Exp(float x)
  {
    return expf(x);
  }
  
  //______________________________________________________________________
  //
  // A collection of fast approximate exp() functions
  // Use these calls if the exp() is in the inner most loop
  
  class fastApproxExponent {
    public: 
      fastApproxExponent(); 
      ~fastApproxExponent();
             
      //__________________________________
      // These methods are fast and reasonably accurate for arguments < 1
      // If the argument is near 0 to 1 and you don't need the precision consider one
      // of these.  Obviously the more terms you include the more accurate it is.
      //
      // http://www.musicdsp.org/showone.php?id=222
      // These are computed using a Taylor series with limited number of terms.  For example exp4():
      //  exp(x) = 1 + x + x^2/2 + x^3/6 + x^4/24 + ...
      //         = (24 + 24x + x^2*12 + x^3*4 + x^4) / 24
      //    (using Horner-scheme:)
      //         = (24 + x * (24 + x * (12 + x * (4 + x)))) * 0.041666666f
      
      inline double exp1(double x) { 
        return (6+x*(6+x*(3+x)))*0.16666666f; 
      }

      inline double exp2(double x) {
        return (24+x*(24+x*(12+x*(4+x))))*0.041666666f;
      }

      inline double exp3(double x) {
        return (120+x*(120+x*(60+x*(20+x*(5+x)))))*0.0083333333f;
      }

      inline double exp4(double x) {
        return (720+x*(720+x*(360+x*(120+x*(30+x*(6+x))))))*0.0013888888f;
      }

      inline double exp5(double x) {
        return (5040+x*(5040+x*(2520+x*(840+x*(210+x*(42+x*(7+x)))))))*0.00019841269f;
      }

      inline double exp6(double x) {
        return (40320+x*(40320+x*(20160+x*(6720+x*(1680+x*(336+x*(56+x*(8+x))))))))*2.4801587301e-5;
      }

      inline double exp7(double x) {
        return (362880+x*(362880+x*(181440+x*(60480+x*(15120+x*(3024+x*(504+x*(72+x*(9+x)))))))))*2.75573192e-6;
      }       

      void populateExp_int(const int low, 
                           const int high);
      //______________________________________________________________________
      //  Minimize the error by splitting the arguement (x)
      //  into the integer and mantisa. 
      // Then we can use: 
      //  exp(z) = exp(x+y)  = exp(x) * exp(y)
      //                         A        B
      //             where -1 < y < 1
      //  exp(x)   has been pre-populated map and is d_exp_integer[x]
      //  exp5(y):  Taylor Series expansion with 5 terms
      //  
      //  Note you must pre-populate d_exp_integer before calling this method
      //  This is 1.2 times faster than std:exp()
      //
      //   This is experimental
      //______________________________________________________________________
      inline double fast_exp(const double z)
      {

        ASSERTMSG( (d_exp_integer.size() > 0), "The d_exp_integer map must be populated before using faster_exp()");
        double x;
        double y = std::modf(z, &x);

        ASSERTRANGE( x, d_int_min, d_int_max);   // is the value of x contained in the map?

        double A = d_exp_integer[(int)x];
        double B = exp5(y); 
       // std::cout << "  fast_exp: z: " << z << " x: " << x << " exp(x): " << A << " y: " << y << " exp(y) " << B<<  " d_int_min: " << d_int_min << " d_int_max: " << d_int_max <<  std::endl;
        return A * B;
      }

      //______________________________________________________________________
      //  Fast approximation to exponential function.
      //                                                               
      // Reference:                                                    
      //    N.N. Schraudolph,                                          
      //    A Fast, Compact Approximation of the Exponential Function,
      //    Neural Computation, 11(4), 1999.                           
      //    http://www.inf.ethz.ch/~schraudo/pubs/
      //
      //    G.C. Cawley,
      //    On a Fast Compact Approximation of the Exponential Function
      //    Neural Computation 12, 2009-2012 (2000)
      //
      //    http://www.cecs.uci.edu/~papers/neuromorphic2008/lectures07/ieee754twiddles.pdf
      //
      //    Avoid this method if the argument is near 0 to 1 ish.  The taylor
      //    series methods are more accurate.
      //______________________________________________________________________
      #define EXP_A 1048576/M_LN2
      #define EXP_C 60801
      inline double Schraudolph_exp( const double x)
      {
        double y = 0.5 * x;     // splitting x in two improve error characteristics.  See pdf
        
        ASSERTRANGE( y, -700, 700);
        union{
          double d;
      #ifdef LITTLE_ENDIAN
          struct{
            int j;
            int i;
          } n;
      #else
          struct{
            int i;
            int j;
          } n;
      #endif     
        } eco;

        eco.n.i = (int)(EXP_A*(y)) + (1072693248 - EXP_C);
        eco.n.j = 0;
        double    exp_x_2 = eco.d;      // exp(x/2)
        return exp_x_2 * exp_x_2;       // exp(x/2 + x/2) = exp(x/2)*exp(x/2
      }

    //__________________________________
    private:
      std::map<int,double> d_exp_integer;       // contains exp(Integer)
      int d_int_min;                            // min & max range of argument.
      int d_int_max;                            // used for bulleproofing
   };
}

#endif
