/*
 * Copyright (c) 2014 The University of Utah
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
   Orion Sky Lawlor, olawlor@acm.org, 9/23/2000

   Computes so-called "Mie scattering" for a homogenous
   sphere of arbitrary size illuminated by coherent
   harmonic radiation.
 */

#include "Particles.h"
#include "TestHelper.h"

#include <iostream>
#include <iomanip>

using namespace std;

int main( int argc, char* argv[] )
{
  TestHelper status(true);
  try{
    complex<double> rCoal;
    rCoal=complex<double>(2.0,-0.6);
    ParticleRadCoeffs P1(rCoal);

    const double absspectralRL002 = P1.abs_spectral_coeff(1e-6, 5e-7);
    //	  cout << "Interp Abs Spectral (1e-6, 5e-7) " << setprecision(12) << absspectralRL002 << endl;
    status( std::abs( absspectralRL002 - 1.3782846611e-12)/absspectralRL002 < 1e-10, "Abs Spectral 1" );


    const double scaspectralRL002 = P1.scattering_spectral_coeff(1e-6, 5e-7);
    //	  cout << "Interp Sca Spectral (1e-6, 5e-7) " << setprecision(12) << scaspectralRL002 << endl;
    status( std::abs( scaspectralRL002 - 1.31439905892e-12)/scaspectralRL002 < 1e-10, "Sca Spectral 1" );

    const double absspectralRL001 = P1.abs_spectral_coeff(1.2e-5, 5e-6);
    //    cout << "Interp Abs Spectral (1.2e-5, 5e-6) " << setprecision(12) << absspectralRL001 << endl;
    status( std::abs( absspectralRL001 - 1.38450195876e-10)/absspectralRL001 < 1e-10, "Abs Spectral 2" );


    const double scaspectralRL001 = P1.scattering_spectral_coeff(1.2e-5, 5e-6);
    //    cout << "Interp Sca Spectral (1.2e-5, 5e-6) " << setprecision(12) << scaspectralRL001 << endl;
    status( std::abs( scaspectralRL001 - 1.22781192143e-10)/scaspectralRL001 < 1e-10, "Sca Spectral 2" );


    const double absspectralRL = P1.abs_spectral_coeff(6e-6, 1.25e-5);
    //	  cout << "Interp Abs Spectral (6e-6, 1.25e-5) " << setprecision(12) << absspectralRL << endl;
    status( std::abs( absspectralRL - 5.20280431039e-10)/absspectralRL < 1e-10, "Abs spectral 3" );

    const double scaspectralRL = P1.scattering_spectral_coeff(6e-6, 1.25e-5);
    //	  cout << "Interp Sca Spectral (6e-6, 1.25e-5) " << setprecision(12) << scaspectralRL << endl;
    status( std::abs( scaspectralRL - 6.35765173434e-10)/scaspectralRL < 1e-10, "Sca Spectral 3");


    const double absplanckRT1200 = P1.planck_abs_coeff(1.25e-5, 1200);
    //	  cout << "Interp Abs Planck (1.25e-5, 1200) " << setprecision(12) << absplanckRT1200 << endl;
    status( std::abs( absplanckRT1200 - 4.67102530629e-10)/absplanckRT1200 < 1e-10, "Abs Planck 1200" );

    const double scaplanckRT1200 = P1.planck_sca_coeff(1.25e-5, 1200);
    //	  cout << "Interp Sca Planck (1.25e-5, 1200) " << setprecision(12) << scaplanckRT1200 << endl;
    status( std::abs( scaplanckRT1200 - 6.07073025146e-10)/scaplanckRT1200 < 1e-10, "Sca Planck 1200" );

    const double absplanckRT1800 = P1.planck_abs_coeff(5e-7, 1800);
    //	  cout << "Interp Abs Planck (5e-7, 1800) " << setprecision(12) << absplanckRT1800 << endl;
    status( std::abs( absplanckRT1800 - 1.34251557074e-12)/absplanckRT1800 < 1e-10, "Abs Planck 1800" );


    const double scaplanckRT1800 = P1.planck_sca_coeff(5e-7, 1800);
    //	  cout << "Interp Sca Planck (5e-7, 1800) " << setprecision(12) << scaplanckRT1800 << endl;
    status( std::abs( scaplanckRT1800 - 1.08655284028e-12)/scaplanckRT1800 < 1e-10, "Sca Planck 1800" );


    const double absplanckRT2000 = P1.planck_abs_coeff(5e-6, 2000);
    //	  	  cout << "Interp Abs Planck (5e-6, 2000) " << setprecision(12) << absplanckRT2000 << endl;
    status( std::abs( absplanckRT2000 - 9.55599200929e-11)/absplanckRT2000 < 1e-10, "Abs Planck 2000" );

    const double scaplanckRT2000 = P1.planck_sca_coeff(5e-6, 2000);
    //		cout << "Interp Sca Planck (5e-6, 2000) " << setprecision(12) << scaplanckRT2000 << endl;
    status( std::abs( scaplanckRT2000 - 1.18580032833e-10)/scaplanckRT2000 < 1e-10, "Sca Planck 2000" );


    const double absrossRT1200 = P1.ross_abs_coeff(1.25e-5, 1200);
    //	  cout << "Interp Abs Ross (1.25e-5, 1200) " << setprecision(12) << absrossRT1200 << endl;
    status( std::abs(absrossRT1200 - 1.18650286699e-10)/absrossRT1200 < 1e-10, "Abs Rosseland 1200" );


    const double scarossRT1200 = P1.ross_sca_coeff(1.25e-5, 1200);
    //	  cout << "Interp Sca Ross (1.25e-5, 1200) " << setprecision(12) << scarossRT1200 << endl;
    status( std::abs( scarossRT1200 - 1.58523581714e-10)/scarossRT1200 < 1e-10, "Sca Rosseland 1200" );

    const double absrossRT1800 = P1.ross_abs_coeff(5e-7, 1800);
    //	  cout << "Interp Abs Ross (5e-7, 1800) " << setprecision(12) << absrossRT1800 << endl;
    status( std::abs(absrossRT1800 - 3.32042200602e-13)/absrossRT1800 < 1e-10, "Abs Rosseland 1800" );

    const double scarossRT1800 = P1.ross_sca_coeff(5e-7, 1800);
    //	  cout << "Interp Sca Ross (5e-7, 1800) " << setprecision(12) << scarossRT1800 << endl;
    status( std::abs( scarossRT1800 - 1.9328544436e-13)/scarossRT1800 < 1e-10, "Sca Rosseland 1800" );


    const double absrossRT2000 = P1.ross_abs_coeff(5e-6, 2000);
    //	  	  cout << "Interp Abs Ross (5e-6, 2000) " << setprecision(12) << absrossRT2000 << endl;
    status( std::abs(absrossRT2000 - 2.31328094937e-11)/absrossRT2000 < 1e-10, "Abs Rosseland 2000" );

    const double scarossRT2000 = P1.ross_sca_coeff(5e-6, 2000);
    //	  	  cout << "Interp Sca Ross (5e-6, 2000) " << setprecision(12) << scarossRT2000 << endl;
    status( std::abs( scarossRT2000 - 2.98789510935e-11)/scarossRT2000 < 1e-10, "Sca Rosseland 2000" );


    // Soot compared to values in Modest Book p. 404 for the same parameters.
    // Equivalent results to the ones in the text. Note here Abs. Coeff has units 1/m
    const complex<double> rSoot(1.89,-0.92);
    const double soot01 = soot_abs_coeff( 3e-6, 1e-5, rSoot );
//    cout << "Soot Abs " << setprecision(12) << soot01 << endl;
    status( std::abs( soot01 - 19.0412262929)/soot01 < 1e-10, "Soot 1" );

    const complex<double> rSootLeeTien(2.21,-1.23);
    const double soot02 = soot_abs_coeff( 3e-6, 1e-5, rSootLeeTien );
//    cout << "Soot Abs " << setprecision(12) << soot02 << endl;
    status( std::abs( soot02 - 17.5455908825)/soot02 < 1e-10, "Soot 2" );

    if( status.ok() ){
      cout << "PASS" << endl;
      return 0;
    }

  }
  catch( std::exception& err ){
    cout << err.what() << endl;
  }

  cout << "FAIL" << endl;
  return -1;
}
