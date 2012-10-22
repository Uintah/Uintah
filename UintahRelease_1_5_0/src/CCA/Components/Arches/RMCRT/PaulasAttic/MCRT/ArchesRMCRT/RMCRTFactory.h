/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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
#ifndef Uintah_Component_Arches_RMCRTFactory_h
#define Uintah_Component_Arches_RMCRTFactory_h
#include <CCA/Components/Arches/MCRT/ArchesRMCRT/RMCRTRRSD.h>
#include <CCA/Components/Arches/MCRT/ArchesRMCRT/RMCRTRRSDStratified.h>
#include <CCA/Components/Arches/MCRT/ArchesRMCRT/RMCRTnoInterpolation.h>

class RMCRTRRSD; 
class RMCRTRRSDStratified;
class RMCRTnoInterpolation;

using namespace std;
namespace Uintah {

  class RMCRTFactory {
    
  public:
     // constructor
    RMCRTFactory();
    
     // destructor
    virtual ~RMCRTFactory();

    virtual int RMCRTsolver(const int &i_n,
			    const int &j_n,
			    const int &k_n,
			    const int &theta_n,
			    const int &phi_n) =0;
    
    static RMCRTFactory *RMCRTModel(string sample_sche);

    /*
    static RMCRTFactory *RMCRTModel(const std::string sample_sche){
      // call RMCRTnoInterpolation, with no stratified sampling,
      // not even with i_n, j_n, k_n, theta_n, and phi_n all = 1
      // this should be faster
      
      if ( sample_sche == "simple_sample"){
	return new RMCRTnoInterpolation;
      }
      else if ( sample_sche == "RRSD"){
	return new RMCRTRRSD;
      }
      else if ( sample_sche == "RRSDstratified") {
	return  new RMCRTRRSDStratified;
      }
      
    }
    */
    
  }; // Class RMCRTFactory
  
}// end of namespace Uintah

#endif
  
