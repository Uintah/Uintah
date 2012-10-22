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
#include <CCA/Components/Arches/MCRT/ArchesRMCRT/RMCRTFactory.h>

using namespace Uintah; 
using namespace std;


RMCRTFactory::RMCRTFactory(){}

RMCRTFactory::~RMCRTFactory(){}


static RMCRTFactory *RMCRTFactory::RMCRTModel(const std::string sample_sche){
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
