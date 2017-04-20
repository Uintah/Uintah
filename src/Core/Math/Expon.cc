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

//______________________________________________________________________
//
#include <Core/Math/Expon.h>
#include <iostream>


using namespace Uintah;

//______________________________________________________________________
//
fastApproxExponent::fastApproxExponent()
{
}

//______________________________________________________________________
//
fastApproxExponent::~fastApproxExponent()
{
}

//______________________________________________________________________
//  Populate the map with exp(integer).  This should be placed inside of a 
//  constructor of the class using it.
void
fastApproxExponent::populateExp_int(const int low, 
                                    const int high)
{
  for(int i=low; i<=high; i++){
    d_exp_integer[i] = std::exp(i);
  }
  
  d_int_min = low;    // used for bulletproofing
  d_int_max = high;
}


