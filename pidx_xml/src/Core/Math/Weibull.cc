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
 *  Weibull.cc: support choosing a random value from a 1D Weibull
 *               distribution (rand), as well as evaluate the probability
 *               of a particular value occuring.
 *
 *  Written by:
 *   Scot Swan
 *   Department of Mechanical Engineering
 *   University of Utah
 *   March 2009
 *
 */

#include <Core/Math/Weibull.h>
#include <Core/Math/MusilRNG.h>

namespace Uintah {

Weibull::Weibull(double WeibMean,
                 double WeibMod,
                 double WeibRefVol,
                 int WeibSeed,
                 double WeibExp) 
  : WeibMean_(WeibMean),
    WeibMod_(WeibMod),
    WeibRefVol_(WeibRefVol),
    WeibExp_(WeibExp),
    mr_(new MusilRNG(WeibSeed))
{
}

Weibull::~Weibull() {
  delete mr_;
}

} // End namespace Uintah
