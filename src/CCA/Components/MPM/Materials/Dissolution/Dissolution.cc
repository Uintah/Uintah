/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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

#include <CCA/Components/MPM/Materials/Dissolution/Dissolution.h>

using namespace Uintah;
using namespace std;

Dissolution::Dissolution(const ProcessorGroup* myworld, MPMLabel* Mlb, 
                         ProblemSpecP ps)
                         : lb(Mlb)
{
}

Dissolution::~Dissolution()
{
}

void Dissolution::setTemperature(double BHTemp)
{
  d_temperature = BHTemp;
}

void Dissolution::setPhase(std::string LCPhase)
{
  // phase is "ramp", "settle" or "hold"  Only do dissolution
  // during the "hold" phase

  d_phase = LCPhase;
}

void Dissolution::setTimeConversionFactor(const double tcf)
{
  // This is the factor to convert 1 Uintah time unit (probably a
  // microsecond) to a Ma.  i.e., if tcf=10, 1 microsecond = 10 Ma
  d_timeConversionFactor = tcf;
}
