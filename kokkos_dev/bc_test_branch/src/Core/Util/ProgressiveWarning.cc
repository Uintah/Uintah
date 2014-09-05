/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <Core/Util/ProgressiveWarning.h>
#include <Core/Util/DebugStream.h>

#include <iostream>

namespace SCIRun {

static DebugStream dbg( "ProgressiveWarning", true );

ProgressiveWarning::ProgressiveWarning( std::string message, int multiplier /* = -1 */, 
                                        std::ostream& stream /* = cerr */ )
{
  d_message = message;
  d_multiplier = multiplier;

  if( d_multiplier == 1 ) {
    std::cout << "Warning: ProgressiveWarning multiplier may not be set to 1... changing to 2.\n";
    d_multiplier = 2;
  }

  if (stream == std::cerr)
    out = &dbg;
  else
    out = &stream;
  
  d_numOccurences = 0;
  d_nextOccurence = 1;
  d_warned = false;
}

bool
ProgressiveWarning::invoke( int numTimes /* = -1 */ )
{
  d_numOccurences += numTimes;
  if (d_numOccurences >= d_nextOccurence && (!d_warned || d_multiplier != -1)) {
    d_warned = true;

    if (d_multiplier != -1) {
      showWarning();
      while (d_nextOccurence <= d_numOccurences)
        d_nextOccurence *= d_multiplier;
    }
    else {
      (*out) << d_message << std::endl;
      (*out) << "  This message will only occur once\n";
    }
    return true;
  }
  return false;
}

void
ProgressiveWarning::changeMessage( std::string message )
{
  d_message = message;
}

void
ProgressiveWarning::showWarning()
{
  (*out) << d_message << std::endl;
  (*out) << "  This warning has occurred " << d_numOccurences << " times\n";
}

} // end namespace SCIRun
