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

/*#############################################################################
  # Error.cc - error messages handler implementation
  ###########################################################################*/

#include "Error.h"
#include "util.h"
#include "DebugStream.h"

std::string
Error::summary(void) const
  /* Print a summary of the object's properties. */
{
  std::ostringstream out;
  out << "Run-time";
  return out.str();
}
  
void
Error::error(const std::ostringstream& msg) const
  /* Print an error message and exit. */
{
  std::cerr << summary() << " error : " << msg.str() << "\n";
  clean();
  exit(1);
}

void
Error::warning(const std::ostringstream& msg) const
  /* Print an warning message but continue running. */
{
  std::cerr << summary() << " warning : " << msg.str() << "\n";
}
