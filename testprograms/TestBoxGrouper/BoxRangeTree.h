/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Package/Uintah/testprograms/TestBoxGrouper/Box.h>
#include <list>
#include <vector>

namespace Uintah {
using namespace SCIRun;

// Just does a simple linear query for testing only.
// Maybe change it to a range tree when testing performance.
class BoxRangeQuerier
{
public:
  typedef std::list<Box*> ResultContainer;
  
  BoxRangeQuerier(const vector<Box*>& boxes)
    : boxes_(boxes) {}

  std::list<Box*> query(const IntVector& low, const IntVector& high);
  std::list<Box*> queryNeighbors(const IntVector& low, const IntVector& high);
private:
  std::vector<Box*> boxes_;
};

} // end namespace Uintah
