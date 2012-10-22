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


#include <CCA/Components/Schedulers/MemoryLog.h>
#include <Core/Grid/Patch.h>
#include <iostream>

using namespace std;

namespace Uintah {
  void logMemory(std::ostream& out, unsigned long& total,
		 const std::string& label, const std::string& name,
		 const std::string& type, const Patch* patch,
		 int material, const std::string& nelems,
		 unsigned long size, void* ptr, int dwid)
  {
    out << label;
    if(dwid != -1)
      out << ":" << dwid;
    char tab = '\t';
    out << tab << name << tab << type << tab;
    if(patch)
      out << patch->getID();
    else
      out << "-";
    out << tab << material << tab << nelems << tab << size << tab << ptr << '\n';
    total += size;
  }
}

