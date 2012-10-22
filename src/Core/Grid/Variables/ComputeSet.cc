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


#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Patch.h>
#include <iostream>


using namespace Uintah;

namespace Uintah {

  ostream& operator<<(ostream& out, const Uintah::PatchSet& ps)
  {
    if(&ps == 0)
      out << "(null PatchSet)";
    else {
      out << "Patches: {";
      for(int i=0;i<ps.size();i++){
        const PatchSubset* pss = ps.getSubset(i);
        if(i != 0)
          out << ", ";
        out << *pss;
      }
      out << "}";
    }
    return out;
  }

  ostream& operator<<(ostream& out, const Uintah::PatchSubset& pss)
  {
    out << "{";
    for(int j=0;j<pss.size();j++){
      if(j != 0)
        out << ",";
      const Patch* patch = pss.get(j);
      out << patch->getID();
    }
    out << "}";
    return out;
  }

  ostream& operator<<(ostream& out, const Uintah::MaterialSubset& mss)
  {
    out << "{";
    for(int j=0;j<mss.size();j++){
      if(j != 0)
        out << ",";
      out << mss.get(j);
    }
    out << "}";
    return out;
  }

  ostream& operator<<(ostream& out, const Uintah::MaterialSet& ms)
  {
    if(&ms == 0)
      out << "(null Materials)";
    else {
      out << "Matls: {";
      for(int i=0;i< ms.size();i++){
        const MaterialSubset* mss = ms.getSubset(i);
        if(i != 0)
          out << ", ";
        out << *mss;
      }
      out << "}";
    }
    return out;
  }

} // end namespace Uintah


  
