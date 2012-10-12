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

#ifndef UINTAH_HOMEBREW_SHAREASSIGNARRAY3_H
#define UINTAH_HOMEBREW_SHAREASSIGNARRAY3_H

#include <Core/Grid/Variables/Array3.h>

#ifndef _WIN32
#include <unistd.h>
#endif
#include <cerrno>

namespace Uintah {

class TypeDescription;

/**************************************

CLASS
ShareAssignArray3
  
Short description...

GENERAL INFORMATION

ShareAssignArray3.h

Version of the Array3 class, but with allowed assignment that shares
the data rather than copies it

Wayne Witzel
Department of Computer Science
University of Utah

Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


KEYWORDS
Array3, Assignment

DESCRIPTION
Long description...
  
WARNING
  
****************************************/

template<class T>
class ShareAssignArray3 : public Array3<T> {
public:
  ShareAssignArray3()
    : Array3<T>() {}
  ShareAssignArray3(const Array3<T>& pv)
    : Array3<T>(pv) {}

  virtual ~ShareAssignArray3() {}
  
  ShareAssignArray3<T>& operator=(const Array3<T>& pv)
  { copyPointer(pv); return *this; }

  ShareAssignArray3<T>& operator=(const ShareAssignArray3<T>& pv)
  { copyPointer(pv); return *this; }
};

} // End namespace Uintah

#endif
