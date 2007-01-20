/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


#ifndef CORE_ALGORITHMS_UTIL_ALGOLIST_H
#define CORE_ALGORITHMS_UTIL_ALGOLIST_H 1

#include <Core/Containers/Handle.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

template<class ALGO>
class AlgoList : public std::vector<SCIRun::Handle<ALGO> > {
public:  
  AlgoList();
  void      lock();
  void      unlock();
  void      add(ALGO* ptr);
  
private:
  Mutex         algolistlock_;
};

template<class ALGO>
AlgoList<ALGO>::AlgoList() :
  algolistlock_("algolist")
{
}

template<class ALGO>
void AlgoList<ALGO>::lock()
{
  algolistlock_.lock();
}

template<class ALGO>
void AlgoList<ALGO>::unlock()
{
  algolistlock_.unlock();
}

template<class ALGO>
void AlgoList<ALGO>::add(ALGO* algoptr)
{
  Handle<ALGO> handle = algoptr;
  push_back(handle);
}

} //end namespace

#endif
