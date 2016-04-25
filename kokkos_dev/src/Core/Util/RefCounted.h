/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#ifndef UINTAH_HOMEBREW_REFCOUNTED_H
#define UINTAH_HOMEBREW_REFCOUNTED_H

#include <atomic>
#include <iostream>

namespace Uintah {
/**************************************

CLASS
   RefCounted

   Short description...

GENERAL INFORMATION

   Task.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


KEYWORDS
   Reference_Counted, RefCounted, Handle

DESCRIPTION
   Long description...

WARNING

****************************************/

struct RefBase {
  virtual ~RefBase() {}
};


class RefCounted
  : public RefBase
{
public:
  RefCounted() = default;

  RefCounted( const RefCounted& ) = delete;
  RefCounted & operator=( const RefCounted& ) = delete;

  RefCounted( RefCounted&& ) = default;
  RefCounted & operator=( RefCounted&& ) = default;

  virtual ~RefCounted()
  {
#ifndef NDEBUG
    if (m_refcount.load(std::memory_order_relaxed)) {
      std::cerr << "Deleted object with non-zero reference count!" << std::endl;
    }
#endif
  }

  void addReference() const
  {
    m_refcount.fetch_add(1, std::memory_order_relaxed);
  }

  bool removeReference() const
  {
    return 1 == m_refcount.fetch_add(-1, std::memory_order_relaxed);
  }

  int getReferenceCount() const
  {
    return m_refcount.load(std::memory_order_relaxed);
  }

private:
  mutable std::atomic<int> m_refcount{0};
};

} // End namespace Uintah

#endif
