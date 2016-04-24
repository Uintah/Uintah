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


#include <Core/Util/RefCounted.h>
#include <Core/Util/Assert.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>

#include <atomic>
#include <mutex>

using namespace Uintah;

static const int          NLOCKS=1024;
static       std::mutex * locks[NLOCKS];
static       bool         initialized = false;
static       std::mutex   initlock{};

static std::atomic<int32t> nextIndex{0};
static std::atomic<int32t> freeIndex{0};


RefCounted::RefCounted()
    : d_refCount(0)
{
  if (!initialized) {
    initlock.lock();
    if (!initialized) {
      for (int i = 0; i < NLOCKS; i++) {
        locks[i] = new std::mutex{};
      }
      initialized = true;
    }
    initlock.unlock();
  }
  d_lockIndex = nextIndex.fetch_add(1, std::memory_order_relaxed) % NLOCKS;
  ASSERT(d_lockIndex >= 0);
}

RefCounted::~RefCounted()
{
  ASSERTEQ(d_refCount, 0);
  int index = ++freeIndex;
  if (index == nextIndex.load(std::memory_order_relaxed)) {
    initlock.lock();
    if (freeIndex.load(nextIndex) == nextIndex.load(nextIndex)) {
      initialized = false;
      for (int i = 0; i < NLOCKS; i++) {
        delete locks[i];
        locks[i] = 0;
      }
    }
    initlock.unlock();
  }
}

void
RefCounted::addReference() const
{
    locks[d_lockIndex]->lock();
    d_refCount++;
    locks[d_lockIndex]->unlock();
}

bool
RefCounted::removeReference() const
{
    locks[d_lockIndex]->lock();
    bool status = (--d_refCount == 0);
    ASSERT(d_refCount >= 0);
    locks[d_lockIndex]->unlock();
    return status;
}
