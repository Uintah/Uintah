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


#include <Core/Util/RefCounted.h>
#include <Core/Util/Assert.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>
using namespace Uintah;
using namespace SCIRun;


RefCounted::RefCounted()
    : d_refCount(0)
{
}

RefCounted::~RefCounted()
{
  ASSERTEQ(d_refCount, 0);
}

void
RefCounted::addReference() const
{
   __sync_add_and_fetch(&d_refCount,1);
}

bool
RefCounted::removeReference() const
{
    bool status = (__sync_sub_and_fetch(&d_refCount,1) == 0);
    return status;
}
