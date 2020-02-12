/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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


/*
 *  Datatype.cc: The Datatype Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 */

#include <Core/Datatypes/Datatype.h>
#include <Core/Parallel/MasterLock.h>
#include <Core/Util/Assert.h>

#include <atomic>

namespace Uintah {

static std::atomic<int32_t> current_generation{1};
static Uintah::MasterLock   init_lock{};


int
Datatype::compute_new_generation()
{
  return current_generation.fetch_add(1, std::memory_order_relaxed);
}



Datatype::Datatype()
  : ref_cnt(0),
    generation(compute_new_generation())
{
}

Datatype::Datatype(const Datatype&)
  : ref_cnt(0),
    generation(compute_new_generation())
{
}

Datatype& Datatype::operator=(const Datatype&)
{
  ASSERT(ref_cnt == 1);
  generation = compute_new_generation();
  return *this;
}

Datatype::~Datatype()
{
}

} // End namespace Uintah

