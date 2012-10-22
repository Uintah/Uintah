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

#include <Core/Datatypes/Archive.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>

namespace Uintah {
using namespace SCIRun;

static Persistent* maker()
{
    return scinew Archive;
}

PersistentTypeID Archive::type_id("Data Archive", "Archive", maker);
#define Archive_VERSION 3
void Archive::io(Piostream&)
{
    NOT_FINISHED("Archive::io(Piostream&)");
}
Archive::Archive() :
  archive(0)
{
}

Archive::Archive(const DataArchiveHandle& ar) :
  archive(ar)
{
}

Archive::~Archive()
{
}
} // End namespace Uintah

