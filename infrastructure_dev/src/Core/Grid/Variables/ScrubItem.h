/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#ifndef UINTAH_HOMEBREW_ScrubItem_H
#define UINTAH_HOMEBREW_ScrubItem_H

#include <Core/Grid/Patch.h>

namespace Uintah {

class VarLabel;

struct ScrubItem {

    ScrubItem* next;
    const VarLabel* label;
    int matl;
    const Patch* patch;
    int dw;
    size_t hash;
    int count;

    ScrubItem(const VarLabel* l, int m, const Patch* p, int dw) :
      label(l), matl(m), patch(p), dw(dw), count(0)
    {
      size_t ptr = (size_t)l;
      hash = ptr ^ (m << 3) ^ (p->getID() << 4) ^ (dw << 2);
    }

    bool operator==(const ScrubItem& d)
    {
      return label == d.label && matl == d.matl && patch == d.patch && dw == d.dw;
    }

};

}

#endif
