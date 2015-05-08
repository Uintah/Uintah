/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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
#ifndef Hash_Table_H
#define Hash_Table_H

#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>

#include <string.h>
#include <vector>
#include <map>

namespace Uintah{
    struct HEntry
    {
		int eID;
        const Patch* pPtr;
		HEntry() : pPtr(NULL) { eID = -1;}
        HEntry(const Patch* value) : pPtr(value) { eID = -1;}
        HEntry(Patch* value) : pPtr(value) { eID = -1;}
    };

    class HashTable
    {
        public:
            HashTable(int& nRow, int& nCol, int& nHie);
            HashTable(IntVector& n);
            ~HashTable();

            void clear();
			void setup_empty();
			void batch_put(IntVector& low, IntVector& high, const Patch* value);
			void batch_get(IntVector& low, IntVector& high, int* bm, int& cur_true, Level::selectType& patches);
        private:
            HEntry *_root;
            int _x, _y, _z, _yz;
			int _patchesCnt;
    };
}

#endif
