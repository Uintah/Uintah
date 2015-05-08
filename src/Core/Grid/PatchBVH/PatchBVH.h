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
#ifndef PATCH_BVH
#define PATCH_BVH

#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>

#include <Core/Grid/PatchBVH/HashTable.h>

#include <vector>
#include <map>
#include <math.h>
#include <time.h>

namespace Uintah {

	class PatchBVH
	{
		public:
			PatchBVH(const std::vector<const Patch*>& patches);
			PatchBVH(const std::vector<Patch*>& patches);
			~PatchBVH();
			IntVector hashBegin(IntVector index);
			IntVector hashEnd(IntVector index);
			void query(const IntVector& low, const IntVector& high, Level::selectType& patches, bool includeExtraCells=false);
			inline void _query(const IntVector& low, const IntVector& high, Level::selectType& patches, bool includeExtraCells=false);
			void clear();

		private:
			HashTable *_patchMap;
			IntVector _low, _high, _rsl;
            IntVector _lowBound, _highBound;
            IntVector _extraLen;
			int _size, _true, *_bm;
			double _ele[3];
			char _hashFlag[3];
	};
}
#endif
