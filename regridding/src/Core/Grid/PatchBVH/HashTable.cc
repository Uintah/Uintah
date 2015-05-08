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
#include <Core/Grid/PatchBVH/HashTable.h>

namespace Uintah {
    HashTable::HashTable(int& nRow, int& nCol, int& nHie) : _x(nRow), _y(nCol), _z(nHie), _yz(nRow * nCol)
    {
        _root = new HEntry[_x * _yz];
		_patchesCnt = 0;
    }
    
    HashTable::HashTable(IntVector& n) : _x(n[0]), _y(n[1]), _z(n[2]), _yz(n[1] * n[2])
    {
        _root = new HEntry[_yz * _x];
		_patchesCnt = 0;
    }
    
    void HashTable::clear()
    {
        delete[] _root;
    }

	void HashTable::setup_empty()
	{
		int valid_dist;
		bool find_valid;
		for (int x = _x - 1; x >= 0; x--)
			for (int y = _y - 1; y >= 0; y--)
			{
				find_valid = false;
				for (int z = _z - 1; z >= 0; z--)
				{
        			int hashVal = x * _yz + y * _z  + z;
					if (_root[hashVal].pPtr)
					{
						find_valid = true;
						valid_dist = 0;
					}
					else if (find_valid)
						_root[hashVal].eID = ++valid_dist;
				}
			}
		return;
	}

	void HashTable::batch_get(IntVector& low, IntVector& high, int* bm, int& cur_true, Level::selectType& patches)
	{
		const Patch* tmp;
		int eid;

		for (int x = low[0]; x < high[0]; x++)
			for (int y = low[1]; y < high[1]; y++)
				for (int z = low[2]; z < high[2]; z++)
				{
					int hashVal = x * _yz + y * _z + z;
					tmp = _root[hashVal].pPtr;
					eid = _root[hashVal].eID;
					if (tmp && cur_true != bm[ eid ])
					{
						patches.push_back(tmp);
						bm[ eid ] = cur_true;
					}
					else if (!tmp)
					{
						if (eid > 0)
							z = z + eid;
						else
							break;
					}
				}
		return;
	}
	
	void HashTable::batch_put(IntVector& low, IntVector& high, const Patch* value)
	{
		if (low[0] >= _x || low[1] >= _y || low[2] >= _z
				|| high[0] < 0 || high[1] < 0 || high[2] < 0)
			return;

		for (int x = low[0]; x < high[0]; x++)
			for (int y = low[1]; y < high[1]; y++)
				for (int z = low[2]; z < high[2]; z++)
				{
        			int hashVal = x * _yz + y * _z  + z;
					_root[hashVal].pPtr = value;
					_root[hashVal].eID = _patchesCnt;
				}
		_patchesCnt++;
		return;
	}

    HashTable::~HashTable()
    {
        clear();
    }
}
