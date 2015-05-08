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
#include <Core/Grid/PatchBVH/PatchBVH.h>
using namespace std;

namespace Uintah {

	PatchBVH::PatchBVH(const std::vector<const Patch*>& patches)
	{
//		BlockTimer(Constructor1);
		if (patches.size() == 0)
			return;

#ifdef PDEBUG
		map<int, int> rec_x;
		map<int, int> rec_y;
		map<int, int> rec_z;
		vector<int> cnt_rsl;
#endif

        _extraLen = patches[0]->getCellLowIndex() 
            - patches[0]->getExtraCellLowIndex();
		_rsl = IntVector(1<<30, 1<<30, 1<<30);
		_low = IntVector(1<<30, 1<<30, 1<<30);
        _high = IntVector(-(1<<30), -(1<<30), -(1<<30));

//		cout << "Size: " << patches.size() << endl;
		for (std::vector<const Patch*>::const_iterator it = patches.begin();
				it < patches.end(); it++)
        {
#ifdef PDEBUG
			if (!rec_x.count((*it)->getCellLowIndex().x()))
				rec_x[(*it)->getCellLowIndex().x()] = 1;
			else
				rec_x[(*it)->getCellLowIndex().x()] ++;
			
			if (!rec_y.count((*it)->getCellLowIndex().y()))
				rec_y[(*it)->getCellLowIndex().y()] = 1;
			else
				rec_y[(*it)->getCellLowIndex().y()] ++;
			
			if (!rec_z.count((*it)->getCellLowIndex().z()))
				rec_z[(*it)->getCellLowIndex().z()] = 1;
			else
				rec_z[(*it)->getCellLowIndex().z()] ++;

			cout << "Low: " << (*it)->getCellLowIndex() 
				<< " High: " << (*it)->getCellHighIndex()
				<< endl;
#endif
			for (int i = 0; i < 3; i ++)
			{
				if (_low[i] > (*it)->getCellLowIndex()[i])
					_low[i] = (*it)->getCellLowIndex()[i];
				if (_high[i] < (*it)->getCellHighIndex()[i])
					_high[i] = (*it)->getCellHighIndex()[i];
				if (_rsl[i] > (*it)->getCellHighIndex()[i]
						- (*it)->getCellLowIndex()[i])
					_rsl[i] =(*it)->getCellHighIndex()[i]
						- (*it)->getCellLowIndex()[i];

#ifdef PDEBUG
				bool tmp_judge = true;
				int k = 0;
				for (unsigned int j = 0; j < cnt_rsl.size(); j ++)
					if (cnt_rsl[j] == (*it)->getCellHighIndex()[k] - (*it)->getCellLowIndex()[k])
					{
						tmp_judge = false;
						break;
					}
				if (tmp_judge)
					cnt_rsl.push_back((*it)->getCellHighIndex()[k] - (*it)->getCellLowIndex()[k]);
#endif
			}
        }
        _lowBound = IntVector(0, 0, 0);
        for (int i = 0; i < 3; i ++)
		{
            _highBound[i] = (_high[i] - _low[i]) / _rsl[i];
			_hashFlag[i] = (_high[i] - _low[i]) % _rsl[i] == 0 ? 0 : 1;
		}
        _patchMap = new HashTable(_highBound);
		for (int i = 0; i < 3; i ++)
			_ele[i] = double(_highBound[i]) / double(_high[i] - _low[i]);
		
		for (std::vector<const Patch*>::const_iterator it = patches.begin();
				it < patches.end(); it++)
		{
			IntVector lowNormal, highNormal;
			lowNormal = hashBegin( (*it)->getCellLowIndex() );
			highNormal = hashBegin( (*it)->getCellHighIndex() );
			(*_patchMap).batch_put(lowNormal, highNormal, (*it));
		}

		if (0.7 * _highBound[0] * _highBound[1] * _highBound[2] > patches.size())
			(*_patchMap).setup_empty();
		_size = patches.size() + 1;
		_bm = new int[_size];
		memset(_bm, 0, sizeof(int) * _size);
		_true = 1;
#ifdef PDEBUG
		cout << "size: " << patches.size() << " lowIndex: " << _low << " highIndex: " << _high << endl;
		for (map<int,int>::iterator it = rec_x.begin(); it != rec_x.end(); it++)
			cout << "lowCellIndexX: " << it->first << " count: " << it->second << endl;
		for (map<int,int>::iterator it = rec_y.begin(); it != rec_y.end(); it++)
			cout << "lowCellIndexY: " << it->first << " count: " << it->second << endl;
		for (map<int,int>::iterator it = rec_z.begin(); it != rec_z.end(); it++)
			cout << "lowCellIndexZ: " << it->first << " count: " << it->second << endl;
		for (unsigned int i = 0; i < cnt_rsl.size(); i ++)
			cout << "RSL_CNT: " << cnt_rsl[i] << endl;

		unsigned int local_size = (_highBound[0] * _highBound[1] * _highBound[2]);
		cout << "patches utilize percent: " << double(patches.size()) / local_size << endl;
#endif
		return;
	}

	PatchBVH::PatchBVH(const std::vector<Patch*>& patches)
	{
//		BlockTimer(Constructor2);
		if (patches.size() == 0)
			return;

#ifdef PDEBUG
		map<int, int> rec_x;
		map<int, int> rec_y;
		map<int, int> rec_z;
		vector<int> cnt_rsl;
#endif

		_extraLen = patches[0]->getCellLowIndex() 
            - patches[0]->getExtraCellLowIndex();
		_rsl = IntVector(1<<30, 1<<30, 1<<30);
		_low = IntVector(1<<30, 1<<30, 1<<30);
        _high = IntVector(-(1<<30), -(1<<30), -(1<<30));
		
//		cout << "Size: " << patches.size() << endl;
	
		for (std::vector<Patch*>::const_iterator it = patches.begin();
				it < patches.end(); it++)
        {
#ifdef PDEBUG
			if (!rec_x.count((*it)->getCellLowIndex().x()))
				rec_x[(*it)->getCellLowIndex().x()] = 1;
			else
				rec_x[(*it)->getCellLowIndex().x()] ++;
			
			if (!rec_y.count((*it)->getCellLowIndex().y()))
				rec_y[(*it)->getCellLowIndex().y()] = 1;
			else
				rec_y[(*it)->getCellLowIndex().y()] ++;
			
			if (!rec_z.count((*it)->getCellLowIndex().z()))
				rec_z[(*it)->getCellLowIndex().z()] = 1;
			else
				rec_z[(*it)->getCellLowIndex().z()] ++;

			cout << "Low: " << (*it)->getCellLowIndex() 
				<< " High: " << (*it)->getCellHighIndex()
				<< endl;
#endif
			for (int i = 0; i < 3; i ++)
			{
				if (_low[i] > (*it)->getCellLowIndex()[i])
					_low[i] = (*it)->getCellLowIndex()[i];
				if (_high[i] < (*it)->getCellHighIndex()[i])
					_high[i] = (*it)->getCellHighIndex()[i];
				if (_rsl[i] > (*it)->getCellHighIndex()[i]
						- (*it)->getCellLowIndex()[i])
					_rsl[i] =(*it)->getCellHighIndex()[i]
						- (*it)->getCellLowIndex()[i];
				
#ifdef PDEBUG
				bool tmp_judge = true;
				int k = 1;
				for (unsigned int j = 0; j < cnt_rsl.size(); j ++)
					if (cnt_rsl[j] == (*it)->getCellHighIndex()[k] - (*it)->getCellLowIndex()[k])
					{
						tmp_judge = false;
						break;
					}
				if (tmp_judge)
					cnt_rsl.push_back((*it)->getCellHighIndex()[k] - (*it)->getCellLowIndex()[k]);
#endif
			}
        }
        _lowBound = IntVector(0, 0, 0);
        for (int i = 0; i < 3; i ++)
		{
            _highBound[i] = (_high[i] - _low[i]) / _rsl[i];
			_hashFlag[i] = (_high[i] - _low[i]) % _rsl[i] == 0 ? 0 : 1;
		}
        _patchMap = new HashTable(_highBound);
		for (int i = 0; i < 3; i ++)
			_ele[i] = double(_highBound[i]) / double(_high[i] - _low[i]);
		
		for (std::vector<Patch*>::const_iterator it = patches.begin();
				it < patches.end(); it++)
		{
			IntVector lowNormal, highNormal;
			lowNormal = hashBegin( (*it)->getCellLowIndex() );
			highNormal = hashBegin( (*it)->getCellHighIndex() );
			(*_patchMap).batch_put(lowNormal, highNormal, (*it));
		}

		if (0.7 * _highBound[0] * _highBound[1] * _highBound[2] > patches.size())
			(*_patchMap).setup_empty();
		_size = patches.size();
		_bm = new int[_size];
		memset(_bm, 0, sizeof(int) * _size);
		_true = 1;
#ifdef PDEBUG
		cout << "size: " << patches.size() << " lowIndex: " << _low << " highIndex: " << _high << endl;
		for (map<int,int>::iterator it = rec_x.begin(); it != rec_x.end(); it++)
			cout << "lowCellIndexX: " << it->first << " count: " << it->second << endl;
		for (map<int,int>::iterator it = rec_y.begin(); it != rec_y.end(); it++)
			cout << "lowCellIndexY: " << it->first << " count: " << it->second << endl;
		for (map<int,int>::iterator it = rec_z.begin(); it != rec_z.end(); it++)
			cout << "lowCellIndexZ: " << it->first << " count: " << it->second << endl;
		for (unsigned int i = 0; i < cnt_rsl.size(); i ++)
			cout << "RSL_CNT: " << cnt_rsl[i] << endl;

		unsigned int local_size = (_highBound[0] * _highBound[1] * _highBound[2]);
		cout << "patches utilize percent: " << double(patches.size()) / double(local_size) << endl;
#endif 
		return;
	}

	PatchBVH::~PatchBVH()
	{
		clear();
	}

	void PatchBVH::clear()
	{
		(*_patchMap).clear();
		delete[] _bm;
	}

	IntVector PatchBVH::hashBegin(IntVector index)
	{
		IntVector ret;
		for (int i = 0; i < 3; i ++)
		{
			if (index[i] < _low[i]) (index[i] = _low[i]);
			if (index[i] > _high[i]) (index[i] = _high[i]);
			if (_hashFlag[i] == 0)
				(ret[i] = _ele[i] * (index[i] - _low[i]));
			else
				(ret[i] = _ele[i] * (index[i] - _low[i] + 1));
			if (ret[i] > _highBound[i])
				(ret[i] = _highBound[i]);
			if (ret[i] < 0)
				ret[i] = 0;
		}
		return ret;
	}
	
	IntVector PatchBVH::hashEnd(IntVector index)
	{
		IntVector ret;
		for (int i = 0; i < 3; i ++)
		{
			if (index[i] < _low[i]) index[i] = _low[i];
			if (index[i] > _high[i]) index[i] = _high[i];
			ret[i] = ceil( _ele[i] * double(index[i] - _low[i]) );
			if (ret[i] > _highBound[i])
				ret[i] = _highBound[i];
			if (ret[i] < 0)
				ret[i] = 0;
		}
		return ret;
	}

	void PatchBVH::query(const IntVector& low, const IntVector& high, Level::selectType& patches, bool includeExtraCells)
	{
		_query(low, high, patches, includeExtraCells);
	}
	inline void PatchBVH::_query(const IntVector& low, const IntVector& high, Level::selectType& patches, bool includeExtraCells)
	{
//		BlockTimer(Query);
		if (high.x() <= low.x() || high.y() <= low.y() || high.z() <= low.z()
				|| high.x() <= _low.x() || high.y() <= _low.y() || high.z() <= _low.z()
				|| low.x() >= _high.x() || low.y() >= _high.y() || low.z() >= _high.z())
			return;
		
        if (includeExtraCells)
		{
			//BlockTimer(BranchTrue);
			IntVector lowExtra = hashBegin(low - _extraLen);
			IntVector highExtra = hashEnd(high + _extraLen);
			(*_patchMap).batch_get(lowExtra, highExtra, _bm, _true, patches);
		}
		else
		{
			//BlockTimer(BranchFalse);
			IntVector lowNormal = (hashBegin(low));
			IntVector highNormal = (hashEnd(high));
			(*_patchMap).batch_get(lowNormal, highNormal, _bm, _true, patches);
		}
		
		_true++;
		if (!_true)
		{
			memset(_bm, 0, sizeof(int) * _size);
			_true = 1;
		}
		return;
	}
}
