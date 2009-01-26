/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#include <Packages/rtrt/Core/Random.h>
#include <Packages/rtrt/Core/MusilRNG.h>

using namespace rtrt;

HashTable<RandomTable::TableInfo, double*> RandomTable::t_random;
HashTable<RandomTable::TableInfo, int*> RandomTable::t_scramble;

int operator<(const RandomTable::TableInfo& t1,
	      const RandomTable::TableInfo& t2)
{
    if(t1.seed < t2.seed)return 1;
    if(t1.size < t2.size)return 1;
    return 0;
}

namespace rtrt {
  int operator==(const RandomTable::TableInfo& t1,
		 const RandomTable::TableInfo& t2)
  {
    return t1.seed==t2.seed && t1.size==t2.size;
  }
} // end namespace rtrt

RandomTable::RandomTable(int seed, int size)
    : tinfo(seed, size)
{

}

double* RandomTable::random()
{
    double* ptr;
    if(!t_random.lookup(tinfo, ptr)){
	ptr=new double[tinfo.size];
	// Must create table
	MusilRNG rng(tinfo.seed);
	for(int i=0;i<tinfo.size;i++){
	    ptr[i]=rng();
	}
	t_random.insert(tinfo, ptr);
    }
    return ptr;
}

int* RandomTable::scramble()
{
    int* ptr;
    if(!t_scramble.lookup(tinfo, ptr)){
	ptr=new int[tinfo.size];
	// Must create table
	for(int i=0;i<tinfo.size;i++)ptr[i]=i;
	MusilRNG rng(tinfo.seed);
	for(int i=0;i<tinfo.size;i++){
	    int r=(int)(rng()*tinfo.size);
	    int tmp=ptr[r];
	    ptr[r]=ptr[i];
	    ptr[i]=tmp;
	}
	t_scramble.insert(tinfo, ptr);
    }
    return ptr;
}

int RandomTable::TableInfo::hash() const
{
    return (seed^size);
}

