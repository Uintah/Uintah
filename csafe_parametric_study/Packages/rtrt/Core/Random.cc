
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

