
#ifndef Math_Random_h
#define Math_Random_h 1

#include <Packages/rtrt/Core/HashTable.h>

namespace rtrt {

class RandomTable {
public:
    struct TableInfo {
	int seed;
	int size;
        inline TableInfo() {
        }
	inline TableInfo(int ns, int nsz){
	    seed=ns; size=nsz;
		}
	friend int operator==(const RandomTable::TableInfo&,
			      const RandomTable::TableInfo&);
	int hash() const;
    };
private:
    TableInfo tinfo;

    static HashTable<RandomTable::TableInfo, double*> t_random;
    static HashTable<RandomTable::TableInfo, int*> t_scramble;
public:
    RandomTable(int, int);
    double* random();
    int* scramble();
};

} // end namespace rtrt

#endif
