
/*
 *  SegFld.h:  The segmented field class
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#ifndef SCI_Packages_DaveW_Datatypes_SegFld_h
#define SCI_Packages_DaveW_Datatypes_SegFld_h 1

#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Containers/Array3.h>
#include <Core/Containers/String.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Core/Persistent/Pstreams.h>

namespace DaveW {
using namespace SCIRun;


class tripleInt {
public:
    inline tripleInt():x(-1),y(-1),z(-1) {};
    inline tripleInt(int x, int y, int z):x(x),y(y),z(z) {};
    inline tripleInt(const tripleInt &copy):x(copy.x),y(copy.y),z(copy.z) {};
    int x;
    int y;
    int z;
};

class SegFld : public ScalarFieldRGint {
public:
    Array1<int> thin;
    Array1<int> comps;
    Array1<Array1<tripleInt> *> compMembers;
public:
    SegFld();
    SegFld(const SegFld&);
    SegFld(ScalarFieldRGchar*);
    virtual ~SegFld();
    virtual ScalarField* clone();

    inline int get_type(int i) {return (i>>28);}
    inline int get_size(int i) {return i%(1<<28);}
    inline int get_index(int type, int size) {return ((type<<28)+size);}

    void audit();
    void printComponents();
    void compress();

    ScalarFieldRGchar* getTypeFld();
    ScalarFieldRG* getBitFld();
    void bldFromChar(ScalarFieldRGchar*);
    void bldFromCharOld(ScalarFieldRGchar*);
//    void setCompsFromGrid();
//    void setGridFromComps();
    void annexComponent(int old_comp, int new_comp);
    void killSmallComponents(int min);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};
typedef LockingHandle<SegFld> SegFldHandle;

} // End namespace DaveW

namespace SCIRun {
void Pio(Piostream&, DaveW::tripleInt&);
}
#endif
