
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

#ifndef SCI_SegFld_h
#define SCI_SegFld_h 1

#include <Classlib/Array1.h>
#include <Classlib/Pstreams.h>
#include <Classlib/String.h>
#include <Datatypes/ScalarFieldRGint.h>

class tripleInt {
public:
    inline tripleInt():x(-1),y(-1),z(-1) {};
    inline tripleInt(int x, int y, int z):x(x),y(y),z(z) {};
    inline tripleInt(const tripleInt &copy):x(copy.x),y(copy.y),z(copy.z) {};
    int x;
    int y;
    int z;
};

void Pio(Piostream&, tripleInt&);

class SegFld : public ScalarFieldRGint {
public:
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
//    void setCompsFromGrid();
//    void setGridFromComps();
    void annexComponent(int old_comp, int new_comp);
    void killSmallComponents(int min);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};
typedef LockingHandle<SegFld> SegFldHandle;

#endif
