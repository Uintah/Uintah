
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

#ifndef SCI_DaveW_Datatypes_SegFld_h
#define SCI_DaveW_Datatypes_SegFld_h 1

#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Array2.h>
#include <SCICore/Containers/Array3.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Containers/Queue.h>
#include <SCICore/Datatypes/ScalarFieldRGchar.h>
#include <SCICore/Datatypes/ScalarFieldRGint.h>
#include <SCICore/Persistent/Pstreams.h>

namespace DaveW {
namespace Datatypes {

using SCICore::Containers::LockingHandle;
using SCICore::Containers::Array1;
using SCICore::Containers::clString;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

using namespace SCICore::Datatypes;

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

void Pio( Piostream &, tripleInt & );

} // End namespace Datatypes
} // End namespace DaveW

//
// $Log$
// Revision 1.3  2000/03/04 00:16:32  dmw
// update some DaveW stuff
//
// Revision 1.2  1999/08/25 03:47:34  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.1  1999/08/23 02:53:00  dmw
// Dave's Datatypes
//
// Revision 1.1  1999/05/03 04:52:05  dmw
// Added and updated DaveW Datatypes/Modules
//
//

#endif
