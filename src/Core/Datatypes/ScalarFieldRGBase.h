
/*
 *  ScalarFieldRGBase.h: Scalar Fields defined on a Regular grid base class
 *
 *  Written by:
 *   Steven G. Parker (& David Weinstein)
 *   Department of Computer Science
 *   University of Utah
 *   March 1994 (& January 1996)
 *
 *  Copyright (C) 1994, 1996 SCI Group
 */

#ifndef SCI_project_ScalarFieldRGBase_h
#define SCI_project_ScalarFieldRGBase_h 1

#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/LockArray3.h>

namespace SCICore {
namespace Datatypes {

using SCICore::Containers::LockingHandle;

typedef LockingHandle< LockArray3<Point> > Array3PointHandle;
typedef unsigned char uchar;

class ScalarFieldRGdouble;
class ScalarFieldRGfloat;
class ScalarFieldRGint;
class ScalarFieldRGshort;
class ScalarFieldRGchar;
class ScalarFieldRGuchar;

class SCICORESHARE ScalarFieldRGBase : public ScalarField {
public:
    enum Representation {
	Double,
	Float,
	Int,
	Short,
	Char,
	Uchar,
	Void
    };
    int nx;
    int ny;
    int nz;

    int is_augmented; // 0 if regular, 1 if "rectalinear"

    Array3PointHandle aug_data; // shared (potentialy)

    ScalarFieldRGBase *next; // so you can link them...

private:
    Representation rep;

public:
    clString getType() const;
    ScalarFieldRGdouble* getRGDouble();
    ScalarFieldRGfloat* getRGFloat();
    ScalarFieldRGint* getRGInt();
    ScalarFieldRGshort* getRGShort();
    ScalarFieldRGchar* getRGChar();
    ScalarFieldRGuchar* getRGUchar();

    Point get_point(int, int, int);
    void locate(const Point&, int&, int&, int&);
    void midLocate(const Point&, int&, int&, int&);
    void set_bounds(const Point &min, const Point &max);
    ScalarFieldRGBase();
    ScalarFieldRGBase(clString);
    ScalarFieldRGBase(const ScalarFieldRGBase&);
    virtual ~ScalarFieldRGBase();
    virtual void compute_bounds();
    virtual void get_boundary_lines(Array1<Point>& lines);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;

    int get_voxel( const Point& p, Point& ivoxel );

    // this code is used for random distribution stuff...

    void cell_pos(int index, int& x, int& y, int& z); // so you can iterate

    virtual void compute_samples(int);  // for random distributions in fields
    virtual void distribute_samples();
};

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/25 03:48:37  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:50  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:25  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:50  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:42  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:12  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:47  dav
// Import sources
//
//

#endif
