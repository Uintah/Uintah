
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

#include <Datatypes/ScalarField.h>
#include <Datatypes/LockArray3.h>

typedef LockingHandle< LockArray3<Point> > Array3PointHandle;

class ScalarFieldRGdouble;
class ScalarFieldRGfloat;
class ScalarFieldRGint;
class ScalarFieldRGshort;
class ScalarFieldRGchar;
class ScalarFieldRGuchar;

class ScalarFieldRGBase : public ScalarField {
public:
    enum Representation {
	Double,
	Float,
	Int,
	Short,
	Char,
	Void,
	Uchar
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

#endif
