
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
#include <Classlib/Array3.h>

class ScalarFieldRGdouble;
class ScalarFieldRGfloat;
class ScalarFieldRGint;
class ScalarFieldRGshort;
class ScalarFieldRGchar;

class ScalarFieldRGBase : public ScalarField {
public:
    enum Representation {
	Double,
	Float,
	Int,
	Short,
	Char,
	Void
    };
    int nx;
    int ny;
    int nz;

private:
    Representation rep;

public:
    clString getType() const;
    ScalarFieldRGdouble* getRGDouble();
    ScalarFieldRGfloat* getRGFloat();
    ScalarFieldRGint* getRGInt();
    ScalarFieldRGshort* getRGShort();
    ScalarFieldRGchar* getRGChar();

    Point get_point(int, int, int);
    void locate(const Point&, int&, int&, int&);
    void midLocate(const Point&, int&, int&, int&);
    void set_bounds(const Point &min, const Point &max);
    ScalarFieldRGBase();
    ScalarFieldRGBase(clString);
    ScalarFieldRGBase(const ScalarFieldRGBase&);
    virtual ~ScalarFieldRGBase();
    virtual void compute_bounds();

    virtual void io(Piostream&);
    static PersistentTypeID type_id;

    int get_voxel( const Point& p, Point& ivoxel );
};

#endif
