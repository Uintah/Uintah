/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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

#ifndef SCI_FieldConverters_ScalarFieldRGBase_h
#define SCI_FieldConverters_ScalarFieldRGBase_h 1

#include <FieldConverters/share/share.h>
#include <Core/Datatypes/LockArray3.h>
#include <FieldConverters/Core/Datatypes/ScalarField.h>

namespace FieldConverters {

using namespace SCIRun;

typedef LockingHandle< LockArray3<Point> > Array3PointHandle;
typedef unsigned char uchar;

template <class T> class ScalarFieldRGT;
typedef class ScalarFieldRGT<double> ScalarFieldRGdouble;
typedef class ScalarFieldRGT<float> ScalarFieldRGfloat;
typedef class ScalarFieldRGT<int> ScalarFieldRGint;
typedef class ScalarFieldRGT<short> ScalarFieldRGshort;
typedef class ScalarFieldRGT<unsigned short> ScalarFieldRGushort;
typedef class ScalarFieldRGT<char> ScalarFieldRGchar;
typedef class ScalarFieldRGT<unsigned char> ScalarFieldRGuchar;

class FieldConvertersSHARE ScalarFieldRGBase : public ScalarField {
public:
    enum Representation {
	Double,
	Float,
	Int,
	Short,
	Ushort,
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
    const char *getType() const;
    ScalarFieldRGdouble* getRGDouble();
    ScalarFieldRGfloat* getRGFloat();
    ScalarFieldRGint* getRGInt();
    ScalarFieldRGshort* getRGShort();
    ScalarFieldRGushort* getRGUshort();
    ScalarFieldRGchar* getRGChar();
    ScalarFieldRGuchar* getRGUchar();

    Point get_point(int, int, int);
    bool locate(int *loc, const Point &p);
    void midLocate(const Point&, int&, int&, int&);
    void set_bounds(const Point &min, const Point &max);

    ScalarFieldRGBase(Representation rep, int x, int y, int z);
    ScalarFieldRGBase(const ScalarFieldRGBase&);

    virtual ~ScalarFieldRGBase();
    virtual void compute_bounds();
    virtual void get_boundary_lines(Array1<Point>& lines);

    virtual double get_value(int i, int j, int k)=0;

    virtual void io(Piostream&);
    static PersistentTypeID type_id;

    int get_voxel( const Point& p, Point& ivoxel );

    // this code is used for random distribution stuff...

    void cell_pos(int index, int& x, int& y, int& z); // so you can iterate

    virtual void compute_samples(int);  // for random distributions in fields
    virtual void distribute_samples();
};

} // end namespace FieldConverters


#endif
