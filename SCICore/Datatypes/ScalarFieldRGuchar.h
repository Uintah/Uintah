
/*
 *  ScalarFieldRGuchar.h: uchar Scalar Fields defined on a Regular grid
 *
 *  Written by:
 *   Steven G. Parker (& David Weinstein)
 *   Department of Computer Science
 *   University of Utah
 *   March 1994 (& January 1996)
 *
 *  Copyright (C) 1994 SCI Group
 *
 *  WARNING: This file was automatically generated from:
 *           ScalarFieldRGtype.h (<- "type" should be in all caps
 *           but I don't want it replaced by the sed during
 *           the generation process.)
 */

#ifndef SCI_project_ScalarFieldRGuchar_h
#define SCI_project_ScalarFieldRGuchar_h 1

#include <CoreDatatypes/ScalarFieldRGBase.h>
#include <Containers/Array3.h>

namespace SCICore {
namespace CoreDatatypes {

using SCICore::Containers::Array3;

class ScalarFieldRGuchar : public ScalarFieldRGBase {
public:
    Array3<unsigned char> grid;

    void resize(int, int, int);
    ScalarFieldRGuchar();
    ScalarFieldRGuchar(const ScalarFieldRGuchar&);
    virtual ~ScalarFieldRGuchar();
    virtual ScalarField* clone();

    virtual void compute_minmax();
    virtual Vector gradient(const Point&);
    virtual int interpolate(const Point&, double&, double epsilon1=1.e-6, double epsilon2=1.e-6);
    virtual int interpolate(const Point&, double&, int&, double epsilon1=1.e-6, double epsilon2=1.e-6, int exhaustive=0);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    
    double get_value( const Point& ivoxel );
    double get_value( int x, int y, int z );

    Vector get_normal( const Point& ivoxel );
    Vector get_normal( int x, int y, int z );

    Vector gradient(int x, int y, int z);

    // this has to be called before augmented stuff (base class)
    
    virtual void fill_gradmags();
};

inline
double
ScalarFieldRGuchar::get_value( const Point& ivoxel )
{
  return grid( ivoxel.x(), ivoxel.y(), ivoxel.z() );
}

inline
double
ScalarFieldRGuchar::get_value( int x, int y, int z )
{
  return grid( x, y, z );
}

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:27  mcq
// Initial commit
//
// Revision 1.4  1999/07/07 21:10:43  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/05/06 19:55:50  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:43  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:12  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//

#endif
