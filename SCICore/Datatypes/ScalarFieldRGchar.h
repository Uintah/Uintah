
/*
 *  ScalarFieldRGchar.h: char Scalar Fields defined on a Regular grid
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

#ifndef SCI_project_ScalarFieldRGchar_h
#define SCI_project_ScalarFieldRGchar_h 1

#include <SCICore/Datatypes/ScalarFieldRGBase.h>
#include <SCICore/Containers/Array3.h>

namespace SCICore {
namespace Datatypes {

using SCICore::Containers::Array3;

class SCICORESHARE ScalarFieldRGchar : public ScalarFieldRGBase {
public:
    Array3<char> grid;

    void resize(int, int, int);
    ScalarFieldRGchar();
    ScalarFieldRGchar(const ScalarFieldRGchar&);
    virtual ~ScalarFieldRGchar();
    virtual ScalarField* clone();

    virtual void compute_minmax();
    virtual Vector gradient(const Point&);
    virtual int interpolate(const Point&, double&, double epsilon1=1.e-6, double epsilon2=1.e-6);
    virtual int interpolate(const Point&, double&, int&, double epsilon1=1.e-6, double epsilon2=1.e-6, int exhaustive=0);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    
    double get_value( const Point& ivoxel );
    virtual double get_value( int x, int y, int z );

    Vector get_normal( const Point& ivoxel );
    Vector get_normal( int x, int y, int z );

    Vector gradient(int x, int y, int z);

    // this has to be called before augmented stuff (base class)
    
    virtual void fill_gradmags();
};

inline
double
ScalarFieldRGchar::get_value( const Point& ivoxel )
{
  return grid( ivoxel.x(), ivoxel.y(), ivoxel.z() );
}

inline
double
ScalarFieldRGchar::get_value( int x, int y, int z )
{
  return grid( x, y, z );
}

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/09/05 23:14:16  dmw
// added virtual accessor method
//
// Revision 1.3  1999/08/25 03:48:38  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:51  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:26  mcq
// Initial commit
//
// Revision 1.4  1999/07/07 21:10:40  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/05/06 19:55:50  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:43  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:12  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//

#endif
