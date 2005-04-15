
/*
 *  ScalarFieldRGCC.h:  Scalar Fields defined on a Regular grid
 *
 *  Written by:
 *   Steven G. Parker (& David Weinstein--Modified by Kurt Zimmerman)
 *   Department of Computer Science
 *   University of Utah
 *   March 1994 (& January 1996)
 *
 *  Copyright (C) 1994 SCI Group
 *
 *  WARNING: This file was automatically generated from:
 *           ScalarFieldRGCCtype.h (<- "type" should be in all caps
 *           but I don't want it replaced by the sed during
 *           the generation process.)
 */

#ifndef SCI_project_ScalarFieldRGCC_h
#define SCI_project_ScalarFieldRGCC_h 1

#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Containers/Array3.h>

namespace SCICore {
namespace Datatypes {

using SCICore::Containers::Array3;

class SCICORESHARE ScalarFieldRGCC : public ScalarFieldRG {
public:
    ScalarFieldRGCC();
    ScalarFieldRGCC(const ScalarFieldRGCC&);
    virtual ~ScalarFieldRGCC();
    virtual ScalarField* clone();

    virtual Vector gradient(const Point&);
    virtual int interpolate(const Point&, double&, double epsilon1=1.e-6, double epsilon2=1.e-6);
    virtual int interpolate(const Point&, double&, int&, double epsilon1=1.e-6, double epsilon2=1.e-6, int exhaustive=0);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    
    virtual Vector gradient(int x, int y, int z);

    // this has to be called before augmented stuff (base class)
    virtual void fill_gradmags();
};

} // End namespace Datatypes
} // End namespace SCICore


#endif
