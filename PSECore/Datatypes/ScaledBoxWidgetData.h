
/*
 *  ScaledBoxWidgetData.h: what a hack...
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#ifndef SCI_CommonDatatypes_ScaledBoxWidgetData_h
#define SCI_CommonDatatypes_ScaledBoxWidgetData_h 1

#include <SCICore/share/share.h>
#include <SCICore/CoreDatatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Geometry/Point.h>

namespace PSECore {
namespace CommonDatatypes {

using SCICore::Containers::LockingHandle;
using SCICore::Geometry::Point;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
using namespace SCICore::CoreDatatypes;

class ScaledBoxWidgetData;
typedef LockingHandle<ScaledBoxWidgetData> ScaledBoxWidgetDataHandle;

class SCICORESHARE ScaledBoxWidgetData : public Datatype {
public:
    Point Center;
    Point R;
    Point D;
    Point I;
    double RatioR;
    double RatioD;
    double RatioI;
    ScaledBoxWidgetData();
    ScaledBoxWidgetData(const Point &, const Point &, const Point &, const Point &, double, double, double);
    virtual ~ScaledBoxWidgetData();
    ScaledBoxWidgetData(const ScaledBoxWidgetData&);
    virtual ScaledBoxWidgetData* clone() const;
    int operator==(const ScaledBoxWidgetData&);
    int operator!=(const ScaledBoxWidgetData&);
    ScaledBoxWidgetData& operator=(const ScaledBoxWidgetData&);
    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace CommonDatatypes
} // End namespace PSECore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:11  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:49  mcq
// Initial commit
//
// Revision 1.4  1999/05/06 20:17:02  dav
// added back PSECore .h files
//
// Revision 1.2  1999/04/27 23:18:36  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:49  dav
// Import sources
//
//

#endif
