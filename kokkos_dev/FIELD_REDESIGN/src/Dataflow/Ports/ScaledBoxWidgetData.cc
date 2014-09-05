//static char *id="@(#) $Id$";

/*
 *  ScaledBoxWidgetData.cc: gotta get the data there somehow...
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <PSECore/Datatypes/ScaledBoxWidgetData.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Malloc/Allocator.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::PersistentSpace;

static Persistent* maker()
{
    return scinew ScaledBoxWidgetData();
}

PersistentTypeID ScaledBoxWidgetData::type_id("Data", "Datatype", maker);

ScaledBoxWidgetData::ScaledBoxWidgetData()
: Center(Point(.5,.5,.5)), R(Point(1,.5,.5)), D(Point(.5,0,.5)), I(Point(.5,.5,0)), RatioR(.5), RatioD(.5), RatioI(.5) {
}

ScaledBoxWidgetData::ScaledBoxWidgetData(const Point &Center, const Point &R,
					 const Point &D, const Point &I, 
					 double RatioR, double RatioD, 
					 double RatioI)
: Center(Center), R(R), D(D), I(I), RatioR(RatioR), RatioD(RatioD), RatioI(RatioI)
{
}

ScaledBoxWidgetData::ScaledBoxWidgetData(const ScaledBoxWidgetData& c)
: Center(c.Center), R(c.R), D(c.D), I(c.I), RatioR(c.RatioR), RatioD(c.RatioD), RatioI(c.RatioI)
{
}

int ScaledBoxWidgetData::operator==(const ScaledBoxWidgetData& s) {
    return (Center==s.Center && R==s.R && I==s.I && D==s.D && RatioR==s.RatioR && RatioD==s.RatioD && RatioI==s.RatioI);
}

int ScaledBoxWidgetData::operator!=(const ScaledBoxWidgetData& s) {
    return (Center!=s.Center || R!=s.R || I!=s.I || D!=s.D || RatioR!=s.RatioR || RatioD!=s.RatioD || RatioI!=s.RatioI);
}

ScaledBoxWidgetData& ScaledBoxWidgetData::operator=(const ScaledBoxWidgetData& s) {
    Center=s.Center;
    R=s.R;
    D=s.D;
    I=s.I;
    RatioR=s.RatioR;
    RatioD=s.RatioD;
    RatioI=s.RatioI;
    return *this;
}

ScaledBoxWidgetData::~ScaledBoxWidgetData()
{
}

ScaledBoxWidgetData* ScaledBoxWidgetData::clone() const
{
    return scinew ScaledBoxWidgetData(*this);
}

#define SCALEDBOXWIDGETDATA_VERSION 1

void ScaledBoxWidgetData::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Geometry::Pio;

    stream.begin_class("Boolean", SCALEDBOXWIDGETDATA_VERSION);
    Pio(stream,Center);
    Pio(stream,R);
    Pio(stream,D);
    Pio(stream,I);
    stream.io(RatioR);
    stream.io(RatioD);
    stream.io(RatioI);
    stream.end_class();
}

} // End namespace Datatypes
} // End namespace PSECore

//
// $Log$
// Revision 1.4  1999/09/08 02:26:42  sparker
// Various #include cleanups
//
// Revision 1.3  1999/08/25 03:48:22  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:11  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:49  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:19  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//
