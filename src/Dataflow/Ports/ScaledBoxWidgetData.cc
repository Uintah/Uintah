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

#include <CommonDatatypes/ScaledBoxWidgetData.h>
#include <Containers/String.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

namespace PSECommon {
namespace CommonDatatypes {

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

} // End namespace CommonDatatypes
} // End namespace PSECommon

//
// $Log$
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
