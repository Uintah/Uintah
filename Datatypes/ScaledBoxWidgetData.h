
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

#ifndef SCI_Datatypes_ScaledBoxWidgetData_h
#define SCI_Datatypes_ScaledBoxWidgetData_h 1

#include <Datatypes/Datatype.h>
#include <Classlib/LockingHandle.h>
#include <Geometry/Point.h>

class ScaledBoxWidgetData;
typedef LockingHandle<ScaledBoxWidgetData> ScaledBoxWidgetDataHandle;

class ScaledBoxWidgetData : public Datatype {
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

#endif
