
/*
 *  WidgetReal.cc:  The first module!
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <WidgetReal/WidgetReal.h>
#include <iostream.h>
extern "C" int abs(int);

WidgetReal::WidgetReal()
: Module("WidgetReal")
{
}

WidgetReal::WidgetReal(const WidgetReal& copy, int deep)
: Module(copy, deep)
{
}

WidgetReal::~WidgetReal()
{
}

Module* make_WidgetReal()
{
    return new WidgetReal;
}

Module* WidgetReal::clone(int deep)
{
    return new WidgetReal(*this, deep);
}

void WidgetReal::execute()
{
    int nloops=10000000;
    int x;
    int ii=0;
    for(int i=0;i<nloops;i++){
	update_progress(i, nloops);
	ii=abs(i+ii);
    }
}
