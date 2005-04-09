
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
#include <Port.h>
#include <iostream.h>
extern "C" int abs(int);

static Module* make_WidgetReal()
{
    return new WidgetReal;
}

static RegisterModule db1("Widgets", "WidgetReal", make_WidgetReal);

WidgetReal::WidgetReal()
: UserModule("WidgetReal")
{
    add_iport(0, "Input", 0);
    add_oport(0, "Output", 0);
}

WidgetReal::WidgetReal(const WidgetReal& copy, int deep)
: UserModule(copy, deep)
{
}

WidgetReal::~WidgetReal()
{
}

Module* WidgetReal::clone(int deep)
{
    return new WidgetReal(*this, deep);
}

void WidgetReal::execute()
{
    int nloops=10000000;
    int ii=0;
    for(int i=0;i<nloops;i++){
	update_progress(i, nloops);
	ii=abs(i+ii);
    }
}
