/*
 *  Datatype.h: The Datatype Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Datatype_h
#define SCI_project_Datatype_h 1

#include <Persistent.h>
#include <Multitask/ITC.h>

class Datatype : public Persistent {
public:
    int ref_cnt;
    Mutex lock;
    Datatype();
    virtual ~Datatype();
};

#endif /* SCI_project_Datatype_h */
