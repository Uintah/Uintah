
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

#include <Core/share/share.h>

#include <Core/Persistent/Persistent.h>
#include <Core/Thread/Mutex.h>

namespace SCIRun {


class SCICORESHARE Datatype : public Persistent {
public:
    int ref_cnt;
    Mutex lock;
    int generation;
    Datatype();
    Datatype(const Datatype&);
    Datatype& operator=(const Datatype&);
    virtual ~Datatype();
};

} // End namespace SCIRun


#endif /* SCI_project_Datatype_h */
