//static char *id="@(#) $Id$";

/*
 *  ITC.cc: Architecture independant parts of InterTask Communication library
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Multitask/ITC.h>
#include <SCICore/Multitask/Task.h>
#include <iostream.h>

namespace SCICore {
namespace Multitask {

struct CrowdMonitor_private {
    ConditionVariable write_waiters;
    ConditionVariable read_waiters;
    int nreaders_waiting;
    int nwriters_waiting;
    int nreaders;
    int nwriters;
    Mutex lock;
    int ndeep;
    Task* owner;
};

CrowdMonitor::CrowdMonitor()
{
    priv=scinew CrowdMonitor_private;
    priv->nreaders_waiting=0;
    priv->nwriters_waiting=0;
    priv->nreaders=0;
    priv->nwriters=0;
    priv->ndeep=0;
    priv->owner=0;
}

CrowdMonitor::~CrowdMonitor()
{
    delete priv;
}

void CrowdMonitor::read_lock()
{
    priv->lock.lock();
    while(priv->nwriters > 0){
	priv->nreaders_waiting++;
	priv->read_waiters.wait(priv->lock);
	priv->nreaders_waiting--;
    }
    priv->nreaders++;
    priv->lock.unlock();
}

void CrowdMonitor::read_unlock()
{
    priv->lock.lock();
    priv->nreaders--;
    if(priv->nreaders == 0 && priv->nwriters_waiting > 0)
	priv->write_waiters.cond_signal();
    priv->lock.unlock();
}

void CrowdMonitor::write_lock()
{
    Task* self=Task::self();
    if(priv->owner == self){
	priv->ndeep++;
	return;
    }
    priv->lock.lock();
    while(priv->nwriters || priv->nreaders){
	// Have to wait...
	priv->nwriters_waiting++;
	priv->write_waiters.wait(priv->lock);
	priv->nwriters_waiting--;
    }
    priv->nwriters++;
    priv->ndeep++;
    priv->owner=self;
    priv->lock.unlock();
}

void CrowdMonitor::write_unlock()
{
    Task* self=Task::self();
    if(priv->owner == self && --priv->ndeep > 0)
	return;
    priv->lock.lock();
    priv->nwriters--;
    if(priv->nwriters_waiting)
	priv->write_waiters.cond_signal(); // Wake one of them up...
    else if(priv->nreaders_waiting)
	priv->read_waiters.broadcast(); // Wait all of them up...
    priv->owner=0;
    priv->lock.unlock();
}

} // End namespace Multitask
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:37  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:06  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:26  dav
// Import sources
//
//

