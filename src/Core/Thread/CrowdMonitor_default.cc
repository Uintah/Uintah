
/*
 *  CrowdMonitor: Multiple reader/single writer locks
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Core/Thread/CrowdMonitor.h>

namespace SCIRun {

struct CrowdMonitor_private {
  ConditionVariable write_waiters;
  ConditionVariable read_waiters;
  Mutex lock;
  int num_readers_waiting;
  int num_writers_waiting;
  int num_readers;
  int num_writers;
  CrowdMonitor_private();
  ~CrowdMonitor_private();
};

CrowdMonitor_private::CrowdMonitor_private()
  : write_waiters("CrowdMonitor write condition"),
    read_waiters("CrowdMonitor read condition"),
    lock("CrowdMonitor lock")
{
  num_readers_waiting=0;
  num_writers_waiting=0;
  num_readers=0;
  num_writers=0;
}

CrowdMonitor_private::~CrowdMonitor_private()
{
}

CrowdMonitor::CrowdMonitor(const char* name)
  : d_name(name)
{
  d_priv=new CrowdMonitor_private;
}

CrowdMonitor::~CrowdMonitor()
{
  delete d_priv;
}

void
CrowdMonitor::readLock()
{
  int oldstate=Thread::couldBlock(d_name);
  d_priv->lock.lock();
  while(d_priv->num_writers > 0){
    d_priv->num_readers_waiting++;
    int s=Thread::couldBlock(d_name);
    d_priv->read_waiters.wait(d_priv->lock);
    Thread::couldBlockDone(s);
    d_priv->num_readers_waiting--;
  }
  d_priv->num_readers++;
  d_priv->lock.unlock();
  Thread::couldBlockDone(oldstate);
}

void
CrowdMonitor::readUnlock()
{
  d_priv->lock.lock();
  d_priv->num_readers--;
  if(d_priv->num_readers == 0 && d_priv->num_writers_waiting > 0)
    d_priv->write_waiters.conditionSignal();
  d_priv->lock.unlock();
}

void
CrowdMonitor::writeLock()
{
  int oldstate=Thread::couldBlock(d_name);
  d_priv->lock.lock();
  while(d_priv->num_writers || d_priv->num_readers){
    // Have to wait...
    d_priv->num_writers_waiting++;
    int s=Thread::couldBlock(d_name);
    d_priv->write_waiters.wait(d_priv->lock);
    Thread::couldBlockDone(s);
    d_priv->num_writers_waiting--;
  }
  d_priv->num_writers++;
  d_priv->lock.unlock();
  Thread::couldBlockDone(oldstate);
} // End namespace SCIRun

void
CrowdMonitor::writeUnlock()
{
  d_priv->lock.lock();
  d_priv->num_writers--;
  if(d_priv->num_writers_waiting)
    d_priv->write_waiters.conditionSignal(); // Wake one of them up...
  else if(d_priv->num_readers_waiting)
    d_priv->read_waiters.conditionBroadcast(); // Wake all of them up...
  d_priv->lock.unlock();
}

} // End namespace SCIRun
