
#ifndef SCI_THREAD_THREADALERT_H
#define SCI_THREAD_THREADALERT_H 1

/**************************************
 
CLASS
   ThreadAlert
   
KEYWORDS
   ThreadAlert
   
DESCRIPTION
   
   When a thread is alerted (see <b>Thread</b>), it throws
   a <b>ThreadAlert</b> exception.  This event is typically
   triggered asynchonously by another thread, and can therefore
   happen at any time.  Threads should catch this exception,
   perform any cleanup operations and rethrow the exception.
 
PATTERNS


WARNING
   
****************************************/

class ThreadAlert {
    friend class Thread;
protected:
    ThreadAlert();
public:
    virtual ~ThreadAlert();
};

#endif
