
#ifndef SCI_THREAD_THREADERROR_H
#define SCI_THREAD_THREADERROR_H

/**************************************
 
CLASS
   ThreadError
   
KEYWORDS
   ThreadError
   
DESCRIPTION
   
   Interface to provide alternate handler for handling thread errors
 
PATTERNS


WARNING
   
****************************************/

class Thread;

class ThreadError {
protected:
    friend class Thread;
    //////////
    // Called when a thead might abort.  See <b>Thread::niceAbort</b>
    virtual char threadAbort(Thread* thread)=0;
public:
    //////////
    // Destroy the ThreadError
    virtual ~ThreadError();
};

#endif
