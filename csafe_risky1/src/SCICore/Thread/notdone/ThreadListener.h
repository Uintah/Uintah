
#ifndef SCI_THREAD_THREADLISTENER_H
#define SCI_THREAD_THREADLISTENER_H 1

/**************************************
 
CLASS
   ThreadListener
   
KEYWORDS
   ThreadListener
   
DESCRIPTION
   An object may implement this interface and register with
   <b>Thread::</b><i>addListener</i> in order to receive events
   regarding thread creation, deletion, and other thread activity.
   These events are received from the <tt>eventQueue</tt> mailbox.
   This interface is still under design and not yet completely
   specified.
 
PATTERNS


WARNING
   
****************************************/

class Thread;
struct ThreadEvent;

class ThreadListener {
    friend class Thread;
    void sendEvent(Thread*, ThreadEvent);
protected:
    ThreadListener();

    virtual ~ThreadListener();
    
    //////////
    // Event tokens are sent by the thread manager to this mailbox. 
    // The listener should recieve these tokens and process them in a
    // timely manner.  Mailbox eventQueue; If the thread library tries
    // to send an event to the eventQueue, and it is full, then the event
    // must be dropped in order to avoid deadlock.  If this occurs, the
    // <i>overflow</i> flag will be set.
    bool overflow;
};

#endif

