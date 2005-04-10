
#ifndef SCI_THREAD_THREADEVENT_H
#define SCI_THREAD_THREADEVENT_H 1

/*
 * Simple struct for specifying thread events that can be sent from
 * the thread scheduler to the <b>ThreadListener</b>s.
 */


#include "Thread.h"
#include <string>

struct ThreadEvent {
    //////////
    //The type of the event.
    enum EventType {
	THREAD_START,
	THREAD_DONE
    };

    //////////
    //The name of the thread thread for THREAD_START and THREAD__DONE events.
    const std::string threadName;

    //////////
    //When the event happened.
    SysClock timestamp;
};

#endif
