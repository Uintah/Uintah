

#ifndef sci_Devices_Tracker_h
#define sci_Devices_Tracker_h 1

#include <Comm/MessageBase.h>
#include <Devices/TrackerServer.h>
#include <Multitask/Task.h>
#include <Multitask/ITC.h>

struct TrackerMessage : public MessageBase {
    TrackerData data;
    void* clientdata;
    TrackerMessage(const TrackerData& data, void* clientdata);
    virtual ~TrackerMessage();
};

class Tracker {
    Tracker(const Tracker&);
    Mailbox<MessageBase*>* mailbox;
    void* clientdata;
    friend class TrackerThread;
public:
    Tracker(Mailbox<MessageBase*>* mailbox, void* data);
    ~Tracker();
};

class TrackerThread : public Task {
public:
    TrackerThread();
    virtual int body(int);
};

#endif
