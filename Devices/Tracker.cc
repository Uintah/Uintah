
#include <Devices/Tracker.h>
#include <Classlib/Array1.h>

static Mutex globlock;
static Array1<Tracker*> clients;
static Semaphore client_sema(0);

TrackerThread::TrackerThread()
: Task("Tracker Server")
{
}

int TrackerThread::body(int)
{
    // This will initialize the flying mouse...
    TrackerData tmp;
    if(!GetTrackerData(tmp))
	return 0;
    while(1){
	// Wait for some clients...
//	client_sema.down();

	TrackerData data;
	GetTrackerData(data); // Block for input from the tracker...
	globlock.lock();
	int n=clients.size();
	for(int i=0;i<n;i++){
	    void* cd=clients[i]->clientdata;
	    clients[i]->mailbox->send(new TrackerMessage(data, cd));
	}
	globlock.unlock();
//	client_sema.up();
    }
}

Tracker::Tracker(Mailbox<MessageBase*>* mailbox, void* clientdata)
: mailbox(mailbox), clientdata(clientdata)
{
    globlock.lock();
    clients.add(this);
    globlock.unlock();
    client_sema.up();
}

Tracker::~Tracker()
{
    globlock.unlock();
    for(int i=0;i<clients.size();i++)
	if(clients[i]==this)
	    break;
    ASSERT(i<clients.size());
    clients.remove(i);
    client_sema.down();
}

TrackerMessage::TrackerMessage(const TrackerData& data, void* clientdata)
: MessageBase(MessageTypes::TrackerMoved), data(data), clientdata(clientdata)
{
}

TrackerMessage::~TrackerMessage()
{
}
