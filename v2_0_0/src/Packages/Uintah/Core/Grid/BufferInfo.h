
#ifndef UINTAH_HOMEBREW_BufferInfo_H
#define UINTAH_HOMEBREW_BufferInfo_H

#include <sci_config.h> // For MPIPP_H on SGI
#include <mpi.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  using namespace std;
  class RefCounted;
  class ProcessorGroup;

  class AfterCommunicationHandler {
  public:
    virtual ~AfterCommunicationHandler() {}
    virtual void finishedCommunication(const ProcessorGroup*) = 0;
  };

  class Sendlist : public AfterCommunicationHandler {
  public:
    Sendlist(Sendlist* next, RefCounted* obj)
      : next(next), obj(obj)
    {}
    virtual ~Sendlist();
    Sendlist* next;
    RefCounted* obj;

    // Sendlist is to be an AfterCommuncationHandler object for the
    // MPI_CommunicationRecord template in MPIScheduler.cc.  The only task
    // it needs to do to handle finished send requests is simply get deleted.
    virtual void finishedCommunication(const ProcessorGroup*) {}

  };

  class  BufferInfo {
  public:
    BufferInfo();
    virtual ~BufferInfo();
    int count() const;
    void get_type(void*&, int&, MPI_Datatype&);

    void add(void* startbuf, int count, MPI_Datatype datatype,
	     bool free_datatype);

    void addSendlist(RefCounted*);
    Sendlist* takeSendlist();
  private:
    BufferInfo(const BufferInfo&);
    BufferInfo& operator=(const BufferInfo&);

  protected:
    Sendlist* sendlist;
    vector<void*> startbufs;
    vector<int> counts;
    vector<MPI_Datatype> datatypes;
    vector<bool> free_datatypes;

    void* buf;
    int cnt;
    MPI_Datatype datatype;

    bool free_datatype;
    bool have_datatype;
  };
}

#endif

