
#ifndef UINTAH_HOMEBREW_BufferInfo_H
#define UINTAH_HOMEBREW_BufferInfo_H

#include <mpi.h>
#include <vector>

namespace Uintah {
  using namespace std;
  class RefCounted;

  struct Sendlist {
    Sendlist* next;
    RefCounted* obj;
    Sendlist(Sendlist* next, RefCounted* obj)
      : next(next), obj(obj)
    {}
    ~Sendlist();
  };

  class  BufferInfo {
  public:
    BufferInfo();
    ~BufferInfo();
    int count() const;
    void get_type(void*&, int&, MPI_Datatype&);

    void add(void* startbuf, int count, MPI_Datatype datatype,
	     bool free_datatype);

    void addSendlist(RefCounted*);
    Sendlist* takeSendlist();
  private:
    BufferInfo(const BufferInfo&);
    BufferInfo& operator=(const BufferInfo&);

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
