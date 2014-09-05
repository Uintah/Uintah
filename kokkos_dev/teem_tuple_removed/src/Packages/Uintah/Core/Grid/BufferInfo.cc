
#include <Packages/Uintah/Core/Grid/BufferInfo.h>
#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>
#include <Core/Util/Assert.h>
#include <Core/Thread/Mutex.h>

using namespace Uintah;

SCIRun::Mutex MPITypeLock( "MPITypeLock" );

BufferInfo::BufferInfo()
{
  have_datatype=false;
  free_datatype=false;
  sendlist=0;
}

BufferInfo::~BufferInfo()
{
 MPITypeLock.lock();
   
  if(free_datatype)
    MPI_Type_free(&datatype);
  for(int i=0;i<(int)datatypes.size();i++){
    if(free_datatypes[i])
      MPI_Type_free(&datatypes[i]);
  }

 MPITypeLock.unlock();
 
  if(sendlist)
    delete sendlist;
}

int
BufferInfo::count() const
{
  return (int)datatypes.size();
}

void
BufferInfo::add(void* startbuf, int count, MPI_Datatype datatype,
		bool free_datatype)
{
  ASSERT(!have_datatype);
  startbufs.push_back(startbuf);
  counts.push_back(count);
  datatypes.push_back(datatype);
  free_datatypes.push_back(free_datatype);
} 

void
BufferInfo::get_type(void*& out_buf, int& out_count,
		     MPI_Datatype& out_datatype)
{
 MPITypeLock.lock();
  ASSERT(count() > 0);
  if(!have_datatype){
    if(count() == 1){
      buf=startbufs[0];
      cnt=counts[0];
      datatype=datatypes[0];
      free_datatype=false; // Will get freed with array
    } else {
      vector<MPI_Aint> indices(count());
      for(int i=0;i<(int)startbufs.size();i++)
	indices[i]=(MPI_Aint)startbufs[i];
      MPI_Type_struct(count(), &counts[0], &indices[0], &datatypes[0],
		      &datatype);
      buf=0;
      cnt=1;
      free_datatype=true;
    }
    have_datatype=true;
  }
  out_buf=buf;
  out_count=cnt;
  out_datatype=datatype;
 MPITypeLock.unlock(); 
}

Sendlist::~Sendlist()
{
  if(obj && obj->removeReference())
    delete obj;

  // A little more complicated than normal, so that this doesn't need
  // to be recursive...
  Sendlist* p = next;
  while(p){
    if(p->obj->removeReference())
      delete p->obj;
    Sendlist* n = p->next;
    p->next=0;  // So that DTOR won't recurse...
    p->obj=0;
    delete p;
    p=n;
  }
}

void BufferInfo::addSendlist(RefCounted* obj)
{
  obj->addReference();
  sendlist=new Sendlist(sendlist, obj);
}

Sendlist* BufferInfo::takeSendlist()
{
  Sendlist* rtn = sendlist;
  sendlist = 0; // They are now responsible for freeing...
  return rtn;
}
