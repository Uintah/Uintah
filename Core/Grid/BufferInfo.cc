
#include <Packages/Uintah/Core/Grid/BufferInfo.h>
#include <Core/Util/Assert.h>

using namespace Uintah;

BufferInfo::BufferInfo()
{
  have_datatype=false;
  free_datatype=false;
}

BufferInfo::~BufferInfo()
{
  if(free_datatype)
    MPI_Type_free(&datatype);
  for(int i=0;i<(int)datatypes.size();i++){
    if(free_datatypes[i])
      MPI_Type_free(&datatypes[i]);
  }
}

int
BufferInfo::count() const
{
  return datatypes.size();
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
}
