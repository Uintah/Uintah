
#include <Packages/Uintah/Core/Parallel/PackBufferInfo.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Util/RefCounted.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/Assert.h>

using namespace Uintah;

#include <iostream>
#include <zlib.h>
#include <string.h>

//this should probably be made dynamic
const unsigned long COMPRESS_BUF_SIZE=10000000;
char compress_buf[COMPRESS_BUF_SIZE];

int PackBufferInfo::compression_level=5;
unsigned long PackBufferInfo::compression_threshold=50;
PackBufferInfo::PackBufferInfo()
  : BufferInfo()
{
  packedBuffer=0;
}

PackBufferInfo::~PackBufferInfo()
{
  if (packedBuffer && packedBuffer->removeReference())
    delete packedBuffer;
}

void
PackBufferInfo::get_type(void*& out_buf, int& out_count,
			 MPI_Datatype& out_datatype, MPI_Comm comm)
{
  MALLOC_TRACE_TAG_SCOPE("PackBufferInfo::get_type");
  ASSERT(count() > 0);
  if(!have_datatype){
    int packed_size;
    int total_packed_size=0;
    for (int i = 0; i < (int)startbufs.size(); i++) {
      if(counts[i]>0)
      {
        MPI_Pack_size(counts[i], datatypes[i], comm, &packed_size);
        total_packed_size += packed_size;
      }
    }
   
    if(total_packed_size>0)
    {
      packedBuffer = scinew PackedBuffer(total_packed_size);
      packedBuffer->addReference();
    }
    
    datatype = MPI_PACKED;
    cnt=total_packed_size;
    buf = packedBuffer->getBuffer();
    have_datatype=true;
  }
  out_buf=buf;
  out_count=cnt;
  out_datatype=datatype;
}

void PackBufferInfo::get_type(void*&, int&, MPI_Datatype&)
{
  // Should use other overload for a PackBufferInfo
  SCI_THROW(SCIRun::InternalError("get_type(void*&, int&, MPI_Datatype&) should not be called on PackBufferInfo objects", __FILE__, __LINE__));
}


void
PackBufferInfo::pack(MPI_Comm comm, int& out_count)
{
  MALLOC_TRACE_TAG_SCOPE("PackBufferInfo::pack");
  ASSERT(have_datatype);

  int position = 0;
  int bufsize = packedBuffer->getBufSize();
  //for each buffer
  for (int i = 0; i < (int)startbufs.size(); i++) {
    //pack into a contigious buffer
    if(counts[i]>0)
      MPI_Pack(startbufs[i], counts[i], datatypes[i], buf, bufsize,
	       &position, comm);
  }
  
  out_count = position;

  //if larger than compression threshold
  if(compression_level>0 && out_count>(int)compression_threshold)
  {
    ASSERT(out_count*1.001+12<COMPRESS_BUF_SIZE);
    unsigned long size=COMPRESS_BUF_SIZE;
    //compress the buffer

    int retval;
    if( (retval=compress2((Bytef*)compress_buf,&size,(const Bytef*)buf,out_count,compression_level)) != Z_OK)
    {
      switch(retval)
      {
        case Z_MEM_ERROR:
          throw SCIRun::InternalError("Compression returned Z_MEM_ERROR", __FILE__, __LINE__);
          break;
        case Z_BUF_ERROR:
          throw SCIRun::InternalError("Compression returned Z_BUF_ERROR", __FILE__, __LINE__);
          break;
        case Z_DATA_ERROR:
          throw SCIRun::InternalError("Compression returned Z_DATA_ERROR", __FILE__, __LINE__);
          break;
        case Z_STREAM_ERROR:
          throw SCIRun::InternalError("Compression returned Z_STREAM_ERROR", __FILE__, __LINE__);
          break;
        default:
          throw SCIRun::InternalError("Compression of MPI message failed", __FILE__, __LINE__);
          break;
      }
    }
    else
    {
      //only use compressed buffer if the size has improved
      if(size<(unsigned long)out_count)
      {
        //copy to buffer
        memcpy(buf,compress_buf,size);
        //set compressed buffer size
        out_count=size;
      }
    }
  }
  
  // When it is all packed, only the buffer necessarily needs to be kept
  // around until after it is sent.
  delete sendlist;
  sendlist = 0;
  addSendlist(packedBuffer);
}

void
PackBufferInfo::unpack(MPI_Comm comm,MPI_Status &status)
{
  MALLOC_TRACE_TAG_SCOPE("PackBufferInfo::unpack");
  ASSERT(have_datatype);
  
  unsigned long bufsize = packedBuffer->getBufSize();

  
  if(compression_level>0 && bufsize>compression_threshold)
  {
    int compressed_size;
    MPI_Get_count(&status,MPI_BYTE,&compressed_size);
  
    //if the recieved size is less than the bufsize then the message was likely compressed
      //there are cases where it still might be uncompressed.
      //for example:  if MPI_Pack_size returned a value that is larger than the actual space needed
    if((unsigned long)compressed_size<bufsize)
    {
      //copy to the buffer
      memcpy(compress_buf,buf,compressed_size);

      int retval;
      //uncompress buffer
      if( (retval=uncompress((Bytef*)buf,&bufsize,(const Bytef*)compress_buf,compressed_size))  != Z_OK)
      {
        switch(retval)
        {
          case Z_MEM_ERROR:
            throw SCIRun::InternalError("Uncompression returned Z_MEM_ERROR", __FILE__, __LINE__);
            break;
          case Z_BUF_ERROR:
            throw SCIRun::InternalError("Uncompression returned Z_BUF_ERROR", __FILE__, __LINE__);
            break;
          case Z_STREAM_ERROR:
            throw SCIRun::InternalError("Uncompression returned Z_STREAM_ERROR", __FILE__, __LINE__);
            break;
          case Z_DATA_ERROR: 
            //this likely means the data was not compressed so just copy the message over
            memcpy(buf,compress_buf,compressed_size);
            break;
          default:
            throw SCIRun::InternalError("Uncompression of MPI message failed", __FILE__, __LINE__);
            break;
        }
      }
      ASSERT(bufsize==(unsigned long)packedBuffer->getBufSize());
    }
  }

  int position = 0;
  for (int i = 0; i < (int)startbufs.size(); i++) {
    if(counts[i]>0)
      MPI_Unpack(buf, bufsize, &position, startbufs[i], counts[i], datatypes[i], comm);
  }
}

