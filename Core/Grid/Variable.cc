#include <Packages/Uintah/Core/Grid/Variable.h>
#include <Core/Exceptions/ErrnoException.h>
#include <Core/Malloc/Allocator.h>
#include <sstream>
#include <zlib.h>
#include <math.h>
#include <errno.h>

using namespace std;
using namespace Uintah;

Variable::Variable()
{
   d_foreign = false;
}

Variable::~Variable()
{
}

void Variable::setForeign()
{
   d_foreign = true;
}

void Variable::emit(OutputContext& oc, string compressionMode)
{
  bool use_rle = false;
  bool use_gzip = false;

  if ((compressionMode == "rle, gzip") || (compressionMode == "gzip, rle")) {
    use_rle = true;
    use_gzip = true;
  }
  else if (compressionMode == "rle")
    use_rle = true;
  else if (compressionMode == "gzip")
    use_gzip = true;
  else if (compressionMode != "") {
    throw InvalidCompressionMode(compressionMode);
  }
  
  ostringstream outstream;

  if (use_rle)
    emitRLE(outstream, oc.varnode);
  else
    emitNormal(outstream, oc.varnode);
  
  unsigned long outstreamsize = outstream.str().size();
  if (use_gzip) {
    // use gzip compression

    // follows compress guidelines: 1% more than source size + 12
    unsigned long bufsize =
      (unsigned long)(ceil((double)outstreamsize * 1.01) + 12); 

    char* buffer = scinew char[bufsize];
    if (compress((Bytef*)buffer, &bufsize,
		 (const Bytef*)outstream.str().c_str(), outstreamsize) != Z_OK)
      cerr << "compress failed in Uintah::Variable::emit\n";
	
    // first write the uncompressed data size
    ssize_t s = ::write(oc.fd, &outstreamsize, sizeof(unsigned long));
    if(s != sizeof(unsigned long))
      throw ErrnoException("Variable::emit (write call)", errno);

    oc.cur += sizeof(unsigned long);

    // then write the compressed data
    s = ::write(oc.fd, buffer, bufsize);
    if(s != bufsize)
      throw ErrnoException("Variable::emit (write call)", errno);

    oc.cur += bufsize;
    delete[] buffer;
  }
  else {
    ::write(oc.fd, outstream.str().c_str(), outstreamsize);
    oc.cur += outstreamsize;
  }
}


void Variable::read(InputContext& ic, long end, string compressionMode)
{
  bool use_rle = false;
  bool use_gzip = false;

  if ((compressionMode == "rle, gzip") || (compressionMode == "gzip, rle")) {
    use_rle = true;
    use_gzip = true;
  }
  else if (compressionMode == "rle")
    use_rle = true;
  else if (compressionMode == "gzip")
    use_gzip = true;
  else if (compressionMode != "") {
    throw InvalidCompressionMode(compressionMode);
  }

  long datasize = end - ic.cur;
  string data;
  string bufferStr;

  data.resize(datasize);
  // casting from const char* below to char* -- use caution
  ssize_t s = ::read(ic.fd, (char*)data.c_str(), datasize);
  if(s != datasize)
    throw ErrnoException("Variable::read (read call)", errno);

  
  ic.cur += datasize;
  
  istringstream instream(data);

  if (use_gzip) {
    // use gzip compression

    // first read the uncompressed data size
    unsigned long uncompressed_size;
    instream.read((char*)&uncompressed_size, sizeof(unsigned long));
    const char* compressed_data = data.c_str() + sizeof(unsigned long);
    long compressed_datasize = datasize - (long)sizeof(unsigned long);

    // casting from const char* below to char* -- use caution
    bufferStr.resize(uncompressed_size);
    char* buffer = (char*)bufferStr.c_str();

    if (uncompress((Bytef*)buffer, &uncompressed_size,
		   (const Bytef*)compressed_data, compressed_datasize) != Z_OK)
       cerr << "uncompress failed in Uintah::Variable::read\n";

    instream.str(bufferStr);
  }

  
  if (use_rle)
    readRLE(instream);
  else
    readNormal(instream);
  ASSERT(instream.fail() == 0);
  ASSERT((unsigned long)instream.tellg() == instream.str().size());
}
