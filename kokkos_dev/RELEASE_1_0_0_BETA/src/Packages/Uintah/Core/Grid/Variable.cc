#include <Packages/Uintah/Core/Grid/Variable.h>
#include <Core/Exceptions/ErrnoException.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
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

void Variable::emit(OutputContext& oc, const string& compressionModeHint)
{
  bool use_rle = false;
  bool use_gzip = false;
  bool try_all = false;

  bool used_rle = false;
  bool used_gzip = false;

  if (compressionModeHint == "tryall") {
    use_rle = false; // try without rle first
    use_gzip = true;
    try_all = true;
  }
  else if ((compressionModeHint == "rle, gzip") ||
	   (compressionModeHint == "gzip, rle")) {
    use_rle = true;
    use_gzip = true;
  }
  else if (compressionModeHint == "rle")
    use_rle = true;
  else if (compressionModeHint == "gzip")
    use_gzip = true;
  else if (compressionModeHint != "" && compressionModeHint != "none") {
    throw InvalidCompressionMode(compressionModeHint);
  }

  used_rle = use_rle;
  used_gzip = use_gzip;
  
  ostringstream outstream;

  if (use_rle)
    emitRLE(outstream, oc.varnode);
  else
    emitNormal(outstream, oc.varnode);

  string preGzip = outstream.str();
  string preGzip2;
  string buffer; // trying to avoid copying the strings back and forth
  string buffer2;
  string* writeoutString = &preGzip;
  if (use_gzip) {
    writeoutString = gzipCompress(&preGzip, &buffer);
    if (writeoutString != &buffer)
      used_gzip = false; // gzip wasn't better, so it wasn't used

    if (try_all) {
      ostringstream outstream2;
      try {
	emitRLE(outstream2, oc.varnode);
	preGzip2 = outstream2.str();
	string* writeoutString2 = gzipCompress(&preGzip2, &buffer2);
	
	if ((*writeoutString2).size() < (*writeoutString).size()) {
	  // rle was making it worse
	  writeoutString->erase(); // erase the old one to save space
	  writeoutString = writeoutString2; 
	  used_rle = true;
	  if (writeoutString2 == &buffer2)
	    used_gzip = true;
	  else
	  used_gzip = false;
	}
	else
	  writeoutString2->erase(); // doesn't get used, so erase it
      }
      catch (InvalidCompressionMode& ) {
	// just trying, if it doesn't like it then forget it
      }
    }
  }

  const char* writebuffer = (*writeoutString).c_str();
  unsigned long writebufferSize = (*writeoutString).size();
  ssize_t s = ::write(oc.fd, writebuffer, writebufferSize);

  if(s != (long)writebufferSize)
    throw ErrnoException("Variable::emit (write call)", errno);
  oc.cur += writebufferSize;

  string compressionMode = compressionModeHint;
  if (try_all || (used_gzip != use_gzip) || (used_rle != use_rle)) {
    // compression mode string changes
    if (used_gzip) {
      if (used_rle)
	compressionMode = "rle, gzip";
      else
	compressionMode = "gzip";
    }
    else if (used_rle) {
      compressionMode = "rle";
    }
    else
      compressionMode = "";
  }

  if (compressionMode != "" && compressionMode != "none")
    appendElement(oc.varnode, "compression", compressionMode);
}

string* Variable::gzipCompress(string* pUncompressed, string* pBuffer)
{
  unsigned long uncompressedSize = pUncompressed->size();

  // follows compress guidelines: 1% more than source size + 12
  unsigned long compressBufsize =
    (unsigned long)(ceil((double)uncompressedSize * 1.01) + 12); 

  pBuffer->resize(compressBufsize + sizeof(unsigned long));
  char* buf = (char*)pBuffer->c_str(); // casting from const, careful
  buf += sizeof(unsigned long); /* the first part will give the size of
				   the uncompressed data */
  
  if (compress((Bytef*)buf, &compressBufsize,
	       (const Bytef*)pUncompressed->c_str(), uncompressedSize) != Z_OK)
    cerr << "compress failed in Uintah::Variable::gzipCompress\n";

  pBuffer->resize(compressBufsize + sizeof(unsigned long));
  if (pBuffer->size() > uncompressedSize) {
    // gzip made it worse -- forget that
    // (this should rarely, if ever, happen, but just in case)
    pBuffer->erase(); /* the other buffer isn't needed, erase it to
			 save space */
    return pUncompressed;
  }
  else {
    // write out the uncompressed size to the first part of the buffer
    char* pbyte = (char*)(&uncompressedSize);
    for (int i = 0; i < (int)sizeof(unsigned long); i++, pbyte++) {
      (*pBuffer)[i] = *pbyte;
    }
    pUncompressed->erase(); /* the original buffer isn't needed, erase it to
			       save space */
    return pBuffer;
  } 
}


void Variable::read(InputContext& ic, long end, const string& compressionMode)
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
  string* uncompressedData = &data;

  data.resize(datasize);
  // casting from const char* below to char* -- use caution
  ssize_t s = ::read(ic.fd, (char*)data.c_str(), datasize);
  if(s != datasize)
    throw ErrnoException("Variable::read (read call)", errno);

  
  ic.cur += datasize;

  if (use_gzip) {
    // use gzip compression

    // first read the uncompressed data size
    unsigned long uncompressed_size;
    istringstream compressedStream(data);
    compressedStream.read((char*)&uncompressed_size, sizeof(unsigned long));
    const char* compressed_data = data.c_str() + sizeof(unsigned long);
    long compressed_datasize = datasize - (long)sizeof(unsigned long);

    // casting from const char* below to char* -- use caution
    bufferStr.resize(uncompressed_size);
    char* buffer = (char*)bufferStr.c_str();

    if (uncompress((Bytef*)buffer, &uncompressed_size,
		   (const Bytef*)compressed_data, compressed_datasize) != Z_OK)
       cerr << "uncompress failed in Uintah::Variable::read\n";

    uncompressedData = &bufferStr;
  }

  istringstream instream(*uncompressedData);
  
  if (use_rle)
    readRLE(instream);
  else
    readNormal(instream);
  ASSERT(instream.fail() == 0);
  ASSERT((unsigned long)instream.tellg() == uncompressedData->size());
}
