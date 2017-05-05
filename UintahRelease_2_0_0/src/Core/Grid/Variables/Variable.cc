/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <Core/Grid/Variables/Variable.h>
#include <Core/Exceptions/InvalidCompressionMode.h>
#include <Core/Disclosure/TypeDescription.h>
#include <CCA/Ports/InputContext.h>
#include <CCA/Ports/OutputContext.h>
#if HAVE_PIDX
#include <CCA/Ports/PIDXOutputContext.h>
#endif
#include <Core/IO/SpecializedRunLengthEncoder.h>
#include <Core/Grid/Patch.h>
#include <Core/Exceptions/ErrnoException.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/Endian.h>
#include <Core/Util/SizeTypeConvert.h>

#include   <sstream>
#include   <iostream>

#include <zlib.h>
#include <cmath>
#include <cerrno>
#include <cstring>
#include <cstdio>


using namespace Uintah;
using namespace std;

Variable::Variable()
{
   d_foreign = false;
   d_valid = true;
}

Variable::~Variable()
{
}

void
Variable::setForeign()
{
   d_foreign = true;
}
//______________________________________________________________________
//
size_t
Variable::emit( OutputContext& oc, const IntVector& l,
                const IntVector& h, const string& compressionModeHint )
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
    cout << "Invalid Compression Mode - throwing exception...\n";
    SCI_THROW(InvalidCompressionMode(compressionModeHint, "", __FILE__, __LINE__));
  }

  used_rle = use_rle;
  used_gzip = use_gzip;

  std::ostringstream outstream;

  if (use_rle) {
    if (!emitRLE(outstream, l, h, oc.varnode)) {
      cout << "Invalid Compression Mode - throwing exception...\n";

      SCI_THROW(InvalidCompressionMode("rle",
                                   virtualGetTypeDescription()->getName(), __FILE__, __LINE__));
    }
  }
  else {
    emitNormal(outstream, l, h, oc.varnode, oc.outputDoubleAsFloat);
  }

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
      if (emitRLE(outstream2, l, h, oc.varnode)) {
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
    }
  }

  errno = -1;

  const char* writebuffer = (*writeoutString).c_str();
  size_t writebufferSize = (*writeoutString).size();
  if(writebufferSize>0)
  {
    ssize_t s = ::write(oc.fd, writebuffer, writebufferSize);

    if(s != (long)writebufferSize) {
      cerr << "\nVariable::emit - write system call failed writing to " << oc.filename
         << " with errno " << errno << ": " << strerror(errno) <<  endl;
      cerr << " * wanted to write: " << writebufferSize << ", but actually wrote " << s << "\n\n";

      SCI_THROW(ErrnoException("Variable::emit (write call)", errno, __FILE__, __LINE__));
    }
    oc.cur += writebufferSize;
  }

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

  if (compressionMode != "" && compressionMode != "none"){
    oc.varnode->appendElement("compression", compressionMode);
  }


  return writebufferSize;
}

#if HAVE_PIDX
//______________________________________________________________________
//
void
Variable::emitPIDX( PIDXOutputContext& pc,
                    unsigned char* pidx_buffer,
                    const IntVector& l,
                    const IntVector& h,
                    const size_t pidx_bufferSize )
{
  // This seems inefficient  -Todd

  ProblemSpecP dummy;
  bool outputDoubleAsFloat = pc.isOutputDoubleAsFloat();
  
  // read the Array3 variable into tmpStream
  std::ostringstream tmpStream;
  emitNormal(tmpStream, l, h, dummy, outputDoubleAsFloat);

  // Create a string from the ostringstream
  string tmpString   = tmpStream.str();
  
  // create a c-string
  const char* writebuffer = tmpString.c_str();
  size_t uda_bufferSize   = tmpString.size();

  // copy the write buffer into the pidx_buffer
  if( uda_bufferSize == pidx_bufferSize) {
    memcpy(pidx_buffer, writebuffer, uda_bufferSize);
  } else {
    throw InternalError("ERROR: Variable::emitPIDX() error reading in buffer", __FILE__, __LINE__);
  }
}

//______________________________________________________________________
//
void
Variable::readPIDX( unsigned char* pidx_buffer,
                    const size_t& pidx_bufferSize,
                    bool swapBytes )
{
  // I don't know if there's a better way to create a istringstream directly from unsigned char*  -Todd
  
  // create a string from pidx_buffer      
  std::string strBuffer(pidx_buffer, pidx_buffer + pidx_bufferSize);

  // create a istringstream from the string 
  istringstream instream(strBuffer);
  
  // push the istringstream into an Array3 variable
  readNormal(instream, swapBytes);

} // end readPIDX()

#endif
//______________________________________________________________________
//
string*
Variable::gzipCompress(string* pUncompressed, string* pBuffer)
{
  unsigned long uncompressedSize = pUncompressed->size();

  // follows compress guidelines: 1% more than source size + 12
  // (round up, so use + 13).
  unsigned long compressBufsize = uncompressedSize * 101 / 100 + 13;

  pBuffer->resize(compressBufsize + sizeof(ssize_t));
  char* buf = const_cast<char*>(pBuffer->c_str()); // casting from const
  buf += sizeof(ssize_t); /* the first part will give the size of
                                   the uncompressed data */

  if (compress((Bytef*)buf, &compressBufsize,
               (const Bytef*)pUncompressed->c_str(), uncompressedSize) != Z_OK)
    cerr << "compress failed in Uintah::Variable::gzipCompress\n";

  pBuffer->resize(compressBufsize + sizeof(ssize_t));
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
    for (int i = 0; i < (int)sizeof(ssize_t); i++, pbyte++) {
      (*pBuffer)[i] = *pbyte;
    }
    pUncompressed->erase(); /* the original buffer isn't needed, erase it to
                               save space */
    return pBuffer;
  }
}
//______________________________________________________________________
//
//<ctc> fix reading files with multiple compression types
void
Variable::read( InputContext& ic, long end, bool swapBytes, int nByteMode,
                const string& compressionMode )
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
    SCI_THROW(InvalidCompressionMode(compressionMode, "", __FILE__, __LINE__));
  }

  long datasize = end - ic.cur;

  // On older UDAs, all variables were saved, even if they had a size
  // of 0.  So this allows us to skip reading 0 sized data.  (FYI, new
  // UDAs should not have this problem.)
  if( datasize == 0 ) return;

  if(datasize>0)
  {
    string data;
    string bufferStr;
    string* uncompressedData = &data;

    data.resize(datasize);
    ssize_t s = ::read(ic.fd, const_cast<char*>(data.c_str()), datasize);

    if(s != datasize) {
      cerr << "Error reading file: " << ic.filename << ", errno=" << errno << '\n';
      SCI_THROW(ErrnoException("Variable::read (read call)", errno, __FILE__, __LINE__));
    }

    ic.cur += datasize;

    //__________________________________
    //              gzip compression
    if (use_gzip) {
      // use gzip compression

      // first read the uncompressed data size
      istringstream compressedStream(data);
      uint64_t uncompressed_size_64;
      compressedStream.read((char*)&uncompressed_size_64, nByteMode);

      unsigned long uncompressed_size = convertSizeType(&uncompressed_size_64, swapBytes, nByteMode);

      if( uncompressed_size > 1000000000 ) {
        cout << "\n";
        cout << "--------------------------------------------------------------------------\n";
        cout << "!!!!!!!! WARNING !!!!!!!! \n";
        cout << "\n";
        cout << "Size of uncompressed variable seems wrong: " << uncompressed_size << "\n";
        cout << "Most likely, the UDA you are trying to read is corrupted due to a problem with\n";
        cout << "libz when it was created... Also, an exception most likely is about to be thrown...\n";
        cout << "--------------------------------------------------------------------------\n";
        cout << "\n\n";
      }

      const char* compressed_data = data.c_str() + nByteMode;

      long compressed_datasize = datasize - (long)(nByteMode);

      // casting from const char* below to char* -- use caution
      bufferStr.resize(uncompressed_size);
      char* buffer = (char*)bufferStr.c_str();

      int result = uncompress( (Bytef*)buffer, &uncompressed_size,
                               (const Bytef*)compressed_data, compressed_datasize );

      if (result != Z_OK) {
        printf( "Uncompress error result is %d\n", result );
        throw InternalError("uncompress failed in Uintah::Variable::read", __FILE__, __LINE__);
      }

      uncompressedData = &bufferStr;
    }  // gzip

    //__________________________________
    //   rle and uncompressed
    istringstream instream(*uncompressedData);

    if (use_rle) {
      readRLE(instream, swapBytes, nByteMode);
    } else {
      readNormal(instream, swapBytes);
    }
    ASSERT(instream.fail() == 0);

  } // end if datasize > 0
} // end read()

//______________________________________________________________________
//
bool
Variable::emitRLE(ostream& /*out*/, const IntVector& /*l*/,
                  const IntVector& /*h*/, ProblemSpecP /*varnode*/)
{
  return false; // not supported by default
}
//______________________________________________________________________
//
void
Variable::readRLE(istream& /*in*/, bool /*swapBytes*/, int /*nByteMode*/)
{
  SCI_THROW(InvalidCompressionMode("rle",
                               virtualGetTypeDescription()->getName(), __FILE__, __LINE__));
}
//______________________________________________________________________
//
void
Variable::offsetGrid(const IntVector& /*offset*/)
{
}
