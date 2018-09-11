/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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


#include <Core/Disclosure/TypeDescription.h>
#include <Core/Exceptions/ErrnoException.h>
#include <Core/Exceptions/InvalidCompressionMode.h>
#include <Core/Grid/Patch.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/SizeTypeConvert.h>

#include <CCA/Ports/InputContext.h>
#include <CCA/Ports/OutputContext.h>
#include <CCA/Ports/PIDXOutputContext.h>

#include <cmath>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>

#include <zlib.h>


using namespace Uintah;


//______________________________________________________________________
//
Variable::Variable()
{
  // do nothing
}

//______________________________________________________________________
//
Variable::~Variable()
{
  // do nothing
}

//______________________________________________________________________
//
void
Variable::setForeign()
{
   d_foreign = true;
}

//______________________________________________________________________
//
size_t
Variable::emit(       OutputContext & oc
              , const IntVector     & l
              , const IntVector     & h
              , const std::string   & compressionModeHint
              )
{
  bool use_gzip = false;
  bool used_gzip = false;
  if (compressionModeHint == "gzip") {
    use_gzip = true;
  }
  else if (compressionModeHint != "" && compressionModeHint != "none") {
    std::cout << "Invalid Compression Mode - throwing exception...\n";
    SCI_THROW(InvalidCompressionMode(compressionModeHint, "", __FILE__, __LINE__));
  }

  used_gzip = use_gzip;

  std::ostringstream outstream;
  emitNormal(outstream, l, h, oc.varnode, oc.outputDoubleAsFloat);

  std::string preGzip = outstream.str();
  std::string buffer;  // trying to avoid copying the strings back and forth
  std::string* writeoutString = &preGzip;

  if (use_gzip) {
    writeoutString = gzipCompress(&preGzip, &buffer);
    if (writeoutString != &buffer) {
      used_gzip = false;  // gzip wasn't better, so it wasn't used
    }
  }

  errno = -1;
  const char* writebuffer = (*writeoutString).c_str();
  size_t writebufferSize = (*writeoutString).size();
  if (writebufferSize > 0) {
    ssize_t s = ::write(oc.fd, writebuffer, writebufferSize);

    if (s != (long)writebufferSize) {
      std::cerr << "\nVariable::emit - write system call failed writing to " << oc.filename << " with errno " << errno << ": "
                << strerror(errno) << std::endl;
      std::cerr << " * wanted to write: " << writebufferSize << ", but actually wrote " << s << "\n\n";

      SCI_THROW(ErrnoException("Variable::emit (write call)", errno, __FILE__, __LINE__));
    }
    oc.cur += writebufferSize;
  }

  std::string compressionMode = compressionModeHint;
  if (used_gzip != use_gzip) {
    // compression mode string changes
    if (used_gzip) {
      compressionMode = "gzip";
    }
    else {
      compressionMode = "";
    }
  }

  if (compressionMode != "" && compressionMode != "none") {
    oc.varnode->appendElement("compression", compressionMode);
  }

  return writebufferSize;
}

//______________________________________________________________________
//
#if HAVE_PIDX
void
Variable::readPIDX( const unsigned char * pidx_buffer
                  , const size_t        & pidx_bufferSize
                  , const bool            swapBytes
                  )
{
  // I don't know if there's a better way to create a istringstream directly from unsigned char*  -Todd
  
  // Create a string from pidx_buffer:
  std::string strBuffer( pidx_buffer, pidx_buffer + pidx_bufferSize );

  // Create an istringstream from the string:
  std::istringstream instream( strBuffer );
  
  // Push the istringstream into an Array3 variable:
  readNormal( instream, swapBytes );

} // end readPIDX()

//______________________________________________________________________
//
void
Variable::emitPIDX(       PIDXOutputContext & /* oc */
                  ,       unsigned char     * /* buffer */
                  , const IntVector         & /* l */
                  , const IntVector         & /* h */
                  , const size_t              /* pidx_bufferSize, used for bullet proofing */
                  )
{
  SCI_THROW( InternalError( "emitPIDX not implemented for this type of variable", __FILE__, __LINE__ ) );
}
#endif

//______________________________________________________________________
//
std::string*
Variable::gzipCompress( std::string* pUncompressed
                      , std::string* pBuffer
                      )
{
  unsigned long uncompressedSize = pUncompressed->size();

  // follows compress guidelines: 1% more than source size + 12 (round up, so use + 13).
  unsigned long compressBufsize = uncompressedSize * 101 / 100 + 13;

  pBuffer->resize(compressBufsize + sizeof(ssize_t));
  char* buf = const_cast<char*>(pBuffer->c_str());  // casting from const
  buf += sizeof(ssize_t);  // the first part will give the size of the uncompressed data

  if (compress((Bytef*)buf, &compressBufsize, (const Bytef*)pUncompressed->c_str(), uncompressedSize) != Z_OK)
    std::cerr << "compress failed in Uintah::Variable::gzipCompress\n";

  pBuffer->resize(compressBufsize + sizeof(ssize_t));
  if (pBuffer->size() > uncompressedSize) {
    // gzip made it worse -- forget that (this should rarely, if ever, happen, but just in case)
    pBuffer->erase();  // the other buffer isn't needed, erase it to save space
    return pUncompressed;
  }
  else {
    // write out the uncompressed size to the first part of the buffer
    char* pbyte = (char*)(&uncompressedSize);
    for (int i = 0; i < (int)sizeof(ssize_t); i++, pbyte++) {
      (*pBuffer)[i] = *pbyte;
    }
    pUncompressed->erase(); // the original buffer isn't needed, erase it to save space

    return pBuffer;
  }
}

//______________________________________________________________________
//
void
Variable::read(       InputContext & ic
              ,       long           end
              ,       bool           swapBytes
              ,       int            nByteMode
              , const std::string  & compressionMode
              )
{
  bool use_gzip = false;

  if (compressionMode == "gzip") {
    use_gzip = true;
  }
  else if (compressionMode != "" && compressionMode != "none") {
    SCI_THROW(InvalidCompressionMode(compressionMode, "", __FILE__, __LINE__));
  }

  long datasize = end - ic.cur;

  // On older UDAs, all variables were saved, even if they had a size
  // of 0.  So this allows us to skip reading 0 sized data.
  // (FYI, new UDAs should not have this problem.)
  if (datasize == 0) {
    return;
  }

  if (datasize > 0) {
    std::string data;
    std::string bufferStr;
    std::string* uncompressedData = &data;

    data.resize(datasize);
    ssize_t s = ::read(ic.fd, const_cast<char*>(data.c_str()), datasize);

    if (s != datasize) {
      std::cerr << "Error reading file: " << ic.filename << ", errno=" << errno << '\n';
      SCI_THROW(ErrnoException("Variable::read (read call)", errno, __FILE__, __LINE__));
    }

    ic.cur += datasize;

    //__________________________________
    // gzip compression
    if (use_gzip) {

      // first read the uncompressed data size
      std::istringstream compressedStream(data);
      uint64_t uncompressed_size_64;
      compressedStream.read((char*)&uncompressed_size_64, nByteMode);

      unsigned long uncompressed_size = convertSizeType(&uncompressed_size_64, swapBytes, nByteMode);
      if (uncompressed_size > 1000000000) {
        std::cout << "\n";
        std::cout << "--------------------------------------------------------------------------\n";
        std::cout << "!!!!!!!! WARNING !!!!!!!! \n";
        std::cout << "\n";
        std::cout << "Size of uncompressed variable seems wrong: " << uncompressed_size << "\n";
        std::cout << "Most likely, the UDA you are trying to read is corrupted due to a problem with\n";
        std::cout << "libz when it was created... Also, an exception most likely is about to be thrown...\n";
        std::cout << "--------------------------------------------------------------------------\n";
        std::cout << "\n\n";
      }

      const char* compressed_data = data.c_str() + nByteMode;
      long compressed_datasize = datasize - (long)(nByteMode);

      // casting from const char* below to char* -- use caution
      bufferStr.resize(uncompressed_size);
      char* buffer = (char*)bufferStr.c_str();

      int result = uncompress((Bytef*)buffer, &uncompressed_size, (const Bytef*)compressed_data, compressed_datasize);
      if (result != Z_OK) {
        printf("Uncompress error result is %d\n", result);
        throw InternalError("uncompress failed in Uintah::Variable::read", __FILE__, __LINE__);
      }

      uncompressedData = &bufferStr;
    }

    //__________________________________
    // uncompressed
    std::istringstream instream(*uncompressedData);
    readNormal(instream, swapBytes);
    ASSERT(instream.fail() == 0);

  }  // end if datasize > 0

} // end read()

//______________________________________________________________________
//
void
Variable::offsetGrid(const IntVector& /*offset*/)
{
}
