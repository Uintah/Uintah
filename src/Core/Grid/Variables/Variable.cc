/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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
              , const std::string   & compressionMode
              )
{
  bool use_gzip = false;

  if (compressionMode == "gzip") {
    use_gzip = true;
  }
  else if (compressionMode != "" && compressionMode != "none") {
    std::cout << "Invalid Compression Mode - throwing exception...\n";
    SCI_THROW(InvalidCompressionMode(compressionMode, "", __FILE__, __LINE__));
  }

  std::ostringstream outstream;
  emitNormal( outstream, l, h, oc.varnode, oc.outputDoubleAsFloat );

  std::string sourceString = outstream.str();
  std::string bufferString;  // trying to avoid copying the strings back and forth
  std::string* writeString = &sourceString;

  if (use_gzip) {
    writeString = gzipCompress( &sourceString, &bufferString );
  }


  //__________________________________
  //  Write the buffer
  errno = -1;
  const char* writeBuffer = (*writeString).c_str();
  size_t writeBufferSize  = (*writeString).size();


  if ( writeBufferSize > 0 ) {
    ssize_t s = ::write( oc.fd, writeBuffer, writeBufferSize );

    if ( s != (long)writeBufferSize ) {
      std::cerr << "\nERROR Variable::emit - write system call failed writing to (" << oc.filename << ") with errno "
                << errno << ": " << strerror(errno) << std::endl;
      std::cerr << " * Write buffer size: (" << writeBufferSize << "), but actually wrote buffer size:(" << s << ")\n\n";

      SCI_THROW(ErrnoException("Variable::emit (write call)", errno, __FILE__, __LINE__));
    }
    oc.cur += writeBufferSize;
  }

  //__________________________________
  //write <compression> gzip </compression> to xml file
  if (use_gzip) {
    oc.varnode->appendElement("compression", compressionMode);
  }

  return writeBufferSize;
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
Variable::gzipCompress( std::string* source_str
                      , std::string* dest_str
                      )
{
  unsigned long source_size = source_str->size();

  // follows compress guidelines: 1% more than source size + 12 (round up, so use + 13).
  unsigned long dest_size = source_size * 101 / 100 + 13;


  dest_str->resize( dest_size + sizeof(ssize_t) );          // increase the size and fill with "\000"

  char* dest_buf = const_cast<char*>( dest_str->c_str() );  // casting from const

  dest_buf += sizeof(ssize_t);                              // the first part will give the size of the source data

                                                            // compress
  int result = compress((Bytef*)dest_buf, &dest_size, (const Bytef*)source_str->c_str(), source_size);


  if ( result != Z_OK ){
    printf("compress error result is %d\n", result);
    throw InternalError("ERROR Uintah::Variable::gzipCompress.   compression failed.", __FILE__, __LINE__);
  }

                                  // Add the source size to the first part of the dest_str
                                  // The source size is needed during decompress()
  dest_str->resize( dest_size + sizeof(ssize_t) );

  char* pbyte = (char*)(&source_size);

  for (int i = 0; i < (int)sizeof(ssize_t); i++, pbyte++) {
    (*dest_str)[i] = *pbyte;
  }

  source_str->erase();         // the original dest_str isn't needed, erase it to save space

  return dest_str;
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

      if ( result != Z_OK ) {
        printf("Uncompress error result is %d\n", result);
        throw InternalError("ERROR Uintah::Variable::read.   Call to uncompress() failed.", __FILE__, __LINE__);
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
