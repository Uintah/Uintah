/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#include <sys/stat.h>
#include <time.h>
       
using namespace Uintah;
using namespace SCIRun;
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
//  In some situations read() will transfer fewer bytes than was requested
//  and it won't indicate an error.  This version will try 10 times
//  and sleep for 10 seconds between attempts
//  http://csapp.cs.cmu.edu/public/ch10-preview.pdf
ssize_t 
Variable::debug_read(int fd, char *usrbuf, size_t nRequestedBytes)
{
  size_t nLeft = nRequestedBytes;
  ssize_t nRead;
  char *bufp = usrbuf;
  int count = 0;
  
  while (nLeft > 0 && count < 10) {
    count ++;
    
    if ( (nRead = ::read(fd, bufp, nLeft)) < 0 ) {
      if (errno == EINTR) {   // interrupted by sig handler return
        nRead = 0;            // and call read() again
      }else{
        return -1;            // errno set by read()
      }
    }
    else if (nRead == 0){
      cerr <<  "    debug_read: at end of file\n";
      break;                  // EOF
    }
    nLeft -= nRead;
    bufp += nRead;
    
    cerr <<  "   debug_read: " << count << " Trying to read nBytes: " << nLeft << " nRequestedBytes " << nRequestedBytes <<  '\n'; 
    sleep(10);
  }
  return (nRequestedBytes - nLeft);     /* return >= 0 */
}
//______________________________________________________________________
//
void
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
  unsigned long writebufferSize = (*writeoutString).size();
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

  if (compressionMode != "" && compressionMode != "none")
    //appendElement(oc.varnode, "compression", compressionMode);
    oc.varnode->appendElement("compression", compressionMode);
}

#if HAVE_PIDX
void
Variable::emit(PIDXOutputContext& pc,
                const IntVector& l,
                const IntVector& h, const string& compressionModeHint,
		 double* pidx_buffer
	      )
{

  // cout << "Start of PIDX emit" << endl;
  ProblemSpecP dummy;

  std::ostringstream outstream;
  emitNormal(outstream, l, h, dummy,false);

  string writeoutString = outstream.str();
  //double *pidx_buffer;

  const char* writebuffer = (writeoutString).c_str();
  unsigned long writebufferSize = (writeoutString).size();

  //cout << "write buffer size = " << writebufferSize/8 << " Name: " << var_name << endl;
 
  int i,zeroCount=0, nonZeroCount=0;
  if(writebufferSize>0) {

    //pidx_buffer = (double *) malloc((writebufferSize/8)*sizeof(double));
    memcpy(pidx_buffer, writebuffer, (writebufferSize/8)*sizeof(double));
    
    //printf("PIDX : %f %f %f %f %f %f %f\n", pidx_buffer[0], pidx_buffer[1], pidx_buffer[2], pidx_buffer[3], pidx_buffer[4], pidx_buffer[5], pidx_buffer[6]);
    //printf("UINTAH : %f %f %f %f %f %f %f\n", (double)writebuffer[0], (double)writebuffer[1], (double)writebuffer[2], (double)writebuffer[3], (double)writebuffer[4], (double)writebuffer[5], (double)writebuffer[6]);
    
//     for(i = 0 ; i < writebufferSize/8 ; i++)
//     {
// 	if((double)pidx_buffer[i] == 0.0)
// 	  zeroCount++;
// 	else
// 	  nonZeroCount++;
// 	printf("Element = %16.16f\n", (double)pidx_buffer[i]);
//     }
    //printf("Zero Count = %d and Non Zero Count %d\n", zeroCount, nonZeroCount);
    

    //    cout << "offsets: " << offset[0] << " " << offset[1] << " " << offset[2] << " "
    //         << offset[3] << " " << offset[4] << endl;
    //    cout << "count: " << count[0] << " " << count[1] << " " << count[2] << " "
    //         << count[3] << " " << count[4] << endl;
    //pidx_buffer = (double*)writebuffer;
       //for (unsigned long i = 0; i< writebufferSize/8; i++) {
	 // pidx_buffer[i] = (double)writebuffer[i];
    //cout << "pidx_buffer[ " << i << "] = " << (double)pidx_buffer[i]/*writebuffer[0]*/ << endl;
        //}


    //pc.variable[vc][mc] = PIDX_variable_global_define(pc.idx_ptr, var_name, /*sample_per_variable_buffer[vc]*/ 1, MPI_DOUBLE);
    //PIDX_variable_local_add(pc.idx_ptr, pc.variable[vc][mc], (int*) offset, (int*) count);
    //PIDX_variable_local_layout(pc.idx_ptr, pc.variable[vc][mc], (double*)pidx_buffer, MPI_DOUBLE);
    //printf("Address [1] : %p\n", pidx_buffer);
     
  }

}

#endif

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

//<ctc> fix reading files with multiple compression types
void
Variable::read( InputContext& ic, long end, bool swapBytes, int nByteMode,
                const string& compressionMode,
                const string& name,
                const int matlIndex )
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
    ssize_t s; 

    s = ::read(ic.fd, const_cast<char*>(data.c_str()), datasize);
    int errsv = errno;
      
    if(s != datasize){
      cerr << "  Error reading file: " << ic.filename << " number of bytes read: " << s << ".  Number of bytes that should have been read: " << datasize <<'\n';
      cerr << "    Name: " << name << " matl: " << matlIndex  << '\n';
      cerr << "    Now trying to recursively read the file.\n";
      s = debug_read( ic.fd, const_cast<char*>(data.c_str()), datasize);
    }   
    //__________________________________
    //  Output file info and throw an exception
    // if there's a problem
    if(s != datasize) {
      struct stat sb;
    
      if ( fstat(ic.fd, &sb) == -1 ) {
        SCI_THROW(ErrnoException("Variable::read (read call)", errno, __FILE__, __LINE__));
      }
      
      printf("__________________________________\n");
      printf("File: %s \n", ic.filename);
      printf("File type:                ");

      switch (sb.st_mode & S_IFMT) {
      case S_IFBLK:  printf("block device\n");       break;
      case S_IFCHR:  printf("character device\n");   break;
      case S_IFDIR:  printf("directory\n");          break;
      case S_IFIFO:  printf("FIFO/pipe\n");          break;
      case S_IFLNK:  printf("symlink\n");            break;
      case S_IFREG:  printf("regular file\n");       break;
      case S_IFSOCK: printf("socket\n");             break;
      default:       printf("unknown?\n");           break;
      }

      printf("  I-node number:            %ld\n", (long) sb.st_ino);
      printf("  Mode:                     %lo (octal)\n",(unsigned long) sb.st_mode);
      printf("  Link count:               %ld\n", (long) sb.st_nlink);
      printf("  Ownership:                UID=%ld   GID=%ld\n",(long) sb.st_uid, (long) sb.st_gid);
      printf("  Preferred I/O block size: %ld bytes\n",(long) sb.st_blksize);
      printf("  File size:                %lld bytes\n",(long long) sb.st_size);
      printf("  Blocks allocated:         %lld\n", (long long) sb.st_blocks);
      printf("  Last status change:       %s", ctime(&sb.st_ctime));
      printf("  Last file access:         %s", ctime(&sb.st_atime));
      printf("  Last file modification:   %s\n", ctime(&sb.st_mtime));
    
      cerr << "  Note if you see: (errno=2: No such file or directory) that may be false!! \n";
      SCI_THROW(ErrnoException("Variable::read (read call)", errsv, __FILE__, __LINE__));
    }
  
    ic.cur += datasize;

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
    }

    istringstream instream(*uncompressedData);
  
    if (use_rle)
      readRLE(instream, swapBytes, nByteMode);
    else
      readNormal(instream, swapBytes);
    ASSERT(instream.fail() == 0);

  } // end if datasize > 0
} // end read()

bool
Variable::emitRLE(ostream& /*out*/, const IntVector& /*l*/,
                  const IntVector& /*h*/, ProblemSpecP /*varnode*/)
{
  return false; // not supported by default
}
  
void
Variable::readRLE(istream& /*in*/, bool /*swapBytes*/, int /*nByteMode*/)
{
  SCI_THROW(InvalidCompressionMode("rle",
                               virtualGetTypeDescription()->getName(), __FILE__, __LINE__));
}

void
Variable::offsetGrid(const IntVector& /*offset*/)
{
}
