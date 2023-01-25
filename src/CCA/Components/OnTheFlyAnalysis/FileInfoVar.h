/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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


#ifndef UINTAH_HOMEBREW_FileInfoVar_H
#define UINTAH_HOMEBREW_FileInfoVar_H

#include <Core/Exceptions/InternalError.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Util/Handle.h>
#include <Core/Util/RefCounted.h>

#include <fstream>

namespace Uintah {

struct FileInfo : public RefCounted {
  
  std::map<std::string, FILE *> files;       // filename is the key
  
  // constructor computes the values
  FileInfo() {};
  
  ~FileInfo(){
    std::map<std::string, FILE *>::iterator it;
    
    for ( it=files.begin() ; it != files.end(); it++ ){
      //std::cout << " closing file " << (*it).first << std::endl;
      fclose((*it).second);
    }
  }
};

typedef Handle<FileInfo> FileInfoP;

  void swapbytes( Uintah::FileInfoP& );

  // Note the general template for SoleVariable::readNormal will not
  // recognize the swapbytes correctly. So specialize it here.
  // Somewhat moot because the swapbytes for FileInfoP is not
  // implemented.
  template<>
  inline void PerPatch<FileInfoP>::readNormal(std::istream& in, bool swapBytes)
  {
    // Note if swap bytes for FileInfoP is implemente then this
    // template specialization can be deleted as the general template
    // will work.
    SCI_THROW(InternalError("Swap bytes for FileInfoP is not implemented", __FILE__, __LINE__));

    ssize_t linesize = (ssize_t)(sizeof(FileInfoP));
    
    FileInfoP val;
    
    in.read((char*) &val, linesize);
    
    if (swapBytes)
      Uintah::swapbytes(val);
    
    value = std::make_shared<FileInfoP>(val);
  }

} // End namespace Uintah

#endif
