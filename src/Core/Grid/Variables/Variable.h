/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef UINTAH_HOMEBREW_Variable_H
#define UINTAH_HOMEBREW_Variable_H

#include <string>
#include <iosfwd>

#include <Core/ProblemSpec/ProblemSpec.h>

namespace SCIRun {
  class IntVector;
}

namespace Uintah {

  class TypeDescription;
  class InputContext;
  class OutputContext;
  class Patch;
  class RefCounted;
  class VarLabel;

/**************************************
     
  CLASS
    Variable

    Short Description...

  GENERAL INFORMATION

    Variable.h

    Steven G. Parker
    Department of Computer Science
    University of Utah
      
    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
    Copyright (C) 2000 SCI Group
      
  KEYWORDS
    Variable
      
  DESCRIPTION
    Long description...
      
  WARNING
      
  ****************************************/
    
class Variable {

public:
  virtual ~Variable();
  
  virtual const TypeDescription* virtualGetTypeDescription() const = 0;
  void setForeign();
  bool isForeign() const {
    return d_foreign;
  }

  //marks a variable as invalid (for example, it is in the process of receiving mpi)
  void setValid() { d_valid=true;} 
  void setInvalid() { d_valid=false;} 
  //returns if a variable is marked valid or invalid
  bool isValid() const {return d_valid;}

  void emit(OutputContext&, const IntVector& l, const IntVector& h,
            const std::string& compressionModeHint);
  void read(InputContext&, long end, bool swapbytes, int nByteMode,
            const std::string& compressionMode);

  virtual void emitNormal(std::ostream& out, const IntVector& l,
                          const IntVector& h, ProblemSpecP varnode, bool outputDoubleAsFloat ) = 0;
  virtual void readNormal(std::istream& in, bool swapbytes) = 0;

  virtual bool emitRLE(std::ostream& /*out*/, const IntVector& l,
                       const IntVector& h, ProblemSpecP /*varnode*/);
  virtual void readRLE(std::istream& /*in*/, bool swapbytes, int nByteMode);
  
  virtual void allocate(const Patch* patch, const IntVector& boundary) = 0;

  virtual void getSizeInfo(std::string& elems, unsigned long& totsize, void*& ptr) const = 0;

  virtual void copyPointer(Variable&) = 0;

  // Only affects grid variables
  virtual void offsetGrid(const IntVector& /*offset*/);

  virtual RefCounted* getRefCounted() = 0;
protected:
  Variable();

private:    
  Variable(const Variable&);
  Variable& operator=(const Variable&);

  // Compresses the string pointed to by pUncompressed and but the
  // resulting compressed data into the string pointed to by pBuffer.
  // Returns the pointer to whichever one is shortest and erases the
  // other one.
  std::string* gzipCompress(std::string* pUncompressed, std::string* pBuffer);
  bool d_foreign;
  //signals of the variable is valid, an mpi variable is not valid until mpi has been recieved
  bool d_valid;
};

} // End namespace Uintah

#endif
