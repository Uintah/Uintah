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

#ifndef CORE_GRID_VARIABLES_VARIABLE_H
#define CORE_GRID_VARIABLES_VARIABLE_H

#include <Core/ProblemSpec/ProblemSpec.h>

#include <sci_defs/pidx_defs.h>

#include <string>
#include <iosfwd>

namespace Uintah {

// forward decls
class TypeDescription;
class InputContext;
class OutputContext;
class Patch;
class PIDXOutputContext;
class RefCounted;
class VarLabel;
class PIDXOutputContext;

/**************************************
     
  CLASS

    Variable


  GENERAL INFORMATION

    Variable.h

    Steven G. Parker
    Department of Computer Science
    University of Utah
      
    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
      
  KEYWORDS

    Variable
      
  ****************************************/
    

class Variable {

public:

  virtual ~Variable();
  
  virtual const TypeDescription* virtualGetTypeDescription() const = 0;

  void setForeign();

  bool isForeign() const { return d_foreign; }

  // marks a variable as valid (for example, it has received all requires via MPI)
  void setValid() { d_valid = true;}

  // marks a variable as invalid (for example, it is still in the process of receiving MPI)
  void setInvalid() { d_valid = false; }

  //returns if a variable is marked valid or invalid
  bool isValid() const { return d_valid; }

  size_t emit(       OutputContext &
             , const IntVector     & l
             , const IntVector     & h
             , const std::string   & compressionModeHint
             );

  void read(       InputContext &
           ,       long           end
           ,       bool           swapbytes
           ,       int            nByteMode
           , const std::string  & compressionMode
           );

#if HAVE_PIDX
  virtual void emitPIDX(       PIDXOutputContext & oc
                       ,       unsigned char     * buffer
                       , const IntVector         & l
                       , const IntVector         & h
                       , const size_t              pidx_bufferSize // buffer size used for bullet proofing.
                       );

  void readPIDX( const unsigned char * pidx_buffer
               , const size_t        & pidx_bufferSize
               , const bool            swapBytes
               );
#endif

  virtual void emitNormal(       std::ostream & out
                         , const IntVector    & l
                         , const IntVector    & h
                         ,       ProblemSpecP   varnode
                         ,       bool           outputDoubleAsFloat
                         ) = 0;

  virtual void readNormal( std::istream& in, bool swapbytes ) = 0;

  virtual void allocate( const Patch* patch, const IntVector& boundary ) = 0;

  virtual void getSizeInfo( std::string& elems, unsigned long& totsize, void*& ptr ) const = 0;

  // used to get size info of the underlying data; this is for host-->device variable copy
  virtual size_t getDataSize() const = 0;

  // used to copy Variables to contiguous buffer prior to bulk host-->device copy
  virtual bool copyOut(void* dst) const = 0;

  virtual void copyPointer( Variable& ) = 0;

  // Only affects grid variables
  virtual void offsetGrid( const IntVector& /*offset*/ );

  virtual RefCounted* getRefCounted() = 0;


protected:

  Variable();


private:

  // eliminate copy, assignment and move
  Variable( const Variable & )            = delete;
  Variable& operator=( const Variable & ) = delete;
  Variable( Variable && )                 = delete;
  Variable& operator=( Variable && )      = delete;

  // Compresses the string pointed to by pUncompressed and but the resulting
  // compressed data into the string pointed to by pBuffer.
  // Returns the pointer to whichever one is shortest and erases the  other one.
  std::string* gzipCompress( std::string* pUncompressed
                           , std::string* pBuffer
                           );

  // states that the variable is from another node - these variables (ghost cells, slabs, corners) are communicated via MPI
  bool d_foreign {false};

  // signals that the variable is valid, an MPI variable is not valid until MPI has been received
  bool d_valid {true};
};

} // namespace Uintah

#endif // CORE_GRID_VARIABLES_VARIABLE_H
