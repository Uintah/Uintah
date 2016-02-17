/*
 * The MIT License
 *
 * Copyright (c) 2013-2016 The University of Utah
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

#ifndef UINTAH_HOMEBREW_PIDXOutputContext_H
#define UINTAH_HOMEBREW_PIDXOutputContext_H

#include <sci_defs/pidx_defs.h>
#if HAVE_PIDX
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <PIDX.h>
#include <mpi.h>
#include <string>
#include <vector>

namespace Uintah {
/**************************************

  CLASS
    PIDXOutputContext

    Short Description...

  GENERAL INFORMATION

    PIDXOutputContext.h

    Sidharth Kumar
    School of Computing
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

    Copyright (C) 2000 SCI Group

  KEYWORDS
    PIDXOutputContext

  DESCRIPTION
    Long description...

  WARNING

  ****************************************/

class PIDXOutputContext {
  public:
    PIDXOutputContext();
    ~PIDXOutputContext();
    
    //  Struct for storing patch extents
    struct patchExtents{      
      IntVector lo_EC;
      IntVector hi_EC;
      IntVector patchSize;
      IntVector patchOffset;
      int totalCells_EC;
      
      //__________________________________
      // debugging
      void print(std::ostream& out){
        out  << "patchExtents: patchOffset: " << patchOffset << " patchSize: " << patchSize << ", totalCells_EC " << totalCells_EC 
             << ", lo_EC: " << lo_EC << ", hi_EC: " << hi_EC << std::endl; 
      }
    };

    void initialize(std::string filename,
                    unsigned int timeStep,
                    MPI_Comm comm);
    
    void setLevelExtents( std::string desc, 
                          IntVector lo,
                          IntVector hi,
                          PIDX_point& level_size );

    void setPatchExtents( std::string desc, 
                          const Patch* patch,
                          const Level* level,
                          const IntVector& boundaryLayer,
                          const TypeDescription* TD,
                          patchExtents& pExtents,
                          PIDX_point& patchOffset,
                          PIDX_point& nPatchCells );

    void checkReturnCode( const int rc,
                          const std::string warn,
                          const char* file, 
                          int line);
                          
    void hardWireBufferValues(unsigned char* patchBuffer, 
                              const patchExtents patchExts,
                              const size_t arraySize,
                              const int samples_per_value );

    void setOutputDoubleAsFloat( bool me){
      d_outputDoubleAsFloat = me;
    }

    bool isOutputDoubleAsFloat(){
      return d_outputDoubleAsFloat;
    }


    std::vector<TypeDescription::Type> getSupportedVariableTypes();

    std::string getDirectoryName(TypeDescription::Type TD);

    template<class T>
    void printBuffer(const std::string & desc,
                     int samples_per_value,
                     SCIRun::IntVector& lo_EC,
                     SCIRun::IntVector& hi_EC,
                     unsigned char* dataPIDX,
                     size_t arraySize );

    std::string filename;
    unsigned int timestep;
    PIDX_file file;
    MPI_Comm comm;
    PIDX_variable **variable;         // is this the variableDescription?  --Todd

    PIDX_access access;
    

  //__________________________________
  //    
  private:

    bool d_isInitialized;
    bool d_outputDoubleAsFloat;
    int d_levelExtents[3];
    
    IntVector getLevelExtents(){
      IntVector levelExtents (d_levelExtents[0],d_levelExtents[1],d_levelExtents[2]);                                                                          
      return levelExtents;                    
    };

  };
} // End namespace Uintah

#endif //HAVE_PIDX
#endif //UINTAH_HOMEBREW_PIDXOutputContext_H
