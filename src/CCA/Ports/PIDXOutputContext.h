/*
 * The MIT License
 *
 * Copyright (c) 2013-2018 The University of Utah
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
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InternalError.h>

#if HAVE_PIDX
#  include <Core/Disclosure/TypeDescription.h>
#  include <Core/Geometry/IntVector.h>
#  include <Core/Grid/Level.h>
#  include <Core/Grid/Patch.h>
#  include <Core/Parallel/Parallel.h>
#  include <Core/Parallel/UintahMPI.h>

#  include <PIDX.h>
#  include <iomanip>  // setw()
#  include <iostream>
#  include <string>
#  include <vector>
#endif

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

#if !HAVE_PIDX

  class  PIDXOutputContext{
    public:
      PIDXOutputContext();
      ~PIDXOutputContext();
      
    class PIDX_flags{
      public:
        // ___________________________________________________
        // Empty methods so you can compile without PIDX

        PIDX_flags()  {}
        ~PIDX_flags() {}
        void print()  {}
        
        void problemSetup( const Uintah::ProblemSpecP& params ){
          std::ostringstream warn;
          warn << " ERROR:  To output with the PIDX file format, you must use the following in your configure line...";
          warn << "                 --with-pidx=<path to PIDX installation>\n";
          throw InternalError(warn.str(), __FILE__, __LINE__);
        }
    };
  };

#else // HAVE_PIDX

class PIDXOutputContext {
  public:  
    PIDXOutputContext();
    ~PIDXOutputContext();

    struct PIDX_IoFlags {
      PIDX_io_type ioType;          // eg: PIDX_RAW_IO (see .../pidx/PIDX_define.h)
      unsigned int compressionType; // eg: PIDX_NO_META_DATA_DUMP (see .../pidx/PIDX_define.h)
      IntVector    restructureBoxSize;
      unsigned int pipeSize;

      IntVector    partitionCount;
      unsigned int blockSize;
      unsigned int blockCount;
    };
    
    //______________________________________________________________________
    // Various flags and options
    class PIDX_flags{
      public:
        PIDX_flags();
        ~PIDX_flags() {}

        PIDX_IoFlags d_checkpointFlags;
        PIDX_IoFlags d_visIoFlags;

        bool d_debugOutput;

        //__________________________________
        // debugging
        void print(){
          std::cout << Parallel::getMPIRank() 
                    << "PIDXFlags: " << std::setw(26) << "\n"
                    << "   checkpoint IO type: " <<  d_checkpointFlags.ioType << "\n"
                    << "   checkpoint compressionType: "<< getCompressTypeName( d_checkpointFlags.compressionType ) << "\n"
                    << "   checkpoint restructure box size: " << d_checkpointFlags.restructureBoxSize << "\n"
                    << "   checkpoint pipe size: " << d_checkpointFlags.pipeSize << "\n"
                    << "   visIo IO type: " <<  d_visIoFlags.ioType << "\n"
                    << "   visIo compressionType: "<< getCompressTypeName( d_visIoFlags.compressionType ) << "\n"
                    << "   visIo restructure box size: " << d_visIoFlags.restructureBoxSize << "\n"
                    << "   visIo partition count: " << d_visIoFlags.partitionCount << "\n"
                    << "   visIo block size: " << d_visIoFlags.blockSize << "\n"
                    << "   visIo block count: " << d_visIoFlags.blockCount << "\n";
        }  

        void problemSetup( const ProblemSpecP& params );
      
      private:
        //__________________________________
        // convert user input into compres type
        unsigned int str2CompressType( const std::string & type );
        
        std::string  getCompressTypeName( const int type );
        
        std::map<std::string, int> compressMap;
    };
    //______________________________________________________________________
    
    //__________________________________
    //  Struct for storing patch extents
    struct patchExtents{      
      IntVector lo_EC;
      IntVector hi_EC;
      IntVector patchSize;
      IntVector patchOffset;
      int totalCells_EC;

      // debugging
      void print(std::ostream& out){
        out  << Parallel::getMPIRank()
             << " patchExtents: patchOffset: " << patchOffset << " patchSize: " << patchSize << ", totalCells_EC " << totalCells_EC 
             << ", lo_EC: " << lo_EC << ", hi_EC: " << hi_EC << "\n"; 
      }
    };
    
    void computeBoxSize( const PatchSubset * patches, 
                         const PIDX_flags    flags,
                               PIDX_point  & newBox );

    void initialize( const std::string  & filename,
                     const unsigned int   timeStep,
                           MPI_Comm       comm,
                           PIDX_flags     flags,
                           PIDX_point     dims,
                     const int            type );

  void initializeParticles( const std::string  & filename, 
                            const unsigned int   timeStep,
                                  MPI_Comm       comm,
                                  PIDX_point     dim,
                            const int            typeOutput );
    
    void setLevelExtents( const std::string & desc, 
                                IntVector     lo,
                                IntVector     hi,
                                PIDX_point  & level_size );

    void setPatchExtents( const std::string     & desc, 
                          const Patch           * patch,
                          const Level           * level,
                          const IntVector       & boundaryLayer,
                          const TypeDescription * TD,
                                patchExtents    & pExtents,
                                PIDX_point      & patchOffset,
                                PIDX_point      & nPatchCells ) const;

    static void checkReturnCode( const int           rc,
                                 const std::string   warn,
                                 const char        * file, 
                                 const int           line );
                          
    void hardWireBufferValues(unsigned char* patchBuffer, 
                              const patchExtents patchExts,
                              const size_t arraySize,
                              const int samples_per_value );

    void setOutputDoubleAsFloat( bool me) { d_outputDoubleAsFloat = me; }

    bool isOutputDoubleAsFloat(){ return d_outputDoubleAsFloat; }


    std::vector<TypeDescription::Type> getSupportedVariableTypes();

    std::string getDirectoryName(TypeDescription::Type TD);

    void printBufferWrap( const std::string           & desc,
                          const TypeDescription::Type   TD,
                          const int                     samples_per_value,
                          const IntVector             & lo_EC,
                          const IntVector             & hi_EC,
                          const unsigned char         * dataPIDX,
                          const size_t                  arraySize ) const;
    template<class T>
    void printBuffer( const std::string       & desc,
                      const std::string       & format,
                      const int                 samples_per_value,
                      const Uintah::IntVector & lo_EC,
                      const Uintah::IntVector & hi_EC,
                      const unsigned char     * dataPIDX,
                      const size_t              arraySize ) const;
                     
    std::string filename;
    unsigned int timestep;
    PIDX_file file;
    PIDX_variable **varDesc;    // variable descriptor array
    PIDX_access access;
    
    // this must match what is specified in DataArchiver.cc
    enum typeOutput { OUTPUT               =  0,
                      CHECKPOINT           =  1,
                      CHECKPOINT_REDUCTION =  3,
                      NONE                 = -9 };

  //__________________________________
  //    
  private:
    bool d_isInitialized;
    bool d_outputDoubleAsFloat;
    int  d_levelExtents[3];
    
    IntVector getLevelExtents() {
      IntVector levelExtents( d_levelExtents[0],d_levelExtents[1],d_levelExtents[2] );
      return levelExtents;                    
    };
    
  };

#endif //HAVE_PIDX

} // end namespace Uintah

#endif //UINTAH_HOMEBREW_PIDXOutputContext_H
