/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#ifndef CCA_COMPONENTS_SCHEDULERS_PROCESSORGROUP_H
#define CCA_COMPONENTS_SCHEDULERS_PROCESSORGROUP_H

#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI
#include <vector>

namespace Uintah {
/**************************************

CLASS
   ProcessorGroup
   

GENERAL INFORMATION

   ProcessorGroup.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   Processor_Group


DESCRIPTION
  

WARNING
  
****************************************/

class Parallel;

class ProcessorGroup {

  public:

    ~ProcessorGroup();

    int size() const { return d_size; }

    int myrank() const { return d_rank; }

    MPI_Comm getComm() const
    {
      return d_comm;
    }

    MPI_Comm getgComm( int i ) const
    {
      if (d_threads < 1 || i == -1) {
        return d_comm;
      }
      else {
        return d_gComms[i];
      }
    }

    void setgComm( int i ) const;

  private:

    const ProcessorGroup* d_parent;

    friend class Parallel;

    ProcessorGroup( const ProcessorGroup* parent,
                          MPI_Comm        comm,
                          bool            allmpi,
                          int             rank,
                          int             size,
                          int             threads );

    int                           d_rank;
    int                           d_size;
    int                           d_threads;
    MPI_Comm                      d_comm;
    mutable std::vector<MPI_Comm> d_gComms;
    bool                          d_allmpi;

    // disable copy and assignment
    ProcessorGroup(const ProcessorGroup&);
    ProcessorGroup& operator=(const ProcessorGroup&);
};

} // End namespace Uintah


#endif // end CCA_COMPONENTS_SCHEDULERS_PROCESSORGROUP_H
