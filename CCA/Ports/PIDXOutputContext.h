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

#include <Core/Disclosure/TypeDescription.h>
#include <sci_defs/pidx_defs.h>
#if HAVE_PIDX
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

class TypeDescription;
    
class PIDXOutputContext {
  public:
    PIDXOutputContext();
    ~PIDXOutputContext();

    void initialize(std::string filename, unsigned int timeStep, int globalExtent[3], MPI_Comm comm);

    std::vector<TypeDescription::Type> getSupportedVariableTypes();
    
    std::string getDirectoryName(TypeDescription::Type TD);

    std::string filename;
    unsigned int timestep;
    PIDX_file file;
    MPI_Comm comm;
    PIDX_variable **variable;

    PIDX_access access;
  private:
    bool d_isInitialized;
    
    

  };
} // End namespace Uintah

#endif //HAVE_PIDX
#endif //UINTAH_HOMEBREW_PIDXOutputContext_H
