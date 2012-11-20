/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifndef UINTAH_GRID_BCData_H
#define UINTAH_GRID_BCData_H

#include <map>
#include <string>
#include <vector>

#include <Core/Util/Handle.h>
#include <Core/Grid/BoundaryConditions/BoundCondBase.h>

namespace Uintah {

/**************************************

CLASS
   BCData
   
   
GENERAL INFORMATION

   BCData.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   BoundCondBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class BCData  {
  public:
    BCData();
    ~BCData();

    // Note, all 'bc's added to this BCData must have the same material index. 
    void addBC( const BoundCondBase * bc );

    const BoundCondBase                     * getBCValue( const std::string & type ) const;
    const std::vector<const BoundCondBase*>   getBCs() const { return d_BCBs; }
    int                                       getMatl() const { return d_matl; }

    void print( int depth = 0 ) const;

    bool exists( const std::string & type ) const;
    bool exists( const std::string & bc_type, const std::string & bc_variable ) const;

    void combine( const BCData & from );
    

    // Used when a BCData is put into a std::set.  
    bool operator() ( const BCData * a,  const BCData * b ) { return a->d_matl < b->d_matl; }

   private:

    int d_matl;  // -2 = uninitialized... all materials = -1.  All BoundCondBase's must be of the same material.

    std::vector<const BoundCondBase*> d_BCBs;
    
    BCData( const BCData & ); // Don't use...  Use a reference, ie:  BCData & new_bcd = BCDataArray->getBCData();

    // Don't call this...
    BCData& operator=( const BCData & );

  };

} // End namespace Uintah

#endif


