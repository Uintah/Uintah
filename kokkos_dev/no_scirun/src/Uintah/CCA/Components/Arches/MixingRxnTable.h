/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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


//----- MixingRxnTable.h --------------------------------------------------

#ifndef Uintah_Component_Arches_MixingRxnTable_h
#define Uintah_Component_Arches_MixingRxnTable_h

// constructor
// destructor
// problemSetup
// getState
// verifyTable
// getIndepVars
// getDepVars

// Uintah includes
#include <Uintah/Core/ProblemSpec/ProblemSpec.h>

// C++ includes
#include     <vector>
#include     <map>
#include     <string>
#include     <stdexcept>

/***************************************************************************
CLASS
    MixingRxnTable 
       
GENERAL INFORMATION
    MixingRxnTable.h - representation of reaction and mixing table data 
    structure (typically an external file in a pre-specified format).

    Author: Charles Reid (charles.reid@utah.edu)
    
    Creation Date : 11-22-2008

    C-SAFE
    
    Copyright U of U 2008

KEYWORDS
    Mixing Table 

DESCRIPTION
    This MixingRxnTable class provides a representation of the mixing 
    and reaction table inside the Arches code (specifically, Properties.cc).  
    The MixingRxnTable class is a generic class that allows for child classes 
    that each provide a specific representation of specific mixing and 
    reaction table formats.  That way, we can use a standard table format, 
    but leave ourselves the flexibility to move to other reaction and mixing 
    table formats in the future.

    Tables can be created in any number of programs (DARS, Cantera, TabProps, 
    etc.).  Implementing this table reader is part of an effort to convert our 
    current table format (which does not use a universal format) to an HDF5 
    B-Splined format, which is what TabProps creates (TabProps is a code 
    written by Dr. James Sutherland and is available through the CRSim software
    SVN repository).

PATTERNS
    None

WARNINGS
    For now, the MixingRxnTable object will be able to read several table formats 
    (or, at least, it will not be the only method available for reading tables).  
    However, ultimately the MixingRxnTable object will be the only interface 
    between Arches and tables, and will only interface with a single, common, 
    uniform table format (HDF5).  However, the format of the MixingRxnTable is 
    such that a new child class can easily be created if a different table 
    format is decided upon.

NOTES

POSSIBLE REVISIONS

***************************************************************************/

namespace Uintah {
class Properties; 
class MixingRxnTable{

public:

  // GROUP: Constructors:
  // Constructs an instance of MixingRxnTable
  MixingRxnTable();

  
  // GROUP: Destructors :
  // Destructor
  virtual ~MixingRxnTable();


  // GROUP: Problem Setup :
  // Set up the problem specs (from the input file)
  // Get table name, then read table data into object
  virtual void problemSetup( const ProblemSpecP& params );


  // GROUP: Actual Action Methods :
  // Actually obtain properties
  virtual const std::vector<double> getState( const double * indepVarValues );


  // GROUP: Verify Methods :
  // Methods used in verifying the table
  virtual void verifyTable( bool diagnosticMode,
                            bool strictMode );


  // GROUP: Get Methods :
  // Get non-state space information from the table

  // Dependent Variables:

  // This getDepVars uses a table whose data has already been loaded
  virtual const std::vector<std::string> & getDepVars();

  // Independent Variables:

  // This getIndepVars uses a table whose data has already been loaded
  virtual const std::vector<std::string> & getIndepVars();

protected :

private:

}; // end class MixingRxnTable
  
} // end namespace Uintah

#endif
