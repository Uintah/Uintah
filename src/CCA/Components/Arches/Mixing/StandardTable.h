/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

//----- StandardTable.h --------------------------------------------------

#ifndef Uintah_Component_Arches_StandardTable_h
#define Uintah_Component_Arches_StandardTable_h

/***************************************************************************
CLASS
    StandardTable 
       
GENERAL INFORMATION
    StandardTable.h - Declaration of StandardTable class

    Author: Padmabhushana R Desam (desam@crsim.utah.edu)
    
    Creation Date : 08-14-2003

    C-SAFE
    
    
KEYWORDS
    Mixing Table 
DESCRIPTION
      Reads the mixing reaction tables created by standalone programs. The current implementation
      is to read specific tables. Future revisons will focus on making this more generic and
      different mixing tables will be standarized with an interface. 
    
PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
    Making it more generic to accomodate different mixing table formats
 
***************************************************************************/

#include <CCA/Components/Arches/Mixing/Stream.h>
#include <CCA/Components/Arches/Mixing/MixingModel.h>

#include <vector>
#include <string>

namespace Uintah {
  class InletStream;
  class TableInterface;
  class VarLabel;
  class ExtraScalarSolver;
  
class StandardTable: public MixingModel{

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructs an instance of StandardTable
      //
      StandardTable(bool calcReactingScalar,
                    bool calcEnthalpy,
                    bool calcVariance);

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Destructor
      //
      virtual ~StandardTable();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      //
      // Set up the problem specification database
      //
      virtual void problemSetup(const ProblemSpecP& params);

      // GROUP: Actual Action Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Carry out actual computation of properties
      //
      virtual void computeProps(const InletStream& inStream,
                                Stream& outStream);

      // GROUP: Get Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      inline bool getCOOutput() const{
        return 0;
      }
      inline bool getSulfurChem() const{
        return 0;
      }
      inline bool getSootPrecursors() const{
        return 0;
      }
      inline bool getTabulatedSoot() const{
        return 0;
      }
      inline double getAdiabaticAirEnthalpy() const{
        return d_H_air;
      }

      inline double getFStoich() const{
        return 0.0;
      }

      inline double getCarbonFuel() const{
        return 0.0;
      }

      inline double getCarbonAir() const{
        return 0.0;
      }

      inline void setCalcExtraScalars(bool calcExtraScalars) {
        d_calcExtraScalars=calcExtraScalars;
      }

      inline void setExtraScalars(std::vector<ExtraScalarSolver*>* extraScalars) {
        d_extraScalars = extraScalars;
      }

protected :

private:

      ///////////////////////////////////////////////////////////////////////
      //
      // Copy Constructor (never instantiated)
      //   [in] 
      //        const StandardTable&   
      //
      StandardTable(const StandardTable&);

      // GROUP: Operators Not Instantiated:
      ///////////////////////////////////////////////////////////////////////
      //
      // Assignment Operator (never instantiated)
      //   [in] 
      //        const StandardTable&   
      //
      StandardTable& operator=(const StandardTable&);

private:
      bool d_calcReactingScalar, d_calcEnthalpy, d_calcVariance;
      int co2_index, h2o_index;
      int T_index, Rho_index, Cp_index, Enthalpy_index, Hs_index;
      double d_H_fuel, d_H_air;
      bool d_adiab_enth_inputs;

    TableInterface* table;
    struct TableValue {
      std::string name;
      int index;
      VarLabel* label;
    };
    std::vector<TableValue*> tablevalues;
    bool d_calcExtraScalars;
    std::vector<ExtraScalarSolver*>* d_extraScalars;


}; // end class StandardTable

} // end namespace Uintah

#endif








