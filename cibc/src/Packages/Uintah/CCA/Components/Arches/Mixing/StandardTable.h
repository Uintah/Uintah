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
    
    Copyright U of U 2000

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

#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixingModel.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  class InletStream;
  class TableInterface;
  class VarLabel;
  
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
      inline double getAdiabaticAirEnthalpy() const{
	return d_H_air;
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
}; // end class StandardTable

} // end namespace Uintah

#endif








