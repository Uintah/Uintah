//----- NewStaticMixingTable.h --------------------------------------------------

#ifndef Uintah_Component_Arches_NewStaticMixingTable_h
#define Uintah_Component_Arches_NewStaticMixingTable_h

/***************************************************************************
CLASS
    NewStaticMixingTable 
       
GENERAL INFORMATION
    NewStaticMixingTable.h - Declaration of NewStaticMixingTable class

    Author: Padmabhushana R Desam (desam@crsim.utah.edu)
    
    Creation Date : 08-14-2003

    C-SAFE
    
    Copyright U of U 2000

KEYWORDS
    Mixing Table 
DESCRIPTION
      Reads and interpolates the mixing reaction tables created to standard format specifications 
      Currently supports non-adiabatic equilibrium tables(3-Dimensions). 
    
PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
        Read and interpolate 4-D tables e.g., non-adiabatic flamelets/extent of reaction tables 
***************************************************************************/

#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixingModel.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <sgi_stl_warnings_off.h>
#include   <vector>
#include   <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

class InletStream;
class ExtraScalarSolver;
   
class NewStaticMixingTable: public MixingModel {

public:

  // GROUP: Constructors:
  ///////////////////////////////////////////////////////////////////////
  //
  // Constructs an instance of NewStaticMixingTable
  //
  NewStaticMixingTable( bool calcReactingScalar,
                        bool calcEnthalpy,
                        bool calcVariance,
                        const ProcessorGroup* myworld );
  
  // GROUP: Destructors :
  ///////////////////////////////////////////////////////////////////////
  //
  // Destructor
  //
  virtual ~NewStaticMixingTable();

  // GROUP: Problem Setup :
  ///////////////////////////////////////////////////////////////////////
  //
  // Set up the problem specification database
  //
  virtual void problemSetup( const ProblemSpecP& params );

  // GROUP: Actual Action Methods :
  ///////////////////////////////////////////////////////////////////////
  //
  // Carry out actual computation of properties
  //
  virtual void computeProps( const InletStream& inStream,
                             Stream& outStream );


  // GROUP: Get Methods :
  ///////////////////////////////////////////////////////////////////////
  //
  inline bool getCOOutput() const{
    return d_co_output;
  }
  inline bool getSulfurChem() const{
    return d_sulfur_chem;
  }
  inline bool getTabulatedSoot() const{
    return d_tabulated_soot;
  }
  
  inline bool getSootPrecursors() const{
    return d_soot_precursors;
  }
  inline double getAdiabaticAirEnthalpy() const{
    return d_H_air;
  }
  inline double getCarbonFuel() const{
    return d_carbon_fuel;
  }
  inline double getCarbonAir() const{
    return d_carbon_air;
  }
  inline double getFStoich() const{
    return d_f_stoich;
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
  //        const NewStaticMixingTable&   
  //
  NewStaticMixingTable(const NewStaticMixingTable&);
  
  // GROUP: Operators Not Instantiated:
  ///////////////////////////////////////////////////////////////////////
  //
  // Assignment Operator (never instantiated)
  //   [in] 
  //        const NewStaticMixingTable&   
  //
  NewStaticMixingTable& operator=(const NewStaticMixingTable&);

private:
  // Looks for needed entry in table and returns that entry. If entry 
  // does not exist and table is dynamic, it calls integrator to compute
  // entry before returning it. If table is static and entry is non-existent,
  // it exits program.
  bool d_calcReactingScalar, d_calcEnthalpy, d_calcVariance;
  double tableLookUp(double mixfrac, double mixfracVars, double heat_loss, int var_index); 
  void readMixingTable( const std::string & inputfile );
  int d_tableDimension;
  std::vector <std::vector <double> > table;
  int d_heatlosscount, d_mixfraccount, d_mixvarcount, d_varscount;
  int co2_index, h2o_index;
  
  int h2s_index, so2_index, so3_index, sulfur_index;
  int s2_index, sh_index, so_index, hso2_index;
  int hoso_index, hoso2_index, sn_index, cs_index;
  int ocs_index, hso_index, hos_index, hsoh_index;
  int h2so_index, hosho_index, hs2_index, h2s2_index;

  int co_index, ch4_index, c2h2_index;
  int co2rate_index, so2rate_index;
  bool d_co_output;
  bool d_sulfur_chem;
  bool d_soot_precursors;
  int soot_index;
  bool d_tabulated_soot;

  std::vector <std::vector<double> > meanMix;
  std::vector<double> heatLoss;
  std::vector<double> variance;
  std::vector<int> eachindepvarcount;
  std::vector<std::string> indepvars_names;
  std::vector<std::string> vars_names;
  std::vector<std::string> vars_units;
  int d_indepvarscount;
  int Hl_index, F_index, Fvar_index;
  int T_index, Rho_index, Cp_index, Hs_index;
  double d_H_fuel, d_H_air;
  double d_f_stoich, d_carbon_fuel, d_carbon_air;
  bool d_calcExtraScalars;
  std::vector<ExtraScalarSolver*>* d_extraScalars;
  
  const ProcessorGroup* d_myworld;
}; // end class NewStaticMixingTable
  
} // end namespace Uintah

#endif
