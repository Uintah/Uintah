//----- MeanMixingModel.h --------------------------------------------------

#ifndef Uintah_Component_Arches_MeanMixingModel_h
#define Uintah_Component_Arches_MeanMixingModel_h

/***************************************************************************
CLASS
    MeanMixingModel
       The MeanMixingModel class sets up a mixing table that contains mean
       values only for all the state space variables.
       
GENERAL INFORMATION
    MeanMixingModel.h - Declaration of MeanMixingModel class

    Author: Jennifer Spinti (spinti@crsim.utah.edu)
    
    Creation Date : 02-04-2002

    C-SAFE
    
    Copyright U of U 2002

KEYWORDS
    
DESCRIPTION
    The MeanMixingModel class is derived from the MixingModel base class. This
    class creates a dynamic mixing table based on mean values. Hence, when this
    class is used, there is no subgrid scale mixing model. All tabulated state 
    space variables are mean values only and are tabulated as functions of the
    independent variables (e.g. mixture fraction, enthalpy, reaction progress
    variables).
 
PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
    None
***************************************************************************/

#include <Packages/Uintah/CCA/Components/Arches/Mixing/Stream.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/MixingModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/DynamicTable.h>
#include <Packages/Uintah/CCA/Components/Arches/Mixing/Integrator.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  class ReactionModel;
  class MixRxnTable;
  class MixRxnTableInfo;
  class InletStream;
  
class MeanMixingModel: public MixingModel, public DynamicTable {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructs an instance of MeanMixingModel
      //   [in] 
      //
      MeanMixingModel();

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Destructor
      //
      virtual ~MeanMixingModel();

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

      /////////////////////////////////////////////////////////////////////////
      // speciesStateSpace returns the state space (dependent) variables,
      // including species composition, given a set of mixture fractions. The 
      // species composition of each stream must be known.
      // The state space variables returned are: density, temperature, heat 
      // capacity, molecular weight, enthalpy, mass fractions 
      // All variables are in SI units with pressure in Pascals.
      // Parameters:
      // [in] mixVar is an array of independent variables
      virtual Stream speciesStateSpace(const std::vector<double>& mixVar);


      // GROUP: Get Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      // Get the number of mixing variables
      //
      inline bool isAdiabatic() const{ 
	return d_adiabatic; 
      }
      inline int getNumMixVars() const{ 
	return d_numMixingVars; 
      }    
      inline int getNumMixStatVars() const{
	return 0;
      }
      inline int getNumRxnVars() const{
	return d_numRxnVars;
      }
      inline int getTableDimension() const{
	return d_tableDimension;
      }
      inline std::string getMixTableType() const{
	return d_tableType;
      }
      //***warning** compute totalvars from number of species and dependent vars
      inline int getTotalVars() const {
	return d_depStateSpaceVars;
      }
      inline ReactionModel* getRxnModel() const {
	return d_rxnModel;
      }
      inline Integrator* getIntegrator() const {
      }  

protected :

private:

      ///////////////////////////////////////////////////////////////////////
      //
      // Copy Constructor (never instantiated)
      //   [in] 
      //        const MeanMixingModel&   
      //
      MeanMixingModel(const MeanMixingModel&);

      // GROUP: Operators Not Instantiated:
      ///////////////////////////////////////////////////////////////////////
      //
      // Assignment Operator (never instantiated)
      //   [in] 
      //        const MeanMixingModel&   
      //
      MeanMixingModel& operator=(const MeanMixingModel&);

private:
      // Looks for needed entry in table and returns that entry. If entry 
      // does not exist and table is dynamic, it calls the reaction model
      // to compute the entry.
      void tableLookUp(int* tableKeyIndex, Stream& stateSpaceVars);
      void convertKeytoFloatValues(int tableKeyIndex[], std::vector<double>& indepVars);
      void computeMeanValues(int* tableKeyIndex, Stream& meanStream);
   
      MixRxnTableInfo* d_tableInfo;
      int d_numMixingVars;
      int d_numRxnVars;
      int d_depStateSpaceVars;
      bool d_adiabatic;
      int d_CO2index;
      int d_H2Oindex;
      std::vector<Stream> d_streams; 
      std::string d_tableType;
      int d_tableDimension;
      // Data structure class that stores the table entries for state-space
      // variables as a function of independent variables.
      // Implemented either as a k-d tree or 2D vector data structure.
      MixRxnTable* d_mixTable;
      ReactionModel* d_rxnModel;
      

}; // end class MeanMixingModel

} // end namespace Uintah

#endif
