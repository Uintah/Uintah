//----- PDFMixingModel.h --------------------------------------------------

#ifndef Uintah_Component_Arches_PDFMixingModel_h
#define Uintah_Component_Arches_PDFMixingModel_h

/***************************************************************************
CLASS
    PDFMixingModel
       Sets up the PDFMixingModel ????
       
GENERAL INFORMATION
    PDFMixingModel.h - Declaration of PDFMixingModel class

    Author: Rajesh Rawat (rawat@crsim.utah.edu)
    
    Creation Date : 05-30-2000

    C-SAFE
    
    Copyright U of U 2000

KEYWORDS
    
DESCRIPTION

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

#include <vector>

namespace Uintah {
class ReactionModel;
class KD_Tree;
class MixRxnTableInfo;
class Integrator;

class PDFMixingModel: public MixingModel, public DynamicTable {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructor taking
      //   [in] 
      //
      PDFMixingModel();

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Destructor
      //
      virtual ~PDFMixingModel();

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
      Stream speciesStateSpace(const std::vector<double>& mixVar);


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
	return d_numMixStatVars;
      }
      inline int getNumRxnVars() const{
	return d_numRxnVars;
      }
      inline int getTableDimension() const{
	return d_tableDimension;
      }
      //***warning** compute totalvars from number of species and dependent vars
      inline int getTotalVars() const {
	return d_depStateSpaceVars;
      }
      inline ReactionModel* getRxnModel() const {
	return d_rxnModel;
      }
      
      inline Integrator* getIntegrator() const {
	return d_integrator;
      }

protected :

private:

      ///////////////////////////////////////////////////////////////////////
      //
      // Copy Constructor (never instantiated)
      //   [in] 
      //        const PDFMixingModel&   
      //
      PDFMixingModel(const PDFMixingModel&);

      // GROUP: Operators Not Instantiated:
      ///////////////////////////////////////////////////////////////////////
      //
      // Assignment Operator (never instantiated)
      //   [in] 
      //        const PDFMixingModel&   
      //
      PDFMixingModel& operator=(const PDFMixingModel&);

private:
      // Looks for needed entry in KDTree and returns that entry. If entry 
      // does not exist, calls integrator to compute entry before returning it.
      Stream tableLookUp(int* tableKeyIndex);

      MixRxnTableInfo* d_tableInfo;
      int d_numMixingVars;
      int d_numMixStatVars;
      int d_numRxnVars;
      int d_depStateSpaceVars;
      bool d_adiabatic;
      std::vector<Stream> d_streams; 
      int d_tableDimension;
      // two dimensional arrays for storing information for linear interpolation
      int **d_tableIndexVec;
      double **d_tableBoundsVec;

      // Data structure class that stores the table entries for state-space
      // variables as a function of independent variables.
      // This could be implemented either as a k-d or a binary tree data structure.
      KD_Tree* d_mixTable;
      // Class that accesses data structure (k-d or binary tree) 
      //DynamicTable* d_mixTableAccess;
      Integrator* d_integrator;
      ReactionModel* d_rxnModel;
      

}; // end class PDFMixingModel

} // end namespace Uintah

#endif


