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
    Revised: Jennifer Spinti (spinti@crsim.utah.edu)
    
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

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  class ReactionModel;
  class MixRxnTable;
  class MixRxnTableInfo;
  class Integrator;
  class InletStream;
  class ChemkinInterface;
  //Limit of variance for which integration will be performed
  const double VARLIMIT = 0.7;
  
class PDFMixingModel: public MixingModel, public DynamicTable {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructs an instance of PDFMixingModel
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
	return d_numMixStatVars;
      }
      inline int getNumRxnVars() const{
	return d_numRxnVars;
      }
      inline int getTableDimension() const{
	return d_tableDimension;
      }
      inline double getStoicPt() const{ 
	return d_stoicValue; 
      }
      inline std::string getMixTableType() const{
	return d_tableType;
      }
      inline std::string getPDFShape() const {
	return d_pdfShape;
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
      // Looks for needed entry in table and returns that entry. If entry 
      // does not exist and table is dynamic, it calls integrator to compute
      // entry before returning it. If table is static and entry is non-existent,
      // it exits program.
      void tableLookUp(int* tableKeyIndex, Stream& stateSpaceVars);
      // Reads static data files created by James' tableGenerator program
      void readStaticTable();
      // Read static data files created by createBetaTable function
      void readBetaTable();
      // Creates static table upon call to problemSetup; hrad coded for one f, gf
      void createBetaTable();
      void convertKeyToFloatValues(int tableKeyIndex[], std::vector<double>& indepVars);
     
      MixRxnTableInfo* d_tableInfo;
      int d_numMixingVars;
      int d_numMixStatVars;
      int d_numRxnVars;
      double d_stoicValue;
      std::string d_pdfShape;
      int d_CO2index;
      int d_H2Oindex;
      std::string d_tableType;	 
      int d_depStateSpaceVars;
      bool d_adiabatic;
      std::vector<Stream> d_streams; 
      int d_tableDimension;
      bool d_dynamic; //If true, table is dynamic; if false, table is static

      // Data structure class that stores the table entries for state-space
      // variables as a function of independent variables.
      // Implemented either as a k-d tree or 2D vector data structure.
      MixRxnTable* d_mixTable;
      Integrator* d_integrator;
      ReactionModel* d_rxnModel;
      ChemkinInterface* d_chemInterf; 
      

}; // end class PDFMixingModel

} // end namespace Uintah

#endif








