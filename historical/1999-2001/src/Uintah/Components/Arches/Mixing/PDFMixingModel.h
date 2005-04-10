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

#include <Uintah/Components/Arches/ArchesLabel.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/MixingModel.h>
#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/CFDInterface.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/VarLabel.h>
#include <SCICore/Geometry/IntVector.h>

#include <vector>

namespace Uintah {
namespace ArchesSpace {

class PDFMixingModel: public MixingModel {

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
      virtual void computeProps(InletStream& inStream,
				Stream& outStream);

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
      // recursive function to linearly interpolate from the KDTree
      Stream& interpolate(int currentDim, int* lowIndex, int* upIndex,
				REAL* lowFactor, REAL* upFactor);
      Stream& tableLookUp(int* tableKeyIndex);

      int d_numMixingVars;
      int d_numMixStatVars;
      int d_rxnVars;
      bool d_adiabatic;
      std::vector<Stream> d_streams; 
      int d_tableDimension;
      // two dimensional arrays for storing information for linear interpolation
      int **d_tableIndexVec;
      double **d_tableBoundsVec;

    // Stores the table information
      MixRxnTableInfo* d_tableInfo;
    // Data structure class that stores the table entries for state-space variables
    // as a function of independent variables.
    // This could be implemented either as a k-d or a binary tree data structure. 
      KD_Tree* d_dynamicTable;
      Integrator* d_integrator;
      ReactionModel* d_rxnModel;
      

}; // end class PDFMixingModel

} // end namespace ArchesSpace
} // end namespace Uintah

#endif

//
// $Log$
// Revision 1.1  2001/01/15 23:38:21  rawat
// added some more classes for implementing mixing model
//
// Revision 1.1  2000/12/18 17:53:10  rawat
// adding mixing model for reacting flows
//
//
