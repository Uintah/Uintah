#ifndef Uintah_Components_Arches_BoundaryCondition_new_h
#define Uintah_Components_Arches_BoundaryCondition_new_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <CCA/Components/Arches/Directives.h>

//===========================================================================

/**
*   @class BoundaryCondition 
*   @author Jeremy Thornock
*   @brief This class sets the boundary conditions for scalars. 
*
*/

namespace Uintah {

class MixingRxnModel; 
class BoundaryCondition_new {

public: 

  //** WARNING: This needs to be duplicated in BoundaryCondition.h for now until BoundaryCondition goes away **//
  enum BC_TYPE { VELOCITY_INLET, MASSFLOW_INLET, VELOCITY_FILE, MASSFLOW_FILE, PRESSURE, OUTLET, WALL, MMWALL, INTRUSION, SWIRL, TURBULENT_INLET }; 

  typedef std::map<IntVector, double> CellToValueMap; 
  typedef std::map<Patch*, vector<CellToValueMap> > PatchToBCValueMap; 
  typedef std::map<std::string, CellToValueMap> ScalarToBCValueMap; 
  typedef std::map< std::string, const VarLabel* > LabelMap; 
  typedef std::map< std::string, double  > DoubleMap; 
  typedef std::map< std::string, DoubleMap > MapDoubleMap;


  BoundaryCondition_new(const int matl_id);

  ~BoundaryCondition_new();
  /** @brief Interface for the input file and set constants */ 
  void  problemSetup( ProblemSpecP& db, std::string eqn_name );

	/** @brief Interface for setting up tabulated BCs */
  void setupTabulatedBC( ProblemSpecP& db, std::string eqn_name, MixingRxnModel* table );

  /** @brief This method sets the boundary value of a scalar to 
             a value such that the interpolated value on the face results
             in the actual boundary condition. */   
  void setScalarValueBC(const ProcessorGroup*,
                        const Patch* patch,
                        CCVariable<double>& scalar, 
                        string varname );
  /** @brief This method set the boundary values of a vector to a 
   * value such that the interpolation or gradient computed between the 
   * interior cell and boundary cell match the boundary condition. */ 
  void setVectorValueBC( const ProcessorGroup*,
    const Patch* patch,
    CCVariable<Vector>& vec, 
    string varname );
  /** @brief This method set the boundary values of a vector to a 
   * value such that the interpolation or gradient computed between the 
   * interior cell and boundary cell match the boundary condition. This is 
   * a specialized case where the boundary value comes from some other vector */
  void setVectorValueBC( const ProcessorGroup*,
    const Patch* patch,
    CCVariable<Vector>& vec, constCCVariable<Vector>& const_vec, 
    string varname );

  /** @brief Sets the area fraction for each minus face according to the boundaries */
  void setAreaFraction( 
    const Patch* patch,
    CCVariable<Vector>& areaFraction, 
    CCVariable<double>& volFraction, 
    constCCVariable<int>& pcell, 
    const int wallType, 
    const int flowType );

  /** @brief Compute the volume weights for the filter cell **/
  void computeFilterVolume( const Patch* patch, 
                            constCCVariable<int>&    cellType, 
                            CCVariable<double>& filterVolume ); 

  void sched_assignTabBCs( SchedulerP& sched, 
                           const PatchSet* patches, 
                           const MaterialSet* matls,
                           const std::string eqnName );

  /** @brief Read in a file for boundary conditions **/ 
  std::map<IntVector, double> readInputFile( std::string file_name ); 

  //new stuff--------------------
  class BCFunctionBase{ 

    public: 

      BCFunctionBase() {};
      virtual ~BCFunctionBase(){}; 

      virtual void setupBC( ProblemSpecP& db, const std::string eqn_name ) = 0; 
      virtual void applyBC( const Patch* patch, Patch::FaceType face, int child, std::string varname, std::string face_name, CCVariable<double>& phi ) = 0; 
 
    protected: 
    
  };

private: 

  typedef std::map< std::string, map<std::string, BCFunctionBase* > > VarToMappedF; 

  class Dirichlet : public BCFunctionBase { 

    public:

      Dirichlet( const int matl_id ) : d_matl_id( matl_id ){}; 
      ~Dirichlet(){};

      void setupBC( ProblemSpecP& db, const std::string eqn_name ){}; 
      void applyBC( const Patch* patch, Patch::FaceType face, int child, std::string varname, std::string face_name, CCVariable<double>& phi); 

    private: 

      const int d_matl_id; 

  }; 

  class Neumann : public BCFunctionBase { 

    public:

      Neumann( const int matl_id ) : d_matl_id( matl_id ){}; 
      ~Neumann(){};

      void setupBC( ProblemSpecP& db, const std::string eqn_name ){}; 
      void applyBC( const Patch* patch, Patch::FaceType face, int child, std::string varname, std::string face_name, CCVariable<double>& phi); 

    private: 

      const int d_matl_id; 

  }; 

  class FromFile : public BCFunctionBase { 

    public:

      FromFile( const int matl_id ) : d_matl_id( matl_id ){}; 
      ~FromFile(){};

      void setupBC( ProblemSpecP& db, const std::string eqn_name ); 
      void applyBC( const Patch* patch, Patch::FaceType face, int child, std::string varname, std::string face_name, CCVariable<double>& phi); 

    private: 

      typedef std::map<IntVector, double> CellToValueMap;                 ///< (i,j,k)   ---> boundary condition value 
      typedef std::map<std::string, CellToValueMap> FaceToBCValueMap;     ///< face name ---> CellToValueMap 

      const int d_matl_id; 
      
      std::map<IntVector, double> readInputFile( std::string file_name );
      FaceToBCValueMap d_face_map; 

  }; 

  class Tabulated : public BCFunctionBase { 

    public:

      Tabulated( const int matl_id ) : d_matl_id( matl_id ){}; 
      ~Tabulated(){};

      void setupBC( ProblemSpecP& db, const std::string eqn_name ); 
      void extra_setupBC( ProblemSpecP& db, const std::string eqn_name, MixingRxnModel* table ); 
      void applyBC( const Patch* patch, Patch::FaceType face, int child, std::string varname, std::string face_name, CCVariable<double>& phi); 

    private: 

      typedef std::map< std::string, double  > DoubleMap;                   ///< dependant var ---> value
      typedef std::map< std::string, DoubleMap > MapDoubleMap;              ///< face name  ---> DoubleMap

      const int d_matl_id; 
      
      MapDoubleMap _tabVarsMap; 

  }; 
  //-----------------------------
 
  //variables
	const int d_matl_id; 

  LabelMap           areaMap;
  MapDoubleMap       _tabVarsMap;
  ScalarToBCValueMap scalar_bc_from_file; 

  void assignTabBCs( const ProcessorGroup*, 
                     const PatchSubset* patches, 
                     const MaterialSubset*, 
                     DataWarehouse*, 
                     DataWarehouse* new_dw,
                     const std::string eqnName );





}; // class BoundaryCondition_new
} // namespace Uintah

#endif 
