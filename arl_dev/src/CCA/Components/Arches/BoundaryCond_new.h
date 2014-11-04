#ifndef Uintah_Components_Arches_BoundaryCondition_new_h
#define Uintah_Components_Arches_BoundaryCondition_new_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <CCA/Components/Arches/Directives.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <Core/Exceptions/InvalidValue.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <boost/shared_ptr.hpp>

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
  //** WARNING!!! ** // 
  enum BC_TYPE { VELOCITY_INLET, MASSFLOW_INLET, VELOCITY_FILE, MASSFLOW_FILE, STABL, PRESSURE, 
    OUTLET, NEUTRAL_OUTLET, WALL, MMWALL, INTRUSION, SWIRL, TURBULENT_INLET }; 
  //** END WARNING!!! **//

  typedef std::map<IntVector, double> CellToValueMap; 
  typedef std::map<Patch*, std::vector<CellToValueMap> > PatchToBCValueMap;
  typedef std::map< std::string, const VarLabel* > LabelMap; 
  typedef std::map< std::string, double  > DoubleMap; 
  typedef std::map< std::string, DoubleMap > MapDoubleMap;
  struct FFInfo{ 
    CellToValueMap values;
    Vector relative_xyz;
    double dx; 
    double dy;
    IntVector relative_ijk;
    std::string default_type;
    std::string name; 
    double default_value;
  }; 
  typedef std::map<std::string, FFInfo> ScalarToBCValueMap; 

  BoundaryCondition_new(const int matl_id);

  ~BoundaryCondition_new();
  /** @brief Interface for the input file and set constants */ 
  void  problemSetup( ProblemSpecP& db, std::string eqn_name );

	/** @brief Interface for setting up tabulated BCs */
  void setupTabulatedBC( ProblemSpecP& db, std::string eqn_name, MixingRxnModel* table );

  /** @brief Create spatial masks **/
  void sched_create_masks(const LevelP& level, SchedulerP& sched, const MaterialSet* matls);
  /** @brief see sched_create_masks **/
  void create_masks(const ProcessorGroup* pg,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw);

  /** @brief This method sets the boundary value of a scalar to 
             a value such that the interpolated value on the face results
             in the actual boundary condition. Note that the boundary condition 
             from the input file can be overridden by the last two arguments. */   
  void setScalarValueBC(const ProcessorGroup*,
                        const Patch* patch,
                        CCVariable<double>& scalar, 
                        const std::string varname, 
                        bool  change_bc=false, 
                        const std::string override_bc="NA")
  {


    using std::vector; 
    using std::string; 

    // This method sets the value of the scalar in the boundary cell
    // so that the boundary condition set in the input file is satisfied. 
    vector<Patch::FaceType>::const_iterator iter;
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);
    Vector Dx = patch->dCell(); 
  
    for (iter = bf.begin(); iter !=bf.end(); iter++){
      Patch::FaceType face = *iter;
  
      //get the face direction
      IntVector insideCellDir = patch->faceDirection(face);
      //get the number of children
      int numChildren = patch->getBCDataArray(face)->getNumberChildren(d_matl_id); //assumed one material
  
      for (int child = 0; child < numChildren; child++){
  
        double bc_value = -9; 
        Vector bc_v_value(0,0,0); 
        std::string bc_s_value = "NA";
  
        Iterator bound_ptr;
        string bc_kind = "NotSet"; 
        string face_name; 
        getBCKind( patch, face, child, varname, d_matl_id, bc_kind, face_name ); 
  
        if ( change_bc == true ){ 
          bc_kind = override_bc; 
        }
  
        bool foundIterator = "false"; 
        if ( bc_kind == "Tabulated" || bc_kind == "FromFile" ){ 
          foundIterator = 
            getIteratorBCValue<std::string>( patch, face, child, varname, d_matl_id, bc_s_value, bound_ptr ); 
        } else {
          foundIterator = 
            getIteratorBCValue<double>( patch, face, child, varname, d_matl_id, bc_value, bound_ptr ); 
        } 
  
        if (foundIterator) {
          // --- notation --- 
          // bp1: boundary cell + 1 or the interior cell one in from the boundary
          if (bc_kind == "Dirichlet") {
  
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector bp1(*bound_ptr - insideCellDir);
              scalar[*bound_ptr] = 2.0*bc_value - scalar[bp1];
            }
  
          } else if (bc_kind == "Neumann") {
  
            IntVector axes = patch->getFaceAxes(face);
            int P_dir = axes[0];  // principal direction
            double plus_minus_one = (double) patch->faceDirection(face)[P_dir];
            double dx = Dx[P_dir];
            
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector bp1(*bound_ptr - insideCellDir); 
              scalar[*bound_ptr] = scalar[bp1] + plus_minus_one * dx * bc_value;
            }
          } else if (bc_kind == "FromFile") { 
  
            ScalarToBCValueMap::iterator i_scalar_bc_storage = scalar_bc_from_file.find( face_name ); 
  
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
  
              IntVector rel_bc = *bound_ptr - i_scalar_bc_storage->second.relative_ijk; 
              CellToValueMap::iterator iter = i_scalar_bc_storage->second.values.find( rel_bc ); //<----WARNING ... May be slow here
              if ( iter != i_scalar_bc_storage->second.values.end() ){ 
  
                double file_bc_value = iter->second; 
                IntVector bp1(*bound_ptr - insideCellDir);
                scalar[*bound_ptr] = 2.0 * file_bc_value - scalar[bp1]; 
  
              } else if ( i_scalar_bc_storage->second.default_type == "Neumann" ){  
          
                IntVector axes = patch->getFaceAxes(face);
                int P_dir = axes[0];  // principal direction
                double plus_minus_one = (double) patch->faceDirection(face)[P_dir];
                double dx = Dx[P_dir];
                IntVector bp1(*bound_ptr - insideCellDir); 
                scalar[*bound_ptr] = scalar[bp1] + plus_minus_one * dx * i_scalar_bc_storage->second.default_value;
  
              } else if ( i_scalar_bc_storage->second.default_type == "Dirichlet" ){ 
  
                IntVector bp1(*bound_ptr - insideCellDir); 
                scalar[*bound_ptr] = 2.0*i_scalar_bc_storage->second.default_value - scalar[bp1];
  
              } 
            }
          } else if ( bc_kind == "Tabulated") {
  
            MapDoubleMap::iterator i_face = _tabVarsMap.find( face_name );
  
            if ( i_face != _tabVarsMap.end() ){ 
  
              DoubleMap::iterator i_var = i_face->second.find( varname ); 
              double tab_bc_value = i_var->second;
  
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir);
                scalar[*bound_ptr] = 2.0 * tab_bc_value - scalar[bp1];
              }
  
            }
          } else if ( bc_kind == "ForcedDirichlet") {
            /* A Dirichlet condition to fix the cell value rather than use interpolate
            This is required to use for cqmom with velocities as internal coordiantes,
            and may help with some radiation physics */
            
            //Here the extra cell should be set to the face value so that the cqmom inversion
            //doesn't return junk, with upwinding of the abscissas this should return correct face value
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              scalar[*bound_ptr] = bc_value;
            }
          } else { 
            throw InvalidValue( "Error: Cannot determine boundary condition type for variable: "+varname, __FILE__, __LINE__);
          }
        }
      }
    }
  }
  /** @brief This method sets the boundary value of a scalar to 
             a value such that the interpolated value on the face results
             in the actual boundary condition. Note that the boundary condition 
             from the input file can be overridden by the last two arguments. */   
  void setExtraCellScalarValueBC(const ProcessorGroup*,
                                 const Patch* patch,
                                 CCVariable<double>& scalar, 
                                 const std::string varname, 
                                 bool  change_bc=false, 
                                 const std::string override_bc="NA")
  {

    using std::vector; 
    using std::string; 

    vector<Patch::FaceType>::const_iterator iter;
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);
    Vector Dx = patch->dCell(); 
  
    for (iter = bf.begin(); iter !=bf.end(); iter++){
      Patch::FaceType face = *iter;
  
      //get the face direction
      IntVector insideCellDir = patch->faceDirection(face);
      //get the number of children
      int numChildren = patch->getBCDataArray(face)->getNumberChildren(d_matl_id); //assumed one material
  
      for (int child = 0; child < numChildren; child++){
  
        double bc_value = -9; 
        Vector bc_v_value(0,0,0); 
        std::string bc_s_value = "NA";
  
        Iterator bound_ptr;
        string bc_kind = "NotSet"; 
        string face_name; 
        getBCKind( patch, face, child, varname, d_matl_id, bc_kind, face_name ); 
  
        if ( change_bc == true ){ 
          bc_kind = override_bc; 
        }
  
        bool foundIterator = "false"; 
        if ( bc_kind == "Tabulated" || bc_kind == "FromFile" ){ 
          foundIterator = 
            getIteratorBCValue<std::string>( patch, face, child, varname, d_matl_id, bc_s_value, bound_ptr ); 
        } else {
          foundIterator = 
            getIteratorBCValue<double>( patch, face, child, varname, d_matl_id, bc_value, bound_ptr ); 
        } 
  
        if (foundIterator) {
          // --- notation --- 
          // bp1: boundary cell + 1 or the interior cell one in from the boundary
          if (bc_kind == "Dirichlet") {
  
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector bp1(*bound_ptr - insideCellDir);
              scalar[*bound_ptr] = bc_value;
            }
  
          } else if (bc_kind == "Neumann") {
  
            IntVector axes = patch->getFaceAxes(face);
            int P_dir = axes[0];  // principal direction
            double plus_minus_one = (double) patch->faceDirection(face)[P_dir];
            double dx = Dx[P_dir];
            
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector bp1(*bound_ptr - insideCellDir); 
              scalar[*bound_ptr] = scalar[bp1] + plus_minus_one * dx * bc_value;
            }
          } else if (bc_kind == "FromFile") { 
  
            ScalarToBCValueMap::iterator i_scalar_bc_storage = scalar_bc_from_file.find( face_name ); 
  
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
  
              IntVector rel_bc = *bound_ptr - i_scalar_bc_storage->second.relative_ijk; 
              CellToValueMap::iterator iter = i_scalar_bc_storage->second.values.find( rel_bc ); //<----WARNING ... May be slow here
              if ( iter != i_scalar_bc_storage->second.values.end() ){ 
  
                double file_bc_value = iter->second; 
                IntVector bp1(*bound_ptr - insideCellDir);
                scalar[*bound_ptr] = file_bc_value; 
  
              } else if ( i_scalar_bc_storage->second.default_type == "Neumann" ){  

                IntVector bp1(*bound_ptr - insideCellDir); 
                scalar[*bound_ptr] = i_scalar_bc_storage->second.default_value;
  
              } else if ( i_scalar_bc_storage->second.default_type == "Dirichlet" ){ 
  
                IntVector bp1(*bound_ptr - insideCellDir); 
                scalar[*bound_ptr] = i_scalar_bc_storage->second.default_value;
  
              } 
            }
          } else if ( bc_kind == "Tabulated") {
  
            MapDoubleMap::iterator i_face = _tabVarsMap.find( face_name );
  
            if ( i_face != _tabVarsMap.end() ){ 
  
              DoubleMap::iterator i_var = i_face->second.find( varname ); 
              double tab_bc_value = i_var->second;
  
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir);
                scalar[*bound_ptr] = tab_bc_value;
              }
  
            }
          } else { 
            throw InvalidValue( "Error: Cannot determine boundary condition type for variable: "+varname, __FILE__, __LINE__);
          }
        }
      }
    }
  }

  /** @brief Check to ensure that valid BCs are set for a specified variable. */   
  void checkForBC( const ProcessorGroup*,
                   const Patch* patch,
                   const std::string varname )
  {

    using std::vector; 
    using std::string; 

    vector<Patch::FaceType>::const_iterator iter;
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);
    Vector Dx = patch->dCell(); 

    for (iter = bf.begin(); iter !=bf.end(); iter++){
      Patch::FaceType face = *iter;
  
      //get the face direction
      IntVector insideCellDir = patch->faceDirection(face);
      //get the number of children
      int numChildren = patch->getBCDataArray(face)->getNumberChildren(d_matl_id); //assumed one material
  
      for (int child = 0; child < numChildren; child++){
  
        double bc_value = -9; 
        Vector bc_v_value(0,0,0); 
        std::string bc_s_value = "NA";
  
        Iterator bound_ptr;
        string bc_kind = "NotSet"; 
        string face_name; 
        getBCKind( patch, face, child, varname, d_matl_id, bc_kind, face_name ); 
  
        bool foundIterator = "false"; 
        if ( bc_kind == "Tabulated" || bc_kind == "FromFile" ){ 
          foundIterator = 
            getIteratorBCValue<std::string>( patch, face, child, varname, d_matl_id, bc_s_value, bound_ptr ); 
        } else {
          foundIterator = 
            getIteratorBCValue<double>( patch, face, child, varname, d_matl_id, bc_value, bound_ptr ); 
        } 
  
        if (foundIterator) {

          if ( bc_kind != "Dirichlet" && bc_kind != "Neumann" && bc_kind != "Tabulated" 
              && bc_kind != "FromFile" && bc_kind != "ForcedDirichlet" ){ 
            throw InvalidValue( "Error: Cannot determine boundary condition type for variable: "+varname+ " with bc type: "+bc_kind, __FILE__, __LINE__);
          }

        } else { 
            throw InvalidValue( "Error: Missing boundary condition for "+ varname, __FILE__, __LINE__);
        }
      }
    }
  }

  /** @brief This method set the boundary values of a vector to a 
   * value such that the interpolation or gradient computed between the 
   * interior cell and boundary cell match the boundary condition. */ 
  void setVectorValueBC( const ProcessorGroup*,
    const Patch* patch,
    CCVariable<Vector>& vec, 
    std::string varname );
  /** @brief This method set the boundary values of a vector to a 
   * value such that the interpolation or gradient computed between the 
   * interior cell and boundary cell match the boundary condition. This is 
   * a specialized case where the boundary value comes from some other vector */
  void setVectorValueBC( const ProcessorGroup*,
    const Patch* patch,
    CCVariable<Vector>& vec, constCCVariable<Vector>& const_vec, 
    std::string varname );

  /** @brief Sets the area fraction for each minus face according to the boundaries */
  void setAreaFraction( 
    const Patch* patch,
    CCVariable<Vector>& areaFraction, 
    CCVariable<double>& volFraction, 
    constCCVariable<int>& pcell, 
    std::vector<int> wallType,
    const int flowType );

  void sched_assignTabBCs( SchedulerP& sched, 
                           const PatchSet* patches, 
                           const MaterialSet* matls,
                           const std::string eqnName );

  /** @brief Read in a file for boundary conditions **/ 
  void readInputFile( std::string file_name, FFInfo& info ); 

  class BCFunctionBase{ 

    public: 

      BCFunctionBase() {};
      virtual ~BCFunctionBase(){}; 

      virtual void setupBC( ProblemSpecP& db, const std::string eqn_name ) = 0; 
      virtual void applyBC( const Patch* patch, Patch::FaceType face, int child, std::string varname, std::string face_name, CCVariable<double>& phi ) = 0; 
 
    protected: 
    
  };

  ScalarToBCValueMap& get_FromFileInfo(){ return scalar_bc_from_file; }; 

  //new stuff--------------------

  enum BCMaskType {BOUNDARY_CELL, BOUNDARY_FACE, FIRST_NORMAL_INTERIOR};

  //copied from Tony...passes the number of ghosts
  /** @brief Copied from Tony and added passed ghosts. This creates a memory window for mask creation. **/ 
  template< typename T >
  static SpatialOps::MemoryWindow
  get_mem_win_for_masks( const Uintah::Patch* const patch, const int nGhost ){

    SpatialOps::IntVec bcMinus, bcPlus;
    Wasatch::get_bc_logicals( patch, bcMinus, bcPlus );
    const SCIRun::IntVector gs = patch->getCellHighIndex(0) - patch->getCellLowIndex(0);
    const SpatialOps::IntVec glob( gs[0] + nGhost*2 + (bcPlus[0] ? T::Location::BCExtra::X : 0),
                                   gs[1] + nGhost*2 + (bcPlus[1] ? T::Location::BCExtra::Y : 0),
                                   gs[2] + nGhost*2 + (bcPlus[2] ? T::Location::BCExtra::Z : 0) );
    const SpatialOps::IntVec extent = glob;
    const SpatialOps::IntVec offset(0,0,0);
    return SpatialOps::MemoryWindow( glob, offset, extent );

  }

  /** @brief For a specific boundary and field type, this struct will hold the mask **/ 
  template <typename FieldT>
  struct MaskContainer { 

    public: 
      typedef std::map<BCMaskType, boost::shared_ptr<SpatialOps::SpatialMask<FieldT> > > MaskStorage; 

      MaskContainer(){
        _mask_storage.clear(); 
      }

      ~MaskContainer(){ 
        _mask_storage.clear();
      }

      /** @brief Create the mask.  Note the mask is a function of the num. of ghost cells...this is going to be tricky **/ 
      void create_mask( const Patch* patch, int nGhosts, const std::vector<SpatialOps::IntVec> ijk, BCMaskType bc_mask_type ){ 
        
        typename MaskStorage::iterator iter = _mask_storage.find(bc_mask_type); 

        if ( iter == _mask_storage.end() ){

          SpatialOps::IntVec bcMinus, bcPlus; 
          Wasatch::get_bc_logicals( patch, bcMinus, bcPlus ); 
          SpatialOps::BoundaryCellInfo bcInfo = SpatialOps::BoundaryCellInfo::build<FieldT>(bcPlus);
          SpatialOps::GhostData gd(nGhosts); 
          const SpatialOps::MemoryWindow window = BoundaryCondition_new::get_mem_win_for_masks<FieldT>( patch, nGhosts );
          //SpatialOps::SpatialMask<FieldT>* mask = new SpatialOps::SpatialMask<FieldT>(window, bcInfo, gd, ijk);

          boost::shared_ptr<SpatialOps::SpatialMask<FieldT> > ptr( scinew SpatialOps::SpatialMask<FieldT>(window, bcInfo, gd, ijk)); 
         
          //_mask_storage.insert(std::make_pair( bc_mask_type, ptr)); 

        }

      }

      /** @brief Get the mask given the enum type.  Note that is must (?) be ghost compatable. Huh? **/ 
      SpatialOps::SpatialMask<FieldT>* get_mask( BCMaskType bc_mask_type ){ 
        typename MaskStorage::iterator iter = _mask_storage.find(bc_mask_type); 
        if ( iter == _mask_storage.end() ){ 
          throw InvalidValue("Mask Error: Cannot find boundary condition information.", __FILE__, __LINE__); 
        } 
        return iter->second; 
      }

      MaskStorage _mask_storage; 

  }; 

  // maps boundary name -> mask contianer
  typedef std::map<const std::string, MaskContainer<SpatialOps::SVolField> > NameToSVolMask;
  // maps patchID -> list of mask containers
  typedef std::map<const int, NameToSVolMask> PatchToSVolMasks; 
  //static PatchToSVolBoundary svol_boundary_info; 
  static PatchToSVolMasks patch_svol_masks; 

  /** @brief This struct contains the helper function to retrieve boundary struct **/ 
  template <typename FieldT>
  struct BCInterfaceStruct;

  /** @brief Public access to the BC struct given the patch ID and a field **/ 
  template <typename FieldT>
  static MaskContainer<FieldT>& get_bc_info( const int patchID, const std::string bc_name ){ 
    return BCInterfaceStruct<FieldT>::get_bc( patchID, bc_name ); }

  /** @brief This struct contains the helper function to retrieve boundary struct **/ 
  template <typename FieldT>
  struct BCInterfaceStruct{ 

    public: 
      /** @brief Generic interface. Overloading should overide this call hence the error.  **/ 
      static MaskContainer<FieldT>& get_bc( ){
        throw InvalidValue("Mask Error: No known BC struct for this variable type.", __FILE__, __LINE__); 
      }; 

      /** @brief SVolField interface for BCBase **/ 
      static MaskContainer<SpatialOps::SVolField>& get_bc( const int patchID, const std::string bc_name ){

        //patch -> map with names, mask
        PatchToSVolMasks::iterator iter = BoundaryCondition_new::patch_svol_masks.find(patchID); 
        if ( iter == BoundaryCondition_new::patch_svol_masks.end() ){ 
          throw InvalidValue("Mask Error: Cannot find SVol masks from PatchToSVolMasks container.", __FILE__, __LINE__); 
        }

        //name -> mask
        NameToSVolMask::iterator name_iter = iter->second.find(bc_name); 
        if ( name_iter == iter->second.end()){ 
          throw InvalidValue("Mask Error: Boundary name doesn't exist in mask map: "+bc_name, __FILE__, __LINE__); 
        }
        return name_iter->second; 

      }; 
  };

  //------------------------------------------------------------------ end NEW --------------------

private: 

  typedef std::map< std::string, std::map<std::string, BCFunctionBase* > > VarToMappedF; 

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

  /** @brief Generate a random name with a fixed length **/  
  void get_random_name(char* s, const int len ){
        static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";

    for (int i = 0; i < len; ++i) {
        s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
    }

    s[len] = 0;
  }

}; // class BoundaryCondition_new

} // namespace Uintah

#endif 
