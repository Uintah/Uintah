#ifndef Uintah_Components_Arches_BoundaryCondition_new_h
#define Uintah_Components_Arches_BoundaryCondition_new_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <CCA/Components/Arches/Directives.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <Core/Exceptions/InvalidValue.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>

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
    OUTLET, NEUTRAL_OUTLET, WALL, MMWALL, INTRUSION, SWIRL, TURBULENT_INLET, PARTMASSFLOW_INLET };
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

  /** @brief Check to make sure there is a BC spec for this variable and setup handoff
              information if needed **/
  void checkBCs( const Patch* patch, const std::string variable, const int matlIndex );

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
  template< class T >
  void setExtraCellScalarValueBC(const ProcessorGroup*,
                                 const Patch* patch,
                                 CCVariable< T >& scalar,
                                 const std::string varname,
                                 bool  change_bc=false,
                                 const std::string override_bc="NA");

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
