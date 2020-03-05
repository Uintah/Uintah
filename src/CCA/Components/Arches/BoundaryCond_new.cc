#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SchedulerP.h>

#include <Core/Exceptions/InvalidState.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Task.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/IO/UintahZlibUtil.h>
#include <Core/Parallel/Parallel.h>
#include <fstream>

//===========================================================================

namespace Uintah{

BoundaryCondition_new::BoundaryCondition_new( const int matl_id):
  d_matl_id(matl_id)
{

}

BoundaryCondition_new::~BoundaryCondition_new()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void BoundaryCondition_new::problemSetup( ProblemSpecP& db, std::string eqn_name )
{

  ProblemSpecP db_root = db->getRootNode();
  ProblemSpecP db_bc   = db_root->findBlock("Grid")->findBlock("BoundaryConditions");

  if ( db_bc ) {
    for ( ProblemSpecP db_face = db_bc->findBlock("Face"); db_face != nullptr; db_face = db_face->findNextBlock("Face") ){

      std::string face_name = "NA";
      db_face->getAttribute("name", face_name );

      for ( ProblemSpecP db_BCType = db_face->findBlock("BCType"); db_BCType != nullptr; db_BCType = db_BCType->findNextBlock("BCType") ){

        std::string name;
        std::string type;
        db_BCType->getAttribute("label", name);
        db_BCType->getAttribute("var", type);

        if ( name == eqn_name ){

          if ( type == "FromFile" ){

            if ( face_name == "NA" ){
              throw ProblemSetupException( "Error: When using FromFile BCs, you must name the <Face> using the name attribute.", __FILE__, __LINE__);
            }

            //Check reference file for this scalar
            std::string file_name;
            db_BCType->require("value", file_name);

            FFInfo bc_values;
            readInputFile( file_name, bc_values );

            Vector rel_xyz;
            db_BCType->require("relative_xyz", rel_xyz);
            bc_values.relative_xyz = rel_xyz;

            db_BCType->findBlock("default")->getAttribute("type",bc_values.default_type);
            db_BCType->findBlock("default")->getAttribute("value",bc_values.default_value);

            scalar_bc_from_file.insert(std::make_pair(face_name, bc_values));

          } else if ( type == "Tabulated" ){

            if ( face_name == "NA" ){
              throw ProblemSetupException( "Error: When using Tabulated BCs, you must name the <Face> using the name attribute.", __FILE__, __LINE__);
            }

          } else if ( type == "ScalarSwirl"){

            // This was developed for DQMOM velocity components with swirl
            // NOTE: this will use the "value" entry as the normal component of the velocity
            swirlInfo my_info;
            db_BCType->require("swirl_no", my_info.swirl_no);
            my_info.swirl_no *= 3./2.;
            // swirl number definition as equation 5.14 from Combustion Aerodynamics 
            // J.M. BEER and N.A. CHIGIER 1983 pag 107 
            // assuming: 
            // constant axial velocity
            // constant density 
            // constant tangential velocity 

            std::string str_vec; // This block sets the default centroid to the origin unless otherwise specified by swirl_cent
            bool Luse_origin =   db_face->getAttribute("origin", str_vec);
            if ( Luse_origin ){
              std::stringstream ss;
              ss << str_vec;
              Vector origin;
              ss >> origin[0] >> origin[1] >> origin[2];
              db_BCType->getWithDefault("swirl_centroid", my_info.swirl_cent, origin);
            } else {
              db_BCType->require("swirl_centroid",my_info.swirl_cent);
            }

            //Must specify which coordinate this scalar represents because the var names are arbitrary
            std::string coord;
            db_BCType->require("swirl_coord", coord);
            if ( coord == "Y") {
              my_info.coord = 1;
            } else if ( coord == "Z"){
              my_info.coord = 2;
            } else {
              throw ProblemSetupException("Error: Coordindate value for scalar swirl must be Y or Z. X is assumed as the flow direction", __FILE__, __LINE__);
            }

            if ( face_name == "NA"){
              ProblemSetupException("Error: Please specificy a face name when using Swirl condition on a scalar.", __FILE__, __LINE__ );
            }

            m_swirl_map.insert(std::make_pair(face_name, my_info));

          }
        }
      }
    }
  }

  //NOTES:
  // each scalar that has an input file for a BC will now be stored in scalar_bc_from_file
  // which is a map <eqn_name, CellToValue>.
  // Next step would ideally be to:
  //   1) create an object to store <patch*, CellToValue> where CellToValue is only on a specific patch
  //   2) create a task that actually populates 1)
  //   3) use the storage container for the BC and iterate through to apply BC's
  //   **RIGHT NOW IT IS CHECKING EACH CELL FOR BC!
  //

}
//______________________________________________________________________
//
//______________________________________________________________________
void
BoundaryCondition_new::setupTabulatedBC( ProblemSpecP& db, std::string eqn_name, MixingRxnModel* table )
{
  ProblemSpecP db_root = db->getRootNode();
  ProblemSpecP db_bc   = db_root->findBlock("Grid")->findBlock("BoundaryConditions");
  _tabVarsMap.clear();

  if ( db_bc ) {

    for ( ProblemSpecP db_face = db_bc->findBlock("Face"); db_face != nullptr; db_face = db_face->findNextBlock("Face") ){

      //first check to see if there are any tabulated BCs on this face//
      bool has_tabulated_bc = false;
      for ( ProblemSpecP db_BCType = db_face->findBlock("BCType"); db_BCType != nullptr; db_BCType = db_BCType->findNextBlock("BCType") ){

          std::string type;
          db_BCType->getAttribute("var", type);

          if ( type == "Tabulated" ){
            has_tabulated_bc = true;
          }
      }

      if ( has_tabulated_bc ){

        std::string face_name="NA";
        db_face->getAttribute("name", face_name);

        std::vector<double> iv;
        std::vector<std::string> allIndepVarNames = table->getAllIndepVars();

        int totalIVs = allIndepVarNames.size();
        int counter=0;

        //Fill the independent variable vector
        for ( int i = 0; i < (int) allIndepVarNames.size(); i++ ){

          std::string iv_name = allIndepVarNames[i];

          for ( ProblemSpecP db_BCType = db_face->findBlock("BCType"); db_BCType != nullptr; db_BCType = db_BCType->findNextBlock("BCType") ){

            std::string name;
            std::string type;
            db_BCType->getAttribute("label",name);
            db_BCType->getAttribute("var", type);

            //probably should add FromFile here too....
            if ( name == iv_name ){

              if ( type != "Dirichlet" ){
                proc0cout << "Cannot set a boundary condition for a dependent variable because " << name << " is not of Dirichlet type." << std::endl;
                throw ProblemSetupException( "Error: When using a tabulated BC, all independent variables must be of type Dirichlet.", __FILE__, __LINE__);
              }

              double value;
              db_BCType->require("value",value);
              iv.push_back(value);

              counter++;

            }
          }
        }

        if ( counter != totalIVs ){
          std::stringstream msg;
          msg << "Error: For BC face named: " << face_name << " the Tabulated option for a variable was used but there are missing IVs boundary specs.";
          throw ProblemSetupException( msg.str(), __FILE__, __LINE__);
        }

        //Get any inerts
        DoubleMap inert_map;

        MixingRxnModel::InertMasterMap master_inert_map = table->getInertMap();
        for ( MixingRxnModel::InertMasterMap::iterator iter = master_inert_map.begin(); iter != master_inert_map.end(); iter++ ){

          for ( ProblemSpecP db_BCType = db_face->findBlock("BCType"); db_BCType != nullptr; db_BCType = db_BCType->findNextBlock("BCType") ){

            std::string name;
            std::string type;
            double value;

            db_BCType->getAttribute("label", name);
            db_BCType->getAttribute("var", type);

            if ( name == iter->first ){

              if ( type != "Dirichlet" ){
                proc0cout << "Cannot set a boundary condition for a dependent variable because " << name << " is not of Dirichlet type." << std::endl;
                throw ProblemSetupException( "Error: When using a tabulated BC, all inert variables must be of type Dirichlet.", __FILE__, __LINE__);
              } else {
                db_BCType->require("value", value);
              }

              inert_map.insert( std::make_pair( name, value ));

            }
          }
        }

        DoubleMap bc_name_to_value;

        for ( ProblemSpecP db_BCType = db_face->findBlock("BCType"); db_BCType != nullptr; db_BCType = db_BCType->findNextBlock("BCType") ){

          std::string name;
          std::string type;
          db_BCType->getAttribute("label", name);
          db_BCType->getAttribute("var", type);

          if ( name == eqn_name ){
            if ( type == "Tabulated" ){

              if ( face_name == "NA" ){
                throw ProblemSetupException( "Error: When using Tabulated BCs, you must name the <Face> using the name attribute.", __FILE__, __LINE__);
              }

              std::string dep_variable = "NA";
              db_BCType->require("value",dep_variable);

              if ( dep_variable == "NA" ){
                throw ProblemSetupException( "Error: When using Tabulated BCs, you must specify the dependent variable in the <value> tag..", __FILE__, __LINE__);
              }

              // get the value from the table
              double tabulate_value = table->getTableValue( iv, dep_variable, inert_map );
              bc_name_to_value.insert( std::make_pair(name, tabulate_value) );

            }
          }
        }
        _tabVarsMap.insert( std::make_pair( face_name, bc_name_to_value ) );
      }
    }
  }
}


//______________________________________________________________________
//
//______________________________________________________________________
void
BoundaryCondition_new::readInputFile( std::string file_name, BoundaryCondition_new::FFInfo& struct_result )
{

  gzFile file = gzopen( file_name.c_str(), "r" );
  if ( file == nullptr ) {
    proc0cout << "Error opening file: " << file_name << " for boundary conditions. Errno: " << errno << std::endl;
    throw ProblemSetupException("Unable to open the given input file: " + file_name, __FILE__, __LINE__);
  }

  struct_result.name = getString( file );

  struct_result.dx = getDouble( file );
  struct_result.dy = getDouble( file );

  int num_points = getInt( file );

  std::map<IntVector, double> values;

  for ( int i = 0; i < num_points; i++ ) {
    int I = getInt( file );
    int J = getInt( file );
    int K = getInt( file );
    double v = getDouble( file );

    IntVector C(I,J,K);

    values.insert( std::make_pair( C, v ));

  }

  struct_result.values = values;

  gzclose( file );

}

//---------------------------------------------------------------------------
// Method: Set Cell Centered Vector BC values
//---------------------------------------------------------------------------
void BoundaryCondition_new::setVectorValueBC( const ProcessorGroup*,
    const Patch* patch,
    CCVariable<Vector>& vec,
    std::string varname )
{
  // This method sets the value of the CELL-CENTERED VECTOR components in the boundary cell
  // so that the boundary condition set in the input file is satisfied.
  std::vector<Patch::FaceType>::const_iterator iter;
  std::vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  Vector Dx = patch->dCell();

  for (iter = bf.begin(); iter !=bf.end(); iter++){
    Patch::FaceType face = *iter;

    //get the face direction
    IntVector insideCellDir = patch->faceDirection(face);
    //get the number of children
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(d_matl_id); //assumed one material
    for (int child = 0; child < numChildren; child++){
      Vector bc_value = Vector(0.0, 0.0, 0.0);
      std::string bc_kind = "NotSet";
      Iterator bound_ptr;

      bool foundIterator =
        getIteratorBCValueBCKind( patch, face, child, varname, d_matl_id, bc_value, bound_ptr, bc_kind);

      double X,Y,Z;

      if (foundIterator) {
        // --- notation ---
        // bp1: boundary cell + 1 or the interior cell one in from the boundary
        if (bc_kind == "Dirichlet") {

          for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {

            IntVector bp1(*bound_ptr - insideCellDir);

            X = 2.0*bc_value.x() - vec[bp1].x();
            Y = 2.0*bc_value.y() - vec[bp1].y();
            Z = 2.0*bc_value.z() - vec[bp1].z();

            vec[*bound_ptr] = Vector(X,Y,Z);

          }

        } else if (bc_kind == "Neumann") {

          IntVector axes = patch->getFaceAxes(face);
          int P_dir = axes[0];  // principal direction
          double plus_minus_one = (double) patch->faceDirection(face)[P_dir];
          double dx = Dx[P_dir];

          for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {

            IntVector bp1(*bound_ptr - insideCellDir);

            X = vec[bp1].x() + plus_minus_one * dx * bc_value.x();
            Y = vec[bp1].y() + plus_minus_one * dx * bc_value.y();
            Z = vec[bp1].z() + plus_minus_one * dx * bc_value.z();

            vec[*bound_ptr] = Vector(X,Y,Z);
          }
        }
      }
    }
  }
}

//---------------------------------------------------------------------------
// Method: Set Cell Centered Vector BC values to another vector value
//---------------------------------------------------------------------------
void BoundaryCondition_new::setVectorValueBC( const ProcessorGroup*,
    const Patch* patch,
    CCVariable<Vector>& vec,
    constCCVariable<Vector>& const_vec,
    std::string varname )
{

  std::vector<Patch::FaceType>::const_iterator iter;
  std::vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  Vector Dx = patch->dCell();

  for (iter = bf.begin(); iter !=bf.end(); iter++){
    Patch::FaceType face = *iter;

    //get the face direction
    IntVector insideCellDir = patch->faceDirection(face);
    //get the number of children
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(d_matl_id); //assumed one material

    for (int child = 0; child < numChildren; child++){
      Vector bc_value = Vector(0.0, 0.0, 0.0);
      std::string bc_kind = "NotSet";
      Iterator bound_ptr;

      bool foundIterator =
        getIteratorBCValueBCKind( patch, face, child, varname, d_matl_id, bc_value, bound_ptr, bc_kind);

      double X,Y,Z;

      if (foundIterator) {
        // --- notation ---
        // bp1: boundary cell + 1 or the interior cell one in from the boundary
        if (bc_kind == "Dirichlet") {
          for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {

            IntVector bp1(*bound_ptr - insideCellDir);

            X = ( const_vec[*bound_ptr].x() );
            Y = ( const_vec[*bound_ptr].y() );
            Z = ( const_vec[*bound_ptr].z() );

            vec[*bound_ptr] = Vector(X,Y,Z);

          }
        } else if (bc_kind == "Neumann") {

          double dvdx, dvdy, dvdz;

          IntVector axes = patch->getFaceAxes(face);
          int P_dir = axes[0];  // principal direction
          double plus_minus_one = (double) patch->faceDirection(face)[P_dir];
          double dx = Dx[P_dir];

          for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {

            IntVector bp1(*bound_ptr - insideCellDir);

            dvdx = -1.0 * plus_minus_one * ( const_vec[bp1].x() - const_vec[*bound_ptr].x() ) / dx;
            dvdy = -1.0 * plus_minus_one * ( const_vec[bp1].y() - const_vec[*bound_ptr].y() ) / dx;
            dvdz = -1.0 * plus_minus_one * ( const_vec[bp1].z() - const_vec[*bound_ptr].z() ) / dx;

            X = vec[bp1].x() + plus_minus_one * dx * dvdx;
            Y = vec[bp1].y() + plus_minus_one * dx * dvdy;
            Z = vec[bp1].z() + plus_minus_one * dx * dvdz;

            vec[*bound_ptr] = Vector(X,Y,Z);

          }
        }
      }
    }
  }
}

//---------------------------------------------------------------------------
// Method: Set the area fraction for all cells
//---------------------------------------------------------------------------
void BoundaryCondition_new::setAreaFraction( const Patch* patch,
                                             CCVariable<Vector>& areaFraction,
                                             CCVariable<double>&  volFraction,
                                             constCCVariable<int>& cellType,
                                             std::vector<int> wallType,
                                             const int flowType )
{

  // areaFraction is a vector with:
  // areaFraction.x = xminus area fraction
  // areaFraction.y = yminus area fraction
  // areaFraction.z = zminus area fraction
  //
  // volFraction is the GAS volume fraction

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

    IntVector c = *iter;
    IntVector cxm = *iter - IntVector(1,0,0);
    IntVector cym = *iter - IntVector(0,1,0);
    IntVector czm = *iter - IntVector(0,0,1);
    IntVector cxp = *iter + IntVector(1,0,0);
    IntVector cyp = *iter + IntVector(0,1,0);
    IntVector czp = *iter + IntVector(0,0,1);

    std::vector<int>::iterator wt_iter;

    for ( wt_iter = wallType.begin(); wt_iter != wallType.end(); wt_iter++ ){

      int type = *wt_iter;

      // curr cell is a wall
      if ( cellType[c] == type ) {

        areaFraction[c] = Vector(0.,0.,0.);
        volFraction[c]  = 0.0;

      }

      // x-minus is a wall but curr cell is flow
      if ( cellType[c] == flowType && cellType[cxm] == type ) {
        Vector T = areaFraction[c];
        T[0] = 0.;
        areaFraction[c] = T;
      }

      // y-minus is a wall but curr cell is flow
      if ( cellType[c] == flowType && cellType[cym] == type ) {
        Vector T = areaFraction[c];
        T[1] = 0.;
        areaFraction[c] = T;
      }

      // z-minus is a wall but curr cell is flowType
      if (cellType[c] == flowType && cellType[czm] == type ) {
        Vector T = areaFraction[c];
        T[2] = 0.;
        areaFraction[c] = T;
      }
    }
  }

  //__________________________________
  // loop over computational domain faces
  std::vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);

  for( std::vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){

    Patch::FaceType face = *iter;

    Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;

    for (CellIterator citer =  patch->getFaceIterator(face, PEC); !citer.done(); citer++) {

      IntVector c = *citer;

      std::vector<int>::iterator wt_iter;

      for ( wt_iter = wallType.begin(); wt_iter != wallType.end(); wt_iter++ ){

        int type = *wt_iter;

        if ( cellType[c] == type ){

          int P_dir = patch->getFaceAxes(face)[0];  //principal dir.

          Vector T = areaFraction[c];
          T[P_dir] = 0.;
          areaFraction[c] = T;
          volFraction[c] = 0.0;

        }
      }
    }
  }
}

//______________________________________________________________________
//          DIRICHLET
//______________________________________________________________________
void BoundaryCondition_new::Dirichlet::applyBC( const Patch* patch, Patch::FaceType face,
                                                int child, std::string varname, std::string face_name,
                                                CCVariable<double>& phi )
{
  double bc_value;
  Iterator bound_ptr;
  std::string bc_kind = "NA";

  bool foundIterator = getIteratorBCValueBCKind( patch, face, child, varname, d_matl_id, bc_value, bound_ptr, bc_kind );

  if (foundIterator) {

    // --- notation ---
    // bp1: boundary cell + 1 or the interior cell one in from the boundary
    IntVector insideCellDir = patch->faceDirection(face);

    for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {

      IntVector bp1(*bound_ptr - insideCellDir);
      phi[*bound_ptr] = 2.0 * bc_value - phi[bp1];

    }
  }
}


//______________________________________________________________________
//          NEUMANN
//______________________________________________________________________
void BoundaryCondition_new::Neumann::applyBC( const Patch* patch, Patch::FaceType face,
                                              int child, std::string varname, std::string face_name,
                                              CCVariable<double>& phi )
{
  double bc_value;
  Iterator bound_ptr;
  std::string bc_kind = "NA";

  bool foundIterator = getIteratorBCValueBCKind( patch, face, child, varname, d_matl_id, bc_value, bound_ptr, bc_kind );

  if (foundIterator) {
    // --- notation ---
    // bp1: boundary cell + 1 or the interior cell one in from the boundary
    IntVector insideCellDir = patch->faceDirection(face);

    IntVector axes = patch->getFaceAxes(face);
    Vector Dx = patch->dCell();
    int P_dir = axes[0];  // principal direction
    double plus_minus_one = (double) patch->faceDirection(face)[P_dir];
    double dx = Dx[P_dir];

    for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
      IntVector bp1(*bound_ptr - insideCellDir);
      phi[*bound_ptr] = phi[bp1] + plus_minus_one * dx * bc_value;
    }
  }
}


//______________________________________________________________________
//  From a file
//______________________________________________________________________
void BoundaryCondition_new::FromFile::setupBC( ProblemSpecP& db, const std::string eqn_name )
{

  ProblemSpecP db_face = db;

  std::string face_name = "NA";
  db_face->getAttribute("name", face_name );

  d_face_map.clear();

  //reparsing the BCType because the abstraction requires that we pass the <Face> node
  //into the setupBC method because there is no "getParentNode" method needed for
  //things like TabulatedBC
  for ( ProblemSpecP db_BCType = db_face->findBlock("BCType"); db_BCType != nullptr; db_BCType = db_BCType->findNextBlock("BCType") ){

    std::string name;
    std::string type;
    db_BCType->getAttribute("label", name);
    db_BCType->getAttribute("var", type);

    if ( name == eqn_name ){

      //Check reference file for this scalar
      std::string file_name;
      db_BCType->require("value", file_name);

      gzFile file = gzopen( file_name.c_str(), "r" );
      int total_variables;

      if ( file == nullptr ) {
        proc0cout << "Error opening file: " << file_name << " for boundary conditions. Errno: " << errno << std::endl;
        throw ProblemSetupException("Unable to open the given input file: " + file_name, __FILE__, __LINE__);
      }

      total_variables = getInt( file );
      std::string eqn_input_file;
      bool found_file = false;
      for ( int i = 0; i < total_variables; i++ ){
        std::string varname  = getString( file );
        eqn_input_file  = getString( file );

        if ( varname == eqn_name ){
          found_file = true;
          break;
        }
      }
      gzclose( file );

      if ( !found_file ){
        std::stringstream err_msg;
        err_msg << "Error: Unable to find BC input file for scalar: " << eqn_name << " Check this file for errors: \n" << file_name << std::endl;
        throw ProblemSetupException( err_msg.str(), __FILE__, __LINE__);
      }

      //If file is found, now create a map from index to value
      CellToValueMap bc_values;
      bc_values = readInputFile( eqn_input_file );

      //scalar_bc_from_file.insert(make_pair(eqn_name, bc_values));
      d_face_map.insert( std::make_pair( face_name, bc_values ));
    }
  }
}

//______________________________________________________________________
//
//______________________________________________________________________
void BoundaryCondition_new::FromFile::applyBC( const Patch* patch, Patch::FaceType face,
                                               int child, std::string varname, std::string face_name,
                                               CCVariable<double>& phi )
{
  double bc_value;
  Iterator bound_ptr;
  std::string bc_kind = "NA";

  bool foundIterator = getIteratorBCValueBCKind( patch, face, child, varname, d_matl_id, bc_value, bound_ptr, bc_kind );

  if (foundIterator) {

    // --- notation ---
    // bp1: boundary cell + 1 or the interior cell one in from the boundary
    IntVector insideCellDir = patch->faceDirection(face);

    FaceToBCValueMap::iterator iter_facetobc = d_face_map.find( face_name );

    for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {

      CellToValueMap::iterator iter = iter_facetobc->second.find( *bound_ptr ); //<----WARNING ... May be slow here
      double file_bc_value = iter->second;

      IntVector bp1(*bound_ptr - insideCellDir);
      phi[*bound_ptr] = 2.0 * file_bc_value - phi[bp1];

    }
  }
}

//______________________________________________________________________
//
//______________________________________________________________________
std::map<IntVector, double>
BoundaryCondition_new::FromFile::readInputFile( std::string file_name )
{

  gzFile file = gzopen( file_name.c_str(), "r" );
  if ( file == nullptr ) {
    proc0cout << "Error opening file: " << file_name << " for boundary conditions. Errno: " << errno << std::endl;
    throw ProblemSetupException("Unable to open the given input file: " + file_name, __FILE__, __LINE__);
  }

  std::string variable = getString( file );
  int         num_points = getInt( file );
  std::map<IntVector, double> result;

  for ( int i = 0; i < num_points; i++ ) {
    int I = getInt( file );
    int J = getInt( file );
    int K = getInt( file );
    double v = getDouble( file );

    IntVector C(I,J,K);

    result.insert( std::make_pair( C, v ));

  }

  gzclose( file );
  return result;
}

//______________________________________________________________________
//  TABULATED
//______________________________________________________________________
void BoundaryCondition_new::Tabulated::setupBC( ProblemSpecP& db, const std::string eqn_name )
{ }

void BoundaryCondition_new::Tabulated::extra_setupBC( ProblemSpecP& db, std::string eqn_name, MixingRxnModel* table )
{

  ProblemSpecP db_root = db->getRootNode();
  ProblemSpecP db_bc   = db_root->findBlock("Grid")->findBlock("BoundaryConditions");

  if ( db_bc ) {

    for ( ProblemSpecP db_face = db_bc->findBlock("Face"); db_face != nullptr; db_face = db_face->findNextBlock("Face") ){

      bool has_tabulated = false;
      std::string face_name = "NOTSET";

      db_face->getAttribute( "name", face_name );

      if ( face_name == "NOTSET" ){
        throw ProblemSetupException("Error: When using Tabulated BCs you must name each <Face>.", __FILE__, __LINE__);
      }

      for ( ProblemSpecP db_BCType = db_face->findBlock("BCType"); db_BCType != nullptr; db_BCType = db_BCType->findNextBlock("BCType") ){
        std::string name;
        std::string type;
        db_BCType->getAttribute("label", name);
        db_BCType->getAttribute("var", type);

        if ( name == eqn_name ){

          if ( type == "Tabulated" ){

            has_tabulated = true;  //don't we already know this by this point?

          }
        }
      }

      if ( has_tabulated ){

        std::vector<double> iv;
        std::vector<std::string> allIndepVarNames = table->getAllIndepVars();

        for ( int i = 0; i < (int) allIndepVarNames.size(); i++ ){
          for ( ProblemSpecP db_BCType = db_face->findBlock("BCType"); db_BCType != nullptr; db_BCType = db_BCType->findNextBlock("BCType") ){

            std::string iv_name = allIndepVarNames[i];
            std::string name;
            std::string type;
            db_BCType->getAttribute("label", name);
            db_BCType->getAttribute("var", type);

            if ( name == iv_name ){

              if ( type != "Dirichlet" ){
                proc0cout << "Cannot set a boundary condition for a dependent variable because " << name << " is not of Dirichlet type." << std::endl;
                throw ProblemSetupException( "Error: When using a tabulated BC, independent variables must be of type Dirichlet or FromFile.", __FILE__, __LINE__);
              }

              double value;
              db_BCType->require("value",value);
              iv.push_back( value );

            }
          }
        }

        //Get any inerts
        DoubleMap inert_map;

        MixingRxnModel::InertMasterMap master_inert_map = table->getInertMap();
        for ( MixingRxnModel::InertMasterMap::iterator iter = master_inert_map.begin(); iter != master_inert_map.end(); iter++ ){

          for ( ProblemSpecP db_BCType = db_face->findBlock("BCType"); db_BCType != nullptr; db_BCType = db_BCType->findNextBlock("BCType") ){

            std::string name;
            std::string type;
            double value;

            db_BCType->getAttribute("label", name);
            db_BCType->getAttribute("var", type);

            if ( name == iter->first ){

              if ( type != "Dirichlet" ){
                proc0cout << "Cannot set a boundary condition for a dependent variable because " << name << " is not of Dirichlet type." << std::endl;
                throw ProblemSetupException( "Error: When using a tabulated BC, all inert variables must be of type Dirichlet.", __FILE__, __LINE__);
              } else {
                db_BCType->require("value", value);
              }

              inert_map.insert( std::make_pair( name, value ));

            }
          }
        }

        std::string face_name = "NA";
        db_face->getAttribute("name", face_name );
        DoubleMap bc_name_to_value;

        for ( ProblemSpecP db_BCType = db_face->findBlock("BCType"); db_BCType != nullptr; db_BCType = db_BCType->findNextBlock("BCType") ){

          std::string name;
          std::string type;
          db_BCType->getAttribute("label", name);
          db_BCType->getAttribute("var", type);

          if ( name == eqn_name ){
            if ( type == "Tabulated" ){

              if ( face_name == "NA" ){
                throw ProblemSetupException( "Error: When using Tabulated BCs, you must name the <Face> using the name attribute.", __FILE__, __LINE__);
              }

              std::string dep_variable = "NA";
              db_BCType->require("value",dep_variable);

              if ( dep_variable == "NA" ){
                throw ProblemSetupException( "Error: When using Tabulated BCs, you must specify the dependent variable in the <value> tag..", __FILE__, __LINE__);
              }

              // get the value from the table
              double tabulate_value = table->getTableValue( iv, dep_variable, inert_map );
              bc_name_to_value.insert( std::make_pair(name, tabulate_value) );

            }
          }
        }
        _tabVarsMap.insert( std::make_pair( face_name, bc_name_to_value ) );
      }

    }
  }
}

//______________________________________________________________________
//
//______________________________________________________________________
void BoundaryCondition_new::Tabulated::applyBC( const Patch* patch, Patch::FaceType face,
                                               int child, std::string varname, std::string face_name,
                                               CCVariable<double>& phi )
{
  Iterator bound_ptr;
  std::string bc_kind = "NA";
  std::string bc_s_value = "NA";
  IntVector insideCellDir = patch->faceDirection(face);

  bool foundIterator = getIteratorBCValue<std::string>( patch, face, child, varname, d_matl_id, bc_s_value, bound_ptr );

  if (foundIterator) {

    MapDoubleMap::iterator i_face = _tabVarsMap.find( face_name );

    if ( i_face != _tabVarsMap.end() ){

      DoubleMap::iterator i_var = i_face->second.find( varname );
      double tab_bc_value = i_var->second;

      for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
        IntVector bp1(*bound_ptr - insideCellDir);
        phi[*bound_ptr] = 2.0 * tab_bc_value - phi[bp1];
      }

    }
  }
}

//______________________________________________________________________
//
//______________________________________________________________________
  template< class T >
  void
  BoundaryCondition_new::setExtraCellScalarValueBC(const ProcessorGroup *,
                                                   const Patch          * patch,
                                                   CCVariable< T >      & scalar,
                                                   const std::string      varname,
                                                   bool  change_bc,
                                                   const std::string override_bc)
  {
    using std::vector;
    using std::string;

    vector<Patch::FaceType>::const_iterator iter;
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    //__________________________________
    //
    for (iter = bf.begin(); iter !=bf.end(); iter++){
      Patch::FaceType face = *iter;

      //get the face direction
      IntVector insideCellDir = patch->faceDirection(face);
      
      //get the number of children
      int numChildren = patch->getBCDataArray(face)->getNumberChildren(d_matl_id); //assumed one material

      for (int child = 0; child < numChildren; child++){

        double bc_value = -9;
        std::string bc_s_value = "NA";

        Iterator bound_ptr;
        std::string bc_kind = "NotSet";
        std::string face_name;
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
        
        //__________________________________
        // bulletproofing
        if( ! foundIterator){
          std::ostringstream warn;
          warn << "ERROR: setExtraCellScalarValueBC: Boundary conditions were not set or specified correctly, face:" 
               << patch->getFaceName(face) << ", variable: " << varname << ", " << " Kind: " << bc_kind  
               << ", numChildren: " << numChildren << " \n";
          throw InternalError(warn.str(), __FILE__, __LINE__);
        }
        
        //__________________________________
        //
        if (bc_kind == "Dirichlet" || bc_kind == "ForcedDirichlet") {

          for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
            scalar[*bound_ptr] = bc_value;
          }
        } 
        //__________________________________
        //
        else if (bc_kind == "Neumann") {

          IntVector axes = patch->getFaceAxes(face);
          int P_dir = axes[0];  // principal direction
          double plus_minus_one = (double) patch->faceDirection(face)[P_dir];
          Vector Dx = patch->dCell();
          double dx = Dx[P_dir];

          for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
            IntVector adjCell(*bound_ptr - insideCellDir);
            scalar[*bound_ptr] = scalar[adjCell] + plus_minus_one * dx * bc_value;
          }
        } 
        //__________________________________
        //
        else if (bc_kind == "FromFile") {

          ScalarToBCValueMap::iterator i_scalar_bc_storage = scalar_bc_from_file.find( face_name );

          for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

            IntVector rel_bc = *bound_ptr - i_scalar_bc_storage->second.relative_ijk;
            CellToValueMap::iterator iter = i_scalar_bc_storage->second.values.find( rel_bc ); //<----WARNING ... May be slow here
            
            if ( iter != i_scalar_bc_storage->second.values.end() ){
              double file_bc_value = iter->second;
              scalar[*bound_ptr] = file_bc_value;

            } 
            else if ( i_scalar_bc_storage->second.default_type == "Neumann" ){
              scalar[*bound_ptr] = i_scalar_bc_storage->second.default_value;

            } 
            else if ( i_scalar_bc_storage->second.default_type == "Dirichlet" ){
              scalar[*bound_ptr] = i_scalar_bc_storage->second.default_value;

            }
          }
        } 
        //__________________________________
        //
        else if ( bc_kind == "Tabulated") {

          MapDoubleMap::iterator i_face = _tabVarsMap.find( face_name );

          if ( i_face != _tabVarsMap.end() ){

            DoubleMap::iterator i_var = i_face->second.find( varname );
            double tab_bc_value = i_var->second;

            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
              scalar[*bound_ptr] = tab_bc_value;
            }
          }
        } 
        else {
          throw InvalidValue( "ERROR: setExtraCellScalarValueBC: Cannot determine boundary condition type for variable: "+varname, __FILE__, __LINE__);
        }
      }
    }
  }


//______________________________________________________________________
//
//______________________________________________________________________
  void
  BoundaryCondition_new::checkBCs(
    const Patch* patch, const std::string variable, const int matlIndex ){

    std::vector<Patch::FaceType> bf;
    std::vector<Patch::FaceType>::const_iterator bf_iter;
    patch->getBoundaryFaces(bf);
    double dx=0;
    double dy=0;

    // Loop over all boundary faces on this patch
    for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++){

      Patch::FaceType face = *bf_iter;

      int numChildren = patch->getBCDataArray(face)->getNumberChildren(matlIndex);
      for (int child = 0; child < numChildren; child++){

        Vector bc_v_value(0,0,0);
        std::string bc_s_value = "NA";

        Iterator bound_ptr;
        std::string bc_kind = "NotSet";
        std::string face_name;


        getBCKind( patch, face, child, variable, matlIndex, bc_kind, face_name );

        std::ofstream outputfile;
        std::stringstream fname;
        fname << "handoff_" << variable << "_" << face_name <<  "." << patch->getID();
        bool file_is_open = false;

        std::string whichface;
        int index=0;
        Vector Dx = patch->dCell();

        if (face == 0){
          whichface = "x-";
          index = 0;
          dx = Dx[1];
          dy = Dx[2];
        } else if (face == 1) {
          whichface = "x+";
          index = 0;
          dx = Dx[1];
          dy = Dx[2];
        } else if (face == 2) {
          whichface = "y-";
          index = 1;
          dx = Dx[2];
          dy = Dx[0];
        } else if (face == 3) {
          whichface = "y+";
          index = 1;
          dx = Dx[2];
          dy = Dx[0];
        } else if (face == 4) {
          whichface = "z-";
          index = 2;
          dx = Dx[0];
          dy = Dx[1];
        } else if (face == 5) {
          whichface = "z+";
          index = 2;
          dx = Dx[0];
          dy = Dx[1];
        }

        if ( bc_kind == "NotSet" ){

          std::cout << "ERROR!:  Missing boundary condition specification!" << std::endl;
          std::cout << "Here are the details:" << std::endl;
          std::cout << "Variable = " << variable << std::endl;
          std::cout << "Face = " << whichface << std::endl;
          std::cout << "Child = " << child << std::endl;
          std::cout << "Material = " << matlIndex << std::endl;
          throw ProblemSetupException("Please correct your <BoundaryCondition> section in your input file for this variable", __FILE__,__LINE__);
        }

        // need to map x,y,z -> i,j,k for the FromFile option
        bool foundIterator = false;
        if ( bc_kind == "FromFile" ){
          foundIterator =
            getIteratorBCValue<std::string>( patch, face, child, variable, matlIndex, bc_s_value, bound_ptr );
        }

        BoundaryCondition_new::ScalarToBCValueMap& scalar_bc_info = get_FromFileInfo();
        BoundaryCondition_new::ScalarToBCValueMap::iterator i_scalar_bc_storage = scalar_bc_info.find( face_name );

        //check the grid spacing:
        if ( i_scalar_bc_storage != scalar_bc_info.end() ){
          proc0cout <<  std::endl << "For scalar handoff file named: " << i_scalar_bc_storage->second.name << std::endl;
          proc0cout <<          "  Grid and handoff spacing relative differences are: ["
            << std::abs(i_scalar_bc_storage->second.dx - dx)/dx << ", "
            << std::abs(i_scalar_bc_storage->second.dy - dy)/dy << "]" << std::endl << std::endl;
        }

        if (foundIterator) {

          //if we are here, then we are of type "FromFile"
          bound_ptr.reset();

          //this should assign the correct normal direction xyz value without forcing the user to have
          //to know what it is.
          if ( index == 0 ) {
            i_scalar_bc_storage->second.relative_xyz[index] = Dx[index]/2.0;
          } else if ( index == 1 ) {
            i_scalar_bc_storage->second.relative_xyz[index] = Dx[index]/2.0;
          } else if ( index == 2 ) {
            i_scalar_bc_storage->second.relative_xyz[index] = Dx[index]/2.0;
          }
          Vector ref_point = i_scalar_bc_storage->second.relative_xyz;
          Point xyz(ref_point[0],ref_point[1],ref_point[2]);

          IntVector ijk = patch->getLevel()->getCellIndex( xyz );

          i_scalar_bc_storage->second.relative_ijk = ijk;
          i_scalar_bc_storage->second.relative_ijk[index] = 0;  //don't allow the normal index to shift

          int face_index_value=10;

          //now check to make sure that there is a bc set for each iterator:
          for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){
            //The next three lines are needed because we are ignoring the user input
            //for the normal index but still loading it into memory
            IntVector mod_bound_ptr = (*bound_ptr);
            face_index_value = mod_bound_ptr[index];
            mod_bound_ptr[index] = (i_scalar_bc_storage->second.values.begin()->first)[index];
            BoundaryCondition_new::CellToValueMap::iterator check_iter = i_scalar_bc_storage->second.values.find(mod_bound_ptr - i_scalar_bc_storage->second.relative_ijk);
            if ( check_iter == i_scalar_bc_storage->second.values.end() ){
              std::stringstream out;
              out <<  "Scalar BC: " << variable << " - No UINTAH boundary cell " << *bound_ptr - i_scalar_bc_storage->second.relative_ijk << " in the handoff file." << std::endl;
              if ( !file_is_open ){
                file_is_open = true;
                outputfile.open(fname.str().c_str());
                outputfile << "Patch Dimentions (exclusive): \n";
                outputfile << " low  = " << patch->getCellLowIndex() << "\n";
                outputfile << " high = " << patch->getCellHighIndex() << "\n";
                outputfile << out.str();
              } else {
                outputfile << out.str();
              }
            }
          }

          //now check the reverse -- does the handoff file have an associated boundary ptr
          BoundaryCondition_new::CellToValueMap temp_map;
          for ( BoundaryCondition_new::CellToValueMap::iterator check_iter = i_scalar_bc_storage->second.values.begin(); check_iter !=
              i_scalar_bc_storage->second.values.end(); check_iter++ ){

            //need to reset the values to get the right [index] int value for the face
            double value = check_iter->second;
            IntVector location = check_iter->first;
            location[index] = face_index_value;

            temp_map.insert(std::make_pair(location, value));

          }

          //reassign the values now with the correct index for the face direction
          i_scalar_bc_storage->second.values = temp_map;

          for ( BoundaryCondition_new::CellToValueMap::iterator check_iter = i_scalar_bc_storage->second.values.begin(); check_iter !=
              i_scalar_bc_storage->second.values.end(); check_iter++ ){

            bool found_it = false;
            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){
              if ( *bound_ptr == (check_iter->first + i_scalar_bc_storage->second.relative_ijk) )
                found_it = true;
            }
            if ( !found_it && patch->containsCell(check_iter->first + i_scalar_bc_storage->second.relative_ijk) ){
              std::stringstream out;
              out << "Scalar BC: " << variable << " - No HANDOFF cell " << check_iter->first << " (relative) in the Uintah geometry object." << std::endl;
              if ( !file_is_open ){
                file_is_open = true;
                outputfile.open(fname.str().c_str());
                outputfile << "Patch Dimentions (exclusive): \n";
                outputfile << " low  = " << patch->getCellLowIndex() << "\n";
                outputfile << " high = " << patch->getCellHighIndex() << "\n";
                outputfile << out.str();
              } else {
                outputfile << out.str();
              }
            }
          }

        }
        if ( file_is_open ){
          std::cout << "\n  Notice: Handoff scalar " << variable << " has warning information printed to file for patch #: " << patch->getID() << "\n";
          outputfile.close();
        }
      }
    }
  }

//______________________________________________________________________
// Explicit template instantiations:
template
void BoundaryCondition_new::setExtraCellScalarValueBC <double> ( const ProcessorGroup* pc,
                                                                 const Patch* patch,
                                                                 CCVariable< double >& scalar,
                                                                 const std::string varname,
                                                                 bool  change_bc,
                                                                 const std::string override_bc);

template
void BoundaryCondition_new::setExtraCellScalarValueBC <float> ( const ProcessorGroup* pc,
                                                                const Patch* patch,
                                                                CCVariable< float >& scalar,
                                                                const std::string varname,
                                                                bool  change_bc,
                                                                const std::string override_bc);

} //namespace Uintah
