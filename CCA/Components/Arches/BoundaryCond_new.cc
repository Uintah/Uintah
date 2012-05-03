#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SchedulerP.h>

#include <Core/Exceptions/InvalidState.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/IO/UintahZlibUtil.h>
#include <Core/Parallel/Parallel.h>


//===========================================================================

using namespace std;
using namespace Uintah;

BoundaryCondition_new::BoundaryCondition_new(const ArchesLabel* fieldLabels):
  d_fieldLabels(fieldLabels)
{} 

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
    for ( ProblemSpecP db_face = db_bc->findBlock("Face"); db_face != 0; 
          db_face = db_face->findNextBlock("Face") ){
      
      for ( ProblemSpecP db_BCType = db_face->findBlock("BCType"); db_BCType != 0; 
          db_BCType = db_BCType->findNextBlock("BCType") ){

        std::string name; 
        std::string type; 
        db_BCType->getAttribute("label", name);
        db_BCType->getAttribute("var", type); 

        if ( name == eqn_name ){ 

          if ( type == "FromFile" ){ 

            //Check reference file for this scalar
            std::string file_name;
            db_BCType->require("inputfile", file_name); 

            gzFile file = gzopen( file_name.c_str(), "r" ); 
            int total_variables;

            if ( file == NULL ) { 
              proc0cout << "Error opening file: " << file_name << " for boundary conditions. Errno: " << errno << endl;
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
              stringstream err_msg; 
              err_msg << "Error: Unable to find BC input file for scalar: " << eqn_name << " Check this file for errors: \n" << file_name << endl;
              throw ProblemSetupException( err_msg.str(), __FILE__, __LINE__);
            } 

            //If file is found, now create a map from index to value
            CellToValueMap bc_values;  
            bc_values = readInputFile( eqn_input_file ); 

            scalar_bc_from_file.insert(make_pair(eqn_name, bc_values)); 

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
}
std::map<IntVector, double>
BoundaryCondition_new::readInputFile( std::string file_name )
{

  gzFile file = gzopen( file_name.c_str(), "r" ); 
  if ( file == NULL ) { 
    proc0cout << "Error opening file: " << file_name << " for boundary conditions. Errno: " << errno << endl;
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

    result.insert( make_pair( C, v ));

  }

  gzclose( file ); 
  return result; 
}
//---------------------------------------------------------------------------
// Method: Set Scalar BC values 
//---------------------------------------------------------------------------
void BoundaryCondition_new::setScalarValueBC( const ProcessorGroup*,
    const Patch* patch,
    CCVariable<double>& scalar, 
    string varname )
{
  // This method sets the value of the scalar in the boundary cell
  // so that the boundary condition set in the input file is satisfied. 
  vector<Patch::FaceType>::const_iterator iter;
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  Vector Dx = patch->dCell(); 

  int archIndex = 0; 
  int mat_id = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

  for (iter = bf.begin(); iter !=bf.end(); iter++){
    Patch::FaceType face = *iter;

    //get the face direction
    IntVector insideCellDir = patch->faceDirection(face);
    //get the number of children
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id); //assumed one material

    for (int child = 0; child < numChildren; child++){
      double bc_value = -9; 
      string bc_kind = "NotSet";
      Iterator bound_ptr;

      bool foundIterator = 
        getIteratorBCValueBCKind( patch, face, child, varname, mat_id, bc_value, bound_ptr, bc_kind); 

      if (foundIterator) {
        // --- notation --- 
        // bp1: boundary cell + 1 or the interior cell one in from the boundary
        if (bc_kind == "Dirichlet") {

          for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
            IntVector bp1(*bound_ptr - insideCellDir);
            scalar[*bound_ptr] = 2.0*bc_value - scalar[bp1];
          }

        } else if (bc_kind == "Neumann") {

          double dx = 0.0;
          double the_sign = 1.0; 

          switch (face) {
            case Patch::xminus:
              dx = Dx.x(); 
              the_sign = -1.0;
              break; 
            case Patch::xplus:
              dx = Dx.x(); 
              break; 
            case Patch::yminus:
              dx = Dx.y(); 
              the_sign = -1.0; 
              break; 
            case Patch::yplus:
              dx = Dx.y(); 
              break; 
            case Patch::zminus:
              dx = Dx.z(); 
              the_sign = -1.0; 
              break; 
            case Patch::zplus:
              dx = Dx.z(); 
              break; 
            default: 
              throw InvalidValue("Error: Face type not recognized.",__FILE__,__LINE__); 
              break; 
          }

          for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
            IntVector bp1(*bound_ptr - insideCellDir); 
            scalar[*bound_ptr] = scalar[bp1] + the_sign * dx * bc_value;
          }
        } else if (bc_kind == "FromFile") { 

          ScalarToBCValueMap::iterator i_scalar_bc_storage = scalar_bc_from_file.find( varname ); 

          for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {

            CellToValueMap::iterator iter = i_scalar_bc_storage->second.find( *bound_ptr ); //<----WARNING ... May be slow here
            if ( iter != i_scalar_bc_storage->second.end() ){ 

              double file_bc_value = iter->second; 
              IntVector bp1(*bound_ptr - insideCellDir);
              scalar[*bound_ptr] = 2.0 * file_bc_value - scalar[bp1]; 

            } else { 
              // THIS NEED TO BE FIXED: 
              //cout << " For cell: " << *bound_ptr << endl;
              //throw InvalidValue("Error: Cell not found in BC input file for scalar: "+ varname,__FILE__,__LINE__); 
              scalar[*bound_ptr] = 0;
            } 
          }
        } 
      }
    }
  }
}

//---------------------------------------------------------------------------
// Method: Set Cell Centered Vector BC values 
//---------------------------------------------------------------------------
void BoundaryCondition_new::setVectorValueBC( const ProcessorGroup*,
    const Patch* patch,
    CCVariable<Vector>& vec, 
    string varname )
{
  // This method sets the value of the CELL-CENTERED VECTOR components in the boundary cell
  // so that the boundary condition set in the input file is satisfied. 
  vector<Patch::FaceType>::const_iterator iter;
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  Vector Dx = patch->dCell(); 

  int archIndex = 0; 
  int mat_id = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

  for (iter = bf.begin(); iter !=bf.end(); iter++){
    Patch::FaceType face = *iter;

    //get the face direction
    IntVector insideCellDir = patch->faceDirection(face);
    //get the number of children
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id); //assumed one material

    for (int child = 0; child < numChildren; child++){
      Vector bc_value = Vector(0.0, 0.0, 0.0);
      string bc_kind = "NotSet";
      Iterator bound_ptr;

      bool foundIterator = 
        getIteratorBCValueBCKind( patch, face, child, varname, mat_id, bc_value, bound_ptr, bc_kind); 

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

          double dx = 0.0;
          double the_sign = 1.0; 

          switch (face) {
            case Patch::xminus:
              dx = Dx.x(); 
              the_sign = -1.0;
              break; 
            case Patch::xplus:
              dx = Dx.x(); 
              break; 
            case Patch::yminus:
              dx = Dx.y(); 
              the_sign = -1.0; 
              break; 
            case Patch::yplus:
              dx = Dx.y(); 
              break; 
            case Patch::zminus:
              dx = Dx.z(); 
              the_sign = -1.0; 
              break; 
            case Patch::zplus:
              dx = Dx.z(); 
              break; 
            default: 
              throw InvalidValue("Error: Face type not recognized.",__FILE__,__LINE__); 
              break; 
          }

          for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
            
            IntVector bp1(*bound_ptr - insideCellDir); 
            
            X = vec[bp1].x() + the_sign * dx * bc_value.x();
            Y = vec[bp1].y() + the_sign * dx * bc_value.y();
            Z = vec[bp1].z() + the_sign * dx * bc_value.z();
          
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
    string varname )
{

  vector<Patch::FaceType>::const_iterator iter;
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  Vector Dx = patch->dCell(); 

  int archIndex = 0; 
  int mat_id = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

  for (iter = bf.begin(); iter !=bf.end(); iter++){
    Patch::FaceType face = *iter;

    //get the face direction
    IntVector insideCellDir = patch->faceDirection(face);
    //get the number of children
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id); //assumed one material

    for (int child = 0; child < numChildren; child++){
      Vector bc_value = Vector(0.0, 0.0, 0.0);
      string bc_kind = "NotSet";
      Iterator bound_ptr;

      bool foundIterator = 
        getIteratorBCValueBCKind( patch, face, child, varname, mat_id, bc_value, bound_ptr, bc_kind); 

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
          
          double dX, dY, dZ; 

          switch (face) {
            case Patch::xminus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir); 

                dX = ( const_vec[bp1].x() - const_vec[*bound_ptr].x() ) / Dx.x(); 
                dY = ( const_vec[bp1].y() - const_vec[*bound_ptr].y() ) / Dx.x(); 
                dZ = ( const_vec[bp1].z() - const_vec[*bound_ptr].z() ) / Dx.x(); 

                X = vec[bp1].x() - Dx.x() * dX; 
                Y = vec[bp1].y() - Dx.x() * dY; 
                Z = vec[bp1].z() - Dx.x() * dZ; 

                vec[*bound_ptr] = Vector(X,Y,Z); 
              }
              break;
            case Patch::xplus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir);

                dX = ( const_vec[*bound_ptr].x() - const_vec[bp1].x() ) / Dx.x();
                dY = ( const_vec[*bound_ptr].y() - const_vec[bp1].y() ) / Dx.x();
                dZ = ( const_vec[*bound_ptr].z() - const_vec[bp1].z() ) / Dx.x();

                X = vec[bp1].x() + Dx.x() * dX; 
                Y = vec[bp1].y() + Dx.x() * dY; 
                Z = vec[bp1].z() + Dx.x() * dZ; 

                vec[*bound_ptr] = Vector(X,Y,Z); 
              }
              break;
#ifdef YDIM
            case Patch::yminus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir); 

                dX = ( const_vec[bp1].x() - const_vec[*bound_ptr].x() ) / Dx.y(); 
                dY = ( const_vec[bp1].y() - const_vec[*bound_ptr].y() ) / Dx.y(); 
                dZ = ( const_vec[bp1].z() - const_vec[*bound_ptr].z() ) / Dx.y(); 

                X = vec[bp1].x() - Dx.y() * dX; 
                Y = vec[bp1].y() - Dx.y() * dY; 
                Z = vec[bp1].z() - Dx.y() * dZ; 

                vec[*bound_ptr] = Vector(X,Y,Z); 
              }
              break;
            case Patch::yplus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir);

                dX = ( const_vec[*bound_ptr].x() - const_vec[bp1].x() ) / Dx.y();
                dY = ( const_vec[*bound_ptr].y() - const_vec[bp1].y() ) / Dx.y();
                dZ = ( const_vec[*bound_ptr].z() - const_vec[bp1].z() ) / Dx.y();

                X = vec[bp1].x() + Dx.y() * dX; 
                Y = vec[bp1].y() + Dx.y() * dY; 
                Z = vec[bp1].z() + Dx.y() * dZ; 

                vec[*bound_ptr] = Vector(X,Y,Z); 
              }
              break;
#endif
#ifdef ZDIM
            case Patch::zminus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir); 

                dX = ( const_vec[bp1].x() - const_vec[*bound_ptr].x() ) / Dx.z(); 
                dY = ( const_vec[bp1].y() - const_vec[*bound_ptr].y() ) / Dx.z(); 
                dZ = ( const_vec[bp1].z() - const_vec[*bound_ptr].z() ) / Dx.z(); 

                X = vec[bp1].x() - Dx.z() * dX; 
                Y = vec[bp1].y() - Dx.z() * dY; 
                Z = vec[bp1].z() - Dx.z() * dZ; 

                vec[*bound_ptr] = Vector(X,Y,Z); 
              }
              break;
            case Patch::zplus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir);

                dX = ( const_vec[*bound_ptr].x() - const_vec[bp1].x() ) / Dx.z();
                dY = ( const_vec[*bound_ptr].y() - const_vec[bp1].y() ) / Dx.z();
                dZ = ( const_vec[*bound_ptr].z() - const_vec[bp1].z() ) / Dx.z();

                X = vec[bp1].x() + Dx.z() * dX; 
                Y = vec[bp1].y() + Dx.z() * dY; 
                Z = vec[bp1].z() + Dx.z() * dZ; 

                vec[*bound_ptr] = Vector(X,Y,Z); 
              }
              break;
#endif
          default: 
            throw InvalidValue("Error: Face type not recognized.",__FILE__,__LINE__); 
            break; 
          }
        }
      }
    }
  }
}

//---------------------------------------------------------------------------
// Method: Set the area fraction for all cells
//---------------------------------------------------------------------------
void BoundaryCondition_new::setAreaFraction( 
    const Patch* patch,
    CCVariable<Vector>& areaFraction, 
    CCVariable<double>&  volFraction, 
    constCCVariable<int>& cellType, 
    const int wallType, 
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

    // curr cell is a wall 
    if ( cellType[c] == wallType ) {
      areaFraction[c] = Vector(0.,0.,0.);
      volFraction[c]  = 0.0;
    }

    // x-minus is a wall but curr cell is flow 
    if ( cellType[c] == flowType && cellType[cxm] == wallType ) {
      Vector T = areaFraction[c]; 
      T[0] = 0.;
      areaFraction[c] = T;
    }

    // y-minus is a wall but curr cell is flow
    if ( cellType[c] == flowType && cellType[cym] == wallType ) {
      Vector T = areaFraction[c]; 
      T[1] = 0.;
      areaFraction[c] = T;
    }

    // z-minus is a wall but curr cell is flowType
    if (cellType[c] == flowType && cellType[czm] == wallType ) {
      Vector T = areaFraction[c]; 
      T[2] = 0.;
      areaFraction[c] = T;
    }
  }
  //__________________________________
  // loop over computational domain faces
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  
  for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){
    Patch::FaceType face = *iter;
    
    Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;
    
    for (CellIterator iter =  patch->getFaceIterator(face, PEC); !iter.done(); iter++) {
      IntVector c = *iter;
      if ( cellType[c] == wallType ){

        int P_dir = patch->getFaceAxes(face)[0];  //principal dir.

        Vector T = areaFraction[c]; 
        T[P_dir] = 0.;
        areaFraction[c] = T; 

      }
    }
  }
}

//---------------------------------------------------------------------------
// Method: Compute the area of the boundary specification
//---------------------------------------------------------------------------
void BoundaryCondition_new::sched_computeBCArea( SchedulerP& sched, 
                                                 const PatchSet* patches, 
                                                 const MaterialSet* matls )
{
  Task* tsk = scinew Task( "BoundaryCondition_new::computeBCArea", this, 
                           &BoundaryCondition_new::computeBCArea ); 

  sched->addTask( tsk, patches, matls ); 

}

void BoundaryCondition_new::computeBCArea( const ProcessorGroup*, 
                                           const PatchSubset* patches, 
                                           const MaterialSubset*, 
                                           DataWarehouse*, 
                                           DataWarehouse* new_dw ) 
{

  for ( int p = 0; p < patches->size(); p++ ) { 

    const Patch* patch = patches->get(p); 
    int archIndex = 0; 
    int indx = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    vector<Patch::FaceType>::const_iterator iter;
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);
    Vector Dx = patch->dCell(); 

    for (iter = bf.begin(); iter !=bf.end(); iter++){
      
      Patch::FaceType face = *iter;
      int numChildren = patch->getBCDataArray(face)->getNumberChildren(indx); //assumed one material

      for (int child = 0; child < numChildren; child++){

        std::string varname = "velocity";
        std::string bc_kind = ""; 
        double bc_value; 
        Iterator bound_ptr;

        bool foundIterator = 
          getIteratorBCValueBCKind( patch, face, child, varname, indx, bc_value, bound_ptr, bc_kind); 

        if (foundIterator) { 

          std::string bc_name = patch->getBCDataArray(face)->getChild(indx, child)->getBCName();  //get name

          // note we are only computing area if a velocity bc is found. 
          
          if ( bc_name == "NotSet" ) {
            std::ostringstream oerr; 
            oerr << "Error: All BCs containing velocity must be named. " << endl 
              << "Please set the name attribute for each <Face> node in <BoundaryConditions>. " << endl;
            throw ProblemSetupException( oerr.str(), __FILE__, __LINE__ );  
          }
       
          double area = 0.0; 

          switch (face) {
            case Patch::xminus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {

                area += Dx.y() * Dx.z(); 

              }
              break;
            case Patch::xplus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {

                area += Dx.y() * Dx.z(); 

              }
              break;
#ifdef YDIM
            case Patch::yminus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {

                area += Dx.x() * Dx.z(); 

              }
              break;
            case Patch::yplus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {

                area += Dx.x() * Dx.z(); 

              }
              break;
#endif
#ifdef ZDIM
            case Patch::zminus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {

                area += Dx.x() * Dx.y(); 

              }
              break;
            case Patch::zplus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {

                area += Dx.x() * Dx.y(); 

              }
              break;
#endif
            default: 
              throw InvalidValue("Error: Face type not recognized.",__FILE__,__LINE__); 
              break; 
          } // end switch statement
        } // found boundary object
      } // iterator over children
    } // iterator over faces
  }
}

