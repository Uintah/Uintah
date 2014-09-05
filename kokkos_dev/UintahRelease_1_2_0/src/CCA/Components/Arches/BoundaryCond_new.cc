#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Exceptions/InvalidState.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Task.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>

//===========================================================================

using namespace std;
using namespace Uintah;
using namespace SCIRun; 

BoundaryCondition_new::BoundaryCondition_new(const ArchesLabel* fieldLabels):
  d_fieldLabels(fieldLabels)
{} 

BoundaryCondition_new::~BoundaryCondition_new()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup 
//---------------------------------------------------------------------------
void BoundaryCondition_new::problemSetup()
{}
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
          switch (face) {
            case Patch::xminus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir); 
                scalar[*bound_ptr] = 2.0*bc_value - scalar[bp1];
              }
              break;
            case Patch::xplus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir);
                scalar[*bound_ptr] = 2.0*bc_value - scalar[bp1];
              }
              break;
#ifdef YDIM
            case Patch::yminus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir);
                scalar[*bound_ptr] = 2.0*bc_value - scalar[bp1];
              }
              break;
            case Patch::yplus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir);
                scalar[*bound_ptr] = 2.0*bc_value - scalar[bp1];
              }
              break;
#endif
#ifdef ZDIM
            case Patch::zminus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir);
                scalar[*bound_ptr] = 2.0*bc_value - scalar[bp1];
              }
              break;
            case Patch::zplus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir);
                scalar[*bound_ptr] = 2.0*bc_value - scalar[bp1];
              }
              break;
#endif
          case Patch::numFaces:
            SCI_THROW(InvalidState("numFaces is not a valid face",__FILE__,__LINE__));
            break;
          case Patch::invalidFace:
            SCI_THROW(InvalidState("invalidFace is not a valid face",__FILE__,__LINE__));
            break;
          }

        } else if (bc_kind == "Neumann") {
          switch (face) {
            case Patch::xminus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir); 
                scalar[*bound_ptr] = scalar[bp1] - Dx.x()*bc_value;
              }
              break;
            case Patch::xplus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir);
                scalar[*bound_ptr] = Dx.x()*bc_value + scalar[bp1];
              }
              break;
#ifdef YDIM
            case Patch::yminus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir); 
                scalar[*bound_ptr] = scalar[bp1] - Dx.y()*bc_value;
              }
              break;
            case Patch::yplus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir);
                scalar[*bound_ptr] = Dx.y()*bc_value + scalar[bp1];
              }
              break;
#endif
#ifdef ZDIM
            case Patch::zminus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir); 
                scalar[*bound_ptr] = scalar[bp1] - Dx.z()*bc_value;
              }
              break;
            case Patch::zplus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir);
                scalar[*bound_ptr] = Dx.z()*bc_value + scalar[bp1];
              }
              break;
#endif
          case Patch::numFaces:
            SCI_THROW(InvalidState("numFaces is not a valid face",__FILE__,__LINE__));
            break;
          case Patch::invalidFace:
            SCI_THROW(InvalidState("invalidFace is not a valid face",__FILE__,__LINE__));
            break;
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
          switch (face) {
            case Patch::xminus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir); 
                X = vec[bp1].x() - Dx.x()*bc_value.x();
                Y = vec[bp1].y() - Dx.x()*bc_value.y();
                Z = vec[bp1].z() - Dx.x()*bc_value.z();
                vec[*bound_ptr] = Vector(X,Y,Z); 
              }
              break;
            case Patch::xplus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir);
                X = Dx.x()*bc_value.x() + vec[bp1].x();
                Y = Dx.x()*bc_value.y() + vec[bp1].y();
                Z = Dx.x()*bc_value.z() + vec[bp1].z();
                vec[*bound_ptr] = Vector(X,Y,Z); 
              }
              break;
#ifdef YDIM
            case Patch::yminus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir); 
                X = vec[bp1].x() - Dx.y()*bc_value.x();
                Y = vec[bp1].y() - Dx.y()*bc_value.y();
                Z = vec[bp1].z() - Dx.y()*bc_value.z();
                vec[*bound_ptr] = Vector(X,Y,Z); 
              }
              break;
            case Patch::yplus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir);
                X = Dx.y()*bc_value.x() + vec[bp1].x();
                Y = Dx.y()*bc_value.y() + vec[bp1].y();
                Z = Dx.y()*bc_value.z() + vec[bp1].z();
                vec[*bound_ptr] = Vector(X,Y,Z); 
              }
              break;
#endif
#ifdef ZDIM
            case Patch::zminus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir); 
                X = vec[bp1].x() - Dx.z()*bc_value.x();
                Y = vec[bp1].y() - Dx.z()*bc_value.y();
                Z = vec[bp1].z() - Dx.z()*bc_value.z();
                vec[*bound_ptr] = Vector(X,Y,Z); 
              }
              break;
            case Patch::zplus:
              for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                IntVector bp1(*bound_ptr - insideCellDir);
                X = Dx.z()*bc_value.x() + vec[bp1].x();
                Y = Dx.z()*bc_value.y() + vec[bp1].y();
                Z = Dx.z()*bc_value.z() + vec[bp1].z();
                vec[*bound_ptr] = Vector(X,Y,Z); 
              }
              break;
#endif
          case Patch::numFaces:
            break;
          case Patch::invalidFace:
            break;
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
          case Patch::numFaces:
            break;
          case Patch::invalidFace:
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
    constCCVariable<int>& cellType, 
    const int wallType, 
    const int flowType )
{

  // areaFraction is a vector with:
  // areaFraction.x = xminus area fraction
  // areaFraction.y = yminus area fraction
  // areaFraction.z = zminus area fraction 

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    
    IntVector c = *iter;
    IntVector cxm = *iter - IntVector(1,0,0);
    IntVector cym = *iter - IntVector(0,1,0);
    IntVector czm = *iter - IntVector(0,0,1); 

    // curr cell is a wall 
    if ( cellType[c] == wallType )
      areaFraction[c] = Vector(0.,0.,0.);

    // x-minus is a wall but curr cell is flow 
    if ( cellType[c] == flowType && cellType[cxm] == wallType ) {
      Vector tempV = areaFraction[c]; 
      tempV[0] = 0.;
      areaFraction[c] = tempV;
    }

    // y-minus is a wall but curr cell is flow
    if ( cellType[c] == flowType && cellType[cym] == wallType ) {
      Vector tempV = areaFraction[c]; 
      tempV[1] = 0.;
      areaFraction[c] = tempV;
    }

    // z-minus is a wall but curr cell is flowType
    if (cellType[c] == flowType && cellType[czm] == wallType ) {
      Vector tempV = areaFraction[c]; 
      tempV[2] = 0.;
      areaFraction[c] = tempV;
    }
  }
}
