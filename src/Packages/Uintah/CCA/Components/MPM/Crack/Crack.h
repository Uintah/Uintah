// Crack.h

#ifndef UINTAH_HOMEBREW_CRACK_H
#define UNITAH_HOMEBREW_CRACK_H

#include <Packages/Uintah/Core/Grid/ComputeSet.h>

#include <iomanip>
#include <iostream>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>

#define MNM 10   // maximum number of materials

namespace Uintah {
using namespace SCIRun;
using std::vector;
using std::string;

   class DataWarehouse;
   class MPMLabel;
   class ProcessorGroup;
   class Patch;
   class VarLabel;
   class Task;

class Crack
{
 public:
 
    // Constructor
     Crack(const ProblemSpecP& ps, SimulationStateP& d_sS,
                           MPMLabel* lb,int n8or27);

    // Destructor
     ~Crack();

    // Basic crack methods
    void addComputesAndRequiresCrackDiscretization(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;  
    void CrackDiscretization(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresParticleVelocityField(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void ParticleVelocityField(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresAdjustCrackContactInterpolated(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void AdjustCrackContactInterpolated(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresAdjustCrackContactIntegrated(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void AdjustCrackContactIntegrated(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresCalculateJIntegral(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void CalculateJIntegral(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresPropagateCracks(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void PropagateCracks(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresInitializeMovingCracks(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void InitializeMovingCracks(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresMoveCracks(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void MoveCracks(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresUpdateCrackExtentAndNormals(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void UpdateCrackExtentAndNormals(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

 private:

     // Prevent copying of this class
     // copy constructor
     Crack(const Crack &con);
     Crack& operator=(const Crack &con);

     // Data members
     SimulationStateP d_sharedState;

     int d_8or27;
     int NGP;
     int NGN;

     // member data of cracks
     string crackType[MNM];                //crack contact type
     double c_mu[MNM];                     //Frcition coefficients
     double separateVol[MNM];              //critical separate volume
     double contactVol[MNM];               //critical contact volume

     //crack geometry
     //quadrilateral segments of cracks
     vector<vector<Point> > rectangles[MNM];
     //resolution of quadrilateral cracks in 1-2 & 2-3 directions 
     vector<int>  rectN12[MNM],rectN23[MNM];       
     //sides at crack front
     vector<vector<short> > rectCrackSidesAtFront[MNM];
     
     //trianglular segements of cracks 
     vector<vector<Point> > triangles[MNM];
     //resolution of triangular cracks in all sides 
     vector<int>  triNCells[MNM];
     //sides at crack front
     vector<vector<short> > triCrackSidesAtFront[MNM];
     Point cmin[MNM],cmax[MNM];            //crack extent

     // crack data after mesh  
     vector<Point> cx[MNM];                //crack node position
     int cnumElems[MNM];                   //number of carck elements
     int cnumNodes[MNM];                   //number of crack points
     vector<IntVector> cElemNodes[MNM];    //nodes of crack elements
     vector<Vector> cElemNorm[MNM];        //crack element normals
     vector<Point> cFrontSegPoints[MNM];   //crack front segments
     vector<short> moved[MNM];             //if crack points moved

     // private methods  
     // Calculate normal of a triangle
     Vector TriangleNormal(const Point&,const Point&,const Point&);
     // Detect if particle and node in same side of a plane
     short Location(const Point&,const Point&,
                       const Point&,const Point&,const Point&);
     //compute signed volume of a tetrahedron
     double Volume(const Point&,const Point&,const Point&,const Point&);
     // a private function
     IntVector CellOffset(const Point&, const Point&, Vector);
     // detect if a line is in another line
     short TwoLinesDuplicate(const Point&,const Point&,const Point&,const Point&);

 protected:
     
     MPMLabel* lb;
};

}//End namespace Uintah

#endif  /* __CRACK_H__*/
