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


    void addComputesAndRequiresCrackAdjustInterpolated(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void CrackContactAdjustInterpolated(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresCrackAdjustIntegrated(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void CrackContactAdjustIntegrated(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

    void addComputesAndRequiresMoveCrack(Task* task,
                                const PatchSet* patches,
                                const MaterialSet* matls) const;
    void MoveCrack(const ProcessorGroup*,
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

     // Data members for cracks
     Point cmin[MNM],cmax[MNM];             //extent of crack plane
     int numElems[MNM];            //number of carck elements    
     int numPts[MNM];              //number of crack points
     double c_mu[MNM];             //Frcition coefficient
     double separateVol[MNM];      //normal volume to seperate
     double contactVol[MNM];       //normal volume to contact
     string crackType[MNM];        //crack contact type
     Point cx[MNM][1000];             //crack position
     IntVector cElemNodes[MNM][1000]; //crack element nodes
     Vector cElemNorm[MNM][1000];     //crack element normals     
 
     //crack geometry
     //quadrilateral segments of cracks
     vector<vector<Point> > allRects[MNM];
 
     //resolution of quadrilateral cracks in 12 direstion 
     vector<int>  allN12[MNM];       
     
     //resolution of quadrilateral cracks in 23 direction 
     vector<int>  allN23[MNM];       
     
     //trianglular segements of cracks 
     vector<vector<Point> > allTris[MNM];

     //resolution of triangular cracks in all sides 
     vector<int>  allNCell[MNM];

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

 protected:
     
     MPMLabel* lb;
};

}//End namespace Uintah

#endif  /* __CRACK_H__*/
