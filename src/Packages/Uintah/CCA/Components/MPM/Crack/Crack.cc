// Crack.cc 
#include "Crack.h"
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h> // for Fracture
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Util/NotFinished.h>
#include <vector>
#include <iostream>
#include <fstream>

using namespace Uintah;
using namespace SCIRun;
using std::vector;
using std::string;

using namespace std;

#define MAX_BASIS 27

MPI_Comm mpi_crack_comm;

Crack::Crack(const ProblemSpecP& ps,SimulationStateP& d_sS,
                           MPMLabel* Mlb,int n8or27)
{
  MPI_Comm_dup( MPI_COMM_WORLD, & mpi_crack_comm );

  // Constructor
  d_sharedState = d_sS;
  lb = Mlb;
  d_8or27=n8or27;
  if(d_8or27==8){
    NGP=1;
    NGN=1;
  } else if(d_8or27==MAX_BASIS){
    NGP=2;
    NGN=2;
  }
 
  // Default values of parameters for fracture analysis 
  rdadx=1.;   // Ratio of crack increment of every time to cell-size
  rJ=-1.;     // Radius of J-path circle
  NJ=2;       // Number of cells J-path away from crack tip
  mS=0;       // MatErial ID for saving J integral
  d_useVolumeIntegral=false; // No using volume integral in J-integral calculation
  d_SMALL_NUM_MPM=1e-200;

  for(Patch::FaceType face = Patch::startFace;
       face<=Patch::endFace; face=Patch::nextFace(face)) {
    GridBCType[face]="None";
  }

  GLP=Point(-9e99,-9e99,-9e99);  // the lowest point of the global grid
  GHP=Point( 9e99, 9e99, 9e99);  // the highest point of the global grid

  d_calFractParameters = "false"; // Flag for calculatinf J
  d_doCrackPropagation = "false"; // Flag for doing crack propagation
  d_outputCrackResults = "false"; // Flag for outputing crack front

  // Read in MPM parameters related to fracture analysis
  ProblemSpecP mpm_soln_ps = ps->findBlock("MPM");
  if(mpm_soln_ps) {
     mpm_soln_ps->get("calculate_fracture_parameters", d_calFractParameters);
     mpm_soln_ps->get("do_crack_propagation", d_doCrackPropagation);
     mpm_soln_ps->get("useVolumeIntegral", d_useVolumeIntegral);
     mpm_soln_ps->get("J_radius", rJ);
     mpm_soln_ps->get("save_J_matID", mS);
     mpm_soln_ps->get("dadx",rdadx);
  }

  if(d_calFractParameters!="false" || d_doCrackPropagation!="false")
    d_outputCrackResults = "true";
 
  // Read in the lowest and highest points of the global grid
  ProblemSpecP grid_level_ps = ps->findBlock("Grid")
                           ->findBlock("Level")->findBlock("Box");
  grid_level_ps->get("lower",GLP);
  grid_level_ps->get("upper",GHP);

  // Read in the BC Types of the global grid
  ProblemSpecP grid_bc_ps = ps->findBlock("Grid")
                           ->findBlock("BoundaryConditions");
  for(ProblemSpecP face_ps = grid_bc_ps->findBlock("Face"); face_ps != 0;
                   face_ps = face_ps->findNextBlock("Face")) {
    map<string,string> values;
    face_ps->getAttributes(values);
    ProblemSpecP bcType_ps = face_ps->findBlock("BCType");
    map<string,string> bc_attr;
    bcType_ps->getAttributes(bc_attr);
    if(values["side"]=="x-")      GridBCType[Patch::xminus]=bc_attr["var"];  
    else if(values["side"]=="x+") GridBCType[Patch::xplus] =bc_attr["var"]; 
    else if(values["side"]=="y-") GridBCType[Patch::yminus]=bc_attr["var"];     
    else if(values["side"]=="y+") GridBCType[Patch::yplus] =bc_attr["var"];     
    else if(values["side"]=="z-") GridBCType[Patch::zminus]=bc_attr["var"];     
    else if(values["side"]=="z+") GridBCType[Patch::zplus] =bc_attr["var"];     
  }

  // Allocate memory for crack geometry data
  int numMPMMatls=0; 
  ProblemSpecP mpm_ps = ps->findBlock("MaterialProperties")->findBlock("MPM");
  for(ProblemSpecP mat_ps=mpm_ps->findBlock("material"); mat_ps!=0;
                   mat_ps=mat_ps->findNextBlock("material") ) numMPMMatls++;
  crackType.resize(numMPMMatls); 
  cmu.resize(numMPMMatls);
  separateVol.resize(numMPMMatls);
  contactVol.resize(numMPMMatls);
  rectangles.resize(numMPMMatls);
  rectN12.resize(numMPMMatls);
  rectN23.resize(numMPMMatls);
  rectCrackSidesAtFront.resize(numMPMMatls);
  triangles.resize(numMPMMatls);
  triNCells.resize(numMPMMatls);
  triCrackSidesAtFront.resize(numMPMMatls);
  arcs.resize(numMPMMatls);
  arcNCells.resize(numMPMMatls);
  arcCrkFrtSegID.resize(numMPMMatls);
  ellipses.resize(numMPMMatls);
  ellipseNCells.resize(numMPMMatls);
  ellipseCrkFrtSegID.resize(numMPMMatls);
  pellipses.resize(numMPMMatls);
  pellipseNCells.resize(numMPMMatls);
  pellipseCrkFrtSegID.resize(numMPMMatls);  
  pellipseExtent.resize(numMPMMatls);
  cmin.resize(numMPMMatls);
  cmax.resize(numMPMMatls);

  // Read in crack parameters, which are placed in element "material"
  int m=0;  // current mat ID
  for(ProblemSpecP mat_ps=mpm_ps->findBlock("material"); mat_ps!=0;
                        mat_ps=mat_ps->findNextBlock("material") ) {
    ProblemSpecP crk_ps=mat_ps->findBlock("crack");
    if(crk_ps==0) crackType[m]="NO_CRACK";
    if(crk_ps!=0) { 
       // Read in crack contact type
       crk_ps->require("type",crackType[m]);
     
       // Read in parameters of crack contact. Use disp criterion 
       // for contact check if separateVol or contactVol less than zero.
       cmu[m]=0.0;   
       separateVol[m]=-1.;
       contactVol[m]=-1.;
       crk_ps->get("separate_volume",separateVol[m]);
       crk_ps->get("contact_volume",contactVol[m]);
       crk_ps->get("mu",cmu[m]);
       if(crackType[m]!="frictional" && crackType[m]!="stick" &&
          crackType[m]!="null") {
          cout << "Unknown crack type: " << crackType[m] << endl;
          exit(1);
       }
        
       // Initialize the arries related to crack geometries
       // for reactangular cracks
       rectangles[m].clear();
       rectN12[m].clear();
       rectN23[m].clear();
       rectCrackSidesAtFront[m].clear();
       // for triangular cracks
       triangles[m].clear();
       triNCells[m].clear();
       triCrackSidesAtFront[m].clear();
       // for arc cracks
       arcs[m].clear();
       arcNCells[m].clear();
       arcCrkFrtSegID[m].clear();
       // for elliptical cracks
       ellipses[m].clear();
       ellipseNCells[m].clear();
       ellipseCrkFrtSegID[m].clear();
       // for partial elliptical cracks
       pellipses[m].clear();
       pellipseExtent[m].clear();
       pellipseNCells[m].clear();
       pellipseCrkFrtSegID[m].clear();

       // Read in geometrical parameters of rectangular cracks
       ProblemSpecP geom_ps=crk_ps->findBlock("crack_segments");
       ReadRectangularCracks(m,geom_ps);
       ReadTriangularCracks(m,geom_ps);
       ReadArcCracks(m,geom_ps);
       ReadEllipticCracks(m,geom_ps);
       ReadPartialEllipticCracks(m,geom_ps);
    } // End of if crk_ps != 0
    m++; // Next material
  }  // End of loop over materials
 
  #if 1  
    OutputInitialCracks(numMPMMatls);
  #endif
}

void Crack::addComputesAndRequiresCrackDiscretization(Task* /*t*/,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
// Do nothing currently 
}

void Crack::CrackDiscretization(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* /*matls*/,
                                DataWarehouse* /*old_dw*/,
                                DataWarehouse* /*new_dw*/)
{
  for(int p=0;p<patches->size();p++) { // All ranks 
    const Patch* patch = patches->get(p);
    int rankSize;
    MPI_Comm_size(mpi_crack_comm,&rankSize);

    // Set radius (rJ) of J-path cirlce or number of cells
    Vector dx = patch->dCell();
    double dx_min=Min(dx.x(),dx.y(),dx.z());
    if(rJ<0.) { // No input from input
      rJ=NJ*dx_min;
    }
    else {      // Input radius of J patch circle
      NJ=(int)(rJ/dx_min);
    }

    // Discretize crack plane 
    // Allocate memory for crack mesh data
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    cs0.resize(numMPMMatls);
    cx.resize(numMPMMatls);
    ce.resize(numMPMMatls);
    cfSegNodes.resize(numMPMMatls);
    cfSegNodesT.resize(numMPMMatls);
    cfSegPtsT.resize(numMPMMatls);
    cfSegV3.resize(numMPMMatls);
    cfSegV2.resize(numMPMMatls);
    cfSegJ.resize(numMPMMatls);
    cfSegK.resize(numMPMMatls);
    cfSegNodesInMat.resize(numMPMMatls);
    cfSegCenterInMat.resize(numMPMMatls);
    cnset.resize(numMPMMatls);
    cfnset.resize(numMPMMatls);
    cfsset.resize(numMPMMatls);

    for(int m = 0; m < numMPMMatls; m++){ 
      cnset[m].resize(rankSize);
      cfnset[m].resize(rankSize);
      cfsset[m].resize(rankSize); 
      // Initialize the arries related to cracks 
      cx[m].clear();
      ce[m].clear();
      cfSegNodes[m].clear();
      cfSegV2[m].clear();
      cmin[m]=Point(9e16,9e16,9e16);
      cmax[m]=Point(-9e16,-9e16,-9e16); 

      if(crackType[m]!="NO_CRACK") {
        // Discretize crack plane
        int nstart0=0;  // Starting node number for each crack geometry

        DiscretizeRectangularCracks(m,nstart0);
        DiscretizeTriangularCracks(m,nstart0);
        DiscretizeArcCracks(m,nstart0);
        DiscretizeEllipticCracks(m,nstart0);
        DiscretizePartialEllipticCracks(m,nstart0);

        // Find extent of crack 
        for(int i=0; i<(int)cx[m].size();i++) {
          cmin[m]=Min(cmin[m],cx[m][i]);
          cmax[m]=Max(cmax[m],cx[m][i]);
        }

        // Smooth crack-front and get tangential vector 
        if(d_calFractParameters!="false" || d_doCrackPropagation!="false")
        SmoothCrackFrontAndGetTangentialVector(m);

        // Get the average length of crack front segments 
        cs0[m]=0.;
        int ncfSegs=(int)cfSegNodes[m].size()/2;
        for(int i=0; i<ncfSegs; i++) {
          int n1=cfSegNodes[m][2*i];
          int n2=cfSegNodes[m][2*i+1];
          cs0[m]+=(cx[m][n1]-cx[m][n2]).length();
        } 
        cs0[m]/=ncfSegs;

        #if 1  
          OutputCrackPlaneMesh(m);
        #endif
      }
    } // End of loop over matls
  } // End of loop over patches
}

void Crack::addComputesAndRequiresParticleVelocityField(Task* t,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{  
  t->requires(Task::OldDW, lb->pXLabel, Ghost::AroundCells, NGN);
  t->computes(lb->gNumPatlsLabel);
  t->computes(lb->GNumPatlsLabel);
  t->computes(lb->GCrackNormLabel);
  t->computes(lb->pgCodeLabel);
}

void Crack::ParticleVelocityField(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* /*matls*/,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int p=0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int numMatls = d_sharedState->getNumMPMMatls();

    // Detect if doing fracture analysis
    double time = d_sharedState->getElapsedTime();
    FindTimeStepForFractureAnalysis(time);

    enum {NO=0,YES};
    enum {SAMESIDE=0,ABOVE_CRACK,BELOW_CRACK};

    Vector dx = patch->dCell(); 
    for(int m=0; m<numMatls; m++) {
      MPMMaterial* mpm_matl=d_sharedState->getMPMMaterial(m);
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      constParticleVariable<Point> px;
      old_dw->get(px, lb->pXLabel, pset);
      ParticleVariable<Short27> pgCode;
      new_dw->allocateAndPut(pgCode,lb->pgCodeLabel,pset);

      NCVariable<int> gNumPatls;
      NCVariable<int> GNumPatls;
      NCVariable<Vector> GCrackNorm;
      new_dw->allocateAndPut(gNumPatls,  lb->gNumPatlsLabel,  dwi, patch);
      new_dw->allocateAndPut(GNumPatls,  lb->GNumPatlsLabel,  dwi, patch);
      new_dw->allocateAndPut(GCrackNorm, lb->GCrackNormLabel, dwi, patch);
      gNumPatls.initialize(0);
      GNumPatls.initialize(0);
      GCrackNorm.initialize(Vector(0,0,0));

      ParticleSubset* psetWGCs = old_dw->getParticleSubset(dwi, patch, 
                                      Ghost::AroundCells, NGN, lb->pXLabel);
      constParticleVariable<Point> pxWGCs;
      old_dw->get(pxWGCs, lb->pXLabel, psetWGCs);

      IntVector ni[MAX_BASIS];

      if((int)ce[m].size()==0) {            // For materials with no cracks
        // set pgCode[idx][k]=1
        for(ParticleSubset::iterator iter=pset->begin();
                                     iter!=pset->end();iter++) {
          for(int k=0; k<d_8or27; k++) pgCode[*iter][k]=1;
        }  
        // Get number of particles around nodes 
        for(ParticleSubset::iterator itr=psetWGCs->begin();
                           itr!=psetWGCs->end();itr++) {
          if(d_8or27==8)
            patch->findCellNodes(pxWGCs[*itr], ni);
          else if(d_8or27==27)
            patch->findCellNodes27(pxWGCs[*itr], ni);
          for(int k=0; k<d_8or27; k++) {
            if(patch->containsNode(ni[k])) 
              gNumPatls[ni[k]]++;
          }
        } //End of loop over partls
      }

      else { // For materials with crack(s)

        //Step 1: Determine if nodes within crack zone 
        Ghost::GhostType  gac = Ghost::AroundCells;
        IntVector g_cmin, g_cmax, cell_idx;
        NCVariable<short> singlevfld;
        new_dw->allocateTemporary(singlevfld, patch, gac, 2*NGN); 
        singlevfld.initialize(0);

        // Get crack extent on grid (g_cmin->g_cmax)
        patch->findCell(cmin[m],cell_idx);
        Point ptmp=patch->nodePosition(cell_idx+IntVector(1,1,1));
        IntVector offset=CellOffset(cmin[m],ptmp,dx);
        g_cmin=cell_idx+IntVector(1-offset.x(),1-offset.y(),1-offset.z());

        patch->findCell(cmax[m],cell_idx);
        ptmp=patch->nodePosition(cell_idx);
        g_cmax=cell_idx+CellOffset(cmax[m],ptmp,dx);

        for(NodeIterator iter=patch->getNodeIterator();!iter.done();iter++) {
          IntVector c=*iter;
          if(c.x()>=g_cmin.x() && c.x()<=g_cmax.x() && c.y()>=g_cmin.y() &&
             c.y()<=g_cmax.y() && c.z()>=g_cmin.z() && c.z()<=g_cmax.z() )
            singlevfld[c]=NO;  // in crack zone
          else
            singlevfld[c]=YES; // in non-crack zone
        }

        /* Step 2: Detect if particle is above, below or in the same side, 
                   count number of particles around nodes */
        NCVariable<int> num0,num1,num2; 
        new_dw->allocateTemporary(num0, patch, gac, 2*NGN);
        new_dw->allocateTemporary(num1, patch, gac, 2*NGN);
        new_dw->allocateTemporary(num2, patch, gac, 2*NGN);
        num0.initialize(0);
        num1.initialize(0);
        num2.initialize(0);
 
        // Generate particle cross code for particles in pset
        for(ParticleSubset::iterator iter=pset->begin();
                                     iter!=pset->end();iter++) {
          particleIndex idx=*iter;
          if(d_8or27==8)
             patch->findCellNodes(px[idx], ni);
          else if(d_8or27==27)
             patch->findCellNodes27(px[idx], ni);

          for(int k=0; k<d_8or27; k++) {
            if(singlevfld[ni[k]]) {          // for nodes in non-crack zone
              pgCode[idx][k]=0;
              num0[ni[k]]++;
            }
            else {                           // for nodes in crack zone
              // Detect if particles above, below or in same side with nodes 
              short  cross=SAMESIDE;
              Vector norm=Vector(0.,0.,0.);

              // Get node position even if ni[k] beyond this patch
              Point gx=patch->nodePosition(ni[k]);

              for(int i=0; i<(int)ce[m].size(); i++) { //loop over crack elems
                //Three vertices of each element
                Point n3,n4,n5;                  
                n3=cx[m][ce[m][i].x()];
                n4=cx[m][ce[m][i].y()];
                n5=cx[m][ce[m][i].z()];

                // If particle and node in same side, continue
                short pPosition = Location(px[idx],gx,n3,n4,n5);
                if(pPosition==SAMESIDE) continue; 

                // Three signed volumes to see if p-g crosses crack
                double v3,v4,v5;
                v3=Volume(gx,n3,n4,px[idx]);
                v4=Volume(gx,n3,n5,px[idx]);
                if(v3*v4>0.) continue;
                v5=Volume(gx,n4,n5,px[idx]);
                if(v3*v5<0.) continue;

                // Particle above crack
                if(pPosition==ABOVE_CRACK && v3>=0. && v4<=0. && v5>=0.) { 
                  if(cross==SAMESIDE || (cross!=SAMESIDE &&
                                (v3==0.||v4==0.||v5==0.) ) ) {
                    cross=ABOVE_CRACK;
                    norm+=TriangleNormal(n3,n4,n5);
                  }
                  else { // no cross
                    cross=SAMESIDE;
                    norm=Vector(0.,0.,0.);
                  }
                }
                // Particle below crack
                if(pPosition==BELOW_CRACK && v3<=0. && v4>=0. && v5<=0.) { 
                  if(cross==SAMESIDE || (cross!=SAMESIDE &&
                                (v3==0.||v4==0.||v5==0.) ) ) {
                    cross=BELOW_CRACK;
                    norm+=TriangleNormal(n3,n4,n5);
                  }
                  else { // no cross
                    cross=SAMESIDE;
                    norm=Vector(0.,0.,0.);
                  }
                }
              } // End of loop over crack elements

              pgCode[idx][k]=cross;
              if(cross==SAMESIDE)    num0[ni[k]]++;
              if(cross==ABOVE_CRACK) num1[ni[k]]++;
              if(cross==BELOW_CRACK) num2[ni[k]]++;
              if(patch->containsNode(ni[k]) && norm.length()>1.e-16) { 
                norm/=norm.length();
                GCrackNorm[ni[k]]+=norm;
              }
            } // End of if(singlevfld)
          } // End of loop over k
        } // End of loop over particles

        // Step 3: count particles around nodes in GhostCells
        for(ParticleSubset::iterator itr=psetWGCs->begin();
                                     itr!=psetWGCs->end();itr++) {
          particleIndex idx=*itr;
          short handled=NO;
          for(ParticleSubset::iterator iter=pset->begin();
                                     iter!=pset->end();iter++) {
            if(pxWGCs[idx]==px[*iter]) {
              handled=YES;
              break;
            }
          }

          if(!handled) {               // particles in GhostCells
            if(d_8or27==8)
               patch->findCellNodes(pxWGCs[idx], ni);
            else if(d_8or27==27)
               patch->findCellNodes(pxWGCs[idx], ni);

            for(int k=0; k<d_8or27; k++) {
              Point gx=patch->nodePosition(ni[k]);
              if(singlevfld[ni[k]]) {          // for nodes in non-crack zone
                num0[ni[k]]++;
              }
              else { 
                short  cross=SAMESIDE; 
                Vector norm=Vector(0.,0.,0.);
                for(int i=0; i<(int)ce[m].size(); i++) { // Loop over crack elems
                  // Three vertices of each element
                  Point n3,n4,n5;                
                  n3=cx[m][ce[m][i].x()];
                  n4=cx[m][ce[m][i].y()];
                  n5=cx[m][ce[m][i].z()];

                  // If particle and node in same side, continue
                  short pPosition = Location(pxWGCs[idx],gx,n3,n4,n5);
                  if(pPosition==SAMESIDE) continue;

                  // Three signed volumes to see if p-g crosses crack
                  double v3,v4,v5;
                  v3=Volume(gx,n3,n4,pxWGCs[idx]);
                  v4=Volume(gx,n3,n5,pxWGCs[idx]);
                  if(v3*v4>0.) continue;
                  v5=Volume(gx,n4,n5,pxWGCs[idx]);
                  if(v3*v5<0.) continue;

                  // Particle above crack
                  if(pPosition==ABOVE_CRACK && v3>=0. && v4<=0. && v5>=0.) {
                    if(cross==SAMESIDE || (cross!=SAMESIDE &&
                                  (v3==0.||v4==0.||v5==0.) ) ) {
                      cross=ABOVE_CRACK;
                      norm+=TriangleNormal(n3,n4,n5);
                    }
                    else { 
                      cross=SAMESIDE;
                      norm=Vector(0.,0.,0.);
                    }
                  }
                  // Particle below crack
                  if(pPosition==BELOW_CRACK && v3<=0. && v4>=0. && v5<=0.) {
                    if(cross==SAMESIDE || (cross!=SAMESIDE &&
                                  (v3==0.||v4==0.||v5==0.) ) ) {
                      cross=BELOW_CRACK;
                      norm+=TriangleNormal(n3,n4,n5);
                    } 
                    else {  
                      cross=SAMESIDE;
                      norm=Vector(0.,0.,0.);
                    }
                  }
                } // End of loop over crack elements

                if(cross==SAMESIDE)    num0[ni[k]]++;
                if(cross==ABOVE_CRACK) num1[ni[k]]++;
                if(cross==BELOW_CRACK) num2[ni[k]]++;
                if(patch->containsNode(ni[k]) && norm.length()>1.e-16) {
                  norm/=norm.length();
                  GCrackNorm[ni[k]]+=norm;
                } 
              } // End of if(singlevfld)
            } // End of loop over k
          } // End of if(!handled)
        } // End of loop over particles

        // Step 4: Convert cross codes to field codes (0 to 1 or 2) 
        for(ParticleSubset::iterator iter=pset->begin();
                                     iter!=pset->end();iter++) {
           particleIndex idx=*iter;
           if(d_8or27==8)
              patch->findCellNodes(px[idx], ni);
           else if(d_8or27==27)
              patch->findCellNodes27(px[idx], ni);

           for(int k=0; k<d_8or27; k++) {
             if(pgCode[idx][k]==0) {
               if((num1[ni[k]]+num2[ni[k]]==0) || num2[ni[k]]!=0) 
                 pgCode[idx][k]=1;
               else if(num1[ni[k]]!=0) 
                 pgCode[idx][k]=2;
               else {
                 cout << "More than two velocity fields found in " 
                      << "Crack::ParticleVeloccityField for node: " 
                      << ni[k] << endl;
                 exit(1);
               } 
             }
           } // End of loop over k
         } // End of loop patls

        for(NodeIterator iter=patch->getNodeIterator();!iter.done();iter++) {
          IntVector c = *iter;
          if(GCrackNorm[c].length()>1.e-16) //unit crack normals
              GCrackNorm[c]/=GCrackNorm[c].length();
                    
          if(num1[c]+num2[c]==0) { // node in non-carck zone
            gNumPatls[c]=num0[c];
            GNumPatls[c]=0;
          }
          else if(num1[c]!=0 && num2[c]!=0) {
            gNumPatls[c]=num1[c];
            GNumPatls[c]=num2[c];
          }
          else {
            if(num1[c]!=0) {
              gNumPatls[c]=num1[c];
              GNumPatls[c]=num0[c];
            }
            if(num2[c]!=0) {
              gNumPatls[c]=num0[c];
              GNumPatls[c]=num2[c];
            }
          }
        } // End of loop over NodeIterator
      } // End of if(numElem[m]!=0)
#if 0 // Output particle velocity field code
      cout << "\n*** Particle velocity field generated "
           << "in Crack::ParticleVelocityField ***" << endl;
      cout << "--patch: " << patch->getID() << endl;
      cout << "--matreial: " << m << endl;
      for(ParticleSubset::iterator iter=pset->begin();
                        iter!=pset->end();iter++) {
        particleIndex idx=*iter;
        cout << "p["<< idx << "]: " << px[idx]<< endl;
        for(int k=0; k<d_8or27; k++) {
          if(pgCode[idx][k]==1)
             cout << setw(10) << "Node: " << k
                  << ",\tvfld: " << pgCode[idx][k] << endl;
          else if(pgCode[idx][k]==2)
             cout << setw(10) << "Node: " << k
                  << ",\tvfld: " << pgCode[idx][k] << " ***" << endl;
          else {
             cout << "Unknown particle velocity code in "
                  << "Crack::ParticleVelocityField" << endl;
             exit(1);
          }
        }  // End of loop over nodes (k)
      }  // End of loop over particles
      cout << "\nNumber of particles around nodes:\n" 
           << setw(18) << "node" << setw(20) << "gNumPatls" 
           << setw(20) << "GNumPatls" << endl;
      for(NodeIterator iter=patch->getNodeIterator();!iter.done();iter++) {
        IntVector c = *iter;
        if(gNumPatls[c]+GNumPatls[c]!=0){ 
        cout << setw(10) << c << setw(20) << gNumPatls[c] 
             << setw(20) << GNumPatls[c] << endl;}
      }
#endif
    } // End of loop numMatls
  } // End of loop patches
}

void Crack::addComputesAndRequiresAdjustCrackContactInterpolated(Task* t,
                                const PatchSet* /*patches*/,
                                const MaterialSet* matls) const
{
  const MaterialSubset* mss = matls->getUnion();

  // Data of primary field
  t->requires(Task::NewDW, lb->gMassLabel,         Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeLabel,       Ghost::None);
  t->requires(Task::NewDW, lb->gNumPatlsLabel,     Ghost::None); 
  t->requires(Task::NewDW, lb->gDisplacementLabel, Ghost::None);

  // Data of additional field
  t->requires(Task::NewDW, lb->GMassLabel,         Ghost::None);
  t->requires(Task::NewDW, lb->GVolumeLabel,       Ghost::None);
  t->requires(Task::NewDW, lb->GNumPatlsLabel,     Ghost::None);
  t->requires(Task::NewDW, lb->GCrackNormLabel,    Ghost::None);
  t->requires(Task::NewDW, lb->GDisplacementLabel, Ghost::None);

  t->modifies(lb->gVelocityLabel, mss);
  t->modifies(lb->GVelocityLabel, mss);

  t->computes(lb->frictionalWorkLabel);

}

void Crack::AdjustCrackContactInterpolated(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){

    enum {NO=0,YES};

    double mua=0.0,mub=0.0;
    double ma,mb,dvan,dvbn,dvat,dvbt,ratioa,ratiob;
    double vol0,normVol;
    Vector va,vb,vc,dva,dvb,ta,tb,na,nb,norm;

    int numMatls = d_sharedState->getNumMPMMatls();
    ASSERTEQ(numMatls, matls->size());

    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double vcell = dx.x()*dx.y()*dx.z();

    // Need access to all velocity fields at once, primary field
    StaticArray<constNCVariable<int> >    gNumPatls(numMatls);
    StaticArray<constNCVariable<double> > gmass(numMatls);
    StaticArray<constNCVariable<double> > gvolume(numMatls);
    StaticArray<constNCVariable<Vector> > gdisplacement(numMatls);
    StaticArray<NCVariable<Vector> >      gvelocity(numMatls);

    // Aceess to additional velocity field, for Farcture
    StaticArray<constNCVariable<int> >    GNumPatls(numMatls);
    StaticArray<constNCVariable<double> > Gmass(numMatls);
    StaticArray<constNCVariable<double> > Gvolume(numMatls);
    StaticArray<constNCVariable<Vector> > GCrackNorm(numMatls);
    StaticArray<constNCVariable<Vector> > Gdisplacement(numMatls);
    StaticArray<NCVariable<Vector> >      Gvelocity(numMatls);
    StaticArray<NCVariable<double> >      frictionWork(numMatls);

    Ghost::GhostType  gnone = Ghost::None;

    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      // Data for primary velocity field 
      new_dw->get(gNumPatls[m], lb->gNumPatlsLabel, dwi, patch, gnone, 0);
      new_dw->get(gmass[m],     lb->gMassLabel,     dwi, patch, gnone, 0);
      new_dw->get(gvolume[m],   lb->gVolumeLabel,   dwi, patch, gnone, 0);
      new_dw->get(gdisplacement[m],lb->gDisplacementLabel,dwi,patch,gnone,0);

      new_dw->getModifiable(gvelocity[m],lb->gVelocityLabel,dwi,patch);

      // Data for second velocity field
      new_dw->get(GNumPatls[m],lb->GNumPatlsLabel,  dwi, patch, gnone, 0);
      new_dw->get(Gmass[m],     lb->GMassLabel,     dwi, patch, gnone, 0);
      new_dw->get(Gvolume[m],   lb->GVolumeLabel,   dwi, patch, gnone, 0);
      new_dw->get(GCrackNorm[m],lb->GCrackNormLabel,dwi, patch, gnone, 0);
      new_dw->get(Gdisplacement[m],lb->GDisplacementLabel,dwi,patch,gnone,0);
      new_dw->getModifiable(Gvelocity[m],lb->GVelocityLabel,dwi,patch);

      new_dw->allocateAndPut(frictionWork[m],lb->frictionalWorkLabel,dwi,patch);
      frictionWork[m].initialize(0.);

      if(crackType[m]=="NO_CRACK") continue;  // no crack in this material
    
      // Loop over nodes to see if there is contact. If yes, adjust velocity field
      for(NodeIterator iter=patch->getNodeIterator();!iter.done();iter++) {
        IntVector c = *iter;
       
        // Only one velocity field
        if(gNumPatls[m][c]==0 || GNumPatls[m][c]==0) continue; 
        // Nodes in non-crack-zone
        norm=GCrackNorm[m][c];
        if(norm.length()<1.e-16) continue;  // should not happen now, but ...

        ma=gmass[m][c];
        va=gvelocity[m][c];
        mb=Gmass[m][c];
        vb=Gvelocity[m][c];
        vc=(va*ma+vb*mb)/(ma+mb);
        short Contact=NO;

        if(separateVol[m]<0. || contactVol[m] <0.) { // Use displacement criterion
          //use displacement criterion
          Vector u1=gdisplacement[m][c];
          Vector u2=Gdisplacement[m][c];
          if(Dot((u2-u1),norm) >0. ) {
            Contact=YES;
          }
        }
        else { // use volume criterion
          // Evaluate the nodal saturated volume (vol0) for general cases
          int numCellsWithPatls=0;
          IntVector cellIndex[8];
          patch->findCellsFromNode(c,cellIndex);
          ParticleSubset* psetWGCs= old_dw->getParticleSubset(dwi, patch,
                                 Ghost::AroundCells, NGN, lb->pXLabel);
          constParticleVariable<Point> pxWGCs;
          old_dw->get(pxWGCs, lb->pXLabel, psetWGCs);

          short cellWithPatls[8];
          for(int k=0; k<8; k++)  cellWithPatls[k]=0;
          for(ParticleSubset::iterator iter=psetWGCs->begin();
                                     iter!=psetWGCs->end();iter++) {
            particleIndex idx=*iter;
            double xp=pxWGCs[idx].x();
            double yp=pxWGCs[idx].y();
            double zp=pxWGCs[idx].z();
            for(int k=0; k<8; k++) { // Loop over 8 cells around the node
              Point l=patch->nodePosition(cellIndex[k]);
              Point h=patch->nodePosition(cellIndex[k]+IntVector(1,1,1));
              if(xp>l.x() && xp<=h.x() && yp>l.y() && yp<=h.y() &&
                 zp>l.z() && zp<=h.z()) cellWithPatls[k]=1;
            } // End of loop over 8 cells
            short allCellsWithPatls=1;
            for(int k=0; k<8; k++) {
              if(cellWithPatls[k]==0) allCellsWithPatls=0;
            }
            if(allCellsWithPatls) break;
          } // End of loop over patls
          for(int k=0; k<8; k++) numCellsWithPatls+=cellWithPatls[k];
          vol0=(float)numCellsWithPatls/8.*vcell;
    
          normVol=(gvolume[m][c]+Gvolume[m][c])/vol0;
          if(normVol>=contactVol[m] || 
            (normVol>separateVol[m] && Dot((vb-va),norm) > 0. )) {
            Contact=YES;
          }
        }

        if(!Contact) { // No contact  
          gvelocity[m][c]=gvelocity[m][c];
          Gvelocity[m][c]=Gvelocity[m][c];
          frictionWork[m][c] += 0.;
        } 
        else { // There is contact, apply contact law
          if(crackType[m]=="null") { // Nothing to do with it
            gvelocity[m][c]=gvelocity[m][c];
            Gvelocity[m][c]=Gvelocity[m][c];
            frictionWork[m][c] += 0.;
          }

          else if(crackType[m]=="stick") { // Assign centerofmass velocity
            gvelocity[m][c]=vc;
            Gvelocity[m][c]=vc;
            frictionWork[m][c] += 0.;
          }

          else if(crackType[m]=="frictional") { // Apply frictional law
            // For velocity field above crack
            Vector deltva(0.,0.,0.);
            dva=va-vc;
            na=norm;
            dvan=Dot(dva,na);
            if((dva-na*dvan).length()>1.e-16)
               ta=(dva-na*dvan)/(dva-na*dvan).length();
            else
               ta=Vector(0.,0.,0.);
            dvat=Dot(dva,ta);
            ratioa=dvat/dvan;
            if( fabs(ratioa)>cmu[m] ) {  // slide
               if(ratioa>0.) mua=cmu[m];
               if(ratioa<0.) mua=-cmu[m];
               deltva=-(na+ta*mua)*dvan;
               gvelocity[m][c]=va+deltva;
               frictionWork[m][c]+=ma*cmu[m]*dvan*dvan*(fabs(ratioa)-cmu[m]);
            }
            else {  // stick
               gvelocity[m][c]=vc;
               frictionWork[m][c] += 0.;
            }

            // For velocity field below crack
            Vector deltvb(0.,0.,0.);
            dvb=vb-vc;
            nb=-norm;
            dvbn=Dot(dvb,nb);
            if((dvb-nb*dvbn).length()>1.e-16)
               tb=(dvb-nb*dvbn)/(dvb-nb*dvbn).length();
            else
               tb=Vector(0.,0.,0.);
            dvbt=Dot(dvb,tb);
            ratiob=dvbt/dvbn;
            if(fabs(ratiob)>cmu[m]) { // slide
               if(ratiob>0.) mub=cmu[m];
               if(ratiob<0.) mub=-cmu[m];
               deltvb=-(nb+tb*mub)*dvbn;
               Gvelocity[m][c]=vb+deltvb;
               frictionWork[m][c]+=mb*cmu[m]*dvbn*dvbn*(fabs(ratiob)-cmu[m]);
            }
            else {// stick
               Gvelocity[m][c]=vc;
               frictionWork[m][c] += 0.;
            }
          }

          else { // Wrong contact type
            cout << "Unknown crack contact type in subroutine " 
                 << "Crack::AdjustCrackContactInterpolated: " 
                 << crackType[m] << endl;
            exit(1);
          }
        } // End of if there is !contact

      } //End of loop over nodes
    } //End of loop over materials
  }  //End of loop over patches
}

void Crack::addComputesAndRequiresAdjustCrackContactIntegrated(Task* t,
                                const PatchSet* /*patches*/,
                                const MaterialSet* matls) const
{
  const MaterialSubset* mss = matls->getUnion();

  t->requires(Task::OldDW, lb->delTLabel);
  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeLabel,        Ghost::None);
  t->requires(Task::NewDW, lb->gNumPatlsLabel,      Ghost::None);
  t->requires(Task::NewDW, lb->gDisplacementLabel,  Ghost::None);
  t->modifies(             lb->gVelocityStarLabel,  mss);
  t->modifies(             lb->gAccelerationLabel,  mss);

  t->requires(Task::NewDW, lb->GMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->GVolumeLabel,        Ghost::None);
  t->requires(Task::NewDW, lb->GNumPatlsLabel,      Ghost::None);
  t->requires(Task::NewDW, lb->GCrackNormLabel,     Ghost::None);
  t->requires(Task::NewDW, lb->GDisplacementLabel,  Ghost::None);
  t->modifies(             lb->GVelocityStarLabel,  mss);
  t->modifies(             lb->GAccelerationLabel,  mss);
  t->modifies(             lb->frictionalWorkLabel, mss);

}

void Crack::AdjustCrackContactIntegrated(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){

    enum {NO=0,YES};

    double mua=0.0,mub=0.0;
    double ma,mb,dvan,dvbn,dvat,dvbt,ratioa,ratiob;
    double vol0,normVol;
    Vector aa,ab,va,vb,vc,dva,dvb,ta,tb,na,nb,norm;

    int numMatls = d_sharedState->getNumMPMMatls();
    ASSERTEQ(numMatls, matls->size());
   
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double vcell = dx.x()*dx.y()*dx.z();

    // Need access to all velocity fields at once
    // Data of primary field
    StaticArray<constNCVariable<double> > gmass(numMatls);
    StaticArray<constNCVariable<double> > gvolume(numMatls);
    StaticArray<constNCVariable<int> >    gNumPatls(numMatls);
    StaticArray<constNCVariable<Vector> > gdisplacement(numMatls);
    StaticArray<NCVariable<Vector> >      gvelocity_star(numMatls);
    StaticArray<NCVariable<Vector> >      gacceleration(numMatls);
    // Data of additional field 
    StaticArray<constNCVariable<double> > Gmass(numMatls);
    StaticArray<constNCVariable<double> > Gvolume(numMatls);
    StaticArray<constNCVariable<int> >    GNumPatls(numMatls);
    StaticArray<constNCVariable<Vector> > GCrackNorm(numMatls);
    StaticArray<constNCVariable<Vector> > Gdisplacement(numMatls);
    StaticArray<NCVariable<Vector> >      Gvelocity_star(numMatls);
    StaticArray<NCVariable<Vector> >      Gacceleration(numMatls);
    // Friction work
    StaticArray<NCVariable<double> >      frictionWork(numMatls);
   
    Ghost::GhostType  gnone = Ghost::None;

    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      // For primary field
      new_dw->get(gmass[m],     lb->gMassLabel,     dwi, patch, gnone, 0);
      new_dw->get(gvolume[m],   lb->gVolumeLabel,   dwi, patch, gnone, 0);
      new_dw->get(gNumPatls[m], lb->gNumPatlsLabel, dwi, patch, gnone, 0);
      new_dw->get(gdisplacement[m],lb->gDisplacementLabel,dwi,patch,gnone,0);

      new_dw->getModifiable(gvelocity_star[m], lb->gVelocityStarLabel,
                                                         dwi, patch);
      new_dw->getModifiable(gacceleration[m],lb->gAccelerationLabel,
                                                         dwi, patch);
      // For additional field
      new_dw->get(Gmass[m],     lb->GMassLabel,     dwi, patch, gnone, 0);
      new_dw->get(Gvolume[m],   lb->GVolumeLabel,   dwi, patch, gnone, 0);
      new_dw->get(GNumPatls[m], lb->GNumPatlsLabel, dwi, patch, gnone, 0);
      new_dw->get(GCrackNorm[m],lb->GCrackNormLabel,dwi, patch, gnone, 0);
      new_dw->get(Gdisplacement[m],lb->GDisplacementLabel,dwi,patch,gnone,0);

      new_dw->getModifiable(Gvelocity_star[m], lb->GVelocityStarLabel,
                                                         dwi, patch);
      new_dw->getModifiable(Gacceleration[m],lb->GAccelerationLabel,
                                                         dwi, patch);
      new_dw->getModifiable(frictionWork[m], lb->frictionalWorkLabel,
                                                         dwi, patch);

      delt_vartype delT;
      old_dw->get(delT, lb->delTLabel);

      if(crackType[m]=="NO_CRACK") continue; // No crack(s) in this material

      for(NodeIterator iter=patch->getNodeIterator();!iter.done();iter++) {
        IntVector c = *iter;

        // For nodes in non-crack zone, there is no contact, just continue
        if(gNumPatls[m][c]==0 || GNumPatls[m][c]==0) continue; 
        norm=GCrackNorm[m][c];
        if(norm.length()<1.e-16) continue;   // should not happen now, but ...

        ma=gmass[m][c];
        va=gvelocity_star[m][c];
        aa=gacceleration[m][c];
        mb=Gmass[m][c];
        vb=Gvelocity_star[m][c];
        ab=Gacceleration[m][c];
        vc=(va*ma+vb*mb)/(ma+mb);
        short Contact=NO;

        if(separateVol[m]<0. || contactVol[m] <0.) { // Use displacement criterion
          // Use displacement criterion
          Vector u1=gdisplacement[m][c];
          //+delT*gvelocity_star[m][c];
          Vector u2=Gdisplacement[m][c];
          //+delT*Gvelocity_star[m][c];
          if(Dot((u2-u1),norm) >0. ) {
            Contact=YES;
          } 
        }
        else { // Use volume criterion
          int numCellsWithPatls=0;
          IntVector cellIndex[8];
          patch->findCellsFromNode(c,cellIndex);
          ParticleSubset* psetWGCs= old_dw->getParticleSubset(dwi, patch,
                                 Ghost::AroundCells, NGN, lb->pXLabel);
          constParticleVariable<Point> pxWGCs;
          old_dw->get(pxWGCs, lb->pXLabel, psetWGCs);

          short cellWithPatls[8];
          for(int k=0; k<8; k++)  cellWithPatls[k]=0;
          for(ParticleSubset::iterator iter=psetWGCs->begin();
                                     iter!=psetWGCs->end();iter++) {
            particleIndex idx=*iter;
            double xp=pxWGCs[idx].x();
            double yp=pxWGCs[idx].y();
            double zp=pxWGCs[idx].z();
            for(int k=0; k<8; k++) { // Loop over 8 cells around the node
              Point l=patch->nodePosition(cellIndex[k]);
              Point h=patch->nodePosition(cellIndex[k]+IntVector(1,1,1));
              if(xp>l.x() && xp<=h.x() && yp>l.y() && yp<=h.y() &&
                 zp>l.z() && zp<=h.z()) cellWithPatls[k]=1;
            } //End of loop over 8 cells
            short allCellsWithPatls=1;
            for(int k=0; k<8; k++) {
              if(cellWithPatls[k]==0) allCellsWithPatls=0;
            }
            if(allCellsWithPatls) break;
          } // End of loop over patls
          for(int k=0; k<8; k++) numCellsWithPatls+=cellWithPatls[k];
          vol0=(float)numCellsWithPatls/8.*vcell;

          normVol=(gvolume[m][c]+Gvolume[m][c])/vol0;

          if(normVol>=contactVol[m] || 
             (normVol>separateVol[m] && Dot((vb-va),norm) > 0.)) {
            Contact=YES;
          }
        }

        if(!Contact) {// No contact
          gvelocity_star[m][c]=gvelocity_star[m][c];
          gacceleration[m][c]=gacceleration[m][c];
          Gvelocity_star[m][c]=Gvelocity_star[m][c];
          Gacceleration[m][c]=Gacceleration[m][c];
          frictionWork[m][c]+=0.0;
        } 
        else { // There is contact, apply contact law
          if(crackType[m]=="null") { // Do nothing
            gvelocity_star[m][c]=gvelocity_star[m][c];
            gacceleration[m][c]=gacceleration[m][c];
            Gvelocity_star[m][c]=Gvelocity_star[m][c];
            Gacceleration[m][c]=Gacceleration[m][c];
            frictionWork[m][c]+=0.0;
          }

          else if(crackType[m]=="stick") { // Assign centerofmass velocity
            gvelocity_star[m][c]=vc;
            gacceleration[m][c]=aa+(vb-va)*mb/(ma+mb)/delT;
            Gvelocity_star[m][c]=vc;
            Gacceleration[m][c]=ab+(va-vb)*ma/(ma+mb)/delT;
            frictionWork[m][c]+=0.0;
          }

          else if(crackType[m]=="frictional") { // Apply friction law
            // For primary field
            Vector deltva(0.,0.,0.);
            dva=va-vc;
            na=norm;
            dvan=Dot(dva,na);
            if((dva-na*dvan).length()>1.e-16)
               ta=(dva-na*dvan)/(dva-na*dvan).length();
            else
               ta=Vector(0.,0.,0.);
            dvat=Dot(dva,ta);
            ratioa=dvat/dvan;
            if( fabs(ratioa)>cmu[m] ) {  // slide
               if(ratioa>0.) mua= cmu[m];
               if(ratioa<0.) mua=-cmu[m];
               deltva=-(na+ta*mua)*dvan;
               gvelocity_star[m][c]=va+deltva;
               gacceleration[m][c]=aa+deltva/delT;
               frictionWork[m][c]+=ma*cmu[m]*dvan*dvan*(fabs(ratioa)-cmu[m]);
            }
            else {   // stick
               gvelocity_star[m][c]=vc;
               gacceleration[m][c]=aa+(vb-va)*mb/(ma+mb)/delT;
               frictionWork[m][c]+=0.0;
            }
            // For additional field
            Vector deltvb(0.,0.,0.);
            dvb=vb-vc;
            nb=-norm;
            dvbn=Dot(dvb,nb);
            if((dvb-nb*dvbn).length()>1.e-16)
               tb=(dvb-nb*dvbn)/(dvb-nb*dvbn).length();
            else
               tb=Vector(0.,0.,0.);
            dvbt=Dot(dvb,tb);
            ratiob=dvbt/dvbn;
            if(fabs(ratiob)>cmu[m]) { // slide
               if(ratiob>0.) mub= cmu[m];
               if(ratiob<0.) mub=-cmu[m];
               deltvb=-(nb+tb*mub)*dvbn;
               Gvelocity_star[m][c]=vb+deltvb;
               Gacceleration[m][c]=ab+deltvb/delT;
               frictionWork[m][c]+=mb*cmu[m]*dvbn*dvbn*(fabs(ratiob)-cmu[m]);
            }
            else {  // stick
               Gvelocity_star[m][c]=vc;
               Gacceleration[m][c]=ab+(va-vb)*ma/(ma+mb)/delT;
               frictionWork[m][c]+=0.0;
            }
          }

          else {
            cout<< "Unknown crack contact type in "
                << "Crack::AdjustCrackContactIntegrated: " 
                << crackType[m] << endl;
            exit(1);
          }
        } // End of if there is !contact
      } //End of loop over nodes
    } //End of loop over materials
  }  //End of loop over patches
}

void Crack::addComputesAndRequiresGetNodalSolutions(Task* t,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  // Required particles' solutions 
  t->requires(Task::NewDW,lb->pMassLabel_preReloc,                gan,NGP);
  t->requires(Task::NewDW,lb->pStressLabel_preReloc,              gan,NGP);
  t->requires(Task::NewDW,lb->pDispGradsLabel_preReloc,           gan,NGP);
  t->requires(Task::NewDW,lb->pStrainEnergyDensityLabel_preReloc, gan,NGP);

  t->requires(Task::NewDW,lb->pgCodeLabel,                        gan,NGP);
  t->requires(Task::NewDW,lb->pKineticEnergyDensityLabel,         gan,NGP);
  t->requires(Task::NewDW,lb->pVelGradsLabel,                     gan,NGP);

  t->requires(Task::OldDW,lb->pXLabel,                            gan,NGP);
  if(d_8or27==27) t->requires(Task::OldDW, lb->pSizeLabel,        gan,NGP);

  // Required nodal solutions
  t->requires(Task::NewDW,lb->gMassLabel,                         gnone);
  t->requires(Task::NewDW,lb->GMassLabel,                         gnone);

  // The nodal solutions to be calculated
  t->computes(lb->gGridStressLabel);
  t->computes(lb->GGridStressLabel);
  t->computes(lb->gStrainEnergyDensityLabel);
  t->computes(lb->GStrainEnergyDensityLabel);
  t->computes(lb->gKineticEnergyDensityLabel);
  t->computes(lb->GKineticEnergyDensityLabel);
  t->computes(lb->gDispGradsLabel);
  t->computes(lb->GDispGradsLabel);
  t->computes(lb->gVelGradsLabel);
  t->computes(lb->GVelGradsLabel);
}

void Crack::GetNodalSolutions(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* ,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw)
{
  /* Compute nodal solutions of stresses, displacement gradients,
     strain energy density and  kinetic energy density by interpolating
     particle's solutions to grid */

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    int numMPMMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      // Get particle's solutions
      constParticleVariable<Short27> pgCode;
      constParticleVariable<Point>   px;
      constParticleVariable<Vector>  psize;
      constParticleVariable<double>  pmass;
      constParticleVariable<double>  pstrainenergydensity;
      constParticleVariable<double>  pkineticenergydensity;
      constParticleVariable<Matrix3> pstress,pdispgrads,pvelgrads;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                             Ghost::AroundNodes, NGP,lb->pXLabel);

      new_dw->get(pmass,                lb->pMassLabel_preReloc,       pset);
      new_dw->get(pstress,              lb->pStressLabel_preReloc,     pset);
      new_dw->get(pdispgrads,           lb->pDispGradsLabel_preReloc,  pset);
      new_dw->get(pstrainenergydensity,
                               lb->pStrainEnergyDensityLabel_preReloc, pset);

      new_dw->get(pgCode,               lb->pgCodeLabel,               pset);
      new_dw->get(pvelgrads,            lb->pVelGradsLabel,            pset);
      new_dw->get(pkineticenergydensity,lb->pKineticEnergyDensityLabel,pset);

      old_dw->get(px,                   lb->pXLabel,                   pset);
      if(d_8or27==27) old_dw->get(psize,lb->pSizeLabel,                pset);

      // Get nodal mass
      constNCVariable<double> gmass, Gmass;
      new_dw->get(gmass, lb->gMassLabel, dwi, patch, Ghost::None, 0);
      new_dw->get(Gmass, lb->GMassLabel, dwi, patch, Ghost::None, 0);

      // Declare nodal variables calculated
      NCVariable<Matrix3> ggridstress,Ggridstress;
      NCVariable<Matrix3> gdispgrads,Gdispgrads;
      NCVariable<Matrix3> gvelgrads,Gvelgrads;
      NCVariable<double>  gstrainenergydensity,Gstrainenergydensity;
      NCVariable<double>  gkineticenergydensity,Gkineticenergydensity;

      new_dw->allocateAndPut(ggridstress, lb->gGridStressLabel,dwi,patch);
      new_dw->allocateAndPut(Ggridstress, lb->GGridStressLabel,dwi,patch);
      new_dw->allocateAndPut(gdispgrads,  lb->gDispGradsLabel, dwi,patch);
      new_dw->allocateAndPut(Gdispgrads,  lb->GDispGradsLabel, dwi,patch);
      new_dw->allocateAndPut(gvelgrads,   lb->gVelGradsLabel,  dwi,patch);
      new_dw->allocateAndPut(Gvelgrads,   lb->GVelGradsLabel,  dwi,patch);
      new_dw->allocateAndPut(gstrainenergydensity,
                                lb->gStrainEnergyDensityLabel, dwi,patch);
      new_dw->allocateAndPut(Gstrainenergydensity,
                                lb->GStrainEnergyDensityLabel, dwi,patch);
      new_dw->allocateAndPut(gkineticenergydensity,
                                lb->gKineticEnergyDensityLabel,dwi,patch);
      new_dw->allocateAndPut(Gkineticenergydensity,
                                lb->GKineticEnergyDensityLabel,dwi,patch);

      ggridstress.initialize(Matrix3(0.));
      Ggridstress.initialize(Matrix3(0.));
      gdispgrads.initialize(Matrix3(0.));
      Gdispgrads.initialize(Matrix3(0.));
      gvelgrads.initialize(Matrix3(0.));
      Gvelgrads.initialize(Matrix3(0.));
      gstrainenergydensity.initialize(0.);
      Gstrainenergydensity.initialize(0.);
      gkineticenergydensity.initialize(0.);
      Gkineticenergydensity.initialize(0.);

      IntVector ni[MAX_BASIS];
      double S[MAX_BASIS];

      if(calFractParameters || doCrackPropagation) {
        for (ParticleSubset::iterator iter = pset->begin();
                             iter != pset->end(); iter++) {
          particleIndex idx = *iter;
 
          // Get the node indices that surround the cell
          if(d_8or27==8){
            patch->findCellAndWeights(px[idx], ni, S);
          }
          else if(d_8or27==27){
            patch->findCellAndWeights27(px[idx], ni, S, psize[idx]);
          }

          for (int k = 0; k < d_8or27; k++){
            if(patch->containsNode(ni[k])){
              double pmassTimesS=pmass[idx]*S[k];
              if(pgCode[idx][k]==1) { 
                ggridstress[ni[k]] += pstress[idx]    * pmassTimesS;
                gdispgrads[ni[k]]  += pdispgrads[idx] * pmassTimesS;
                gvelgrads[ni[k]]   += pvelgrads[idx]  * pmassTimesS;
                gstrainenergydensity[ni[k]]  += pstrainenergydensity[idx] *
                                                        pmassTimesS;
                gkineticenergydensity[ni[k]] += pkineticenergydensity[idx] *
                                                        pmassTimesS;
              }
              else if(pgCode[idx][k]==2) {
                Ggridstress[ni[k]] += pstress[idx]    * pmassTimesS;
                Gdispgrads[ni[k]]  += pdispgrads[idx] * pmassTimesS;
                Gvelgrads[ni[k]]   += pvelgrads[idx]  * pmassTimesS;
                Gstrainenergydensity[ni[k]]  += pstrainenergydensity[idx] *
                                                        pmassTimesS;
                Gkineticenergydensity[ni[k]] += pkineticenergydensity[idx] *
                                                        pmassTimesS;
              }
            }
          } // End of loop over k
        } // End of loop over particles

        for(NodeIterator iter=patch->getNodeIterator();!iter.done();iter++) {
          IntVector c = *iter;
          // For primary field
          ggridstress[c]           /= gmass[c];
          gdispgrads[c]            /= gmass[c];
          gvelgrads[c]             /= gmass[c];
          gstrainenergydensity[c]  /= gmass[c];
          gkineticenergydensity[c] /= gmass[c];
          // For additional field
          Ggridstress[c]           /= Gmass[c];
          Gdispgrads[c]            /= Gmass[c];
          Gvelgrads[c]             /= Gmass[c];
          Gstrainenergydensity[c]  /= Gmass[c];
          Gkineticenergydensity[c] /= Gmass[c];
        }
      } // End if(calFractParameters || doCrackPropagation)
    }
  }
}

void Crack::addComputesAndRequiresCrackFrontSegSubset(Task* /*t*/,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
// Currently do nothing
}

void Crack::CrackFrontSegSubset(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* /*matls*/,
                      DataWarehouse* /*old_dw*/,
                      DataWarehouse* /*new_dw*/)
{ // Create cfsset -- store crack-front seg numbers           
  for(int p=0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);

    int pid,patch_size;
    MPI_Comm_size(mpi_crack_comm,&patch_size);
    MPI_Comm_rank(mpi_crack_comm,&pid);
                
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m=0; m<numMPMMatls; m++) {
      /* Task 1: Collect crack-front segs in each patch and
                 broadcast to all ranks 
      */
      // cfsset -- store crack-front segs
      cfsset[m][pid].clear();
      int ncfSegs=(int)cfSegNodes[m].size()/2;
      for(int j=0; j<ncfSegs; j++) {
        int n1=cfSegNodes[m][2*j];
        int n2=cfSegNodes[m][2*j+1];
        Point cent=cx[m][n1]+(cx[m][n2]-cx[m][n1])/2.;
        if(patch->containsPoint(cent)) {
          cfsset[m][pid].push_back(j);
        } 
      } // End of loop over j

      MPI_Barrier(mpi_crack_comm);

      // Broadcast cfsset of each patch to all ranks
      for(int i=0; i<patch_size; i++) {
        int num; // number of crack-front segs in patch i
        if(i==pid) num=cfsset[m][i].size();
        MPI_Bcast(&num,1,MPI_INT,i,mpi_crack_comm);
        cfsset[m][i].resize(num);
        MPI_Bcast(&cfsset[m][i][0],num,MPI_INT,i,mpi_crack_comm);
      }

      MPI_Barrier(mpi_crack_comm);

      /* Task 2: Collect crack-front nodes in each patch and
                 broadcast to all ranks
      */
      // cfnset -- store Crack-front node indice
      cfnset[m][pid].clear();
      for(int j=0; j<(int)cfSegNodes[m].size(); j++) {
        int node=cfSegNodes[m][j];
        if(patch->containsPoint(cx[m][node])) 
          cfnset[m][pid].push_back(j);
      }

      MPI_Barrier(mpi_crack_comm);

      // Broadcast cfnset of each patch to all ranks
      for(int i=0; i<patch_size; i++) {
        int num; // number of crack-front nodes in patch i
        if(i==pid) num=cfnset[m][i].size();
        MPI_Bcast(&num,1,MPI_INT,i,mpi_crack_comm);
        cfnset[m][i].resize(num);
        MPI_Bcast(&cfnset[m][i][0],num,MPI_INT,i,mpi_crack_comm);
      }

    } // End of loop over matls
  } // End of loop over patches
}       
     
void Crack::addComputesAndRequiresCalculateFractureParameters(Task* t,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
  // Requires nodal solutions
  int NGC=NJ+NGN+1;
  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::NewDW, lb->gMassLabel,                gac,NGC);
  t->requires(Task::NewDW, lb->GMassLabel,                gac,NGC);
  t->requires(Task::NewDW, lb->GNumPatlsLabel,            gac,NGC);
  t->requires(Task::NewDW, lb->gDisplacementLabel,        gac,NGC);
  t->requires(Task::NewDW, lb->GDisplacementLabel,        gac,NGC);
  t->requires(Task::NewDW, lb->gGridStressLabel,          gac,NGC);
  t->requires(Task::NewDW, lb->GGridStressLabel,          gac,NGC);
  t->requires(Task::NewDW, lb->gDispGradsLabel,           gac,NGC);
  t->requires(Task::NewDW, lb->GDispGradsLabel,           gac,NGC);
  t->requires(Task::NewDW, lb->gStrainEnergyDensityLabel, gac,NGC);
  t->requires(Task::NewDW, lb->GStrainEnergyDensityLabel, gac,NGC);
  t->requires(Task::NewDW, lb->gKineticEnergyDensityLabel,gac,NGC);
  t->requires(Task::NewDW, lb->GKineticEnergyDensityLabel,gac,NGC);

  // Required for area integral
  t->requires(Task::NewDW, lb->gAccelerationLabel,        gac,NGC);
  t->requires(Task::NewDW, lb->GAccelerationLabel,        gac,NGC);
  t->requires(Task::NewDW, lb->gVelGradsLabel,            gac,NGC);
  t->requires(Task::NewDW, lb->GVelGradsLabel,            gac,NGC);
  t->requires(Task::NewDW, lb->gVelocityLabel,            gac,NGC);
  t->requires(Task::NewDW, lb->GVelocityLabel,            gac,NGC);

  if(d_8or27==27) 
    t->requires(Task::OldDW, lb->pSizeLabel, Ghost::None);

  /*
  Computes J integral for each crack-front segments, specified by 
  cfSegNodes[MNM]. J integral will be stored in 
  cfSegJ[MNM], a data member in Crack.h. For both end points 
  of each segment, they have the value of J integral of this segment. 
  J integrals will be converted to stress intensity factors
  for hypoelastic materials.
  */
}

void Crack::CalculateFractureParameters(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double dx_max=Max(dx.x(),dx.y(),dx.z());

    // Variables related to MPI
    int pid,patch_size;
    MPI_Comm_size(mpi_crack_comm,&patch_size);
    MPI_Comm_rank(mpi_crack_comm,&pid);
    MPI_Datatype MPI_VECTOR=fun_getTypeDescription((Vector*)0)->getMPIType();

    enum {NO=0,YES};
    enum {R=0,L};

    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m=0;m<numMatls;m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();

      int dwi = matls->get(m);
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      Ghost::GhostType  gac = Ghost::AroundCells;

      constNCVariable<double> gmass,Gmass;
      constNCVariable<int>     GnumPatls;
      constNCVariable<Vector>  gdisp,Gdisp;
      constNCVariable<Matrix3> ggridStress,GgridStress;
      constNCVariable<Matrix3> gdispGrads,GdispGrads;
      constNCVariable<double>  gW,GW;
      constNCVariable<double>  gK,GK;
      constNCVariable<Vector>  gacc,Gacc;
      constNCVariable<Vector>  gvel,Gvel;
      constNCVariable<Matrix3> gvelGrads,GvelGrads;

      // Get nodal solutions
      int NGC=NJ+NGN+1; 
      new_dw->get(gmass,      lb->gMassLabel,         dwi,patch,gac,NGC);
      new_dw->get(Gmass,      lb->GMassLabel,         dwi,patch,gac,NGC);
      new_dw->get(GnumPatls,  lb->GNumPatlsLabel,     dwi,patch,gac,NGC);
      new_dw->get(gdisp,      lb->gDisplacementLabel, dwi,patch,gac,NGC);
      new_dw->get(Gdisp,      lb->GDisplacementLabel, dwi,patch,gac,NGC);
      new_dw->get(ggridStress,lb->gGridStressLabel,   dwi,patch,gac,NGC);
      new_dw->get(GgridStress,lb->GGridStressLabel,   dwi,patch,gac,NGC);
      new_dw->get(gdispGrads, lb->gDispGradsLabel,    dwi,patch,gac,NGC);
      new_dw->get(GdispGrads, lb->GDispGradsLabel,    dwi,patch,gac,NGC);
      new_dw->get(gW,lb->gStrainEnergyDensityLabel,   dwi,patch,gac,NGC);
      new_dw->get(GW,lb->GStrainEnergyDensityLabel,   dwi,patch,gac,NGC);
      new_dw->get(gK,lb->gKineticEnergyDensityLabel,  dwi,patch,gac,NGC);
      new_dw->get(GK,lb->GKineticEnergyDensityLabel,  dwi,patch,gac,NGC);

      new_dw->get(gacc,       lb->gAccelerationLabel, dwi,patch,gac,NGC);
      new_dw->get(Gacc,       lb->GAccelerationLabel, dwi,patch,gac,NGC);
      new_dw->get(gvel,       lb->gVelocityLabel,     dwi,patch,gac,NGC);
      new_dw->get(Gvel,       lb->GVelocityLabel,     dwi,patch,gac,NGC);
      new_dw->get(gvelGrads,  lb->gVelGradsLabel,     dwi,patch,gac,NGC);
      new_dw->get(GvelGrads,  lb->GVelGradsLabel,     dwi,patch,gac,NGC);

      constParticleVariable<Vector> psize;
      if(d_8or27==27) old_dw->get(psize, lb->pSizeLabel, pset);

      // Allocate memory for cfSegJ and cfSegK
      int cfNodeSize=(int)cfSegNodes[m].size();
      cfSegJ[m].resize(cfNodeSize);
      cfSegK[m].resize(cfNodeSize);
      if(calFractParameters || doCrackPropagation) {
        for(int i=0; i<patch_size; i++) {// Loop over all patches
          int num=cfnset[m][i].size(); // number of crack-front nodes in patch i 

          if(num>0) { // If there is crack-front node(s) in patch i
            Vector* cfJ=new Vector[num];
            Vector* cfK=new Vector[num];

            if(pid==i) { // Calculte J & K by proc i
              for(int l=0; l<num; l++) { // Calculate J & K over crack-front nodes
                int idx=cfnset[m][i][l];     // crack-front node index
                int node=cfSegNodes[m][idx]; // node

                int pre_idx=-1;
                for(int ij=0; ij<l; ij++) {
                  if(node==cfSegNodes[m][cfnset[m][i][ij]]) {
                    pre_idx=ij;
                    break;
                  }
                }

                if(pre_idx<0) { // Not operated
                  /* Step 1: Define crack front segment coordinates
                   v1,v2,v3: direction consies of new axes X',Y' and Z'
                   Origin located at the point at which J&K is calculated
                  */
                  
                  // Two segs connected by the node
                  int segs[2];
                  FindSegsFromNode(m,node,segs);

                  // Position at which to calculate J & K
                  Point origin;
                  double x0,y0,z0;
                  if(segs[L]>=0 && segs[R]>=0) { // for middle nodes
                    origin=cx[m][node];
                  }
                  else { // for edge nodes
                    int nd1=-1,nd2=-1;
                    if(segs[R]<0) { // for right-edge nodes
                      nd1=cfSegNodes[m][2*segs[L]];
                      nd2=cfSegNodes[m][2*segs[L]+1];
                    }
                    if(segs[L]<0) { // for left-edge nodes
                      nd1=cfSegNodes[m][2*segs[R]];
                      nd2=cfSegNodes[m][2*segs[R]+1];  
                    }
                    Point pt1=cx[m][nd1];
                    Point pt2=cx[m][nd2]; 
                    origin=pt1+(pt2-pt1)/2.;
                  }
                  x0=origin.x();  y0=origin.y();  z0=origin.z();

                  // Direction-cosines of the node
                  Vector v1,v2,v3;
                  Vector v2T=Vector(0.,0.,0.);
                  double l1,m1,n1,l2,m2,n2,l3,m3,n3;
                  for(int j=R; j<=L; j++) {
                    if(segs[j]>=0) v2T+=cfSegV2[m][segs[j]];
                  }
                  v2=v2T/v2T.length();
                  v3=-cfSegV3[m][idx];
                  Vector v23=Cross(v2,v3);
                  v1=v23/v23.length();
                  Vector v31=Cross(v3,v1);
                  v2=v31/v31.length();
                  l1=v1.x(); m1=v1.y(); n1=v1.z();
                  l2=v2.x(); m2=v2.y(); n2=v2.z();
                  l3=v3.x(); m3=v3.y(); n3=v3.z();

                  // Coordinates transformation matrix from global to local(T)
                  // and the one from local to global (TT)
                  Matrix3 T =Matrix3(l1,m1,n1,l2,m2,n2,l3,m3,n3);
                  Matrix3 TT=Matrix3(l1,l2,l3,m1,m2,m3,n1,n2,n3);
 
                  /* Step 2 Define the center of J-integral contour
                  */
                  Point JCircleCenter=origin;
                  // See if the center is too close to the boundary                 
                  Point ptsNearOrigin[3]={origin+TT*(rJ*v1),
                                          origin+TT*(rJ*v2),origin+TT*(-rJ*v2)};
                  short ptsInMat[3]={YES,YES,YES};
                  // Get the node indices that surround the cell
                  IntVector ni[MAX_BASIS];
                  double S[MAX_BASIS];
                  for(int j=0; j<3; j++) {
                    if(d_8or27==8)
                      patch->findCellAndWeights(ptsNearOrigin[j], ni, S);
                    else if(d_8or27==27)
                      patch->findCellAndWeights27(ptsNearOrigin[j], ni, S, psize[j]);

                    for(int k = 0; k < d_8or27; k++) {
                      double totalMass=gmass[ni[k]]+Gmass[ni[k]];
                      if(totalMass<5*d_SMALL_NUM_MPM) {
                        ptsInMat[j]=NO;
                        break;
                      }
                    }
                  } // End of loop over j
                  // Move the center if it is too close to the boundary
                  if(!ptsInMat[0]) JCircleCenter=origin+TT*(-0.9*rJ*v1);
                  if(!ptsInMat[1]) JCircleCenter=origin+TT*(-0.9*rJ*v2);
                  if(!ptsInMat[2]) JCircleCenter=origin+TT*(0.9*rJ*v2);

                  /* Step 3: Find parameters A[14] of J-path circle with equation
                     A0x^2+A1y^2+A2z^2+A3xy+A4xz+A5yz+A6x+A7y+A8z+A9-r^2=0 and
                     A10x+A11y+A12z+A13=0 */
                  double A[14];    
                  FindJPathCircle(JCircleCenter,v1,v2,v3,A); 
         
                  /* Step 4: Find intersection(crossPt) between J-ptah and crack plane
                  */
                  Point crossPt;
                  double d_rJ=rJ;
                  while(!FindIntersectionOfJPathAndCrackPlane(m,d_rJ,A,crossPt)) 
                    d_rJ/=2.;
                  if(fabs(d_rJ-rJ)/rJ>1.e-6) {
                    cout << "!!! Radius of J-contour decreased from " << rJ << " to " 
                         << d_rJ << " for seg "<< i << " of mat " << m << endl; 
                  }

                  // Get coordinates of intersection in local system (xcprime,ycprime)
                  double xc,yc,zc,xcprime,ycprime,scprime;
                  xc=crossPt.x(); yc=crossPt.y(); zc=crossPt.z();
                  xcprime=l1*(xc-x0)+m1*(yc-y0)+n1*(zc-z0);
                  ycprime=l2*(xc-x0)+m2*(yc-y0)+n2*(zc-z0);
                  scprime=sqrt(xcprime*xcprime+ycprime*ycprime);
    
                  /* Step 5: Put integral points in J-path circle and do initialization
                  */
                  double xprime,yprime,x,y,z;
                  double PI=3.141592654;
                  int nSegs=16;                       // Number of segments on J-path circle
                  Point*   X  = new Point[nSegs+1];   // Integral points
                  double*  W  = new double[nSegs+1];  // Strain energy density
                  double*  K  = new double[nSegs+1];  // Kinetic energy density
                  Matrix3* ST = new Matrix3[nSegs+1]; // Stresses in global coordinates
                  Matrix3* DG = new Matrix3[nSegs+1]; // Disp grads in global coordinates
                  Matrix3* st = new Matrix3[nSegs+1]; // Stresses in local coordinates
                  Matrix3* dg = new Matrix3[nSegs+1]; // Disp grads in local coordinates
    
                  for(int j=0; j<=nSegs; j++) {       // Loop over points on the circle
                    double angle,cosTheta,sinTheta;
                    angle=2*PI*(float)j/(float)nSegs;
                    cosTheta=(xcprime*cos(angle)-ycprime*sin(angle))/scprime;
                    sinTheta=(ycprime*cos(angle)+xcprime*sin(angle))/scprime;
                    // Coordinates of integral points in local coordinates
                    xprime=d_rJ*cosTheta;
                    yprime=d_rJ*sinTheta;
                    // Coordinates of integral points in global coordinates
                    x=l1*xprime+l2*yprime+x0;
                    y=m1*xprime+m2*yprime+y0;
                    z=n1*xprime+n2*yprime+z0;
                    X[j] = Point(x,y,z);
                    W[j]  = 0.0;
                    K[j]  = 0.0;
                    ST[j] = Matrix3(0.);
                    DG[j] = Matrix3(0.);
                    st[j] = Matrix3(0.);
                    dg[j] = Matrix3(0.);
                  }
 
                  /* Step 6: Evaluate solutions at integral points in global coordinates
                  */
                  for(int j=0; j<=nSegs; j++) {
                    if(d_8or27==8) 
                      patch->findCellAndWeights(X[j],ni,S);
                    else if(d_8or27==27)
                      patch->findCellAndWeights27(X[j],ni,S,psize[j]);
 
                    for(int k=0; k<d_8or27; k++) {
                      if(GnumPatls[ni[k]]!=0 && j<nSegs/2) {  //below crack
                        W[j]  += GW[ni[k]]          * S[k];
                        K[j]  += GK[ni[k]]          * S[k];
                        ST[j] += GgridStress[ni[k]] * S[k];
                        DG[j] += GdispGrads[ni[k]]  * S[k];
                      }
                      else { //above crack or in non-crack zone
                        W[j]  += gW[ni[k]]          * S[k];
                        K[j]  += gK[ni[k]]          * S[k];
                        ST[j] += ggridStress[ni[k]] * S[k];
                        DG[j] += gdispGrads[ni[k]]  * S[k];
                      }
                    } // End of loop over k
                  } // End of loop over j
 
                  /* Step 7: Transform the solutions to crack-front coordinates
                  */
                  for(int j=0; j<=nSegs; j++) {
                    for(int i1=0; i1<3; i1++) {
                      for(int j1=0; j1<3; j1++) {
                        for(int i2=0; i2<3; i2++) {
                          for(int j2=0; j2<3; j2++) {
                            st[j](i1,j1) += T(i1,i2)*T(j1,j2)*ST[j](i2,j2);
                            dg[j](i1,j1) += T(i1,i2)*T(j1,j2)*DG[j](i2,j2);
                          } 
                        } 
                      } // End of loop over j1
                    } // End of loop over i1
                  } // End of loop over j
 
                  /* Step 8: Get function values at integral points
                  */
                  double* f1ForJx = new double[nSegs+1];
                  double* f1ForJy = new double[nSegs+1];
                  for(int j=0; j<=nSegs; j++) {  
                    double angle,cosTheta,sinTheta;
                    double t1,t2,t3;
   
                    angle=2*PI*(float)j/(float)nSegs;
                    cosTheta=(xcprime*cos(angle)-ycprime*sin(angle))/scprime;
                    sinTheta=(ycprime*cos(angle)+xcprime*sin(angle))/scprime;
                    t1=st[j](0,0)*cosTheta+st[j](0,1)*sinTheta;
                    t2=st[j](1,0)*cosTheta+st[j](1,1)*sinTheta;
                    t3=st[j](2,0)*cosTheta+st[j](2,1)*sinTheta;
 
                    Vector t123=Vector(t1,t2,0./*t3*/); // plane state
                    Vector dgx=Vector(dg[j](0,0),dg[j](1,0),dg[j](2,0));
                    Vector dgy=Vector(dg[j](0,1),dg[j](1,1),dg[j](2,1));
  
                    f1ForJx[j]=(W[j]+K[j])*cosTheta-Dot(t123,dgx);
                    f1ForJy[j]=(W[j]+K[j])*sinTheta-Dot(t123,dgy);
                  }
 
                  /* Step 9: Get J integral
                  */
                  double Jx1=0.,Jy1=0.; 
                  for(int j=0; j<nSegs; j++) {   // Loop over segments
                    Jx1 += f1ForJx[j] + f1ForJx[j+1];
                    Jy1 += f1ForJx[j] + f1ForJy[j+1];
                  } // End of loop over segments
                  Jx1 *= d_rJ*PI/nSegs;
                  Jy1 *= d_rJ*PI/nSegs; 
 
                  /* Step 10: Release dynamic arries for this crack front segment
                  */
                  delete [] X;
                  delete [] W;        delete [] K;
                  delete [] ST;       delete [] DG;   
                  delete [] st;       delete [] dg;
                  delete [] f1ForJx;  delete [] f1ForJy;
 
                  /* Step 11: Effect of the area integral in J-integral formula
                  */ 
                  double Jx2=0.,Jy2=0.;
                  if(d_useVolumeIntegral) {
                    // Define integral points in the area enclosed by J-integral contour
                    int nc=(int)(d_rJ/dx_max);
                    if(d_rJ/dx_max-nc>=0.5) nc++;
                    if(nc<2) nc=2; // Cell number J-path away from the origin
                    double* c=new double[4*nc];
                    for(int j=0; j<4*nc; j++) 
                      c[j]=(float)(-4*nc+1+2*j)/4/nc*d_rJ;

                    int count=0;  // number of integral points
                    Point* x=new Point[16*nc*nc];
                    Point* X=new Point[16*nc*nc];
                    for(int i1=0; i1<4*nc; i1++) {
                      for(int j1=0; j1<4*nc; j1++) {
                        Point pq=Point(c[i1],c[j1],0.);
                        if(pq.asVector().length()<d_rJ) {
                          x[count]=pq;
                          X[count]=origin+TT*pq.asVector();
                          count++;
                        }
                      }
                    }
                    delete [] c;

                    // Get the solution at the integral points in local system
                    Vector* acc=new Vector[count];      // accelerations
                    Vector* vel=new Vector[count];      // velocities
                    Matrix3* dg = new Matrix3[count];   // displacement gradients
                    Matrix3* vg = new Matrix3[count];   // velocity gradients
  
                    for(int j=0; j<count; j++) {
                      // Get the solutions in global system
                      Vector ACC=Vector(0.,0.,0.);
                      Vector VEL=Vector(0.,0.,0.);
                      Matrix3 DG=Matrix3(0.0);
                      Matrix3 VG=Matrix3(0.0);
  
                      if(d_8or27==8)
                        patch->findCellAndWeights(X[j],ni,S);
                      else if(d_8or27==27)
                        patch->findCellAndWeights27(X[j],ni,S,psize[j]);
  
                      for(int k=0; k<d_8or27; k++) {
                        if(GnumPatls[ni[k]]!=0 && x[j].y()<0.) { // below crack 
                          // Valid only for stright crack within J-path, usually true
                          ACC += Gacc[ni[k]]       * S[k];
                          VEL += Gvel[ni[k]]       * S[k];
                          DG  += GdispGrads[ni[k]] * S[k];
                          VG  += GvelGrads[ni[k]]  * S[k];
                        }
                        else {  // above crack or in non-crack zone
                          ACC += gacc[ni[k]]       * S[k];
                          VEL += gvel[ni[k]]       * S[k];
                          DG  += gdispGrads[ni[k]] * S[k];
                          VG  += gvelGrads[ni[k]]  * S[k];
                        }
                      } // End of loop over k

                      // Transfer into the local system
                      acc[j] = T*ACC;
                      vel[j] = T*VEL;
 
                      dg[j]=Matrix3(0.0);
                      vg[j]=Matrix3(0.0);
                      for(int i1=0; i1<3; i1++) {
                        for(int j1=0; j1<3; j1++) {
                          for(int i2=0; i2<3; i2++) {
                            for(int j2=0; j2<3; j2++) {
                              dg[j](i1,j1) += T(i1,i2)*T(j1,j2)*DG(i2,j2);
                              vg[j](i1,j1) += T(i1,i2)*T(j1,j2)*VG(i2,j2);
                            }  
                          }  
                        } // End of loop over j1
                      } // End of loop over i1
                    } // End of loop over j
 
                    double f2ForJx=0.,f2ForJy=0.;
                    double rho=mpm_matl->getInitialDensity();
                    for(int j=0; j<count;j++) {
                      // Zero components in z direction for plane state
                      Vector dgx=Vector(dg[j](0,0),dg[j](1,0),0./*dg[j](2,0)*/);
                      Vector dgy=Vector(dg[j](0,1),dg[j](1,1),0./*dg[j](2,1)*/);
                      Vector vgx=Vector(vg[j](0,0),vg[j](1,0),0./*vg[j](2,0)*/);
                      Vector vgy=Vector(vg[j](0,1),vg[j](1,1),0./*vg[j](2,1)*/);
                      f2ForJx+=rho*(Dot(acc[j],dgx)-Dot(vel[j],vgx));
                      f2ForJy+=rho*(Dot(acc[j],dgy)-Dot(vel[j],vgy));
                    }
  
                    double Jarea=PI*d_rJ*d_rJ;
                    Jx2=f2ForJx/count*Jarea;
                    Jy2=f2ForJy/count*Jarea;

                    delete [] x;    delete [] X;
                    delete [] acc;  delete [] vel;
                    delete [] dg;   delete [] vg;

                  } // End of if(useVoluemIntegral)
 
                  cfJ[l]=Vector(Jx1+Jx2,Jy1+Jy2,0.);

                  /* Step 12: Convert J to K 
                  */
                  // Task 11a: Find COD near crack tip (point(-d,0,0) in local coordinates)
                  double d;
                  if(d_doCrackPropagation!="false")  // For crack propagation
                    d=rdadx*dx_max;
                  else  // For calculation of crack-tip parameters
                    d=d_rJ/2.;

                  double x_d=l1*(-d)+x0;
                  double y_d=m1*(-d)+y0;
                  double z_d=n1*(-d)+z0;
                  Point  p_d=Point(x_d,y_d,z_d);
         
                  // Get displacements at point p_d 
                  Vector disp_a=Vector(0.);
                  Vector disp_b=Vector(0.);
                  if(d_8or27==8)
                    patch->findCellAndWeights(p_d,ni,S);
                  else if(d_8or27==27)
                    patch->findCellAndWeights27(p_d,ni,S,psize[0]);
                  for(int k=0; k<d_8or27; k++) {
                    disp_a += gdisp[ni[k]] * S[k];
                    disp_b += Gdisp[ni[k]] * S[k];
                  }  

                  // Tranform to local system
                  Vector disp_a_prime=T*disp_a;
                  Vector disp_b_prime=T*disp_b;
 
                  // Crack opening displacements
                  Vector D = disp_a_prime - disp_b_prime;
  
                  // Task 11: Get crack propagating velocity, currently just set it to zero
                  Vector C=Vector(0.,0.,0.);   
 
                  // Convert J-integral into stress intensity factors 
                  Vector SIF;
                  cm->ConvertJToK(mpm_matl,cfJ[l],C,D,SIF);      
                  cfK[l]=SIF; 
                } // End if not operated
                else { // if operated
                  cfJ[l]=cfJ[pre_idx];
                  cfK[l]=cfK[pre_idx];
                } 
              } // End of loop over nodes(l) for calculating J & K
            } // End if(pid==i)

            // Broadcast J & K calculated by proc i to all ranks
            MPI_Bcast(cfJ,num,MPI_VECTOR,i,mpi_crack_comm);
            MPI_Bcast(cfK,num,MPI_VECTOR,i,mpi_crack_comm);

            // Save data in cfsegJ and cfSegK
            for(int l=0; l<num; l++) {
              int idx=cfnset[m][i][l];
              cfSegJ[m][idx]=cfJ[l];
              cfSegK[m][idx]=cfK[l];
            }

            // Release dynamic arries
            delete [] cfJ;
            delete [] cfK; 
          } // End if(num>0)
        } // End of loop over ranks (i)

        // Output fracture parameters and crack-front position
        if(m==mS && pid==0 && (calFractParameters || doCrackPropagation)) {
          OutputCrackFrontResults(m);
        }

      } // End if(calFractParameters || doCrackPropagation)
    } // End of loop over matls
  } // End of loop patches
}

// Output fracture parameters and crack-front position
void Crack::OutputCrackFrontResults(const int& m)
{ 
  //static double timeforoutputcrack=0.0;

  double time=d_sharedState->getElapsedTime();
  ofstream outCrkFrt("CrackFrontResults.dat", ios::app);

//  if(time>=timeforoutputcrack) {
    for(int i=0;i<(int)cfSegNodes[m].size();i++) {
      int    node=cfSegNodes[m][i];
      int segs[2];
      FindSegsFromNode(m,node,segs);
      Point  cp=cx[m][node];
      Vector cfPara=cfSegK[m][i];
      outCrkFrt << setw(15) << time
                << setw(5)  << i/2
                << setw(10)  << node
                << setw(15) << cp.x()
                << setw(15) << cp.y()
                << setw(15) << cp.z()
                << setw(15) << cfPara.x()
                << setw(15) << cfPara.y()
                << setw(15) << cfPara.z();
      if(cfPara.x()!=0.) {  
        outCrkFrt << setw(15) << cfPara.y()/cfPara.x() << endl;
      }
      else {
        outCrkFrt << setw(15) << "inf" << endl;
      }
      if(segs[1]<0) { // End of the sub-crack
        outCrkFrt << endl;
      }
    }
//    timeforoutputcrack+=d_outputInterval;
//  }
}

void Crack::addComputesAndRequiresPropagateCrackFrontNodes(Task* t,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const 
{ 
  Ghost::GhostType  gac = Ghost::AroundCells;
  int NGC=2*NGN;
  t->requires(Task::NewDW, lb->gMassLabel, gac, NGC);
  t->requires(Task::NewDW, lb->GMassLabel, gac, NGC);
  if(d_8or27==27)
   t->requires(Task::OldDW,lb->pSizeLabel, Ghost::None);
}

void Crack::PropagateCrackFrontNodes(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* /*matls*/,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
  for(int p=0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double dx_max=Max(dx.x(),dx.y(),dx.z());

    int pid,patch_size;
    MPI_Comm_rank(mpi_crack_comm, &pid);
    MPI_Comm_size(mpi_crack_comm, &patch_size);

    enum {NO=0,YES};
    enum {R=0,L};  // the right and left sides

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m=0; m<numMPMMatls; m++) {
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();

      // Get nodal mass information
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch); 

      Ghost::GhostType  gac = Ghost::AroundCells;
      constNCVariable<double> gmass,Gmass;
      int NGC=2*NGN;
      new_dw->get(gmass, lb->gMassLabel, dwi, patch, gac, NGC);
      new_dw->get(Gmass, lb->GMassLabel, dwi, patch, gac, NGC);

      constParticleVariable<Vector> psize;
      if(d_8or27==27) old_dw->get(psize, lb->pSizeLabel, pset);

      if(doCrackPropagation) {
        // cfSegPtsT -- crack front points after propagation
        cfSegPtsT[m].clear();

        IntVector ni[MAX_BASIS];
        double S[MAX_BASIS];

        // Step 1: Update crack front segments, discard dead segements 

        // Step 1a: Detect if crack-front nodes inside materials
        cfSegNodesInMat[m].resize(cfSegNodes[m].size());
        for(int i=0; i<(int)cfSegNodes[m].size();i++) {
          cfSegNodesInMat[m][i]=YES;
        }

        for(int i=0; i<patch_size; i++) { 
          int num=cfnset[m][i].size();
          short* inMat=new short[num];

          if(pid==i) { // Rank i does it
            for(int j=0; j<num; j++) {
              int idx=cfnset[m][i][j];
              int node=cfSegNodes[m][idx]; 
              Point pt=cx[m][node];
              inMat[j]=YES;

              // Get the node indices that surround the cell
              if(d_8or27==8)
                patch->findCellAndWeights(pt, ni, S);
              else if(d_8or27==27)
                patch->findCellAndWeights27(pt, ni, S, psize[j]);

              for(int k = 0; k < d_8or27; k++) {
                double totalMass=gmass[ni[k]]+Gmass[ni[k]];
                if(totalMass<5*d_SMALL_NUM_MPM) {
                  inMat[j]=NO;
                  break;
                }
              }
            } // End of loop over j
          } // End of if(pid==i)

          MPI_Bcast(&inMat[0],num,MPI_SHORT,i,mpi_crack_comm);

          for(int j=0; j<num; j++) {
            int idx=cfnset[m][i][j];
            cfSegNodesInMat[m][idx]=inMat[j];
          }
          delete [] inMat; 
        } // End of loop over patches

        MPI_Barrier(mpi_crack_comm);

        // Step 1b: See if crack-front segment center inside materials
        //          for single crack-front seg problems
        int ncfSegs=(int)cfSegNodes[m].size()/2;
        if(ncfSegs==1) {
          cfSegCenterInMat[m].resize(ncfSegs);
          for(int i=0; i<ncfSegs;i++){
            cfSegCenterInMat[m][i]=YES;
          }

          for(int i=0; i<patch_size; i++) {
            int num=cfsset[m][i].size();
            short* inMat=new short[num];
            if(pid==i) {
              for(int j=0; j<num; j++) {
                int seg=cfsset[m][i][j];
                int nd1=cfSegNodes[m][2*seg];
                int nd2=cfSegNodes[m][2*seg+1];
                Point center=cx[m][nd1]+(cx[m][nd2]-cx[m][nd1])/2.;
                inMat[j]=YES;

                // Get the node indices that surround the cell
                if(d_8or27==8)
                  patch->findCellAndWeights(center, ni, S);
                else if(d_8or27==27)
                  patch->findCellAndWeights27(center, ni, S, psize[j]);

                for(int k = 0; k < d_8or27; k++) {
                  double totalMass=gmass[ni[k]]+Gmass[ni[k]];
                  if(totalMass<5*d_SMALL_NUM_MPM) {
                    inMat[j]=NO;
                    break;
                  }
                }
              } // End of loop over j
            } // End of if(pid==i)

            MPI_Bcast(&inMat[0],num,MPI_SHORT,i,mpi_crack_comm);

            for(int j=0; j<num; j++) {
              int seg=cfsset[m][i][j];
              cfSegCenterInMat[m][seg]=inMat[j];
            }
            delete [] inMat;
          } // End of loop over i 
        } // End of if(ncfSegs==1) 

        MPI_Barrier(mpi_crack_comm);

        // Step 1c: Store crack-front nodes and J&K in temporary arraies
        int old_size=(int)cfSegNodes[m].size();
        int*    cfSegNdT = new int[old_size];
        Vector* cfSegJT  = new Vector[old_size];
        Vector* cfSegKT  = new Vector[old_size];
        Vector* cfSegNmT = new Vector[old_size/2];
        for(int i=0; i<old_size; i++) {
          cfSegNdT[i]=cfSegNodes[m][i];
          cfSegJT[i]=cfSegJ[m][i];
          cfSegKT[i]=cfSegK[m][i];
          if(i<old_size/2) cfSegNmT[i]=cfSegV2[m][i];
        }
        cfSegNodes[m].clear();
        cfSegV2[m].clear();
        cfSegK[m].clear();
        cfSegJ[m].clear();

        // Step 1d: Collect the active crack-front segs & parameters
        for(int i=0; i<old_size/2; i++) { // Loop over crack-front segs 
          short thisSegActive=NO;
          int nd1=cfSegNdT[2*i];
          int nd2=cfSegNdT[2*i+1];
          if(old_size/2==1) { // for single seg problems 
            // Remain active if any of two ends and center inside
            if(cfSegNodesInMat[m][2*i] || cfSegNodesInMat[m][2*i+1] ||
               cfSegCenterInMat[m][i]) thisSegActive=YES;
          }
          else { // for multiple seg problems
            // Remain active if any of two ends inside  
            if(cfSegNodesInMat[m][2*i] || cfSegNodesInMat[m][2*i+1])
              thisSegActive=YES;
          }
          if(thisSegActive) { 
            cfSegNodes[m].push_back(nd1);
            cfSegNodes[m].push_back(nd2); 
            cfSegJ[m].push_back(cfSegJT[2*i]);
            cfSegJ[m].push_back(cfSegJT[2*i+1]);
            cfSegK[m].push_back(cfSegKT[2*i]);
            cfSegK[m].push_back(cfSegKT[2*i+1]);
            cfSegV2[m].push_back(cfSegNmT[i]);
          }
          else { // The segment is dead
            if(pid==0) {
              cout << "!!! Crack-front seg " << i << "(" << nd1  
                   << cx[m][nd1] << "-->" << nd2 << cx[m][nd2]
                   << ") of Mat " << m << " is dead." << endl; 
            }
          }
        } // End of loop over crack-front segs     
        delete [] cfSegNdT;
        delete [] cfSegNmT;
        delete [] cfSegJT;
        delete [] cfSegKT;

        // If all crack-front segs dead, the material is broken.
        if(cfSegNodes[m].size()/2<=0 && pid==0) {
         cout << "!!! Material " << m
              << " is broken. Program terminated." << endl;
         exit(1);
        }

        // Step 2: Detect if crack front nodes propagate (cp) 
        //         and propagate them virtually (da) 
        short*  cp=new  short[(int)cfSegNodes[m].size()];
        Vector* da=new Vector[(int)cfSegNodes[m].size()];

        for(int i=0; i<(int)cfSegNodes[m].size(); i++) { 
          int node=cfSegNodes[m][i];

          // Find the segment(s) connected by the node:
          int segs[2];
          FindSegsFromNode(m,node,segs);

          // Detect if the node has been operated  
          int pre_idx=-1;
          for(int ij=0; ij<i; ij++) {
            if(node==cfSegNodes[m][ij]) {
              pre_idx=ij;
              break;
            }
          }
          //if(i>0 && node==cfSegNodes[m][i-1]) pre_idx=i-1;

          if(pre_idx<0) { // for the nodes not operated
            // Direction-cosines of the node
            Vector v1,v2,v3;
            Vector v2T=Vector(0.,0.,0.);
            for(int j=R; j<=L; j++) {
              if(segs[j]>=0) v2T+=cfSegV2[m][segs[j]];    
            }
            v2=v2T/v2T.length();
            v3=-cfSegV3[m][i];
            Vector v23=Cross(v2,v3);
            v1=v23/v23.length(); 
            Vector v31=Cross(v3,v1);
            v2=v31/v31.length();            
            // Coordinates transformation matrix from local to global
            Matrix3 T=Matrix3(v1.x(), v2.x(), v3.x(),
                              v1.y(), v2.y(), v3.y(),
                              v1.z(), v2.z(), v3.z());

            // Determine if the node propagates 
            cp[i]=NO;
            double KI  = cfSegK[m][i].x();
            double KII = cfSegK[m][i].y();
            if(cm->CrackSegmentPropagates(KI,KII)) cp[i]=YES;
                
            // Propagate the node virtually  
            double theta=cm->GetPropagationDirection(KI,KII);
            double dl=rdadx*dx_max;
            Vector da_local=Vector(dl*cos(theta),dl*sin(theta),0.);
            da[i]=T*da_local;
          } // End of if(!operated)
          else { // if(operated)
            cp[i]=cp[pre_idx];
            da[i]=da[pre_idx];
          }  // End of if(!operated) {} else {}
        } // End of loop over cfSegNodes 

	// Step 3: Determine the propagation extent for each node
	int min_idx=0,max_idx=-1;
        int cfNodeSize=cfSegNodes[m].size();
        for(int i=0; i<cfNodeSize; i++) {
          int node=cfSegNodes[m][i];
          Point pt=cx[m][node];

          // Detect if the node has been operated
          int pre_idx=-1;
          for(int ij=0; ij<i; ij++) {
            if(node==cfSegNodes[m][ij]) {
              pre_idx=ij;
              break;
            }
          }
  
          if(pre_idx<0) { // not operated
            // Find the segments coonected by the node
            int segs[2];
            FindSegsFromNode(m,node,segs);
            //  Find the minimum and maximum indices of this crack
            if(i>max_idx) { 
              int segsT[2];
              min_idx=i;
              segsT[R]=segs[R];
              while(segsT[R]>=0 && min_idx>0)
                FindSegsFromNode(m,cfSegNodes[m][--min_idx],segsT);
              max_idx=i;
              segsT[L]=segs[L];
              while(segsT[L]>=0 && max_idx<cfNodeSize-1) 
                FindSegsFromNode(m,cfSegNodes[m][++max_idx],segsT);
            }

            // Count the nodes which propagate among (2ns+1) nodes around pt
            int ns=(max_idx-min_idx+1)/10+2;  
            int np=0;
            for(int j=-ns; j<=ns; j++) { 
              int cidx=i+2*j;
              if(cidx<min_idx && cp[min_idx]) np++;
              if(cidx>max_idx && cp[max_idx]) np++;
              if(cidx>=min_idx && cidx<=max_idx && cp[cidx]) np++;
            } 
     
            // New position of pt after virtual propagation
            double fraction=(double)np/(2*ns+1);
            Point new_pt=pt+fraction*da[i];         

            // Step 4: Deal with the boundary nodes: extending new_pt
            //         by (fraction*rdadx*dx_max) if it is inside of material
            if((segs[R]<0||segs[L]<0) && (new_pt-pt).length()/dx_max>0.01) {
              // Check if new_pt is inside the material
              short newPtInMat=YES;

              // patchID of the node before propagation
              int procID=-1;
               for(int i1=0; i1<patch_size; i1++) {
                 for(int j1=0; j1<(int)cfnset[m][i1].size(); j1++){
                   int nodeij=cfSegNodes[m][cfnset[m][i1][j1]];
                   if(node==nodeij) {
                     procID=i1;
                     break;
                   }
                 }
               }
               if(pid==procID) {
                 IntVector ni[MAX_BASIS];
                 double S[MAX_BASIS];
                 if(d_8or27==8)
                   patch->findCellAndWeights(new_pt, ni, S);
                 else if(d_8or27==27)
                   patch->findCellAndWeights27(new_pt, ni, S, psize[0]);

                 for(int k = 0; k < d_8or27; k++) {
                   double totalMass=gmass[ni[k]]+Gmass[ni[k]];
                   if(totalMass<5*d_SMALL_NUM_MPM) {
                     newPtInMat=NO;
                     break;
                   }
                 }
               }
               MPI_Bcast(&newPtInMat,1,MPI_SHORT,procID,mpi_crack_comm);

               if(newPtInMat) {
                 Point tmp_pt=new_pt;
                 int n1=-1,n2=-1;
                 if(segs[R]<0) { // right edge nodes
                   n1=cfSegNodes[m][2*segs[L]+1];
                   n2=cfSegNodes[m][2*segs[L]];
                 }
                 else if(segs[L]<0) { // left edge nodes
                   n1=cfSegNodes[m][2*segs[R]];
                   n2=cfSegNodes[m][2*segs[R]+1];
                 }

                 Vector v=TwoPtsDirCos(cx[m][n1],cx[m][n2]);
                 new_pt=tmp_pt+v*(fraction*rdadx*dx_max);

                 // Check if it beyond the global gird
                 FindIntersectionLineAndGridBoundary(tmp_pt,new_pt);
               }
             }

            // Push back the new_pt
            cfSegPtsT[m].push_back(new_pt);
          } // End if(!operated)
          else {
            Point pre_pt=cfSegPtsT[m][pre_idx];
            cfSegPtsT[m].push_back(pre_pt);
          }
        } // End of loop cfSegNodes

        // Apply symmetric BCs to new crack-front points
        for(int i=0; i<(int)cfSegNodes[m].size();i++) {
          Point pt=cx[m][cfSegNodes[m][i]];
          ApplyBCsForCrackPoints(dx,pt,cfSegPtsT[m][i]);
        }

        // Release dynamic arraies
        delete [] cp;
        delete [] da;
      } // End of if(doCrackPropagation)

    } // End of loop over matls
  } // End of loop over patches
}

void Crack::FindSegsFromNode(const int& m,const int& node, int segs[])
{
  // segs[R] -- the seg on the right of the node
  // segs[L] -- the seg on the left of the node

  enum {R=0,L};
  segs[R]=segs[L]=-1;

  int ncfSegs=(int)cfSegNodes[m].size()/2;
  for(int j=0; j<ncfSegs; j++) {
    int node0=cfSegNodes[m][2*j];
    int node1=cfSegNodes[m][2*j+1];
    if(node==node1) // the right seg
      segs[R]=j;
    if(node==node0) // the left seg
      segs[L]=j;
  } // End of loop over j

  if(segs[R]<0 && segs[L]<0) {
    cout << " Failure to find the crack-front segments for node "
         << node << ". Program terminated." << endl;
    for(int j=0; j<ncfSegs; j++) {
      cout << "seg=" << j << ": [" << cfSegNodes[m][2*j]<< ","
           << cfSegNodes[m][2*j+1] << "]" << endl;
    }
    exit(1);
  }
}
 
void Crack::addComputesAndRequiresConstructNewCrackFrontElems(Task* /*t*/,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
  // Nothing to do currently
}

void Crack::ConstructNewCrackFrontElems(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* /*matls*/,
                      DataWarehouse* /*old_dw*/,
                      DataWarehouse* /*new_dw*/)
{
  for(int p=0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double dx_max=Max(dx.x(),dx.y(),dx.z());

    double PI=3.141592654;
    enum {NO=0,YES};
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m=0; m<numMPMMatls; m++) {
      if(doCrackPropagation) {

        // Combine crack front nodes if they propagates a little
        for(int i=0; i<(int)cfSegNodes[m].size(); i++) {
           int node=cfSegNodes[m][i];
           // crack plane normal at this node
           int segs[2];
           FindSegsFromNode(m,node,segs);
           Vector v2,v2T=Vector(0.,0.,0.);
           for(int j=0; j<=1; j++) {
             if(segs[j]>=0) v2T+=cfSegV2[m][segs[j]];
           }
           v2=v2T/v2T.length();

           // crack propagation increment(dis) and direction(vp) 
           double dis=(cfSegPtsT[m][i]-cx[m][node]).length();
           Vector vp=TwoPtsDirCos(cx[m][node],cfSegPtsT[m][i]);
           // Crack propa angle(in degree) measured from crack plane
           double angle=90-acos(Dot(vp,v2))*180/PI;
           if(dis<0.2*(rdadx*dx_max) || fabs(angle)<10)
             cx[m][node]=cfSegPtsT[m][i];
        }

        // Clear up the temporary crack front segment nodes
        cfSegNodesT[m].clear();

        int ncfSegs=cfSegNodes[m].size()/2;
        for(int i=0; i<ncfSegs; i++) { // Loop over front segs
          // crack front nodes and points 
          int n1,n2,n1p,n2p,nc;
          Point p1,p2,p1p,p2p,pc;
          n1=cfSegNodes[m][2*i];
          n2=cfSegNodes[m][2*i+1];
          p1=cx[m][n1];
          p2=cx[m][n2];

          p1p=cfSegPtsT[m][2*i];
          p2p=cfSegPtsT[m][2*i+1];
          pc =p1p+(p2p-p1p)/2.;

          // length of crack front segment after propagation 
          double l12=(p2p-p1p).length();

          // seven cases
          short sp=YES, ep=YES; 
          if((p1p-p1).length()/dx_max<0.01) sp=NO; // p1 no propagating
          if((p2p-p2).length()/dx_max<0.01) ep=NO; // p2 no propagating

          short CASE=0;             // no propagation
          if(l12/cs0[m]<2.) {    // no break the seg
            if( sp && !ep) CASE=1;  // p1 propagates, p2 doesn't
            if(!sp &&  ep) CASE=2;  // p2 propagates, p1 doesn't
            if( sp &&  ep) CASE=3;  // Both of p1 and p2 propagate
          }
          else {                    // break the seg
            if( sp && !ep) CASE=4;  // p1 propagates, p2 doesn't
            if(!sp &&  ep) CASE=5;  // p2 propagates, p1 doesn't
            if( sp &&  ep) CASE=6;  // Both of p1 and p2 propagate
          }

          if(CASE>3 && patch->getID()==0)
            cout << "!!! mat " << m << ", cfSeg " << i
                << " will be split into two segments (CASE " 
                << CASE << ")" << endl;

          // Detect if the seg is the first seg of a crack
          short firstSeg=YES;
          if(i>0 && n1==cfSegNodes[m][2*(i-1)+1]) firstSeg=NO;

          switch(CASE) {
            case 0:
              cfSegNodesT[m].push_back(n1);
              cfSegNodesT[m].push_back(n2);
              break;
            case 1: 
              // the new crack point 
              if(!firstSeg) n1p=(int)cx[m].size()-1;
              else {n1p=(int)cx[m].size(); cx[m].push_back(p1p);} 
              // the new crack element
              ce[m].push_back(IntVector(n1,n1p,n2));
              // the new crack front-seg nodes 
              cfSegNodesT[m].push_back(n1p);
              cfSegNodesT[m].push_back(n2);
              break;
            case 2: 
              n2p=(int)cx[m].size();
              // the new crack point
              cx[m].push_back(p2p);
              // the new crack element
              ce[m].push_back(IntVector(n1,n2p,n2));
              // the new crack front-seg nodes
              cfSegNodesT[m].push_back(n1);
              cfSegNodesT[m].push_back(n2p);
              break;
            case 3: 
              // the new crack point
              if(!firstSeg) n1p=(int)cx[m].size()-1;
              else {n1p=(int)cx[m].size(); cx[m].push_back(p1p);}
              n2p=n1p+1;
              cx[m].push_back(p2p);
              // the new crack elements
              ce[m].push_back(IntVector(n1,n1p,n2));
              ce[m].push_back(IntVector(n1p,n2p,n2));
              // the new crack front-seg nodes
              cfSegNodesT[m].push_back(n1p);
              cfSegNodesT[m].push_back(n2p);
              break;
            case 4:   
              // the new crack point
              if(!firstSeg) n1p=(int)cx[m].size()-1;
              else {n1p=(int)cx[m].size(); cx[m].push_back(p1p);}
              nc=n1p+1;
              cx[m].push_back(pc);
              // the new crack elements
              ce[m].push_back(IntVector(n1,n1p,nc));
              ce[m].push_back(IntVector(n1,nc,n2));
              // the new crack front-seg nodes, a new seg generated
              cfSegNodesT[m].push_back(n1p);
              cfSegNodesT[m].push_back(nc);
              cfSegNodesT[m].push_back(nc);
              cfSegNodesT[m].push_back(n2);
              break;
            case 5:
              nc=(int)cx[m].size();
              n2p=nc+1;
              // the new crack points
              cx[m].push_back(pc);
              cx[m].push_back(p2p);
              // the new crack elements
              ce[m].push_back(IntVector(n1,nc,n2));
              ce[m].push_back(IntVector(n2,nc,n2p));
              // the new crack front-seg nodes, a new seg generated 
              cfSegNodesT[m].push_back(n1);
              cfSegNodesT[m].push_back(nc);
              cfSegNodesT[m].push_back(nc);
              cfSegNodesT[m].push_back(n2p);
              break;
            case 6:  
              // the new crack point
              if(!firstSeg) n1p=(int)cx[m].size()-1;
              else {n1p=(int)cx[m].size(); cx[m].push_back(p1p);}
              nc =n1p+1;
              n2p=n1p+2;
              cx[m].push_back(pc);
              cx[m].push_back(p2p);
              // the new crack elements
              ce[m].push_back(IntVector(n1,n1p,nc));
              ce[m].push_back(IntVector(n1,nc,n2));
              ce[m].push_back(IntVector(n2,nc,n2p));
              // the new crack front-seg nodes, a new seg generated 
              cfSegNodesT[m].push_back(n1p);
              cfSegNodesT[m].push_back(nc);
              cfSegNodesT[m].push_back(nc);
              cfSegNodesT[m].push_back(n2p);
              break;
          }
        } // End of loop over crack front segments

        // Reset crack front segment nodes after crack propagation
        cfSegNodes[m].clear();
        for(int i=0; i<(int)cfSegNodesT[m].size(); i++) {
          cfSegNodes[m].push_back(cfSegNodesT[m][i]);
        }
      } // End of if(doCrackpropagation)
    } // End of loop over matls
  } // End of loop over patches
}

void Crack::addComputesAndRequiresCrackPointSubset(Task* /*t*/,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
// Currently do nothing 
}

void Crack::CrackPointSubset(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* /*matls*/,
                      DataWarehouse* /*old_dw*/,
                      DataWarehouse* /*new_dw*/)
{
  for(int p=0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    int pid,patch_size;
    MPI_Comm_rank(mpi_crack_comm, &pid);
    MPI_Comm_size(mpi_crack_comm, &patch_size);

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m=0; m<numMPMMatls; m++) {
      cnset[m][pid].clear();
      // Collect crack nodes in each patch
      for(int i=0; i<(int)cx[m].size(); i++) {
        if(patch->containsPoint(cx[m][i])) {
          cnset[m][pid].push_back(i);
        } 
      } // End of loop over nodes

      MPI_Barrier(mpi_crack_comm);

      // Broadcast cnset of each patch to all ranks
      for(int i=0; i<patch_size; i++) {
        int num; // number of crack nodes in patch i
        if(i==pid) num=cnset[m][i].size();
        MPI_Bcast(&num,1,MPI_INT,i,mpi_crack_comm);
        cnset[m][i].resize(num);
        MPI_Bcast(&cnset[m][i][0],num,MPI_INT,i,mpi_crack_comm);
      }
    
    } // End of loop over matls
  } // End of loop over patches
}

void Crack::addComputesAndRequiresMoveCracks(Task* t,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  Ghost::GhostType  gac = Ghost::AroundCells;
  int NGC=2*NGN;
  t->requires(Task::NewDW, lb->gMassLabel,         gac, NGC);
  t->requires(Task::NewDW, lb->gNumPatlsLabel,     gac, NGC);
  t->requires(Task::NewDW, lb->gVelocityStarLabel, gac, NGC);
  t->requires(Task::NewDW, lb->GMassLabel,         gac, NGC);
  t->requires(Task::NewDW, lb->GNumPatlsLabel,     gac, NGC);
  t->requires(Task::NewDW, lb->GVelocityStarLabel, gac, NGC);

  if(d_8or27==27)
   t->requires(Task::OldDW,lb->pSizeLabel, Ghost::None);
}

void Crack::MoveCracks(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
  for(int p=0; p<patches->size(); p++){
    int pid,patch_size;
    MPI_Comm_rank(mpi_crack_comm, &pid);
    MPI_Comm_size(mpi_crack_comm, &patch_size);
    MPI_Datatype MPI_POINT=fun_getTypeDescription((Point*)0)->getMPIType();

    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double dx_min=Min(dx.x(),dx.y(),dx.z());
 
    enum {NO=0,YES};

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label() );

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){ // loop over matls    
      if((int)ce[m].size()==0) // for materials with no cracks
        continue; 

      /* Get the necessary information
      */
      MPMMaterial* mpm_matl=d_sharedState->getMPMMaterial(m);
      int dwi=mpm_matl->getDWIndex();
      ParticleSubset* pset=old_dw->getParticleSubset(dwi,patch);
      constParticleVariable<Vector> psize;
      if(d_8or27==27) 
        old_dw->get(psize,lb->pSizeLabel,pset);

      Ghost::GhostType  gac = Ghost::AroundCells;
      constNCVariable<double> gmass,Gmass;
      constNCVariable<int>    gnum,Gnum;
      constNCVariable<Vector> gvelocity_star, Gvelocity_star;
      int NGC=2*NGN;
      new_dw->get(gmass,         lb->gMassLabel,        dwi,patch,gac,NGC);
      new_dw->get(gnum,          lb->gNumPatlsLabel,    dwi,patch,gac,NGC);
      new_dw->get(gvelocity_star,lb->gVelocityStarLabel,dwi,patch,gac,NGC);
      new_dw->get(Gmass,         lb->GMassLabel,        dwi,patch,gac,NGC);
      new_dw->get(Gnum,          lb->GNumPatlsLabel,    dwi,patch,gac,NGC);
      new_dw->get(Gvelocity_star,lb->GVelocityStarLabel,dwi,patch,gac,NGC);

      for(int i=0; i<patch_size; i++) { // Loop over all patches
        int numNodes=cnset[m][i].size();
        if(numNodes>0) {
          Point* cptmp=new Point[numNodes]; 
          if(pid==i) { // Proc i update the position of nodes in the patch
            for(int j=0; j<numNodes; j++) {
              int idx=cnset[m][i][j];
              Point pt=cx[m][idx];

              double mg,mG;
              Vector vg,vG;
              Vector vcm = Vector(0.0,0.0,0.0);

              // Get element nodes and shape functions          
              IntVector ni[MAX_BASIS];
              double S[MAX_BASIS];
              if(d_8or27==8) 
                patch->findCellAndWeights(pt, ni, S);
              else if(d_8or27==27) 
                patch->findCellAndWeights27(pt, ni, S, psize[idx]);

              // Calculate center-of-velocity (vcm) 
              // Sum of shape functions from nodes with particle(s) around them
              // This part is necessary for pt located outside the body 
              double sumS=0.0;  
              for(int k =0; k < d_8or27; k++) {
                Point pi=patch->nodePosition(ni[k]);
                if(RealGlobalGridContainsNode(dx_min,pi) &&  //ni[k] in real grid
                   (gnum[ni[k]]+Gnum[ni[k]]!=0)) sumS += S[k];
              }
              if(sumS>1.e-6) {   
                for(int k = 0; k < d_8or27; k++) {
                  Point pi=patch->nodePosition(ni[k]);
                  if(RealGlobalGridContainsNode(dx_min,pi) &&  
                             (gnum[ni[k]]+Gnum[ni[k]]!=0)) { 
                    mg = gmass[ni[k]];
                    mG = Gmass[ni[k]];
                    vg = gvelocity_star[ni[k]];
                    vG = Gvelocity_star[ni[k]];
                    vcm += (mg*vg+mG*vG)/(mg+mG)*S[k]/sumS;
                  }
                }
              }

              // Update the position
              cptmp[j]=pt+vcm*delT;

              // Apply symmetric BCs for the new position 
              ApplyBCsForCrackPoints(dx,pt,cptmp[j]);
            } // End of loop over numNodes 
          } // End if(pid==i)
       
          // Broadcast the updated position to all ranks 
          MPI_Bcast(cptmp,numNodes,MPI_POINT,i,mpi_crack_comm);
       
          // Save the updated potion back  
          for(int j=0; j<numNodes; j++) {
            int idx=cnset[m][i][j];
            cx[m][idx]=cptmp[j];
          }

          delete [] cptmp;

        } // End of if(numNodes>0)
      } // End of loop over patch_size

      // Detect if crack points outside the global grid
      for(int i=0; i<(int)cx[m].size();i++) {
        if(!RealGlobalGridContainsNode(dx_min,cx[m][i])) {
          cout << "cx[" << m << "," << i << "]=" << cx[m][i] 
               << " outside the global grid." 
               << " Program terminated." << endl;
          exit(1);
        }
      }  
    } // End of loop over matls
  }
}

short Crack::RealGlobalGridContainsNode(const double& dx,const Point& pt)
{
  // return true if pt within the real global grid or
  // around it (within 1% of the cell size dx)
  double px=pt.x(),  py=pt.y(),  pz=pt.z();
  double lx=GLP.x(), ly=GLP.y(), lz=GLP.z();
  double hx=GHP.x(), hy=GHP.y(), hz=GHP.z();

  return ((px>lx || fabs(px-lx)/dx<0.01) && (px<hx || fabs(px-hx)/dx<0.01) &&
          (py>ly || fabs(py-ly)/dx<0.01) && (py<hy || fabs(py-hy)/dx<0.01) &&
          (pz>lz || fabs(pz-lz)/dx<0.01) && (pz<hz || fabs(pz-hz)/dx<0.01));
}

// If p2 is outside the global grid, find the intersection between p1-p2 
// and grid boundary, and store it in p2
void Crack::FindIntersectionLineAndGridBoundary(const Point& p1, Point& p2) 
{
  double lx=GLP.x(), ly=GLP.y(), lz=GLP.z();
  double hx=GHP.x(), hy=GHP.y(), hz=GHP.z();

  double x1=p1.x(), y1=p1.y(), z1=p1.z();
  double x2=p2.x(), y2=p2.y(), z2=p2.z();

  Vector v=TwoPtsDirCos(p1,p2);
  double l=v.x(), m=v.y(), n=v.z();

  if(x2>hx || x2<lx) {
    if(x2>hx) x2=hx;
    if(x2<lx) x2=lx;
    y2=y1+(x2-x1)/l*m;
    z2=z1+(x2-x1)/l*n;
  }

  if(y2>hy || y2<ly) {
    if(y2>hy) y2=hy;
    if(y2<ly) y2=ly;
    x2=x1+(y2-y1)/m*l;
    z2=z1+(y2-y1)/m*n;
  }

  if(z2>hz || z2<lz) {
    if(z2>hz) z2=hz;
    if(z2<lz) z2=lz;
    x2=x1+(z2-z1)/n*l;
    y2=y1+(z2-z1)/n*m;
  }
  p2=Point(x2,y2,z2);
}

void Crack::addComputesAndRequiresUpdateCrackExtentAndNormals(Task* /*t*/,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
// Currently do nothing
}

void Crack::UpdateCrackExtentAndNormals(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* /*matls*/,
                      DataWarehouse* /*old_dw*/,
                      DataWarehouse* /*new_dw*/)
{
  for(int p=0; p<patches->size(); p++){
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m=0; m<numMPMMatls; m++) {
      // Update crack extent for this material  
      cmin[m]=Point(9.e16,9.e16,9.e16);
      cmax[m]=Point(-9.e16,-9.e16,-9.e16);
      for(int i=0; i<(int)cx[m].size(); i++) {
        cmin[m]=Min(cmin[m],cx[m][i]);
        cmax[m]=Max(cmax[m],cx[m][i]);
      } // End of loop over crack points 

      // Update the outward normals for crack front segments
      int ncfSegs=(int)cfSegNodes[m].size()/2;
      cfSegV2[m].resize(ncfSegs);
      for(int i=0; i<ncfSegs; i++) {
        int elemID=-1;
        int n1=cfSegNodes[m][2*i];
        int n2=cfSegNodes[m][2*i+1];
        // Find the crack element ID of the front segment
        Vector ceNorm;
        for(int j=0; j<(int)ce[m].size(); j++) {
          int numDupNodes=0;
          int n3=ce[m][j].x();
          int n4=ce[m][j].y();
          int n5=ce[m][j].z();
          if(n1==n3 || n1==n4 || n1==n5) numDupNodes++;
          if(n2==n3 || n2==n4 || n2==n5) numDupNodes++;
          if(numDupNodes==2) {
            elemID=j;
            ceNorm=TriangleNormal(cx[m][n3],cx[m][n4],cx[m][n5]);
            break;
          }
        } // End of loop over j
        if(elemID>=0) {
          cfSegV2[m][i]=ceNorm;
        } 
        else {
          cout << " Failure to find cfSegNorm of (mat " << m 
               << ", seg " << i << "). Program terminated." << endl;
          exit(1);
        }
      } // End of loop over ncfSegs 

      // Smooth crack-front and get tangential vectors
      if(d_calFractParameters!="false" || d_doCrackPropagation!="false")
      SmoothCrackFrontAndGetTangentialVector(m);

    } // End of loop over matls
  } // End of loop patches
}

// *** PRIVATE METHODS BELOW ***

// Calculate outward normal of a triangle
Vector Crack::TriangleNormal(const Point& p1, 
            const Point& p2, const Point& p3)
{
  double x21,x31,y21,y31,z21,z31;
  double a,b,c;
  Vector norm;

  x21=p2.x()-p1.x();
  x31=p3.x()-p1.x();
  y21=p2.y()-p1.y();
  y31=p3.y()-p1.y();
  z21=p2.z()-p1.z();
  z31=p3.z()-p1.z();

  a=y21*z31-z21*y31;
  b=x31*z21-z31*x21;
  c=x21*y31-y21*x31;
  if(Vector(a,b,c).length()>1.e-16)
     norm=Vector(a,b,c)/Vector(a,b,c).length();
  else
     norm=Vector(a,b,c);

  return norm;
}

// Detect if two points are in same side of a plane
short Crack::Location(const Point& p, const Point& g, 
        const Point& n1, const Point& n2, const Point& n3) 
{
  /* p,g -- two points (usually particle and node)
     n1,n2,n3 -- three points on the plane
  */
  double x1,y1,z1,x2,y2,z2,x3,y3,z3,xp,yp,zp,xg,yg,zg;
  double x21,y21,z21,x31,y31,z31,a,b,c,d,dp,dg; 
  short cross;
 
  x1=n1.x(); y1=n1.y(); z1=n1.z();
  x2=n2.x(); y2=n2.y(); z2=n2.z();
  x3=n3.x(); y3=n3.y(); z3=n3.z();
  xp=p.x();  yp=p.y();  zp=p.z();
  xg=g.x();  yg=g.y();  zg=g.z();

  x21=x2-x1; y21=y2-y1; z21=z2-z1;
  x31=x3-x1; y31=y3-y1; z31=z3-z1;

  a=y21*z31-z21*y31;
  b=z21*x31-x21*z31;
  c=x21*y31-y21*x31;
  d=-a*x1-b*y1-c*z1;

  dp=a*xp+b*yp+c*zp+d;
  dg=a*xg+b*yg+c*zg+d;

    if(fabs(dg)<1.e-16) { // node on crack plane
     if(dp>0.) 
        cross=1;        // p above carck
     else
        cross=2;        // p below crack
  }
  else {                // node not on crack plane
     if(dp*dg>0.) 
        cross=0;        // p, g on same side
     else if(dp>0.) 
        cross=1;        // p above, g below
     else 
        cross=2;        // p below, g above
  }

  return cross;
}

// Compute signed volume of a tetrahedron
double Crack::Volume(const Point& p1, const Point& p2, 
                     const Point& p3, const Point& p)
{ 
   // p1,p2,p3 -- three corners on bottom, and p -- vertex
   double vol;
   double x1,y1,z1,x2,y2,z2,x3,y3,z3,x,y,z;

   x1=p1.x(); y1=p1.y(); z1=p1.z();
   x2=p2.x(); y2=p2.y(); z2=p2.z();
   x3=p3.x(); y3=p3.y(); z3=p3.z();
   x = p.x(); y = p.y(); z = p.z();

   vol=-(x1-x2)*(y3*z-y*z3)-(x3-x)*(y1*z2-y2*z1)
       +(y1-y2)*(x3*z-x*z3)+(y3-y)*(x1*z2-x2*z1)
       -(z1-z2)*(x3*y-x*y3)-(z3-z)*(x1*y2-x2*y1);

   if(fabs(vol)<1.e-16) 
     return (0.);
   else 
     return(vol);
}
  
IntVector Crack::CellOffset(const Point& p1, const Point& p2, Vector dx)
{
  int nx,ny,nz;
  if(fabs(p1.x()-p2.x())/dx.x()<1e-6) // p1.x()=p2.x()
    nx=NGN-1;
  else
    nx=NGN;
  if(fabs(p1.y()-p2.y())/dx.y()<1e-6) // p1.y()=p2.y()
    ny=NGN-1;
  else
    ny=NGN;
  if(fabs(p1.z()-p2.z())/dx.z()<1e-6) // p1.z()=p2.z()
    nz=NGN-1;
  else
    nz=NGN;

  return IntVector(nx,ny,nz);
}

// Detect if line p3-p4 included in line p1-p2
short Crack::TwoLinesDuplicate(const Point& p1,const Point& p2,
                               const Point& p3,const Point& p4)
{
   double l12,l31,l32,l41,l42;
   double x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4;
   x1=p1.x(); y1=p1.y(); z1=p1.z();
   x2=p2.x(); y2=p2.y(); z2=p2.z();
   x3=p3.x(); y3=p3.y(); z3=p3.z();
   x4=p4.x(); y4=p4.y(); z4=p4.z();

   l12=sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
   l31=sqrt((x3-x1)*(x3-x1)+(y3-y1)*(y3-y1)+(z3-z1)*(z3-z1));
   l32=sqrt((x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)+(z3-z2)*(z3-z2));
   l41=sqrt((x4-x1)*(x4-x1)+(y4-y1)*(y4-y1)+(z4-z1)*(z4-z1));
   l42=sqrt((x4-x2)*(x4-x2)+(y4-y2)*(y4-y2)+(z4-z2)*(z4-z2));

   if(fabs(l31+l32-l12)/l12<1.e-6 && fabs(l41+l42-l12)/l12<1.e-6 && l41>l31)
     return 1;
   else
     return 0;
}

void Crack::FindJPathCircle(const Point& origin, const Vector& v1,
                    const Vector& v2,const Vector& v3, double A[])
{
   /* Find the parameters A0-A13 of J-patch circle with equation
      A0x^2+A1y^2+A2z^2+A3xy+A4xz+A5yz+A6x+A7y+A8z+A9-r^2=0
      and A10x+A11y+A12z+A13=0
      where r is radius of the circle */

   double x0,y0,z0;
   double l1,m1,n1,l2,m2,n2,l3,m3,n3;

   x0=origin.x(); y0=origin.y(); z0=origin.z();

   l1=v1.x(); m1=v1.y(); n1=v1.z();
   l2=v2.x(); m2=v2.y(); n2=v2.z();
   l3=v3.x(); m3=v3.y(); n3=v3.z();

   // parameters of the circle
   double term1,term2;
   term1=l1*x0+m1*y0+n1*z0;
   term2=l2*x0+m2*y0+n2*z0;

   A[0]=l1*l1+l2*l2;
   A[1]=m1*m1+m2*m2;
   A[2]=n1*n1+n2*n2;
   A[3]=2*(l1*m1+l2*m2);
   A[4]=2*(l1*n1+l2*n2);
   A[5]=2*(m1*n1+m2*n2);
   A[6]=-2*(l1*term1+l2*term2);
   A[7]=-2*(m1*term1+m2*term2);
   A[8]=-2*(n1*term1+n2*term2);
   A[9]=A[0]*x0*x0+A[1]*y0*y0+A[2]*z0*z0+A[3]*x0*y0+A[4]*x0*z0+A[5]*y0*z0;

   A[10]=l3;
   A[11]=m3; 
   A[12]=n3;
   A[13]=-(l3*x0+m3*y0+n3*z0);
}

bool Crack::FindIntersectionOfJPathAndCrackPlane(const int& m,
              const double& radius, const double M[],Point& crossPt)
{
   /* Find intersection between J-path circle and crack plane.
          J-patch circle equations:
      Ax^2+By^2+Cz^2+Dxy+Exz+Fyz+Gx+Hy+Iz+J-r^2=0 and a1x+b1y+c1z+d1=0
          crack plane equation:
      a2x+b2y+c2z+d2=0
          m -- material ID, r -- radius of J-path circle. 
         Parameters of J-path circle stroed in array M.
   */
   double A,B,C,D,E,F,G,H,I,J,a1,b1,c1,d1;
   A=M[0];      a1=M[10];
   B=M[1];      b1=M[11];
   C=M[2];      c1=M[12];
   D=M[3];      d1=M[13];
   E=M[4]; 
   F=M[5]; 
   G=M[6]; 
   H=M[7]; 
   I=M[8]; 
   J=M[9];

   int numCross=0;
   crossPt=Point(-9e32,-9e32,-9e32);
   for(int i=0; i<(int)ce[m].size(); i++) {  // Loop over crack segments
     // Find equation of crack segment: a2x+b2y+c2z+d2=0
     double a2,b2,c2,d2;   // parameters of a 3D plane
     Point pt1,pt2,pt3;    // three vertices of the segment
     pt1=cx[m][ce[m][i].x()];
     pt2=cx[m][ce[m][i].y()];
     pt3=cx[m][ce[m][i].z()];
     FindPlaneEquation(pt1,pt2,pt3,a2,b2,c2,d2);

     /* Define crack-segment coordinates (X',Y',Z')
        The origin located at p1, and X'=p1->p2
        v1,v2,v3 -- dirction cosines of new axes X',Y' and Z'
     */  
     Vector v1,v2,v3;
     double term1 = sqrt(a2*a2+b2*b2+c2*c2);
     v2=Vector(a2/term1,b2/term1,c2/term1);
     v1=TwoPtsDirCos(pt1,pt2);
     Vector v12=Cross(v1,v2);
     v3=v12/v12.length();        // right-hand system
     // Transform matrix from global to local
     Matrix3 T=Matrix3(v1.x(),v1.y(),v1.z(),v2.x(),v2.y(),v2.z(),
                       v3.x(),v3.y(),v3.z());

     /* Find intersection between J-path circle and crack plane
        first combine a1x+b1y+c1z+d1=0 And a2x+b2y+c2z+d2=0, get
        x=p1*z+q1 & y=p2*z+q2 (CASE 1) or 
        x=p1*y+q1 & z=p2*y+q2 (CASE 2) or 
        y=p1*x+q1 & z=p2*y+q2 (CASE 3), depending on the equations
        then combine with equation of the circle, getting the intersection
     */
     int CASE=0;
     double delt1,delt2,delt3,p1,q1,p2,q2;
     double abar,bbar,cbar,abc;
     Point crossPt1,crossPt2;
 
     delt1=a1*b2-a2*b1;
     delt2=a1*c2-a2*c1;
     delt3=b1*c2-b2*c1;
     if(fabs(delt1)>=fabs(delt2) && fabs(delt1)>=fabs(delt3)) CASE=1;
     if(fabs(delt2)>=fabs(delt1) && fabs(delt2)>=fabs(delt3)) CASE=2;
     if(fabs(delt3)>=fabs(delt1) && fabs(delt3)>=fabs(delt2)) CASE=3;

     double x1=0.,y1=0.,z1=0.,x2=0.,y2=0.,z2=0.;
     switch(CASE) {
       case 1:
         p1=(b1*c2-b2*c1)/delt1;
         q1=(b1*d2-b2*d1)/delt1;
         p2=(a2*c1-a1*c2)/delt1;
         q2=(a2*d1-a1*d2)/delt1;
         abar=p1*p1*A+p2*p2*B+C+p1*p2*D+p1*E+p2*F;
         bbar=2*p1*q1*A+2*p2*q2*B+(p1*q2+p2*q1)*D+q1*E+q2*F+p1*G+p2*H+I;
         cbar=q1*q1*A+q2*q2*B+q1*q2*D+q1*G+q2*H+J-radius*radius;
         abc=bbar*bbar-4*abar*cbar;
         if(abc<0.0) continue;  // no solution, skip to the next segment
         // the first solution
         z1=0.5*(-bbar+sqrt(abc))/abar;
         x1=p1*z1+q1;
         y1=p2*z1+q2;
         crossPt1=Point(x1,y1,z1);
         // the second solution
         z2=0.5*(-bbar-sqrt(abc))/abar;
         x2=p1*z2+q1;
         y2=p2*z2+q2;
         crossPt2=Point(x2,y2,z2);
         break;
       case 2:
         p1=(b2*c1-b1*c2)/delt2;
         q1=(c1*d2-c2*d1)/delt2;
         p2=(a2*b1-a1*b2)/delt2;
         q2=(a2*d1-a1*d2)/delt2;
         abar=p1*p1*A+B+p2*p2*C+p1*D+p1*p2*E+p2*F;
         bbar=2*p1*q1*A+2*p2*q2*C+q1*D+(p1*q2+p2*q1)*E+q2*F+p1*G+H+p2*I;
         cbar=q1*q1*A+q2*q2*C+q1*q2*E+q1*G+q2*I+J-radius*radius;
         abc=bbar*bbar-4*abar*cbar;
         if(abc<0.0) continue;  // no solution, skip to the next segment
         // the first solution
         y1=0.5*(-bbar+sqrt(abc))/abar;
         x1=p1*y1+q1;
         z1=p2*y1+q2;
         crossPt1=Point(x1,y1,z1);
         //the second solution
         y2=0.5*(-bbar-sqrt(abc))/abar;
         x2=p1*y2+q1;
         z2=p2*y2+q2;
         crossPt2=Point(x2,y2,z2);
         break;
       case 3:
         p1=(a2*c1-a1*c2)/delt3;
         q1=(c1*d2-c2*d1)/delt3;
         p2=(a1*b2-a2*b1)/delt3;
         q2=(b2*d1-b1*d2)/delt3;
         abar=A+p1*p1*B+p2*p2*C+p1*D+p2*E+p1*p2*F;
         bbar=2*p1*q1*B+2*p2*q2*C+q1*D+q2*E+(p1*q2+p2*q1)*F+G+p1*H+p2*I;
         cbar=q1*q1*B+q2*q2*C+q1*q2*F+q1*H+q2*I+J-radius*radius;
         abc=bbar*bbar-4*abar*cbar;
         if(abc<0.0) continue;  // no solution, skip to the next segment
         // the first solution
         x1=0.5*(-bbar+sqrt(abc))/abar;
         y1=p1*x1+q1;
         z1=p2*x1+q2;
         crossPt1=Point(x1,y1,z1);
         // the second solution
         x2=0.5*(-bbar-sqrt(abc))/abar;
         y2=p1*x2+q1;
         z2=p2*x2+q2;
         crossPt2=Point(x2,y2,z2);
         break;
     }

     /* Detect if crossPt1 & crossPt2 in the triangular segment.
        Transform and rotate the coordinates of crossPt1 and crossPt2 into
        crack-segment coordinates (X', Y' and Z')
     */
     Point p1p,p2p,p3p,crossPt1p,crossPt2p;
     p1p     =Point(0.,0.,0.)+T*(pt1-pt1);
     p2p     =Point(0.,0.,0.)+T*(pt2-pt1);
     p3p     =Point(0.,0.,0.)+T*(pt3-pt1);
     crossPt1p=Point(0.,0.,0.)+T*(crossPt1-pt1);
     crossPt2p=Point(0.,0.,0.)+T*(crossPt2-pt1);
     if(PointInTriangle(crossPt1p,p1p,p2p,p3p)) {
       numCross++;
       crossPt=crossPt1;
     }
     if(PointInTriangle(crossPt2p,p1p,p2p,p3p)) {
       numCross++;
       crossPt=crossPt2;
     }
   } // End of loop over crack segments

   if(numCross==0) 
     return 0;
   else 
     return 1;
}

// Direction cosines of a line from p1 to p2
Vector Crack::TwoPtsDirCos(const Point& p1,const Point& p2)
{
  double dx,dy,dz,ds;
  dx=p2.x()-p1.x(); 
  dy=p2.y()-p1.y(); 
  dz=p2.z()-p1.z();
  ds=sqrt(dx*dx+dy*dy+dz*dz);
  return Vector(dx/ds, dy/ds, dz/ds);   
}

// Find the equation of a plane with three points on it
void Crack::FindPlaneEquation(const Point& p1,const Point& p2, 
            const Point& p3, double& a,double& b,double& c,double& d)
{
  // Determine parameters a,b,c,d for plane equation ax+by+cz+d=0
  double x21,x31,y21,y31,z21,z31;
  
  x21=p2.x()-p1.x();
  y21=p2.y()-p1.y();
  z21=p2.z()-p1.z();

  x31=p3.x()-p1.x();
  y31=p3.y()-p1.y();
  z31=p3.z()-p1.z();

  a=y21*z31-z21*y31;
  b=x31*z21-z31*x21;
  c=x21*y31-y21*x31;
  d=-p1.x()*a-p1.y()*b-p1.z()*c;
}   

// Detect if a point is within a triangle (2D case)
short Crack::PointInTriangle(const Point& p,const Point& pt1,
                           const Point& pt2,const Point& pt3)  
{
  // y=0 for all points

  double x1,z1,x2,z2,x3,z3,x,z;
  double area_p1p2p,area_p2p3p,area_p3p1p,area_p123;

  x1=pt1.x(); z1=pt1.z();
  x2=pt2.x(); z2=pt2.z();
  x3=pt3.x(); z3=pt3.z();
  x =p.x();   z =p.z();

  area_p1p2p=x1*z2+x2*z+x*z1-x1*z-x2*z1-x*z2;
  area_p2p3p=x2*z3+x3*z+x*z2-x2*z-x3*z2-x*z3;
  area_p3p1p=x3*z1+x1*z+x*z3-x3*z-x1*z3-x*z1;

  area_p123=fabs(x1*z2+x2*z3+x3*z1-x1*z3-x2*z1-x3*z2);

  if(fabs(area_p1p2p)/area_p123<1.e-3) area_p1p2p=0.;
  if(fabs(area_p2p3p)/area_p123<1.e-3) area_p2p3p=0.;
  if(fabs(area_p3p1p)/area_p123<1.e-3) area_p3p1p=0.;

  return (area_p1p2p<=0. && area_p2p3p<=0. && area_p3p1p<=0.);
}

// Detect if doing fracture analysis
void Crack::FindTimeStepForFractureAnalysis(double time)
{
  static double timeforcalculateJK=0.0;
  static double timeforpropagation=0.0;

  if(d_calFractParameters=="true") {
    if(time>=timeforcalculateJK) {
      calFractParameters=1;
      timeforcalculateJK+=d_outputInterval;
    }
    else {
     calFractParameters=0;
    }
  }
  else if(d_calFractParameters=="false") {
   calFractParameters=0;
  }
  else if(d_calFractParameters=="every_time_step"){
    calFractParameters=1;
  }

  double propagationInterval=0.0;
  if(d_doCrackPropagation=="true") {
    if(time>=timeforpropagation){
      doCrackPropagation=1;
      timeforpropagation+=propagationInterval;
    }
    else {
      doCrackPropagation=0;
    }
  }
  else if(d_doCrackPropagation=="false") {
    doCrackPropagation=0;
  }
  else if(d_doCrackPropagation=="every_time_step"){
    doCrackPropagation=1;
  }
}

void Crack::ApplyBCsForCrackPoints(const Vector& cs,
                        const Point& old_pt,Point& new_pt)
{
  // cs -- cell size
  for(Patch::FaceType face = Patch::startFace;
       face<=Patch::endFace; face=Patch::nextFace(face)) {
    if(GridBCType[face]=="symmetry") {
      if( face==Patch::xminus && fabs(old_pt.x()-GLP.x())/cs.x()<1.e-2 )
        new_pt(0)=GLP.x(); // On symmetric face x-
      if( face==Patch::xplus  && fabs(old_pt.x()-GHP.x())/cs.x()<1.e-2 )
        new_pt(0)=GHP.x(); // On symmetric face x+
      if( face==Patch::yminus && fabs(old_pt.y()-GLP.y())/cs.y()<1.e-2 )
        new_pt(1)=GLP.y(); // On symmetric face y-
      if( face==Patch::yplus  && fabs(old_pt.y()-GHP.y())/cs.y()<1.e-2 )
        new_pt(1)=GHP.y(); // On symmetric face y+
      if( face==Patch::zminus && fabs(old_pt.z()-GLP.z())/cs.z()<1.e-2 )
        new_pt(2)=GLP.z(); // On symmetric face z-
      if( face==Patch::zplus  && fabs(old_pt.z()-GHP.z())/cs.z()<1.e-62 ) 
        new_pt(2)=GHP.z(); // On symmetric face z+
    }
  }
}

void Crack::ReadRectangularCracks(const int& m,const ProblemSpecP& geom_ps)
{
  for(ProblemSpecP quad_ps=geom_ps->findBlock("quadrilateral");
       quad_ps!=0; quad_ps=quad_ps->findNextBlock("quadrilateral")) {
    int n12=1,n23=1;
    Point p;
    vector<Point> thisRectPts;
    vector<short> thisRectCrackSidesAtFront;

    // Four vertices of the quadrilateral
    quad_ps->require("p1",p);
    thisRectPts.push_back(p);
    quad_ps->require("p2",p);
    thisRectPts.push_back(p);
    quad_ps->require("p3",p);
    thisRectPts.push_back(p);
    quad_ps->require("p4",p);
    thisRectPts.push_back(p);
    rectangles[m].push_back(thisRectPts);
    thisRectPts.clear();

    // Mesh resolution
    quad_ps->get("resolution_p1_p2",n12);
    rectN12[m].push_back(n12);
    quad_ps->get("resolution_p2_p3",n23);
    rectN23[m].push_back(n23);
    // Crack front
    short Side;
    string cfsides;
    quad_ps->get("crack_front_sides",cfsides);
    if(cfsides.length()==4) {
      for(string::const_iterator iter=cfsides.begin();
                        iter!=cfsides.end(); iter++) {
        if(*iter=='Y' || *iter=='y')      Side=1;
        else if(*iter=='N' || *iter=='n') Side=0;
        else {
          cout << " Wrong specification for crack_front_sides." << endl;
          exit(1);
        }
        thisRectCrackSidesAtFront.push_back(Side);
      }
    }
    else if(cfsides.length()==0) {
      thisRectCrackSidesAtFront.push_back(0);
      thisRectCrackSidesAtFront.push_back(0);
      thisRectCrackSidesAtFront.push_back(0);
      thisRectCrackSidesAtFront.push_back(0);
    }
    else {
      cout << " The length of string crack_front_sides for "
           << "quadrilaterals should be 4." << endl;
      exit(1);
    }
    rectCrackSidesAtFront[m].push_back(thisRectCrackSidesAtFront);
    thisRectCrackSidesAtFront.clear();
  } // End of quadrilateral
}

void Crack::ReadTriangularCracks(const int& m,const ProblemSpecP& geom_ps)
{
  for(ProblemSpecP tri_ps=geom_ps->findBlock("triangle");
       tri_ps!=0; tri_ps=tri_ps->findNextBlock("triangle")) {
    int n=1;
    Point p;
    vector<Point> thisTriPts;
    vector<short> thisTriCrackSidesAtFront;

    // Three vertices of the triangle
    tri_ps->require("p1",p);
    thisTriPts.push_back(p);
    tri_ps->require("p2",p);
    thisTriPts.push_back(p);
    tri_ps->require("p3",p);
    thisTriPts.push_back(p);
    triangles[m].push_back(thisTriPts);
    thisTriPts.clear();
    tri_ps->get("resolution",n);
    triNCells[m].push_back(n);

    // Crack front
    short Side;
    string cfsides;
    tri_ps->get("crack_front_sides",cfsides);
    if(cfsides.length()==3) {
      for(string::const_iterator iter=cfsides.begin();
                        iter!=cfsides.end(); iter++) {
        if( *iter=='Y' || *iter=='n')     Side=1;
        else if(*iter=='N' || *iter=='n') Side=0;
        else {
          cout << " Wrong specification for crack_front_sides." << endl;
          exit(1);
        }
        thisTriCrackSidesAtFront.push_back(Side);
      }
    }
    else if(cfsides.length()==0) {
      thisTriCrackSidesAtFront.push_back(0);
      thisTriCrackSidesAtFront.push_back(0);
      thisTriCrackSidesAtFront.push_back(0);
    }
    else {
      cout << " The length of string crack_front_sides for"
           << " triangles should be 3." << endl;
      exit(1);
    }
    triCrackSidesAtFront[m].push_back(thisTriCrackSidesAtFront);
    thisTriCrackSidesAtFront.clear();
  } // End of triangles
}

void Crack::ReadArcCracks(const int& m,const ProblemSpecP& geom_ps)
{
  for(ProblemSpecP arc_ps=geom_ps->findBlock("arc");
       arc_ps!=0; arc_ps=arc_ps->findNextBlock("arc")) {
    int n;
    Point p;
    int cfsID=9999;   // All segments by default
    vector<Point> thisArcPts;

    // Three points on the arc
    arc_ps->require("start_point",p);
    thisArcPts.push_back(p);
    arc_ps->require("middle_point",p);
    thisArcPts.push_back(p);
    arc_ps->require("end_point",p);
    thisArcPts.push_back(p);

    // Resolution on circumference
    arc_ps->require("resolution_circumference",n);
    arcNCells[m].push_back(n);
    arcs[m].push_back(thisArcPts);
    thisArcPts.clear();

    // Crack front segment ID
    arc_ps->get("crack_front_segment_ID",cfsID);
    arcCrkFrtSegID[m].push_back(cfsID);
  } // End of arc
}

void Crack::ReadEllipticCracks(const int& m,const ProblemSpecP& geom_ps)
{
  for(ProblemSpecP ellipse_ps=geom_ps->findBlock("ellipse");
      ellipse_ps!=0; ellipse_ps=ellipse_ps->findNextBlock("ellipse")) {
    int n;
    Point p;
    int cfsID=9999;  // All segments by default
    vector<Point> thisEllipsePts;

    // Three points on the arc
    ellipse_ps->require("point1_axis1",p);
    thisEllipsePts.push_back(p);
    ellipse_ps->require("point_axis2",p);
    thisEllipsePts.push_back(p);
    ellipse_ps->require("point2_axis1",p);
    thisEllipsePts.push_back(p);

    // Resolution on circumference
    ellipse_ps->require("resolution_circumference",n);
    ellipseNCells[m].push_back(n);
    ellipses[m].push_back(thisEllipsePts);
    thisEllipsePts.clear();

    // Crack front segment ID
    ellipse_ps->get("crack_front_segment_ID",cfsID);
    ellipseCrkFrtSegID[m].push_back(cfsID);
  } // End of ellipses
}

void Crack::ReadPartialEllipticCracks(const int& m,
                        const ProblemSpecP& geom_ps)
{
  for(ProblemSpecP pellipse_ps=geom_ps->findBlock("partial_ellipse");
      pellipse_ps!=0; pellipse_ps=
      pellipse_ps->findNextBlock("partial_ellipse")) {
    int n;
    Point p;
    string Extent;
    int cfsID=9999;  // All segments by default
    vector<Point> thispEllipsePts;

    // Center,two points on major and minor axes
    pellipse_ps->require("center",p);
    thispEllipsePts.push_back(p);
    pellipse_ps->require("point_axis1",p);
    thispEllipsePts.push_back(p);
    pellipse_ps->require("point_axis2",p);
    thispEllipsePts.push_back(p);
    pellipses[m].push_back(thispEllipsePts);
    thispEllipsePts.clear();

    // Extent of the partial ellipse (quarter or half)
    pellipse_ps->require("extent",Extent);
    pellipseExtent[m].push_back(Extent);

    // Resolution on circumference
    pellipse_ps->require("resolution_circumference",n);
    pellipseNCells[m].push_back(n);

    // Crack front segment ID
    pellipse_ps->get("crack_front_segment_ID",cfsID);
    pellipseCrkFrtSegID[m].push_back(cfsID);
  } // End of ellipses
}

void Crack::OutputInitialCracks(const int& numMatls)
{
  int pid;
  MPI_Comm_rank(mpi_crack_comm, &pid);
  if(pid==0) { //output from the first rank
    cout << "*** Crack information output from rank "
         << pid << " ***" << endl;
    for(int m=0; m<numMatls; m++) {
      if(crackType[m]=="NO_CRACK")
        cout << "\nMaterial " << m << ": no crack exists" << endl;
      else {
        cout << "\nMaterial " << m << ":\n" 
             << "   Crack contact type: " << crackType[m] << endl;
        if(crackType[m]=="frictional")
          cout << "   Frictional coefficient: " << cmu[m] << endl;

        if(crackType[m]!="null") {
          if(separateVol[m]<0. || contactVol[m]<0.)
            cout  << "   Check crack contact by displacement criterion" << endl;
          else {
            cout  << "Check crack contact by volume criterion with\n"
                  << "            separate volume = " << separateVol[m]
                  << "\n            contact volume = " << contactVol[m] << endl;
          }
        }
 
        cout <<"\nCrack geometry:" << endl;
        // Triangular cracks
        for(int i=0;i<(int)rectangles[m].size();i++) {
          cout << "Rectangle " << i << ": meshed by [" << rectN12[m][i]
               << ", " << rectN23[m][i] << ", " << rectN12[m][i]
               << ", " << rectN23[m][i] << "]" << endl;
          for(int j=0;j<4;j++)
            cout << "   pt " << j+1 << ": " << rectangles[m][i][j] << endl;
          for(int j=0;j<4;j++) {
            if(rectCrackSidesAtFront[m][i][j]) {
              int j2=(j+2<5 ? j+2 : 1);
              cout << "   side " << j+1 << " (p" << j+1 << "-" << "p" << j2
                   << ") is a crack front." << endl;
            }
          }
        }

        // Triangular cracks
        for(int i=0;i<(int)triangles[m].size();i++) {
          cout << "Triangle " << i << ": meshed by [" << triNCells[m][i]
               << ", " << triNCells[m][i] << ", " << triNCells[m][i] 
               << "]" << endl;
          for(int j=0;j<3;j++)
            cout << "   pt " << j+1 << ": " << triangles[m][i][j] << endl;
          for(int j=0;j<3;j++) {
            if(triCrackSidesAtFront[m][i][j]) {
              int j2=(j+2<4 ? j+2 : 1);
              cout << "   side " << j+1 << " (p" << j+1 << "-" << "p" << j2
                   << ") is a crack front." << endl;
            }
          }
        }

        // Arc cracks
        for(int i=0;i<(int)arcs[m].size();i++) {
          cout << "Arc " << i << ": meshed by " << arcNCells[m][i]
               << " cells on the circumference.\n"
               << "   crack front segment ID: " << arcCrkFrtSegID[m][i]
               << "\n   start, middle and end points of the arc:"  << endl;
          for(int j=0;j<3;j++)
            cout << "   pt " << j+1 << ": " << arcs[m][i][j] << endl;
        }

        // Elliptic cracks
        for(int i=0;i<(int)ellipses[m].size();i++) {
          cout << "Ellipse " << i << ": meshed by " << ellipseNCells[m][i]
               << " cells on the circumference.\n"
               << "   crack front segment ID: " << ellipseCrkFrtSegID[m][i]
               << endl;
          cout << "   end point on axis1: " << ellipses[m][i][0] << endl;
          cout << "   end point on axis2: " << ellipses[m][i][1] << endl;
          cout << "   another end point on axis1: " << ellipses[m][i][2]
               << endl;
        }

        // Partial elliptic cracks
        for(int i=0;i<(int)pellipses[m].size();i++) {
          cout << "Partial ellipse " << i << " (" << pellipseExtent[m][i]
               << "): meshed by " << pellipseNCells[m][i]
               << " cells on the circumference.\n"
               << "   crack front segment ID: " << pellipseCrkFrtSegID[m][i]
               << endl;
          cout << "   center: " << pellipses[m][i][0] << endl;
          cout << "   end point on axis1: " << pellipses[m][i][1] << endl;
          cout << "   end point on axis2: " << pellipses[m][i][2] << endl;
        }
      } // End of if(crackType...)
    } // End of loop over materials
  }
}

void Crack::DiscretizeRectangularCracks(const int& m,int& nstart0)
{
  int k,i,j,ni,nj,n1,n2,n3;
  int nstart1,nstart2,nstart3;
  Point p1,p2,p3,p4,pt;
  Vector norm;

  for(k=0; k<(int)rectangles[m].size(); k++) {  // Loop over quadrilaterals
    // Resolutions for the quadrilateral
    ni=rectN12[m][k];
    nj=rectN23[m][k];
    // Four vertices for the quadrilateral
    p1=rectangles[m][k][0];
    p2=rectangles[m][k][1];
    p3=rectangles[m][k][2];
    p4=rectangles[m][k][3];

    // Nodes on sides p2-p3 and p1-p4
    Point* side23=new Point[2*nj+1];
    Point* side14=new Point[2*nj+1];
    for(j=0; j<=2*nj; j++) {
      side23[j]=p2+(p3-p2)*(float)j/(2*nj);
      side14[j]=p1+(p4-p1)*(float)j/(2*nj);
    }

    // Generate crack points
    for(j=0; j<=nj; j++) {
      for(i=0; i<=ni; i++) {
        pt=side14[2*j]+(side23[2*j]-side14[2*j])*(float)i/ni;
        cx[m].push_back(pt);
      }
      if(j!=nj) {
        for(i=0; i<ni; i++) {
          int jj=2*j+1;
          pt=side14[jj]+(side23[jj]-side14[jj])*(float)(2*i+1)/(2*ni);
          cx[m].push_back(pt);
        }
      }  // End of if j!=nj
    } // End of loop over j

    // Create elements and get normals for quadrilaterals
    for(j=0; j<nj; j++) {
      nstart1=nstart0+(2*ni+1)*j;
      nstart2=nstart1+(ni+1);
      nstart3=nstart2+ni;
      for(i=0; i<ni; i++) {
        /* There are four elements in each sub-rectangle */
        // For the 1st element (n1,n2,n3 three nodes of the element)
        n1=nstart2+i;
        n2=nstart1+i;
        n3=nstart1+(i+1);
        ce[m].push_back(IntVector(n1,n2,n3));
        // For the 2nd element
        n1=nstart2+i;
        n2=nstart3+i;
        n3=nstart1+i;
        ce[m].push_back(IntVector(n1,n2,n3));
        // For the 3rd element
        n1=nstart2+i;
        n2=nstart1+(i+1);
        n3=nstart3+(i+1);
        ce[m].push_back(IntVector(n1,n2,n3));
        // For the 4th element
        n1=nstart2+i;
        n2=nstart3+(i+1);
        n3=nstart3+i;
        ce[m].push_back(IntVector(n1,n2,n3));
      }  // End of loop over i
    }  // End of loop over j
    nstart0+=((2*ni+1)*nj+ni+1);
    delete [] side14;
    delete [] side23;

    // Collect crack-front segments
    int seg0=0;
    for(j=0; j<4; j++) {
      if(!rectCrackSidesAtFront[m][k][j]) {
        seg0=j+1;
        break;
      }
    }
    for(int l=0; l<4; l++) { // Loop over sides of the quad
      j=seg0+l;
      if(j>3) j-=4;
      if(rectCrackSidesAtFront[m][k][j]) {
        int j1 = (j!=3 ? j+1 : 0);
        Point pt1=rectangles[m][k][j];
        Point pt2=rectangles[m][k][j1];  
        for(i=0; i<(int)ce[m].size(); i++) {
          int ii=i;
          if(j>1) ii=ce[m].size()-(i+1);
          n1=ce[m][ii].x();
          n2=ce[m][ii].y();
          n3=ce[m][ii].z();
          for(int s=0; s<3; s++) { // Loop over sides of the elem
            int sn=n1,en=n2;
            if(s==1) {sn=n2; en=n3;}
            if(s==2) {sn=n3; en=n1;}
            if(TwoLinesDuplicate(pt1,pt2,cx[m][sn],cx[m][en])) {
              cfSegNodes[m].push_back(sn);
              cfSegNodes[m].push_back(en);
              norm=TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]);
              cfSegV2[m].push_back(norm);
            }
          }
        } // End of loop over i
      }      
    } // End of loop over j
  } // End of loop over quadrilaterals
}

void Crack::DiscretizeTriangularCracks(const int&m, int& nstart0)
{
  int neq=1; 
  int k,i,j;
  int nstart1,nstart2,n1,n2,n3;
  Point p1,p2,p3,pt;
  Vector norm;

  for(k=0; k<(int)triangles[m].size(); k++) {  // Loop over all triangles
    // Three vertices of the triangle
    p1=triangles[m][k][0];
    p2=triangles[m][k][1];
    p3=triangles[m][k][2];

    // Mesh resolution of the triangle
    neq=triNCells[m][k];

    // Create temprary arraies
    Point* side12=new Point[neq+1];
    Point* side13=new Point[neq+1];

    // Generate node coordinates
    for(j=0; j<=neq; j++) {
      side12[j]=p1+(p2-p1)*(float)j/neq;
      side13[j]=p1+(p3-p1)*(float)j/neq;
    }
    for(j=0; j<=neq; j++) {
      for(i=0; i<=j; i++) {
        double w=0.0;
        if(j!=0) w=(float)i/j;
        pt=side12[j]+(side13[j]-side12[j])*w;
        cx[m].push_back(pt);
      } // End of loop over i
    } // End of loop over j

    // Generate elements and their normals
    for(j=0; j<neq; j++) {
      nstart1=nstart0+j*(j+1)/2;
      nstart2=nstart0+(j+1)*(j+2)/2;
      for(i=0; i<j; i++) {
        // Left element
        n1=nstart1+i;
        n2=nstart2+i;
        n3=nstart2+(i+1);
        ce[m].push_back(IntVector(n1,n2,n3));
        // Right element
        n1=nstart1+i;
        n2=nstart2+(i+1);
        n3=nstart1+(i+1);
        ce[m].push_back(IntVector(n1,n2,n3));
      } // End of loop over i
      n1=nstart0+(j+1)*(j+2)/2-1;
      n2=nstart0+(j+2)*(j+3)/2-2;
      n3=nstart0+(j+2)*(j+3)/2-1;
      ce[m].push_back(IntVector(n1,n2,n3));
    } // End of loop over j
    // Add number of nodes in this trianglular segment
    nstart0+=(neq+1)*(neq+2)/2;
    delete [] side12;
    delete [] side13;

    // Collect crack-front segments
    int seg0=0;
    for(j=0; j<3; j++) {
      if(!triCrackSidesAtFront[m][k][j]) {
        seg0=j+1;
        break;
      }
    }
    for(int l=0; l<3; l++) { // Loop over sides of the triangle
      j=seg0+l;
      if(j>2) j-=3;
      if(triCrackSidesAtFront[m][k][j]) {
        int j1 = (j!=2 ? j+1 : 0);
        Point pt1=triangles[m][k][j];
        Point pt2=triangles[m][k][j1];
        for(i=0; i<(int)ce[m].size(); i++) {
          int ii=i;
          if(j>1) ii=ce[m].size()-(i+1);
          n1=ce[m][ii].x();
          n2=ce[m][ii].y();
          n3=ce[m][ii].z();
          for(int s=0; s<3; s++) { // Loop over sides of the elem
            int sn=n1,en=n2;
            if(s==1) {sn=n2; en=n3;}
            if(s==2) {sn=n3; en=n1;}
            if(TwoLinesDuplicate(pt1,pt2,cx[m][sn],cx[m][en])) {
              cfSegNodes[m].push_back(sn);
              cfSegNodes[m].push_back(en);
              norm=TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]);
              cfSegV2[m].push_back(norm);
            }
          }
        } // End of loop over i
      }
    } // End of loop over l
  } // End of loop over triangles
}

void Crack::DiscretizeArcCracks(const int& m, int& nstart0)
{
  for(int k=0; k<(int)arcs[m].size(); k++) {  // Loop over all arcs
    // Three points of the arc
    Point p1=arcs[m][k][0];
    Point p2=arcs[m][k][1];
    Point p3=arcs[m][k][2];
    double x1,y1,z1,x2,y2,z2,x3,y3,z3;
    x1=p1.x(); y1=p1.y(); z1=p1.z();
    x2=p2.x(); y2=p2.y(); z2=p2.z();
    x3=p3.x(); y3=p3.y(); z3=p3.z();

    // Find center of the arc
    double a1,b1,c1,d1,a2,b2,c2,d2,a3,b3,c3,d3;
    a1=2*(x2-x1); b1=2*(y2-y1); c1=2*(z2-z1);
    d1=x1*x1-x2*x2+y1*y1-y2*y2+z1*z1-z2*z2;
    a2=2*(x3-x1); b2=2*(y3-y1); c2=2*(z3-z1);
    d2=x1*x1-x3*x3+y1*y1-y3*y3+z1*z1-z3*z3;
    FindPlaneEquation(p1,p2,p3,a3,b3,c3,d3);

    double delt,deltx,delty,deltz;
    delt  = Matrix3(a1,b1,c1,a2,b2,c2,a3,b3,c3).Determinant();
    deltx = Matrix3(-d1,b1,c1,-d2,b2,c2,-d3,b3,c3).Determinant();
    delty = Matrix3(a1,-d1,c1,a2,-d2,c2,a3,-d3,c3).Determinant();
    deltz = Matrix3(a1,b1,-d1,a2,b2,-d2,a3,b3,-d3).Determinant();
    double x0,y0,z0;
    x0=deltx/delt;  y0=delty/delt;  z0=deltz/delt;
    Point origin=Point(x0,y0,z0);
    double radius=sqrt((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)+(z1-z0)*(z1-z0));

    // Define local coordinates
    Vector v1,v2,v3;
    double temp=sqrt(a3*a3+b3*b3+c3*c3);
    v3=Vector(a3/temp,b3/temp,c3/temp);
    v1=TwoPtsDirCos(origin,p1);
    Vector v31=Cross(v3,v1);
    v2=v31/v31.length();
    double lx,mx,nx,ly,my,ny;
    lx=v1.x();  mx=v1.y();  nx=v1.z();
    ly=v2.x();  my=v2.y();  ny=v2.z();

    // Angle of the arc
    double angleOfArc;
    double PI=3.141592654;
    double x3prime,y3prime;
    x3prime=lx*(x3-x0)+mx*(y3-y0)+nx*(z3-z0);
    y3prime=ly*(x3-x0)+my*(y3-y0)+ny*(z3-z0);
    double cosTheta=x3prime/radius;
    double sinTheta=y3prime/radius;
    double thetaQ=fabs(asin(y3prime/radius));
    if(sinTheta>=0.) {
      if(cosTheta>=0) angleOfArc=thetaQ;
      else angleOfArc=PI-thetaQ;
    }
    else {
      if(cosTheta<=0.) angleOfArc=PI+thetaQ;
      else angleOfArc=2*PI-thetaQ;
    }

    // Node points
    cx[m].push_back(origin);
    for(int j=0;j<=arcNCells[m][k];j++) {  // Loop over points
      double thetai=angleOfArc*j/arcNCells[m][k];
      double xiprime=radius*cos(thetai);
      double yiprime=radius*sin(thetai);
      double xi=lx*xiprime+ly*yiprime+x0;
      double yi=mx*xiprime+my*yiprime+y0;
      double zi=nx*xiprime+ny*yiprime+z0;
      cx[m].push_back(Point(xi,yi,zi));
    } // End of loop over points

    // Crack elements
    for(int j=1;j<=arcNCells[m][k];j++) {  // Loop over segs
      int n1=nstart0;
      int n2=nstart0+j;
      int n3=nstart0+(j+1);
      ce[m].push_back(IntVector(n1,n2,n3));
      // Crack front segments
      if(arcCrkFrtSegID[m][k]==9999 || arcCrkFrtSegID[m][k]==j) {
        cfSegNodes[m].push_back(n2);
        cfSegNodes[m].push_back(n3);
        Vector thisSegNorm=TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]);
        cfSegV2[m].push_back(thisSegNorm);
      }
    }
    nstart0+=arcNCells[m][k]+2;
  } // End of loop over arcs
}

void Crack::DiscretizeEllipticCracks(const int& m, int& nstart0)
{
  for(int k=0; k<(int)ellipses[m].size(); k++) {
    // Three points of the ellipse
    Point p1=ellipses[m][k][0];
    Point p2=ellipses[m][k][1];
    Point p3=ellipses[m][k][2];
    // Center and half axial lengths of the ellipse
    double x0,y0,z0,a,b;
    Point origin=p3+(p1-p3)*0.5;
    x0=origin.x();
    y0=origin.y();
    z0=origin.z();
    a=(p1-origin).length();
    b=(p2-origin).length();
    // Local coordinates
    Vector v1,v2,v3;
    v1=TwoPtsDirCos(origin,p1);
    v2=TwoPtsDirCos(origin,p2);
    Vector v12=Cross(v1,v2);
    v3=v12/v12.length();
    double lx,mx,nx,ly,my,ny;
    lx=v1.x();  mx=v1.y();  nx=v1.z();
    ly=v2.x();  my=v2.y();  ny=v2.z();
     // Crack nodes
    cx[m].push_back(origin);
    for(int j=0;j<ellipseNCells[m][k];j++) {  // Loop over points
      double PI=3.141592654;
      double thetai=j*(2*PI)/ellipseNCells[m][k];
      double xiprime=a*cos(thetai);
      double yiprime=b*sin(thetai);
      double xi=lx*xiprime+ly*yiprime+x0;
      double yi=mx*xiprime+my*yiprime+y0;
      double zi=nx*xiprime+ny*yiprime+z0;
      cx[m].push_back(Point(xi,yi,zi));
    } // End of loop over points
    // Crack elements
    for(int j=1;j<=ellipseNCells[m][k];j++) {  // Loop over segs
      int j1 = (j==ellipseNCells[m][k]? 1 : j+1);
      int n1=nstart0;
      int n2=nstart0+j;
      int n3=nstart0+j1;
      ce[m].push_back(IntVector(n1,n2,n3));
      // Crack front segments
      if(ellipseCrkFrtSegID[m][k]==9999 || ellipseCrkFrtSegID[m][k]==j) {
        cfSegNodes[m].push_back(n2);
        cfSegNodes[m].push_back(n3);
        Vector thisSegNorm=TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]);
        cfSegV2[m].push_back(thisSegNorm);
      }
    }
    nstart0+=ellipseNCells[m][k]+1;
  } // End ofloop over ellipses
}
 
void Crack::DiscretizePartialEllipticCracks(const int& m, int& nstart0)
{
  for(int k=0; k<(int)pellipses[m].size(); k++) {
    double extent=0.0;
    if(pellipseExtent[m][k]=="quarter") extent=0.25;
    else if(pellipseExtent[m][k]=="half") extent=0.5;

    // Center, end points on major and minor axes
    Point origin=pellipses[m][k][0];
    Point major_p=pellipses[m][k][1];
    Point minor_p=pellipses[m][k][2];
    double x0,y0,z0,a,b;
    x0=origin.x();
    y0=origin.y();
    z0=origin.z();
    a=(major_p-origin).length();
    b=(minor_p-origin).length();
    // Local coordinates
    Vector v1,v2,v3;
    v1=TwoPtsDirCos(origin,major_p);
    v2=TwoPtsDirCos(origin,minor_p);
    Vector v12=Cross(v1,v2);
    v3=v12/v12.length();
    double lx,mx,nx,ly,my,ny;
    lx=v1.x();  mx=v1.y();  nx=v1.z();
    ly=v2.x();  my=v2.y();  ny=v2.z();

    // Crack nodes
    cx[m].push_back(origin);
    for(int j=0;j<=pellipseNCells[m][k];j++) {  // Loop over points
      double PI=3.141592654;
      double thetai=j*(2*PI*extent)/pellipseNCells[m][k];
      double xiprime=a*cos(thetai);
      double yiprime=b*sin(thetai);
      double xi=lx*xiprime+ly*yiprime+x0;
      double yi=mx*xiprime+my*yiprime+y0;
      double zi=nx*xiprime+ny*yiprime+z0;
      cx[m].push_back(Point(xi,yi,zi));
    } // End of loop over points

    // Crack elements
    for(int j=1;j<=pellipseNCells[m][k];j++) {  // Loop over segs
      int n1=nstart0;
      int n2=nstart0+j;
      int n3=nstart0+j+1;
      ce[m].push_back(IntVector(n1,n2,n3));
      // Crack front segments
      if(pellipseCrkFrtSegID[m][k]==9999 || pellipseCrkFrtSegID[m][k]==j) {
        cfSegNodes[m].push_back(n2);
        cfSegNodes[m].push_back(n3);
        Vector thisSegNorm=TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]);
        cfSegV2[m].push_back(thisSegNorm);
      }
    }
    nstart0+=pellipseNCells[m][k]+2;
  } // End of discretizing partial ellipses
}

void Crack::OutputCrackPlaneMesh(const int& m)
{
  int pid;
  MPI_Comm_rank(mpi_crack_comm, &pid);
  if(pid==0) { // Output from the first rank
    cout << "\n*** Crack mesh information output from rank "
         << pid << " ***" << endl;
    cout << "MatID: " << m << endl;
    cout << "  Number of crack elems: " << (int)ce[m].size()
         << "\n  Number of crack nodes: " << (int)cx[m].size()
         << "\n  Number of crack-front elems: "
         << (int)cfSegNodes[m].size()/2 << endl;

    cout << "  Element nodes and normals (" << (int)ce[m].size()
         << " elems in total):" << endl;
    for(int i=0; i<(int)ce[m].size(); i++) {
      cout << "     Elem " << i << ": " << ce[m][i] << endl;
    }

    cout << "  Crack nodes (" << (int)cx[m].size()
         << " nodes in total):" << endl;
    for(int i=0; i<(int)cx[m].size(); i++) {
      cout << "     Node " << i << ": " << cx[m][i] << endl;
    }

    cout << "  Crack front elems (" << (int)cfSegNodes[m].size()/2
         << " elems in total)" << endl;
    for(int i=0; i<(int)cfSegNodes[m].size();i++) {
      cout << "     Seg " << i/2 << ": "
           << cfSegNodes[m][i] << cx[m][cfSegNodes[m][i]] 
           << ", V2: " << cfSegV2[m][i/2] 
           << ", V3: " << cfSegV3[m][i] << endl;
    }

    cout << "\n  Average length of crack front segs, cs0[m]="
         << cs0[m] << endl;

    cout << "\n  Crack extent: " << cmin[m] << "-->"
         <<  cmax[m] << endl << endl;
  }
}
 
short Crack::SmoothCrackFrontAndGetTangentialVector(const int& mm)
{ 
  short flag=1;       // Smooth successfully
  double ep=1.e-10;   // Tolerance
  enum {R=0,L};       // Right and left

  int i=-1,l=-1,k=-1;

  int cfNodeSize=(int)cfSegNodes[mm].size();
  cfSegV3[mm].clear();
  cfSegV3[mm].resize(cfNodeSize);

  // Minimum and maximum index of each sub-crack
  int min_idx=0,max_idx=-1,numSegs=-1;
  vector<Point>  pts; // Crack-front point subset of the sub-crack
  vector<Vector> V3;  // Crack-front point tangential vector
  vector<double> dis; // Arc length from the starting point 
  vector<int>    idx;  

  for(k=0; k<cfNodeSize;k++) {
    // Step 1: Collect crack points for current sub-crack
    int node=cfSegNodes[mm][k];
    int segs[2];
    FindSegsFromNode(mm,node,segs);

    if(k>max_idx) { // The next sub-crack
      int segsT[2];
      // The minimum index of the sub-crack
      min_idx=k;
      segsT[R]=segs[R];
      while(segsT[R]>=0 && min_idx>0)
        FindSegsFromNode(mm,cfSegNodes[mm][--min_idx],segsT);

      // The maximum index of the sub-crack
      max_idx=k;
      segsT[L]=segs[L];
      while(segsT[L]>=0 && max_idx<cfNodeSize-1) 
        FindSegsFromNode(mm,cfSegNodes[mm][++max_idx],segsT);

      // Allocate memory for the sub-crack
      numSegs=(max_idx-min_idx+1)/2;
      pts.resize(numSegs+1);
      V3.resize(numSegs+1);
      dis.resize(numSegs+1);
      idx.resize(max_idx+1);
    }

    if(k>=min_idx && k<=max_idx) { // For the sub-crack
      short pre_idx=-1;
      for(int ij=0; ij<k; ij++) {
        if(node==cfSegNodes[mm][ij]) {pre_idx=ij; break;}
      }
      if(pre_idx<0) { 
        int ki=(k-min_idx+1)/2;
        pts[ki]=cx[mm][cfSegNodes[mm][k]];
        // Arc length
        if(k==min_idx) dis[ki]=0.;
        else dis[ki]=dis[ki-1]+(pts[ki]-pts[ki-1]).length();
      }
      idx[k]=(k-min_idx+1)/2;
      if(k<max_idx) continue; // Collect next point 
    } 

    // Step 2: Define how to smooth the sub-crack
    int n=numSegs+1;          // number of points (>=2)
    int m=(int)(numSegs/2)+2; // number of intervals (>=2)     
    int n1=7*m-3;

    // Arries starting from 1
    double* S=new double[n+1]; // arc-length to the first point
    double* X=new double[n+1]; // x indexed from 1
    double* Y=new double[n+1]; // y indexed from 1
    double* Z=new double[n+1]; // z indexed from 1
    for(i=1; i<=n; i++) {
      S[i]=dis[i-1];
      X[i]=pts[i-1].x();
      Y[i]=pts[i-1].y(); 
      Z[i]=pts[i-1].z();
    }

    int*    g=new int[n+1];    // segID
    int*    j=new int[m+1];    // number of points 
    double* s=new double[m+1]; // positions of intervals 
    double* ex=new double[n1+1];  
    double* ey=new double[n1+1]; 
    double* ez=new double[n1+1];

    // Positins of the intervals
    s[1]=S[1]-(S[2]-S[1])/5.;
    for(l=2; l<=m; l++) s[l]=s[1]+(S[n]-s[1])/m*(l-1);

    // Number of points in each seg & the segs to which
    // the points belongs
    for(l=1; l<=m; l++) { // Loop over segs
      j[l]=0; // Number of points in the seg
      for(i=1; i<=n; i++) {
        if((l<m  && S[i]>s[l] && S[i]<=s[l+1]) ||
           (l==m && S[i]>s[l] && S[i]<=S[n])) {
          j[l]++; // Number of points in seg l
          g[i]=l; // Seg ID of point i
        }
      }
    }

    // Step 3: Smooth the sub-crack points
    if(CubicSpline(n,m,n1,S,X,s,j,ex,ep) &&
       CubicSpline(n,m,n1,S,Y,s,j,ey,ep) &&
       CubicSpline(n,m,n1,S,Z,s,j,ez,ep)) {// Smooth successfully
      for(i=1; i<=n; i++) {
        l=g[i];
        double t=0.,dtdS=0.;
        if(l<m)  {
          t=2*(S[i]-s[l])/(s[l+1]-s[l])-1.;
          dtdS=2./(s[l+1]-s[l]);
        }
        if(l==m) {
          t=2*(S[i]-s[l])/(S[n]-s[l])-1.;
          dtdS=2./(S[n]-s[l]);
        }

        double Xv0,Xv1,Xv2,Xv3,Yv0,Yv1,Yv2,Yv3,Zv0,Zv1,Zv2,Zv3;
        Xv0=ex[7*l-6]; Xv1=ex[7*l-5]; Xv2=ex[7*l-4]; Xv3=ex[7*l-3];
        Yv0=ey[7*l-6]; Yv1=ey[7*l-5]; Yv2=ey[7*l-4]; Yv3=ey[7*l-3];
        Zv0=ez[7*l-6]; Zv1=ez[7*l-5]; Zv2=ez[7*l-4]; Zv3=ez[7*l-3];

        double t0,t1,t2,t3,t0p,t1p,t2p,t3p;
        t0 =1.; t1 =t;    t2 =2*t*t-1.; t3 =4*t*t*t-3*t;
        t0p=0.; t1p=dtdS; t2p=4*t*dtdS; t3p=(12.*t*t-3.)*dtdS;

        V3[i-1].x(Xv1*t1p+Xv2*t2p+Xv3*t3p);
        V3[i-1].y(Yv1*t1p+Yv2*t2p+Yv3*t3p);
        V3[i-1].z(Zv1*t1p+Zv2*t2p+Zv3*t3p);
        pts[i-1].x(Xv0*t0+Xv1*t1+Xv2*t2+Xv3*t3);
        pts[i-1].y(Yv0*t0+Yv1*t1+Yv2*t2+Yv3*t3);
        pts[i-1].z(Zv0*t0+Zv1*t1+Zv2*t2+Zv3*t3);
      }
    }
    else { // Not smooth successfully, use the raw data
      flag=0;
      for(i=0; i<n; i++) {
        Point pt1=(i==0   ? pts[i]: pts[i-1]);
        Point pt2=(i==n-1 ? pts[i]: pts[i+1]);
        V3[i]=TwoPtsDirCos(pt1,pt2);
      }   
    }
          
    delete [] g;
    delete [] j;
    delete [] s;
    delete [] ex;
    delete [] ey;
    delete [] ez;
    delete [] S;
    delete [] X;
    delete [] Y;
    delete [] Z;

    // Step 4: Store tangential vectors and modify cx
    for(i=min_idx;i<=max_idx;i++) { // Loop over 
      int ki=idx[i]; 
      int nd=cfSegNodes[mm][i]; 
      cx[mm][nd]=pts[ki];
      cfSegV3[mm][i]=V3[ki]/V3[ki].length();
    }
    pts.clear();
    idx.clear();
    dis.clear();
    V3.clear();

  } // End of loop over k

  return flag;
}

short Crack::CubicSpline(const int& n, const int& m, const int& n1,
                         double x[], double y[], double z[],
                         int j[], double e[], const double& ep)
{
  short flag=1;
  int i,k,n3,l,j1,nk,lk,llk,jj,lly,nnj,mmi,nn,ii,my,jm,ni,nij;
  double h1,h2,xlk,xlk1,a1,a2,a3,a4,t;

  double** f=new double*[n1+1];
  for(i=0; i<n1+1; i++) f[i]=new double[14];

  for(i=1; i<=n1; i++) {
    e[i]=0.;
    for(k=1; k<=13; k++) f[i][k]=0.;
  }

  n3=0;
  for(l=1; l<=m; l++) {
    if(l<m)
      h1=1./(z[l+1]-z[l]);
    else
      h1=1./(x[n]-z[m]);

    j1=j[l];
    for(k=1; k<=j1; k++) {
      nk=n3+k;
      xlk=2.*(x[nk]-z[l])*h1-1.;
      xlk1=xlk*xlk;
      a1=1.;
      a2=xlk;
      a3=2.*xlk1-1.;
      a4=(4.*xlk1-3.)*xlk;
      e[7*l-6]+=a1*y[nk];
      e[7*l-5]+=a2*y[nk];
      e[7*l-4]+=a3*y[nk];
      e[7*l-3]+=a4*y[nk];
      f[7*l-6][7]+=a1*a1;
      f[7*l-5][7]+=a2*a2;
      f[7*l-4][7]+=a3*a3;
      f[7*l-3][7]+=a4*a4;
      f[7*l-6][8]+=a1*a2;
      f[7*l-5][8]+=a2*a3;
      f[7*l-4][8]+=a3*a4;
      f[7*l-6][9]+=a1*a3;
      f[7*l-5][9]+=a2*a4;
      f[7*l-6][10]+=a1*a4;
    }

    f[7*l-5][6]=f[7*l-6][8];
    f[7*l-4][5]=f[7*l-6][9];
    f[7*l-3][4]=f[7*l-6][10];

    f[7*l-4][6]=f[7*l-5][8];
    f[7*l-3][5]=f[7*l-5][9];

    f[7*l-3][6]=f[7*l-4][8];
    
    f[7*l-6][4]=-0.5;
    f[7*l-4][2]=-0.5;
    f[7*l-5][3]=0.5;
    f[7*l-3][1]=0.5;
    f[7*l-6][11]=0.5;
    f[7*l-5][10]=0.5;
    f[7*l-4][9]=0.5;
    f[7*l-3][8]=0.5;
    f[7*l-5][4]=-h1;
    f[7*l-5][11]=h1;
    f[7*l-4][3]=4.*h1;
    f[7*l-4][10]=4.*h1;
    f[7*l-4][11]=8.*h1*h1;
    f[7*l-4][4]=-8.*h1*h1;
    f[7*l-3][2]=-9.*h1;
    f[7*l-3][9]=9.*h1;
    f[7*l-3][3]=48.*h1*h1;
    f[7*l-3][10]=48.*h1*h1;

    if(l<=m-1) {
      if(l<m-1)
        h2=1./(z[l+2]-z[l+1]);
      else
        h2=1./(x[n]-z[m]);

      f[7*l-2][3]=1.;
      f[7*l-2][4]=1.;
      f[7*l-2][5]=1.;
      f[7*l-2][6]=1.;
      f[7*l-2][11]=1.;
      f[7*l-2][13]=1.;
      f[7*l-2][10]=-1.;
      f[7*l-2][12]=-1.;
      f[7*l-1][3]=2.*h1;
      f[7*l-1][4]=8.*h1;
      f[7*l-1][5]=18.*h1;
      f[7*l-1][10]=-2.*h2;
      f[7*l-1][11]=8.*h2;
      f[7*l-1][12]=-18.*h2;
      f[7*l][3]=16.*h1*h1;
      f[7*l][4]=96.*h1*h1;
      f[7*l][10]=-16.*h2*h2;
      f[7*l][11]=96.*h2*h2;
    }
    n3+=j[l];
  }

  lk=7;
  llk=lk-1;
  for(jj=1; jj<=llk; jj++) {
    lly=lk-jj;
    nnj=n1+1-jj;
    for(i=1; i<=lly; i++) {
      for(k=2; k<=13; k++) f[jj][k-1]= f[jj][k];
      f[jj][13]=0.;
      mmi=14-i;
      f[nnj][mmi]=0.;
    }
  }

  nn=n1-1;
  for(i=1; i<=nn; i++) {
    k=i;
    ii=i+1;
    for(my=ii; my<=lk; my++) {
      if(fabs(f[my][1])<=fabs(f[k][1])) continue;
      k=my;
    }

    if(k!=i) {
      t=e[i];
      e[i]=e[k];
      e[k]=t;
      for(jj=1; jj<=13; jj++) {
        t=f[i][jj];
        f[i][jj]=f[k][jj];
        f[k][jj]=t;
      }
    }

    if(ep>=fabs(f[i][1])) {
      flag=0;
      return flag; // unsuccessful
    }
    else {
      e[i]/=f[i][1];
      for(jj=2; jj<=13; jj++) f[i][jj]/=f[i][1];

      ii=i+1;
      for(my=ii; my<=lk; my++) {
        t=f[my][1];
        e[my]-=t*e[i];
        for(jj=2; jj<=13; jj++) f[my][jj-1]=f[my][jj]-t*f[i][jj];
        f[my][13]=0.;
      }

      if(lk==n1) continue;
      lk++;
    }
  }

  e[n1]/=f[n1][1];
  jm=2;
  nn=n1-1;
  for(i=1; i<=nn; i++) {
    ni=n1-i;
    for(jj=2; jj<=jm; jj++) {
      nij=ni-1+jj;
      e[ni]-=f[ni][jj]*e[nij];
    }
    if(jm==13) continue;
    jm++;
  }

  return flag;
}
