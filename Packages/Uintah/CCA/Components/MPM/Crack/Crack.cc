#include "Crack.h"
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
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Util/NotFinished.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>

using namespace Uintah;
using namespace SCIRun;
using std::vector;
using std::string;

using namespace std;

#define MAX_BASIS 27

Crack::Crack(const ProblemSpecP& ps,SimulationStateP& d_sS,
                           MPMLabel* Mlb,int n8or27)
{
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

  // Read in crack parameters, which are placed in element "material" 
  ProblemSpecP mpm_ps = ps->findBlock("MaterialProperties")->findBlock("MPM");

  int m=0; // current material ID 
  for( ProblemSpecP mat_ps=mpm_ps->findBlock("material"); mat_ps!=0;
                   mat_ps=mat_ps->findNextBlock("material") ) {

    // variables for this material
    vector<vector<Point> > rectangles;
    vector<vector<Point> > triangles;
    vector<int> n12;
    vector<int> n23;
    vector<int> ncell;

    ProblemSpecP crk_ps=mat_ps->findBlock("crack");
 
    if(crk_ps==0) crackType[m]="NO_CRACK";
 
    if(crk_ps!=0) { 
       // read in crack contact type, frictional coefficient,
       // contcat volume and separate volume 
       crk_ps->require("type",crackType[m]);
     
       // default values of critical volumes   
       separateVol[m]=1.;
       contactVol[m]=1.;
       if(crackType[m]=="frictional" || crackType[m]=="stick") {
          if(crackType[m]=="frictional") {
             crk_ps->require("mu",c_mu[m]);
          }
          else if(crackType[m]=="stick") {
             c_mu[m]=0.0;
          }
          crk_ps->get("separateVol",separateVol[m]);
          crk_ps->get("contactVol",contactVol[m]);
       }
       else if(crackType[m]=="null") {
          c_mu[m]=0.;
       }
       else {
          cout << "Unkown crack contact type in subroutine Crack::Crack: " 
               << crackType[m] << endl;
          exit(1);
       }
        
       // read in crack geometry  
       cmin[m]=Point(9e16,9e16,9e16);
       cmax[m]=Point(-9e16,-9e16,-9e16);
       ProblemSpecP geom_ps=crk_ps->findBlock("cracksegments");
       // read in quadrilaterals
       for(ProblemSpecP quad_ps=geom_ps->findBlock("quadrilateral");
          quad_ps!=0; quad_ps=quad_ps->findNextBlock("quadrilateral")) {
          Point p;   
          vector<Point> thisRect;
          int n;
          quad_ps->require("pt1",p);
          thisRect.push_back(p);
          cmin[m]=Min(p,cmin[m]);
          cmax[m]=Max(p,cmax[m]); 
          quad_ps->require("pt2",p);
          thisRect.push_back(p);
          cmin[m]=Min(p,cmin[m]);
          cmax[m]=Max(p,cmax[m]);
          quad_ps->require("pt3",p);
          thisRect.push_back(p);
          cmin[m]=Min(p,cmin[m]);
          cmax[m]=Max(p,cmax[m]);
          quad_ps->require("pt4",p);
          thisRect.push_back(p);
          cmin[m]=Min(p,cmin[m]);
          cmax[m]=Max(p,cmax[m]);
          rectangles.push_back(thisRect);
          thisRect.clear();
           
          quad_ps->require("n12",n);  // n12, n23 are numbers of sub-cells
          n12.push_back(n);
          quad_ps->require("n23",n);
          n23.push_back(n);
        
       }
       allRects[m]=rectangles;
       allN12[m]=n12;
       allN23[m]=n23;
       rectangles.clear();
       n12.clear();
       n23.clear();
 
       // read in triangles         
       for(ProblemSpecP tri_ps=geom_ps->findBlock("triangle");
             tri_ps!=0; tri_ps=tri_ps->findNextBlock("triangle")) {
          Point p;
          vector<Point> thisTri;
          int n;

          tri_ps->require("pt1",p);
          thisTri.push_back(p);
          cmin[m]=Min(p,cmin[m]);
          cmax[m]=Max(p,cmax[m]);
          tri_ps->require("pt2",p);
          thisTri.push_back(p);
          cmin[m]=Min(p,cmin[m]);
          cmax[m]=Max(p,cmax[m]);
          tri_ps->require("pt3",p);
          thisTri.push_back(p);
          cmin[m]=Min(p,cmin[m]);
          cmax[m]=Max(p,cmax[m]);
          triangles.push_back(thisTri);
          thisTri.clear();
          tri_ps->require("ncell",n);
          ncell.push_back(n);
       }
       allTris[m]=triangles;
       allNCell[m]=ncell;
       ncell.clear();
       triangles.clear();
    } // End of if crk_ps != 0

    m++; // next material
  }  // End of loop materials

  int numMatls=m;  // total number of materials

#if 1  // output crack parameters
  cout << "*** Crack Information ***" << endl;
  for(m=0; m<numMatls; m++) {
     if(crackType[m]=="NO_CRACK") {
        cout << "\n--- material: " << m << ", no crack exists" << endl;
     }
     else {
        cout << " --- material: " << m << ", crack contact type: " 
             << "\'" << crackType[m] << "\'" << endl;
        if(crackType[m]=="frictional") {
           cout << "    frictional coefficient: " << c_mu[m] << endl;
        }
        if(separateVol[m]<0. || contactVol[m]<0.) {
          cout  << "\nCheck crack contact by displacement criterion"
                << endl;
        }
        else {
          cout  << "\nCheck crack contact by volume criterion"
                << ", separate volume = " << separateVol[m]
                << ", contact volume = " << contactVol[m] << endl;
        }
      
        cout <<"\nCrack segments:" << endl;
        for(int i=0;i<(int)allRects[m].size();i++) {
           cout << "\nRectangle " << i << ": meshed by [" << allN12[m][i] 
                << ", " << allN23[m][i] << "]" << endl;
           for(int j=0;j<4;j++) {
              cout << "pt " << j << ": " << allRects[m][i][j] << endl;
           }
        } 
        for(int i=0;i<(int)allTris[m].size();i++) {
           cout << "\nTriangle " << i << ": " << "ncell=" 
                << allNCell[m][i] << endl;
           for(int j=0;j<3;j++) {
              cout << "pt " << j << ": " << allTris[m][i][j] << endl;
           }
        }
        cout << "\ncrack extent: lower: " << cmin[m]
             << ", upper: " << cmax[m] << endl << endl;
     } // End of if(crackType...)
  } // End of loop over materials
#endif 

}

void Crack::addComputesAndRequiresCrackDiscretization(Task* t,
                                const PatchSet* patches,
                                const MaterialSet* matls) const
{
// do nothing currently 
}

void Crack::CrackDiscretization(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  int cn,ce,id;
  int k,i,j,ni,nj,n1,n2,n3;
  int nstart0,nstart1,nstart2,nstart3;
  double w;
  Point p1,p2,p3,p4,pt,p_1,p_2;

  for(int p=0;p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    int numMPMMatls=d_sharedState->getNumMPMMatls();

    for(int m = 0; m < numMPMMatls; m++){ 

       if(crackType[m]=="NO_CRACK") { // no crack for this material
          numElems[m] = 0;
          numPts[m]   = 0;
          continue;
       }

       cn = 0;  // current node
       ce = 0;  // icurrent element

       //Discretize quadrilaterals
       nstart0=0;  // starting node number for each level (in j direction)
       for(k=0; k<(int)allRects[m].size(); k++) {  // loop over quadrilaterals
         // resolutions for the quadrilateral 
         ni=allN12[m][k];       
         nj=allN23[m][k]; 
         // four points for the quadrilateral 
         p1=allRects[m][k][0];   
         p2=allRects[m][k][1];
         p3=allRects[m][k][2];
         p4=allRects[m][k][3];

         // create temprary arraies
         Point* side2=new Point[2*nj+1];
         Point* side4=new Point[2*nj+1];

         // generate side-node coordinates
         for(j=0; j<=2*nj; j++) {
           side2[j]=p2+(p3-p2)*(float)j/(2*nj);
           side4[j]=p1+(p4-p1)*(float)j/(2*nj);
         }

         for(j=0; j<=nj; j++) {
           for(i=0; i<=ni; i++) {  
              w=(float)i/(float)ni;
              p_1=side4[2*j];
              p_2=side2[2*j];
              pt=p_1+(p_2-p_1)*w;
              cx[m][cn++]=pt;
           }
           if(j!=nj) {
              for(i=0; i<ni; i++) {
                 w=(float)(2*i+1)/(float)(2*ni);
                 p_1=side4[2*j+1];
                 p_2=side2[2*j+1];
                 pt=p_1+(p_2-p_1)*w;
                 cx[m][cn++]=pt;
              }
           }  // End of if j!=nj
         } // End of loop over j

         // create elements and get normals for quadrilaterals
         for(j=0; j<nj; j++) {
           nstart1=nstart0+(2*ni+1)*j;
           nstart2=nstart1+(ni+1);
           nstart3=nstart2+ni;
           for(i=0; i<ni; i++) {
              // there are four segments in each sub-rectangle
              // for 1st segment, n1,n2,n3 three nodes of the triangle 
              n1=nstart2+i; 
              n2=nstart1+i; 
              n3=nstart1+(i+1);
              cElemNodes[m][ce]=IntVector(n1,n2,n3);
              cElemNorm[m][ce++]=TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]);
              // for 2nd segment
              n1=nstart2+i;
              n2=nstart3+i;
              n3=nstart1+i;
              cElemNodes[m][ce]=IntVector(n1,n2,n3);
              cElemNorm[m][ce++]=TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]);
              // for 3 rd segment
              n1=nstart2+i;
              n2=nstart1+(i+1);
              n3=nstart3+(i+1);
              cElemNodes[m][ce]=IntVector(n1,n2,n3);
              cElemNorm[m][ce++]=TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]);
              // for 4th element 
              n1=nstart2+i;
              n2=nstart3+(i+1);
              n3=nstart3+i;
              cElemNodes[m][ce]=IntVector(n1,n2,n3);
              cElemNorm[m][ce++]=TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]);
           }  // end of loop over i
         }  // end of loop over j
         nstart0+=((2*ni+1)*nj+ni+1);  
         delete [] side4;
         delete [] side2;
       } // End ofloop over quadrilaterals 

       // discretize triangluar segments 
       for(k=0; k<(int)allTris[m].size(); k++) {  // loop over all triangles
         p1=allTris[m][k][0];
         p2=allTris[m][k][1];
         p3=allTris[m][k][2];

         // create temprary arraies
         Point* side12=new Point[allNCell[m][k]+1];
         Point* side13=new Point[allNCell[m][k]+1];

         // generate node coordinates
         for(j=0; j<=allNCell[m][k]; j++) {
           w=(float)j/(float)allNCell[m][k];
           side12[j]=p1+(p2-p1)*w;
           side13[j]=p1+(p3-p1)*w;
         }
        
         for(j=0; j<=allNCell[m][k]; j++) {
           for(i=0; i<=j; i++) {
             p_1=side12[j];
             p_2=side13[j];
             if(j==0) w=0.0;
             else w=(float)i/(float)j;
             pt=p_1+(p_2-p_1)*w;
             cx[m][cn++]=pt;
           } // End of loop over i
         } // End of loop over j
 
         // generate elements and their normals
         for(j=0; j<allNCell[m][k]; j++) {
           nstart1=nstart0+j*(j+1)/2;
           nstart2=nstart0+(j+1)*(j+2)/2;
           for(i=0; i<j; i++) {
             //left element
             n1=nstart1+i;
             n2=nstart2+i;
             n3=nstart2+(i+1);
             cElemNodes[m][ce]=IntVector(n1,n2,n3);
             cElemNorm[m][ce++]=TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]);
             //right element
             n1=nstart1+i;
             n2=nstart2+(i+1);
             n3=nstart1+(i+1);
             cElemNodes[m][ce]=IntVector(n1,n2,n3);
             cElemNorm[m][ce++]=TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]);
           } // End of loop over i
           n1=nstart0+(j+1)*(j+2)/2-1;
           n2=nstart0+(j+2)*(j+3)/2-2;
           n3=nstart0+(j+2)*(j+3)/2-1;
           cElemNodes[m][ce]=IntVector(n1,n2,n3);
           cElemNorm[m][ce++]=TriangleNormal(cx[m][n1],cx[m][n2],cx[m][n3]);
         } // End of loop over j
         //add number of nodes in this trianglular segment
         nstart0+=(allNCell[m][k]+1)*(allNCell[m][k]+2)/2;
         delete [] side12;
         delete [] side13;
       } // End of loop over triangles

       numPts[m]=cn;     // number of crack points in this materials
       numElems[m]=ce;   // number of crack segments in this material 
 
       cout << "*** Crack elements information \n" << "MatID: " << m << endl;
       for(int mp=0; mp<numElems[m]; mp++) {
         n1=cElemNodes[m][mp].x();
         n2=cElemNodes[m][mp].y();
         n3=cElemNodes[m][mp].z();
         cout << "   Elem " << mp 
              << ": " << n1 << cx[m][n1] << ", " << n2 << cx[m][n2]
              << ", " << n3 << cx[m][n3] << endl;
       } 
     } // End of loop over matls
   } // End of loop over patches
}

void Crack::addComputesAndRequiresParticleVelocityField(Task* t,
                                const PatchSet* patches,
                                const MaterialSet* matls) const
{  
  //t->requires(Task::OldDW, lb->pXLabel, Ghost::None);
  t->requires(Task::OldDW, lb->pXLabel, Ghost::AroundCells, NGN);
  t->computes(lb->gNumPatlsLabel);
  t->computes(lb->GNumPatlsLabel);
  t->computes(lb->GCrackNormLabel);
  t->computes(lb->pgCodeLabel);
}

void Crack::ParticleVelocityField(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  //double time0, time1;
  //time0 = clock();

  for(int p=0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int numMatls = d_sharedState->getNumMPMMatls();

    Ghost::GhostType  gan = Ghost::AroundNodes;
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

      IntVector ni[MAX_BASIS];

      if(numElems[m]==0) { // for materials wothout any carck
        for(ParticleSubset::iterator iter=pset->begin();
                                     iter!=pset->end();iter++) {
          particleIndex idx=*iter;
          if(d_8or27==8)
            patch->findCellNodes(px[idx], ni);
          else if(d_8or27==27)
            patch->findCellNodes27(px[idx], ni);
          for(int k=0; k<d_8or27; k++) {
            pgCode[idx][k]=1;
            gNumPatls[ni[k]]++;
          }
        } //End of loop over partls
      } // End of if(numElems[m]==0

      if(numElems[m]!=0) { // for materials with crack
        //Step 1: determine if nodes in crack zone 
        int nx, ny, nz;   
        IntVector gmin, gmax, cellIdx;
        NCVariable<short> singlevfld;
        new_dw->allocateTemporary(singlevfld, patch, // Ghost cells needed !
                          Ghost::AroundCells, NGN);
        singlevfld.initialize(0);
        //cout << "cmin=" << cmin[m] << ", cmax=" << cmax[m] << endl;

        patch->findCell(cmin[m],cellIdx);
        Point ptmp=patch->nodePosition(cellIdx+IntVector(1,1,1));
        if(fabs(cmin[m].x()-ptmp.x())/dx.x()<1e-6) // ptmp.x()=cmin.x()
          nx=NGN-1; 
        else  
          nx=NGN;
        if(fabs(cmin[m].y()-ptmp.y())/dx.y()<1e-6) // ptmp.y()=cmin.y()
          ny=NGN-1; 
        else 
          ny=NGN;
        if(fabs(cmin[m].z()-ptmp.z())/dx.z()<1e-6) // ptmp.z()=cmin.z()
          nz=NGN-1;
        else
          nz=NGN; 
        gmin=cellIdx+IntVector(1-nx,1-ny,1-nz);

        if(gmin.x()<patch->getNodeLowIndex().x()) 
           gmin.x(patch->getNodeLowIndex().x());
        if(gmin.y()<patch->getNodeLowIndex().y())
           gmin.y(patch->getNodeLowIndex().y());
        if(gmin.z()<patch->getNodeLowIndex().z())
           gmin.z(patch->getNodeLowIndex().z());

        patch->findCell(cmax[m],cellIdx);
        ptmp=patch->nodePosition(cellIdx);
        if(fabs(cmax[m].x()-ptmp.x())/dx.x()<1e-6) // ptmp.x()=cmax.x()
          nx=NGN-1; 
        else
          nx=NGN;
        if(fabs(cmax[m].y()-ptmp.y())/dx.y()<1e-6) // ptmp.y()=cmax.y()
          ny=NGN-1;
        else
          ny=NGN;
        if(fabs(cmax[m].z()-ptmp.z())/dx.z()<1e-6) // ptmp.z()=cmax.z()
          nz=NGN-1;
        else 
          nz=NGN;
        gmax=cellIdx+IntVector(nx,ny,nz);

        if(gmax.x()>patch->getNodeHighIndex().x())    
           gmax.x(patch->getNodeHighIndex().x());
        if(gmax.y()>patch->getNodeHighIndex().y())
           gmax.y(patch->getNodeHighIndex().y());
        if(gmax.z()>patch->getNodeHighIndex().z())
           gmax.z(patch->getNodeHighIndex().z());
        //cout << "gmin=" << gmin << ", gmax=" << gmax << endl;

        for(NodeIterator iter=patch->getNodeIterator();!iter.done();iter++) {
          IntVector c=*iter;
          if(c.x()>=gmin.x() && c.x()<=gmax.x() && c.y()>=gmin.y() &&
            c.y()<=gmax.y() && c.z()>=gmin.z() && c.z()<=gmax.z() )
            singlevfld[c]=0; // in crack zone
          else
            singlevfld[c]=1; // in non-crack zone, single velocity field
            //cout << "c=" << c << ", singlevfld=" << singlevfld[c] << endl;
        }
            
        // Step 2: Detect if particle is above, below or in the same side 
        NCVariable<int> num0,num1,num2; 
        new_dw->allocateTemporary(num0, patch, Ghost::AroundCells, 2*NGN);
        new_dw->allocateTemporary(num1, patch, Ghost::AroundCells, 2*NGN);
        new_dw->allocateTemporary(num2, patch, Ghost::AroundCells, 2*NGN);
        num0.initialize(0);
        num1.initialize(0);
        num2.initialize(0);
 
        //Generate particle cross code for particles in pset
        for(ParticleSubset::iterator iter=pset->begin();
                                     iter!=pset->end();iter++) {
          particleIndex idx=*iter;
          if(d_8or27==8)
             patch->findCellNodes(px[idx], ni);
          else if(d_8or27==27)
             patch->findCellNodes27(px[idx], ni);

          for(int k=0; k<d_8or27; k++) {
            //for nodes in non-crack zone of cracked materials
            if(singlevfld[ni[k]]) {
              pgCode[idx][k]=0;
              num0[ni[k]]++;
            }
            else { //for nodes in crack zone
              enum   {NON_CRACK=0,ABOVE_CRACK,BELOW_CRACK};
              short  cross=NON_CRACK;
              Vector norm=Vector(0.,0.,0.);
              Point  gx=patch->nodePosition(ni[k]);
              for(int i=0; i<numElems[m]; i++) {  //loop over crack elements
                Point n3,n4,n5; //three vertices of the triangular element
                n3=cx[m][cElemNodes[m][i].x()];
                n4=cx[m][cElemNodes[m][i].y()];
                n5=cx[m][cElemNodes[m][i].z()];

                short position = NotSameSide(px[idx],gx,n3,n4,n5);
                if(position==NON_CRACK) continue; // particle and node in same side
                // three volumes for determining if p-g line crosses in crack elements
                double v3,v4,v5;
                v3=Volume(gx,n3,n4,px[idx]);
                v4=Volume(gx,n3,n5,px[idx]);
                if(v3*v4>0.) continue;
                v5=Volume(gx,n4,n5,px[idx]);
                if(v3*v5<0.) continue;

                if(position==ABOVE_CRACK && v3>=0. && v4<=0. && v5>=0.) { //above crack
                  if(cross==NON_CRACK || (cross!=NON_CRACK &&
                                (v3==0.||v4==0.||v5==0.) ) ) {
                    cross=ABOVE_CRACK;
                    norm+=cElemNorm[m][i];
                  }
                  else { // no cross
                    cross=NON_CRACK;
                    norm=Vector(0.,0.,0.);
                  }
                }

                if(position==BELOW_CRACK && v3<=0. && v4>=0. && v5<=0.) { //below crack
                  if(cross==NON_CRACK || (cross!=NON_CRACK &&
                                (v3==0.||v4==0.||v5==0.) ) ) {
                    cross=BELOW_CRACK;
                    norm+=cElemNorm[m][i];
                  }
                  else { // no cross
                    cross=NON_CRACK;
                    norm=Vector(0.,0.,0.);
                  }
                }
              } // End of loop over elements

              pgCode[idx][k]=cross;
              if(cross==NON_CRACK)   num0[ni[k]]++;
              if(cross==ABOVE_CRACK) num1[ni[k]]++;
              if(cross==BELOW_CRACK) num2[ni[k]]++;
              if(norm.length()>1.e-16) norm/=norm.length();
              if(norm.length()>1.e-16) GCrackNorm[ni[k]]+=norm;
            } //End of if(singlevfld)
          } // End of loop over k
        } // End of loop over particles
 
        //Handle particles in GhostCells
        ParticleSubset* psetWGCs= old_dw->getParticleSubset(dwi, patch,
                                 Ghost::AroundCells, NGN, lb->pXLabel);
        constParticleVariable<Point> pxWGCs;
        old_dw->get(pxWGCs, lb->pXLabel, psetWGCs);

        for(ParticleSubset::iterator iter=psetWGCs->begin();
                                     iter!=psetWGCs->end();iter++) {
          particleIndex idx=*iter;
 
          //Detect if particles in Ghost Cells
          short pincell=1; //Particles in Ghost Cells
          for(ParticleSubset::iterator iter1=pset->begin();
                                     iter1!=pset->end();iter1++) {
            particleIndex idx1=*iter1;
            if(pxWGCs[idx]==px[idx1]) {
              pincell=0;  //particles not in Ghost Cells
              break;
            }
          }

          if(pincell) { //for particles in Ghost Cells
            if(d_8or27==8)
              patch->findCellNodes(pxWGCs[idx], ni);
            else if(d_8or27==27)
              patch->findCellNodes27(pxWGCs[idx], ni);
        
            for(int k=0; k<d_8or27; k++) {
              //for nodes in non-crack zone of cracked materials  
              if(singlevfld[ni[k]]) 
                num0[ni[k]]++;
              else { //for nodes in crack zone
                enum   {NON_CRACK=0,ABOVE_CRACK,BELOW_CRACK};
                short  cross=NON_CRACK;
                Vector norm=Vector(0.,0.,0.);
                Point  gx=patch->nodePosition(ni[k]);
                for(int i=0; i<numElems[m]; i++) {  //loop over crack elements
                  Point n3,n4,n5; //three vertices of the triangular element
                  n3=cx[m][cElemNodes[m][i].x()];
                  n4=cx[m][cElemNodes[m][i].y()];
                  n5=cx[m][cElemNodes[m][i].z()];
                          
                  short position = NotSameSide(pxWGCs[idx],gx,n3,n4,n5);
                  if(position==NON_CRACK) continue; // particle and node in same side 
                  // three volumes for determining if p-g line crosses in crack elements  
                  double v3,v4,v5;
                  v3=Volume(gx,n3,n4,pxWGCs[idx]);
                  v4=Volume(gx,n3,n5,pxWGCs[idx]);
                  if(v3*v4>0.) continue;
                  v5=Volume(gx,n4,n5,pxWGCs[idx]);
                  if(v3*v5<0.) continue;
                                    
                  if(position==ABOVE_CRACK && v3>=0. && v4<=0. && v5>=0.) { //above crack
                    if(cross==NON_CRACK || (cross!=NON_CRACK && 
                                  (v3==0.||v4==0.||v5==0.) ) ) {
                      cross=ABOVE_CRACK;
                      norm+=cElemNorm[m][i];
                    }
                    else { // no cross
                      cross=NON_CRACK;
                      norm=Vector(0.,0.,0.);
                    }
                  }
                                        
                  if(position==BELOW_CRACK && v3<=0. && v4>=0. && v5<=0.) { //below crack
                    if(cross==NON_CRACK || (cross!=NON_CRACK && 
                                  (v3==0.||v4==0.||v5==0.) ) ) { 
                      cross=BELOW_CRACK;
                      norm+=cElemNorm[m][i];
                    }
                    else { // no cross
                      cross=NON_CRACK;
                      norm=Vector(0.,0.,0.);
                    }
                  } 
                } // End of loop over elements
                            
                if(cross==NON_CRACK)   num0[ni[k]]++;
                if(cross==ABOVE_CRACK) num1[ni[k]]++;
                if(cross==BELOW_CRACK) num2[ni[k]]++;
                if(norm.length()>1.e-16) norm/=norm.length();
                if(norm.length()>1.e-16) GCrackNorm[ni[k]]+=norm;
              } //End of if(singlevfld) 
            } // End of loop over k 
          } // End of if(pincell) 
        } // End of loop over particles
             
        //Step 3: convert cross codes to field codes 
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
                  cout << "Three velocity fields found in " 
                       << "Crack::ParticleVeloccityField for node: " << ni[k] << endl;
                  exit(1);
                } 

              }
           }// End of loop over k
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
 
#if 0 // output particle velocity field code
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
  //time1=clock()-time0;
  //time1/=CLOCKS_PER_SEC;
  //cout << "***time for particleVelocityField (1+2) = " << time1 << endl;
}

void Crack::addComputesAndRequiresCrackAdjustInterpolated(Task* t,
                                const PatchSet* patches,
                                const MaterialSet* matls) const
{
  const MaterialSubset* mss = matls->getUnion();

  //data of primary field
  t->requires(Task::NewDW, lb->gMassLabel,         Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeLabel,       Ghost::None);
  t->requires(Task::NewDW, lb->gNumPatlsLabel,     Ghost::None); 
  t->requires(Task::NewDW, lb->gDisplacementLabel, Ghost::None);

  //data of additional field
  t->requires(Task::NewDW, lb->GMassLabel,         Ghost::None);
  t->requires(Task::NewDW, lb->GVolumeLabel,       Ghost::None);
  t->requires(Task::NewDW, lb->GNumPatlsLabel,     Ghost::None);
  t->requires(Task::NewDW, lb->GCrackNormLabel,    Ghost::None);
  t->requires(Task::NewDW, lb->GDisplacementLabel, Ghost::None);

  t->modifies(lb->gVelocityLabel, mss);
  t->modifies(lb->GVelocityLabel, mss);

  t->computes(lb->frictionalWorkLabel);

}

void Crack::CrackContactAdjustInterpolated(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  //double time0, time1;
  //time0=clock();

  double mua,mub;
  double ma,mb,dvan,dvbn,dvat,dvbt,ratioa,ratiob;
  double vol0,normVol;
  Vector va,vb,vc,dva,dvb,ta,tb,na,nb,norm;

  int numMatls = d_sharedState->getNumMPMMatls();
  ASSERTEQ(numMatls, matls->size());

  for(int p=0;p<patches->size();p++){

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

    Ghost::GhostType  gan   = Ghost::AroundNodes;
    Ghost::GhostType  gnone = Ghost::None;

    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      // data for primary velocity field 
      new_dw->get(gNumPatls[m], lb->gNumPatlsLabel, dwi, patch, gnone, 0);
      new_dw->get(gmass[m],     lb->gMassLabel,     dwi, patch, gnone, 0);
      new_dw->get(gvolume[m],   lb->gVolumeLabel,   dwi, patch, gnone, 0);
      new_dw->get(gdisplacement[m],lb->gDisplacementLabel,dwi,patch,gnone,0);

      new_dw->getModifiable(gvelocity[m],lb->gVelocityLabel,dwi,patch);

      // data for second velocity field
      new_dw->get(GNumPatls[m],lb->GNumPatlsLabel,  dwi, patch, gnone, 0);
      new_dw->get(Gmass[m],     lb->GMassLabel,     dwi, patch, gnone, 0);
      new_dw->get(Gvolume[m],   lb->GVolumeLabel,   dwi, patch, gnone, 0);
      new_dw->get(GCrackNorm[m],lb->GCrackNormLabel,dwi, patch, gnone, 0);
      new_dw->get(Gdisplacement[m],lb->GDisplacementLabel,dwi,patch,gnone,0);
      new_dw->getModifiable(Gvelocity[m],lb->GVelocityLabel,dwi,patch);

      new_dw->allocateAndPut(frictionWork[m],lb->frictionalWorkLabel,dwi,patch);
      frictionWork[m].initialize(0.);

      if(crackType[m]=="NO_CRACK") continue;  // no crack in this material
      // loop over nodes to see if there is contact. If yes, adjust velocity field
      for(NodeIterator iter=patch->getNodeIterator();!iter.done();iter++) {
        IntVector c = *iter;
       
        //only one velocity field
        if(gNumPatls[m][c]==0 || GNumPatls[m][c]==0) continue; 
        //nodes in non-crack-zone
        norm=GCrackNorm[m][c];
        if(norm.length()<1.e-16) continue;  // should not happen now, but ...

        ma=gmass[m][c];
        va=gvelocity[m][c];
        mb=Gmass[m][c];
        vb=Gvelocity[m][c];
        vc=(va*ma+vb*mb)/(ma+mb);
        short Contact=0;

        if(separateVol[m]<0. || contactVol[m] <0.) { 
          //use displacement criterion
          Vector u1=gdisplacement[m][c];
          Vector u2=Gdisplacement[m][c];
          //cout << "Interpolated--node:" << c << ", u1=" << u1 << ", u2=" << u2 << endl;
          if(Dot((u2-u1),norm) >0. ) {
            Contact=1;
          }
        }
        else { // use volume criterion
          //evaluate the nodal saturated volume (vol0) for general cases
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
            for(int k=0; k<8; k++) { //loop over 8 cells around the node
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
            Contact=1;
            //cout << "Interpolated---node:" << c << ", normVol:" << normVol << endl;
          }
        }

        if(!Contact) { // no contact  
          gvelocity[m][c]=gvelocity[m][c];
          Gvelocity[m][c]=Gvelocity[m][c];
          frictionWork[m][c] += 0.;
        } 
        else { // there is contact, apply contact law
          if(crackType[m]=="null") { // nothing to do with it
            gvelocity[m][c]=gvelocity[m][c];
            Gvelocity[m][c]=Gvelocity[m][c];
            frictionWork[m][c] += 0.;
          }

          else if(crackType[m]=="stick") { // assign centerofmass velocity
            gvelocity[m][c]=vc;
            Gvelocity[m][c]=vc;
            frictionWork[m][c] += 0.;
          }

          else if(crackType[m]=="frictional") { // apply frictional law
            //for velocity field above crack
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
            if( fabs(ratioa)>c_mu[m] ) {  // slide
               if(ratioa>0.) mua=c_mu[m];
               if(ratioa<0.) mua=-c_mu[m];
               deltva=-(na+ta*mua)*dvan;
               gvelocity[m][c]=va+deltva;
               frictionWork[m][c]+=ma*c_mu[m]*dvan*dvan*(fabs(ratioa)-c_mu[m]);
            }
            else {  // stick
               gvelocity[m][c]=vc;
               frictionWork[m][c] += 0.;
            }

            //for velocity field below crack
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
            if(fabs(ratiob)>c_mu[m]) { // slide
               if(ratiob>0.) mub=c_mu[m];
               if(ratiob<0.) mub=-c_mu[m];
               deltvb=-(nb+tb*mub)*dvbn;
               Gvelocity[m][c]=vb+deltvb;
               frictionWork[m][c]+=mb*c_mu[m]*dvbn*dvbn*(fabs(ratiob)-c_mu[m]);
            }
            else {// stick
               Gvelocity[m][c]=vc;
               frictionWork[m][c] += 0.;
            }
          }

          else { // wrong contact type
            cout << "Unknown crack contact type in subroutine " 
                 << "Crack::CrackContactAdjustInterpolated: " 
                 << crackType[m] << endl;
            exit(1);
          }
        } // End of if there is !contact

      } //End of loop over nodes
    } //End of loop over materials
  }  //End of loop over patches
  //time1=clock()-time0;
  //time1/=CLOCKS_PER_SEC;
  //cout << "***time for crackAdjustInterpolated = " << time1 << endl;
}

void Crack::addComputesAndRequiresCrackAdjustIntegrated(Task* t,
                                const PatchSet* patches,
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

void Crack::CrackContactAdjustIntegrated(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  //double time0, time1;
  //time0=clock();

  double mua,mub;
  double ma,mb,dvan,dvbn,dvat,dvbt,ratioa,ratiob;
  double vol0,normVol;
  Vector aa,ab,va,vb,vc,dva,dvb,ta,tb,na,nb,norm;

  int numMatls = d_sharedState->getNumMPMMatls();
  ASSERTEQ(numMatls, matls->size());

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double vcell = dx.x()*dx.y()*dx.z();

    // Need access to all velocity fields at once
    // data of primary field
    StaticArray<constNCVariable<double> > gmass(numMatls);
    StaticArray<constNCVariable<double> > gvolume(numMatls);
    StaticArray<constNCVariable<int> >    gNumPatls(numMatls);
    StaticArray<constNCVariable<Vector> > gdisplacement(numMatls);
    StaticArray<NCVariable<Vector> >      gvelocity_star(numMatls);
    StaticArray<NCVariable<Vector> >      gacceleration(numMatls);
    // data of additional field 
    StaticArray<constNCVariable<double> > Gmass(numMatls);
    StaticArray<constNCVariable<double> > Gvolume(numMatls);
    StaticArray<constNCVariable<int> >    GNumPatls(numMatls);
    StaticArray<constNCVariable<Vector> > GCrackNorm(numMatls);
    StaticArray<constNCVariable<Vector> > Gdisplacement(numMatls);
    StaticArray<NCVariable<Vector> >      Gvelocity_star(numMatls);
    StaticArray<NCVariable<Vector> >      Gacceleration(numMatls);
    // friction work
    StaticArray<NCVariable<double> >      frictionWork(numMatls);
   
    Ghost::GhostType  gan   = Ghost::AroundNodes;
    Ghost::GhostType  gnone = Ghost::None;

    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      // for primary field
      new_dw->get(gmass[m],     lb->gMassLabel,     dwi, patch, gnone, 0);
      new_dw->get(gvolume[m],   lb->gVolumeLabel,   dwi, patch, gnone, 0);
      new_dw->get(gNumPatls[m], lb->gNumPatlsLabel, dwi, patch, gnone, 0);
      new_dw->get(gdisplacement[m],lb->gDisplacementLabel,dwi,patch,gnone,0);

      new_dw->getModifiable(gvelocity_star[m], lb->gVelocityStarLabel,
                                                         dwi, patch);
      new_dw->getModifiable(gacceleration[m],lb->gAccelerationLabel,
                                                         dwi, patch);
      // for additional field
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

      if(crackType[m]=="NO_CRACK") continue; // no crack in this material
                                             // nothing to do with it
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
        short Contact=0;

        if(separateVol[m]<0. || contactVol[m] <0.) {
          //use displacement criterion
          Vector u1=gdisplacement[m][c];
          //+delT*gvelocity_star[m][c];
          Vector u2=Gdisplacement[m][c];
          //+delT*Gvelocity_star[m][c];
          //cout << "Integrated--node:" << c << ", u1=" << u1 << ", u2=" << u2 << endl;
          if(Dot((u2-u1),norm) >0. ) {
            Contact=1;
          } 
        }
        else { //use volume criterion
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
            for(int k=0; k<8; k++) { //loop over 8 cells around the node
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
            Contact=1;
            //cout << "Intergrated--node:" << c << ", normVol:" << normVol << endl;
          }
        }

        if(!Contact) {//no contact
          gvelocity_star[m][c]=gvelocity_star[m][c];
          gacceleration[m][c]=gacceleration[m][c];
          Gvelocity_star[m][c]=Gvelocity_star[m][c];
          Gacceleration[m][c]=Gacceleration[m][c];
          frictionWork[m][c]+=0.0;
        } 
        else { // there is contact, apply contact law
          if(crackType[m]=="null") { // do nothing
            gvelocity_star[m][c]=gvelocity_star[m][c];
            gacceleration[m][c]=gacceleration[m][c];
            Gvelocity_star[m][c]=Gvelocity_star[m][c];
            Gacceleration[m][c]=Gacceleration[m][c];
            frictionWork[m][c]+=0.0;
          }

          else if(crackType[m]=="stick") { //  assign centerofmass velocity
            gvelocity_star[m][c]=vc;
            gacceleration[m][c]=aa+(vb-va)*mb/(ma+mb)/delT;
            Gvelocity_star[m][c]=vc;
            Gacceleration[m][c]=ab+(va-vb)*ma/(ma+mb)/delT;
            frictionWork[m][c]+=0.0;
          }

          else if(crackType[m]=="frictional") { // apply friction law
            //for primary field
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
            if( fabs(ratioa)>c_mu[m] ) {  // slide
               if(ratioa>0.) mua= c_mu[m];
               if(ratioa<0.) mua=-c_mu[m];
               deltva=-(na+ta*mua)*dvan;
               gvelocity_star[m][c]=va+deltva;
               gacceleration[m][c]=aa+deltva/delT;
               frictionWork[m][c]+=ma*c_mu[m]*dvan*dvan*(fabs(ratioa)-c_mu[m]);
            }
            else {   // stick
               gvelocity_star[m][c]=vc;
               gacceleration[m][c]=aa+(vb-va)*mb/(ma+mb)/delT;
               frictionWork[m][c]+=0.0;
            }
            //for additional field
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
            if(fabs(ratiob)>c_mu[m]) { // slide
               if(ratiob>0.) mub= c_mu[m];
               if(ratiob<0.) mub=-c_mu[m];
               deltvb=-(nb+tb*mub)*dvbn;
               Gvelocity_star[m][c]=vb+deltvb;
               Gacceleration[m][c]=ab+deltvb/delT;
               frictionWork[m][c]+=mb*c_mu[m]*dvbn*dvbn*(fabs(ratiob)-c_mu[m]);
            }
            else {  // stick
               Gvelocity_star[m][c]=vc;
               Gacceleration[m][c]=ab+(va-vb)*ma/(ma+mb)/delT;
               frictionWork[m][c]+=0.0;
            }
          }

          else {
            cout<< "Unknown crack contact type in "
                << "Crack::CrackContactAdjustIntegrated: " 
                << crackType[m] << endl;
            exit(1);
          }
        } // End of if there is !contact
      } //End of loop over nodes
    } //End of loop over materials
  }  //End of loop over patches
  //time1=clock()-time0;
  //time1/=CLOCKS_PER_SEC;
  //cout << "***time for crackAdjustIntegrated = " << time1 << endl;
}

void Crack::addComputesAndRequiresMoveCrack(Task* t,
                                const PatchSet* patches,
                                const MaterialSet* matls) const
{
  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::NewDW, lb->gMassLabel,             gac,NGN);
  t->requires(Task::NewDW, lb->gVelocityStarLabel,     gac,NGN);
  t->requires(Task::NewDW, lb->GMassLabel,             gac,NGN);
  t->requires(Task::NewDW, lb->GVelocityStarLabel,     gac,NGN);

  if(d_8or27==27)
   t->requires(Task::OldDW,lb->pSizeLabel, Ghost::None);
}

void Crack::MoveCrack(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
 //double time0, time1;
 //time0 = clock();

 //move crack position patch by patch
 int numMPMMatls=d_sharedState->getNumMPMMatls();
 for(int p=0; p<patches->size(); p++){
   const Patch* patch = patches->get(p);
   delt_vartype delT;
   old_dw->get(delT, d_sharedState->get_delt_label() );

   IntVector l=patch->getNodeLowIndex();
   IntVector h=patch->getNodeHighIndex();
   Point lp=patch->nodePosition(l);
   Point hp=patch->nodePosition(h);

   for(int m = 0; m < numMPMMatls; m++){ // loop over matls    
     if(numElems[m]==0) continue; // for materials with no cracks
     MPMMaterial* mpm_matl=d_sharedState->getMPMMaterial(m);
     int dwi=mpm_matl->getDWIndex();
     ParticleSubset* pset=old_dw->getParticleSubset(dwi,patch);
     constParticleVariable<Vector> psize;
     if(d_8or27==27) 
       old_dw->get(psize,lb->pSizeLabel,pset);

     Ghost::GhostType  gac = Ghost::AroundCells;
     constNCVariable<double> gmass;
     constNCVariable<double> Gmass;
     constNCVariable<Vector> gvelocity_star;
     constNCVariable<Vector> Gvelocity_star;
     new_dw->get(gmass,         lb->gMassLabel,        dwi,patch,gac,NGP);
     new_dw->get(gvelocity_star,lb->gVelocityStarLabel,dwi,patch,gac,NGP);
     new_dw->get(Gmass,         lb->GMassLabel,        dwi,patch,gac,NGP);
     new_dw->get(Gvelocity_star,lb->GVelocityStarLabel,dwi,patch,gac,NGP);

     //move crack points
     for(int i=0; i<numPts[m]; i++) { //loop over crack points
       if(cx[m][i].x()>=lp.x() && cx[m][i].x()<hp.x() &&
          cx[m][i].y()>=lp.y() && cx[m][i].y()<hp.y() &&
          cx[m][i].z()>=lp.z() && cx[m][i].z()<hp.z() ) { // in the patch
                 
         // move crack with centerofmass velocity field
         IntVector ni[MAX_BASIS];
         double S[MAX_BASIS];
         if(d_8or27==8) 
           patch->findCellAndWeights(cx[m][i], ni, S);
         else if(d_8or27==27) 
           patch->findCellAndWeights27(cx[m][i], ni, S, psize[i]);
                
         Vector vcm = Vector(0.0,0.0,0.0);
         for(int k = 0; k < d_8or27; k++) {
           double mg = gmass[ni[k]];
           double mG = Gmass[ni[k]];
           Vector vg = gvelocity_star[ni[k]];
           Vector vG = Gvelocity_star[ni[k]];
           vcm += (mg*vg+mG*vG)/(mg+mG)*S[k];
         }
         cx[m][i] += vcm*delT;
         //cout << "i=" << i << ", cx=" << cx[m][i] << endl;
       } // End of if in patch p
     } // End of loop numPts[m]

     // update crack element normals
     for(int i=0; i<numElems[m]; i++) {
       // n3, n4, n5 three nodes of the element
       int n3=cElemNodes[m][i].x();
       int n4=cElemNodes[m][i].y();
       int n5=cElemNodes[m][i].z();
       cElemNorm[m][i]=TriangleNormal(cx[m][n3],cx[m][n4],cx[m][n5]);
     } // End of loop crack elements

   } //End of loop over matls
 } //End of loop over patches
  //time1=clock()-time0;
  //time1/=CLOCKS_PER_SEC;
  //cout << "***time for moveCrack = " << time1 << endl;
}


// private functions

// calculate outward normal of a triangle
Vector Crack::TriangleNormal(const Point& p1, 
                     const Point& p2, const Point& p3)
{
  double x21,x31,y21,y31,z21,z31;
  double a,b,c;
  Vector normal;

  x21=p2.x()-p1.x();
  x31=p3.x()-p1.x();
  y21=p2.y()-p1.y();
  y31=p3.y()-p1.y();
  z21=p2.z()-p1.z();
  z31=p3.z()-p1.z();

  a=y21*z31-z21*y31;
  b=x31*z21-z31*x21;
  c=x21*y31-y21*x31;
  if(Vector(a,b,c).length() >1.e-16)
     normal=Vector(a,b,c)/Vector(a,b,c).length();
  else
     normal=Vector(a,b,c);
  return normal;
}

//detect if two points are in same side of a plane
short Crack::NotSameSide(const Point& p, const Point& g, 
        const Point& n1, const Point& n2, const Point& n3) 
{ //p,g -- two points(particle and node)
  //n1,n2,n3 -- three points on the plane
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
        cross=1;       // p above carck
     else
        cross=2;       // p below crack
  }
  else {                // node not on crack plane
     if(dp*dg>0.) 
        cross=0;       // p, g on same side
     else if(dp>0.) 
        cross=1;       // p above, g below
     else 
        cross=2;       // p below, g above
  }
  return cross;
}

//compute signed volume of a tetrahedron
double Crack::Volume(const Point& p1, const Point& p2, 
                          const Point& p3, const Point& p)
{  //p1,p2,p3 three corners on bottom; p vertex
   double vol;
   double x1,y1,z1,x2,y2,z2,x3,y3,z3,x,y,z;

   x1=p1.x(); y1=p1.y(); z1=p1.z();
   x2=p2.x(); y2=p2.y(); z2=p2.z();
   x3=p3.x(); y3=p3.y(); z3=p3.z();
   x = p.x(); y = p.y(); z = p.z();

   vol=-(x1-x2)*(y3*z-y*z3)-(x3-x)*(y1*z2-y2*z1)
       +(y1-y2)*(x3*z-x*z3)+(y3-y)*(x1*z2-x2*z1)
       -(z1-z2)*(x3*y-x*y3)-(z3-z)*(x1*y2-x2*y1);

   if(fabs(vol)<1.e-16) return (0.);
   else return(vol);
}
  

