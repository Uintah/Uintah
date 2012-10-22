/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/********************************************************************************
    Crack.cc
    PART TWO: PARTICLE-NODE PAIR VELOCITY FIELDS 

    Created by Yajun Guo in 2002-2005.
********************************************************************************/

#include "Crack.h"
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Containers/StaticArray.h>
#include <vector>
#include <iostream>
#include <fstream>

using namespace Uintah;
using namespace std;

using std::vector;
using std::string;

#define MAX_BASIS 27

void Crack::addComputesAndRequiresParticleVelocityField(
            Task* t, const PatchSet* /*patches*/,
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

      if((int)ce[m].size()==0) { // For materials without crack
        // set pgCode[idx][k]=1
        for(ParticleSubset::iterator iter=pset->begin();
                                     iter!=pset->end();iter++) {
          for(int k=0; k<n8or27; k++) pgCode[*iter][k]=1;
        }
        // Get number of particles around a node
        for(ParticleSubset::iterator itr=psetWGCs->begin();
                           itr!=psetWGCs->end();itr++) {
          if(n8or27==8)
            patch->findCellNodes(pxWGCs[*itr], ni);
          else if(n8or27==27)
            patch->findCellNodes27(pxWGCs[*itr], ni);
          for(int k=0; k<n8or27; k++) {
            if(patch->containsNode(ni[k]))
              gNumPatls[ni[k]]++;
          }
        } 
      } // End of if(...)
      else { // For materials with crack(s)
              
        // Step 1: Detect if nodes are in crack zone
       
        Ghost::GhostType  gac = Ghost::AroundCells;
        IntVector g_cmin, g_cmax, cell_idx;
        NCVariable<short> singlevfld;
        new_dw->allocateTemporary(singlevfld,patch,gac,2*NGN);
        singlevfld.initialize(0);

        // Determine crack extent on grid (g_cmin->g_cmax)
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

        // Step 2: Detect if a particle is above, below or in the same side,
        //         and count the particles around nodes.
       
        NCVariable<int> num0,num1,num2;
        new_dw->allocateTemporary(num0, patch, gac, 2*NGN);
        new_dw->allocateTemporary(num1, patch, gac, 2*NGN);
        new_dw->allocateTemporary(num2, patch, gac, 2*NGN);
        num0.initialize(0);
        num1.initialize(0);
        num2.initialize(0);

        // Determine how particle-node segment crosses crack plane 
        for(ParticleSubset::iterator iter=pset->begin();
                              iter!=pset->end();iter++) {
          particleIndex idx=*iter;
          if(n8or27==8)
             patch->findCellNodes(px[idx], ni);
          else if(n8or27==27)
             patch->findCellNodes27(px[idx], ni);

          for(int k=0; k<n8or27; k++) {
            if(singlevfld[ni[k]]) { // for nodes in non-crack zone
              pgCode[idx][k]=0;
              num0[ni[k]]++;
            }
            else { // for nodes in crack zone
              // Detect if particles are above, below or in same side with nodes
              short  cross=SAMESIDE;
              Vector norm=Vector(0.,0.,0.);

              // Get node position even if ni[k] beyond this patch
              Point gx=patch->nodePosition(ni[k]);

              for(int i=0; i<(int)ce[m].size(); i++) {
                //Three vertices of the element
                Point n3,n4,n5;
                n3=cx[m][ce[m][i].x()];
                n4=cx[m][ce[m][i].y()];
                n5=cx[m][ce[m][i].z()];

                // If particle and node are in same side, continue
                short pgc=ParticleNodeCrackPLaneRelation(px[idx],gx,n3,n4,n5);
                if(pgc==SAMESIDE) continue;

                // Three signed volumes to see if p-g segment crosses crack
                double v3,v4,v5;
                v3=Volume(gx,n3,n4,px[idx]);
                v4=Volume(gx,n3,n5,px[idx]);
                if(v3*v4>0.) continue;
                v5=Volume(gx,n4,n5,px[idx]);
                if(v3*v5<0.) continue;

                // Particle above crack
                if(pgc==ABOVE_CRACK && v3>=0. && v4<=0. && v5>=0.) {
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
                if(pgc==BELOW_CRACK && v3<=0. && v4>=0. && v5<=0.) {
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
          } 
        } // End of loop over particles

        // Step 3: count particles around nodes in GhostCells
       
        for(ParticleSubset::iterator itr=psetWGCs->begin();
                              itr!=psetWGCs->end();itr++) {
          particleIndex idx=*itr;
          short operated=NO;
          for(ParticleSubset::iterator iter=pset->begin();
                               iter!=pset->end();iter++) {
            if(pxWGCs[idx]==px[*iter]) {
              operated=YES;
              break;
            }
          }

          if(!operated) {// Particles are in GhostCells
            if(n8or27==8)
               patch->findCellNodes(pxWGCs[idx], ni);
            else if(n8or27==27)
               patch->findCellNodes(pxWGCs[idx], ni);

            for(int k=0; k<n8or27; k++) {
              Point gx=patch->nodePosition(ni[k]);
              if(singlevfld[ni[k]]) { // for nodes in non-crack zone
                num0[ni[k]]++;
              }
              else {
                short  cross=SAMESIDE;
                Vector norm=Vector(0.,0.,0.);
                for(int i=0; i<(int)ce[m].size(); i++) { 
                  // Three vertices of each element
                  Point n3,n4,n5;
                  n3=cx[m][ce[m][i].x()];
                  n4=cx[m][ce[m][i].y()];
                  n5=cx[m][ce[m][i].z()];

                  // If particle and node in same side, continue
                  short pgc=ParticleNodeCrackPLaneRelation(pxWGCs[idx],gx,n3,n4,n5);
                  if(pgc==SAMESIDE) continue;

                  // Three signed volumes to see if p-g segment crosses crack
                  double v3,v4,v5;
                  v3=Volume(gx,n3,n4,pxWGCs[idx]);
                  v4=Volume(gx,n3,n5,pxWGCs[idx]);
                  if(v3*v4>0.) continue;
                  v5=Volume(gx,n4,n5,pxWGCs[idx]);
                  if(v3*v5<0.) continue;

                  // Particle above crack
                  if(pgc==ABOVE_CRACK && v3>=0. && v4<=0. && v5>=0.) {
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
                  if(pgc==BELOW_CRACK && v3<=0. && v4>=0. && v5<=0.) {
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

        // Step 4: Determine particle-node-pair velocity field
       
        for(ParticleSubset::iterator iter=pset->begin();
                              iter!=pset->end();iter++) {
           particleIndex idx=*iter;
           if(n8or27==8)
              patch->findCellNodes(px[idx], ni);
           else if(n8or27==27)
              patch->findCellNodes27(px[idx], ni);

           for(int k=0; k<n8or27; k++) {
             if(pgCode[idx][k]==0) {
               if((num1[ni[k]]+num2[ni[k]]==0) || num2[ni[k]]!=0)
                 pgCode[idx][k]=1;
               else if(num1[ni[k]]!=0)
                 pgCode[idx][k]=2;
               else {
                 cout << "Error: more than two velocity fields found in "
                      << "Crack::ParticleVeloccityField for node: "
                      << ni[k] << endl;
                 exit(1);
               }
             }
           } // End of loop over k
         } // End of loop patls

        for(NodeIterator iter=patch->getNodeIterator();!iter.done();iter++) {
          IntVector c = *iter;
          if(GCrackNorm[c].length()>1.e-16) // unit vector
              GCrackNorm[c]/=GCrackNorm[c].length();

          if(num1[c]+num2[c]==0) { // for nodes in non-carck zone
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
      } 
    
#if 0 
      // Output particle velocity field code
      cout << "\n*** Particle velocity field generated "
           << "in Crack::ParticleVelocityField ***" << endl;
      cout << "--patch: " << patch->getID() << endl;
      cout << "--matreial: " << m << endl;
      for(ParticleSubset::iterator iter=pset->begin();
                        iter!=pset->end();iter++) {
        particleIndex idx=*iter;
        cout << "p["<< idx << "]: " << px[idx]<< endl;
        for(int k=0; k<n8or27; k++) {
          if(pgCode[idx][k]==1)
             cout << setw(10) << "Node: " << k
                  << ",\tvfld: " << pgCode[idx][k] << endl;
          else if(pgCode[idx][k]==2)
             cout << setw(10) << "Node: " << k
                  << ",\tvfld: " << pgCode[idx][k] << " ***" << endl;
          else {
             cout << "Error: unknown particle velocity code in "
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

// Detect the relation between two points and a plane
short Crack::ParticleNodeCrackPLaneRelation(const Point& p,
                            const Point& g, const Point& n1,
                           const Point& n2, const Point& n3)
{
  // p, g     -- two points (usually particle and node)
  // n1,n2,n3 -- three points on the plane

  short cross;
  // cross=0 if p and g are on the same side of the plane,
  // cross=1 if p-g crosses the plane and p is above the plane,
  // cross=2 if p-g crosses the plane and p is below the plane.

  double x1,y1,z1,x2,y2,z2,x3,y3,z3,xp,yp,zp,xg,yg,zg;
  double x21,y21,z21,x31,y31,z31,a,b,c,d,dp,dg;

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
      cross=1;  // p above carck
    else
      cross=2;  // p below crack
  }
  else { // node not on crack plane
    if(dp*dg>0.)
      cross=0;  // p, g on same side
    else if(dp>0.)
      cross=1;  // p above, g below
    else
      cross=2;  // p below, g above
  }

  return cross;
}

// Compute signed volume of a tetrahedron
double Crack::Volume(const Point& p1, const Point& p2,
                     const Point& p3, const Point& p)
{
   // p1,p2,p3 -- three corners on bottom. p -- vertex

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

