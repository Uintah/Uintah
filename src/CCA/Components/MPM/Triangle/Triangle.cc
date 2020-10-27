/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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
#include <CCA/Components/MPM/Triangle/Triangle.h>
#include <CCA/Components/MPM/Triangle/TriangleMaterial.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Patch.h>
#include <Core/Math/Matrix3.h>
#include <fstream>

using namespace Uintah;
using namespace std;

//______________________________________________________________________
Triangle::Triangle(TriangleMaterial* tm, MPMFlags* flags, MaterialManagerP& ss)
{
  d_lb = scinew MPMLabel();

  d_flags = flags;

  d_materialManager = ss;

  registerPermanentTriangleState(tm);
}

Triangle::~Triangle()
{
  delete d_lb;
}
//______________________________________________________________________
ParticleSubset* 
Triangle::createTriangles(TriangleMaterial* matl,
                          particleIndex numTriangles,
                          const Patch* patch,
                          DataWarehouse* new_dw,
                          const string fileroot)
{
  int dwi = matl->getDWIndex();
  Vector dx = patch->dCell();
  double gridLength = dx.minComponent();
  Matrix3 Identity; Identity.Identity();
  ParticleSubset* subset = allocateVariables(numTriangles,dwi,patch,new_dw);
  ParticleSubset::iterator iter = subset->begin();
  for(;iter != subset->end(); iter++){
       particleIndex idx = *iter;
    triangleAreaAtNodes[idx] = Vector(0.,0.,0.0);
  }

  // Open tri and pts files
  string ptsfilename = fileroot + ".pts";
  string trifilename = fileroot + ".tri";

  std::ifstream pts(ptsfilename.c_str());
  if (!pts ){
    throw ProblemSetupException(
      "ERROR Opening pts file "+ptsfilename+" in createTriangles \n",
                                                         __FILE__, __LINE__);
  }

  std::ifstream tri(trifilename.c_str());
  if (!tri ){
    throw ProblemSetupException(
      "ERROR Opening tri file "+trifilename+" in createTriangles \n",
                                                         __FILE__, __LINE__);
  }

  // Read in pts files
  vector<double> px, py, pz;
  double p1,p2,p3 = 0.0;
  particleIndex start = 0;
  int numpts = 0;
  while (pts >> p1 >> p2 >> p3) {
   px.push_back(p1);
   py.push_back(p2);
   pz.push_back(p3);
   numpts++;
  } // while lines in the pts file

  // Create a set to hold the triangle indices for each point
  // Also, create a place to hold the "vertex area" for each point
  vector<set<int> > triangles(numpts);
  vector<double> ptArea(numpts);

  // Read in tri file
  // Put the triangle index into a set 
  // Keep track of which triangles each point is part of by inserting
  // the triangle index into each set
  // Compute the area of each triangle
  vector<int> i0, i1, i2;
  vector<long64> TID;
  int ip0,ip1,ip2;
  unsigned int numtri = 0;
  vector<double> triAreaNow(numtri);
  while (tri >> ip0 >> ip1 >> ip2) {
   long64 tid = numtri;
   i0.push_back(ip0);
   i1.push_back(ip1);
   i2.push_back(ip2);
   TID.push_back(tid);
   triangles[ip0].insert(numtri);
   triangles[ip1].insert(numtri);
   triangles[ip2].insert(numtri);
   Point P0(px[ip0], py[ip0], pz[ip0]);
   Point P1(px[ip1], py[ip1], pz[ip1]);
   Point P2(px[ip2], py[ip2], pz[ip2]);
   Vector A = P1-P0;
   Vector B = P2-P0;
   Vector C = P2-P1;
   if(A.length() > 3.*gridLength || 
      B.length() > 3.*gridLength ||
      C.length() > 3.*gridLength){
    ostringstream warn;
    warn <<"Triangle " << numtri << " in " << trifilename 
         << " is too large relative to the grid cell size\n" 
         << "Its points are = \n"
         << P0 << "\n" << P1 << "\n" << P2 << "\n";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
   }
   triAreaNow.push_back(0.5*Cross(A,B).length());
   numtri++;
  } // while lines in the tri file

  // Compute the area at each vertex by putting 1/3 of the area of each
  // triangle that touches a point into ptArea
  double totalArea = 0;
  for(int i = 0; i<numpts; i++){
    for (set<int>::iterator it1 = triangles[i].begin(); 
                            it1!= triangles[i].end();  it1++){
     ptArea[i]+=triAreaNow[*it1]/3.; 
    }
    totalArea+=ptArea[i];
    if(triangles[i].size() > 30){
       cout << "This node has " << triangles[i].size() << " triangles" << endl;
    }
  } // while lines in the pts file

  vector<int>       useInPen(numpts,1);
  vector<IntVector> useInPenVector(numtri,IntVector(99,99,99));

  // make triangles from subsequent points if their midpoint is on patch
  // Choose which points to use in penalty contact so that each point is
  // used just once.
  for(unsigned int i = 0; i<numtri; i++){
      useInPenVector[i] = IntVector(useInPen[i0[i]],
                                    useInPen[i1[i]],
                                    useInPen[i2[i]]);
      useInPen[i0[i]]=0;
      useInPen[i1[i]]=0;
      useInPen[i2[i]]=0;
  }

  for(unsigned int i = 0; i<numtri; i++){
    Point P0(px[i0[i]], py[i0[i]], pz[i0[i]]);
    Point P1(px[i1[i]], py[i1[i]], pz[i1[i]]);
    Point P2(px[i2[i]], py[i2[i]], pz[i2[i]]);
    Point test((px[i0[i]]+px[i1[i]]+px[i2[i]])/3.,
               (py[i0[i]]+py[i1[i]]+py[i2[i]])/3.,
               (pz[i0[i]]+pz[i1[i]]+pz[i2[i]])/3.);
    if(patch->containsPoint(test)){
      particleIndex pidx   = start;
      triangle_pos[pidx]   = test;
      triangleID[pidx]     = TID[i];
      triangleArea[pidx]   = triAreaNow[i];
      triangleMidToNode0[pidx] = P0 - test;
      triangleMidToNode1[pidx] = P1 - test;
      triangleMidToNode2[pidx] = P2 - test;

      triangleAreaAtNodes[pidx]+=Vector(ptArea[i0[i]], 
                                        ptArea[i1[i]], 
                                        ptArea[i2[i]]);

      triangleUseInPenalty[pidx] = useInPenVector[i];

      triangleSize[pidx]    = Identity;
      triangleDefGrad[pidx] = Identity;
      start++;
#if 0
      Vector r0 = P1 - P0;
      Vector r1 = P2 - P0;
      Vector r2 = -.1*Cross(r1,r0);
      Matrix3 size =Matrix3(r0.x()/dx.x(), r1.x()/dx.x(), r2.x()/dx.x(),
                            r0.y()/dx.y(), r1.y()/dx.y(), r2.y()/dx.y(),
                            r0.z()/dx.z(), r1.z()/dx.z(), r2.z()/dx.z());

      double Jsize = size.Determinant();
      if(Jsize <= 0.0){
       cout << "negative J" << endl;
      }
      triangleSize[pidx]    = size;
#endif
    }
  }

  tri.close();
  pts.close();

  return subset;
}

//__________________________________
//
ParticleSubset* 
Triangle::allocateVariables(particleIndex numTriangles, 
                               int dwi, const Patch* patch,
                               DataWarehouse* new_dw)
{
  ParticleSubset* subset =new_dw->createParticleSubset(numTriangles,dwi,patch);

  new_dw->allocateAndPut(triangle_pos,   d_lb->pXLabel,                 subset);
  new_dw->allocateAndPut(triangleID,     d_lb->triangleIDLabel,         subset);
  new_dw->allocateAndPut(triangleSize,   d_lb->pSizeLabel,              subset);
  new_dw->allocateAndPut(triangleDefGrad,d_lb->pDeformationMeasureLabel,subset);
  new_dw->allocateAndPut(triangleMidToNode0,
                                         d_lb->triMidToN0VectorLabel,   subset);
  new_dw->allocateAndPut(triangleMidToNode1,
                                         d_lb->triMidToN1VectorLabel,   subset);
  new_dw->allocateAndPut(triangleMidToNode2,
                                         d_lb->triMidToN2VectorLabel,   subset);
  new_dw->allocateAndPut(triangleUseInPenalty,
                                         d_lb->triUseInPenaltyLabel,    subset);
  new_dw->allocateAndPut(triangleArea,   d_lb->triAreaLabel,            subset);
  new_dw->allocateAndPut(triangleAreaAtNodes,
                                         d_lb->triAreaAtNodesLabel,     subset);

  return subset;
}

//__________________________________
//
particleIndex 
Triangle::countTriangles(const Patch* patch, const string fileroot)
{
  particleIndex sum = 0;

  string ptsfilename = fileroot + ".pts";
  string trifilename = fileroot + ".tri";

  std::ifstream pts(ptsfilename.c_str());
  if (!pts ){
    throw ProblemSetupException(
      "ERROR Opening pts file "+ptsfilename+" in countTriangles \n",
                                                         __FILE__, __LINE__);
  }

  std::ifstream tri(trifilename.c_str());
  if (!tri ){
    throw ProblemSetupException(
      "ERROR Opening tri file "+trifilename+" in countTriangles \n",
                                                         __FILE__, __LINE__);
  }

  vector<double> px, py, pz;
  double p1,p2,p3 = 0.0;
  int numpts = 0;
  while (pts >> p1 >> p2 >> p3) {
    px.push_back(p1);
    py.push_back(p2);
    pz.push_back(p3);
    numpts++;
  } // while lines in the pts file

  vector<int> i0, i1, i2;
  vector<long64> TID;
  int ip0,ip1,ip2;
  unsigned int numtri = 0;
  while (tri >> ip0 >> ip1 >> ip2) {
    long64 tid = numtri;
    i0.push_back(ip0);
    i1.push_back(ip1);
    i2.push_back(ip2);
    TID.push_back(tid);
    numtri++;
  } // while lines in the tri file

  // make triangles from the three tri points if their midpoint is on patch
  for(unsigned int i = 0; i<numtri; i++){
    Point test((px[i0[i]]+px[i1[i]]+px[i2[i]])/3.,
               (py[i0[i]]+py[i1[i]]+py[i2[i]])/3.,
               (pz[i0[i]]+pz[i1[i]]+pz[i2[i]])/3.);
    
    if(patch->containsPoint(test)){
      sum++;
    }
  }
  
  tri.close();
  pts.close();

  return sum;
}
//__________________________________
//
vector<const VarLabel* > Triangle::returnTriangleState()
{
  return d_triangle_state;
}
//__________________________________
//
vector<const VarLabel* > Triangle::returnTriangleStatePreReloc()
{
  return d_triangle_state_preReloc;
}
//__________________________________
//
void Triangle::registerPermanentTriangleState(TriangleMaterial* lsmat)
{
  d_triangle_state.push_back(d_lb->triangleIDLabel);
  d_triangle_state.push_back(d_lb->pSizeLabel);
  d_triangle_state.push_back(d_lb->pDeformationMeasureLabel);
  d_triangle_state.push_back(d_lb->pScaleFactorLabel);
  d_triangle_state.push_back(d_lb->triMidToN0VectorLabel);
  d_triangle_state.push_back(d_lb->triMidToN1VectorLabel);
  d_triangle_state.push_back(d_lb->triMidToN2VectorLabel);
  d_triangle_state.push_back(d_lb->triUseInPenaltyLabel);
  d_triangle_state.push_back(d_lb->triAreaAtNodesLabel);
  d_triangle_state.push_back(d_lb->triAreaLabel);

  d_triangle_state_preReloc.push_back(d_lb->triangleIDLabel_preReloc);
  d_triangle_state_preReloc.push_back(d_lb->pSizeLabel_preReloc);
  d_triangle_state_preReloc.push_back(d_lb->pDeformationMeasureLabel_preReloc);
  d_triangle_state_preReloc.push_back(d_lb->pScaleFactorLabel_preReloc);
  d_triangle_state_preReloc.push_back(d_lb->triMidToN0VectorLabel_preReloc);
  d_triangle_state_preReloc.push_back(d_lb->triMidToN1VectorLabel_preReloc);
  d_triangle_state_preReloc.push_back(d_lb->triMidToN2VectorLabel_preReloc);
  d_triangle_state_preReloc.push_back(d_lb->triUseInPenaltyLabel_preReloc);
  d_triangle_state_preReloc.push_back(d_lb->triAreaAtNodesLabel_preReloc);
  d_triangle_state_preReloc.push_back(d_lb->triAreaLabel_preReloc);
}
//__________________________________
//
void Triangle::scheduleInitialize(const LevelP& level,
                                     SchedulerP& sched, MaterialManagerP &mm)
{
  Task* t = scinew Task("Triangle::initialize",
                  this, &Triangle::initialize);

  t->computes(d_lb->pXLabel);
  t->computes(d_lb->pSizeLabel);
  t->computes(d_lb->triangleIDLabel);
  t->computes(d_lb->triMidToN0VectorLabel);
  t->computes(d_lb->triMidToN1VectorLabel);
  t->computes(d_lb->triMidToN2VectorLabel);
  t->computes(d_lb->triUseInPenaltyLabel);
  t->computes(d_lb->triAreaLabel);
  t->computes(d_lb->triAreaAtNodesLabel);
  t->computes(d_lb->pDeformationMeasureLabel);
  t->computes(d_lb->triangleCountLabel);

  sched->addTask(t, level->eachPatch(), mm->allMaterials("Triangle"));
}

//__________________________________
//
void Triangle::initialize(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset* triangle_matls,
                             DataWarehouse* ,
                             DataWarehouse* new_dw)
{
  particleIndex totalTriangles=0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    //printTask(patches,patch,cout_doing,"Doing initialize for Triangles\t");

    for(int m=0;m<triangle_matls->size();m++){
      TriangleMaterial* triangle_matl = 
        (TriangleMaterial*) d_materialManager->getMaterial("Triangle", m);
      string filename = triangle_matl->getTriangleFilename();
      particleIndex numTriangles = countTriangles(patch,filename);
      totalTriangles+=numTriangles;

      createTriangles(triangle_matl, numTriangles,
                      patch, new_dw, filename);
    }
    new_dw->put(sumlong_vartype(totalTriangles), d_lb->triangleCountLabel);
//    cout << "Total Triangles =  " << totalTriangles << endl;
  }
}
