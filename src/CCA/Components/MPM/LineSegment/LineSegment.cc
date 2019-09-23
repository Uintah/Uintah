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
#include <CCA/Components/MPM/LineSegment/LineSegment.h>
#include <CCA/Components/MPM/LineSegment/LineSegmentMaterial.h>
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
LineSegment::LineSegment(LineSegmentMaterial* tm, MPMFlags* flags,
                           MaterialManagerP& ss)
{
  d_lb = scinew MPMLabel();

  d_flags = flags;

  d_materialManager = ss;

  registerPermanentLineSegmentState(tm);
}

LineSegment::~LineSegment()
{
  delete d_lb;
}
//______________________________________________________________________
ParticleSubset* 
LineSegment::createLineSegments(LineSegmentMaterial* matl,
                                particleIndex numLineSegments,
                                const Patch* patch,
                                DataWarehouse* new_dw,
                                const string filename)
{
  int dwi = matl->getDWIndex();
  Matrix3 Identity; Identity.Identity();
  ParticleSubset* subset = allocateVariables(numLineSegments,dwi,patch,new_dw);

  if(filename!="") {
    std::ifstream is(filename.c_str());
    if (!is ){
      throw ProblemSetupException(
        "ERROR Opening line segment file "+filename+" in createLineSegments \n",
                                                            __FILE__, __LINE__);
    }

    Vector dx=patch->dCell();

    vector<double> px, py, pz;
    vector<long64> TID;
    double p1,p2,p3 = 0.0;
    string line;
    particleIndex start = 0;
    int count = 0;
    while (getline(is, line)) {
     istringstream ss(line);
     string token;
     long64 tid = count;
     ss >> token;
     p1 = stof(token);
     ss >> token;
     p2 = stof(token);
     px.push_back(p1);
     py.push_back(p2);
     pz.push_back(p3);
     TID.push_back(tid);
     count++;
    } // while lines in the file

    // make line segments from subsequent points if their midpoint is on patch
    for(unsigned int i = 0; i<px.size()-1; i++){
      Point test(0.5*(px[i]+px[i+1]),0.5*(py[i]+py[i+1]),0.5*(pz[i]+pz[i+1]));
      if(patch->containsPoint(test)){
        particleIndex pidx   = start;
        lineseg_pos[pidx]    = test;
        linesegID[pidx]      = TID[i];
        Vector lsETE = Vector(px[i+1]-px[i], py[i+1]-py[i], pz[i+1]-pz[i]);
        linesegMidToEnd[pidx] = 0.5*lsETE;
        Matrix3 size =Matrix3(lsETE.x()/dx.x(), 0.1*lsETE.y()/dx.y(), 0.0,
                              lsETE.y()/dx.x(), -.1*lsETE.x()/dx.y(), 0.0,
                                           0.0,                  0.0, 1.0);

        double Jsize = size.Determinant();
        if(Jsize <= 0.0){
          Matrix3 size =Matrix3(lsETE.x()/dx.x(), -.1*lsETE.y()/dx.y(), 0.0,
                                lsETE.y()/dx.x(), 0.1*lsETE.x()/dx.y(), 0.0,
                                             0.0,                  0.0, 1.0);
        }
        linesegSize[pidx]    = size;
        linesegDefGrad[pidx] = Identity;
        start++;
      }
    }
    
    // make a line segment from the first and last points in the file if
    // their midpoint is on patch
    int last = px.size()-1;
    Point test(0.5*(px[last]+px[0]),0.5*(py[last]+py[0]),0.5*(pz[last]+pz[0]));
    if(patch->containsPoint(test)){
      particleIndex pidx   = start;
      lineseg_pos[pidx]    = test;
      linesegID[pidx]      = TID[last];
      Vector lsETE = Vector(px[0]-px[last], 
                            py[0]-py[last], 
                            pz[0]-pz[last]);
      linesegMidToEnd[pidx]=0.5*lsETE;
      Matrix3 size =Matrix3(lsETE.x()/dx.x(), 0.1*lsETE.y()/dx.y(), 0.0,
                            lsETE.y()/dx.x(), -.1*lsETE.x()/dx.y(), 0.0,
                                         0.0,                  0.0, 1.0);
      double Jsize = size.Determinant();
      if(Jsize <= 0.0){
        Matrix3 size =Matrix3(lsETE.x()/dx.x(), -.1*lsETE.y()/dx.y(), 0.0,
                              lsETE.y()/dx.x(), 0.1*lsETE.x()/dx.y(), 0.0,
                                           0.0,                  0.0, 1.0);
      }
      linesegSize[pidx]    = size;
      linesegDefGrad[pidx] = Identity;
      start++;
    }
    is.close();
  }

  return subset;
}

//__________________________________
//
ParticleSubset* 
LineSegment::allocateVariables(particleIndex numLineSegments, 
                               int dwi, const Patch* patch,
                               DataWarehouse* new_dw)
{
  ParticleSubset* subset = 
                    new_dw->createParticleSubset(numLineSegments,dwi,patch);

  new_dw->allocateAndPut(lineseg_pos,    d_lb->pXLabel,                 subset);
  new_dw->allocateAndPut(linesegID,      d_lb->linesegIDLabel,          subset);
  new_dw->allocateAndPut(linesegSize,    d_lb->pSizeLabel,              subset);
  new_dw->allocateAndPut(linesegDefGrad, d_lb->pDeformationMeasureLabel,subset);
  new_dw->allocateAndPut(linesegMidToEnd,d_lb->lsMidToEndVectorLabel,   subset);

  return subset;
}

//__________________________________
//
particleIndex 
LineSegment::countLineSegments(const Patch* patch, const string filename)
{
  particleIndex sum = 0;

  if(filename!="") {
    std::ifstream is(filename.c_str());
    if (!is ){
      throw ProblemSetupException(
         "ERROR Opening line segment file "+filename+" in countLineSegments\n",
                                                           __FILE__, __LINE__);
    }

    string line;

    vector<double> px, py, pz;
    while (getline(is, line)) {
     istringstream ss(line);
     string token;
     double f1,f2,f3 = 0.0;
//     ss >> token;
     ss >> token;
     f1 = stof(token);
     ss >> token;
     f2 = stof(token);
//     ss >> token;
//     f3 = stof(token);
     px.push_back(f1);
     py.push_back(f2);
     pz.push_back(f3);

/*
     // Skip over fields not needed to count line segments
     ss >> token;
     ss >> token;
     ss >> token;

     ss >> token;
     ss >> token;
     ss >> token;

     ss >> token;
     ss >> token;
     ss >> token;
*/
    } // while lines in file

    // make line segments from subsequent points if their midpoint is on patch
    for(unsigned int i = 0; i<px.size()-1; i++){
      Point test(0.5*(px[i]+px[i+1]),0.5*(py[i]+py[i+1]),0.5*(pz[i]+pz[i+1]));
      if(patch->containsPoint(test)){
        sum++;
      }
    }
    
    // make a line segment from the first and last points in the file if
    // their midpoint is on patch
    int last = px.size()-1;
    Point test(0.5*(px[last]+px[0]),
               0.5*(py[last]+py[0]),
               0.5*(pz[last]+pz[0]));
    if(patch->containsPoint(test)){
      sum++;
    }

    is.close();
  }

  return sum;
}
//__________________________________
//
vector<const VarLabel* > LineSegment::returnLineSegmentState()
{
  return d_lineseg_state;
}
//__________________________________
//
vector<const VarLabel* > LineSegment::returnLineSegmentStatePreReloc()
{
  return d_lineseg_state_preReloc;
}
//__________________________________
//
void LineSegment::registerPermanentLineSegmentState(LineSegmentMaterial* lsmat)
{
  d_lineseg_state.push_back(d_lb->linesegIDLabel);
  d_lineseg_state.push_back(d_lb->pSizeLabel);
  d_lineseg_state.push_back(d_lb->pDeformationMeasureLabel);
  d_lineseg_state.push_back(d_lb->pScaleFactorLabel);
  d_lineseg_state.push_back(d_lb->lsMidToEndVectorLabel);
  d_lineseg_state_preReloc.push_back(d_lb->linesegIDLabel_preReloc);
  d_lineseg_state_preReloc.push_back(d_lb->pSizeLabel_preReloc);
  d_lineseg_state_preReloc.push_back(d_lb->pDeformationMeasureLabel_preReloc);
  d_lineseg_state_preReloc.push_back(d_lb->pScaleFactorLabel_preReloc);
  d_lineseg_state_preReloc.push_back(d_lb->lsMidToEndVectorLabel_preReloc);
}
//__________________________________
//
void LineSegment::scheduleInitialize(const LevelP& level,
                                     SchedulerP& sched, MaterialManagerP &mm)
{
  Task* t = scinew Task("LineSegment::initialize",
                  this, &LineSegment::initialize);

  t->computes(d_lb->pXLabel);
  t->computes(d_lb->pSizeLabel);
  t->computes(d_lb->linesegIDLabel);
  t->computes(d_lb->lsMidToEndVectorLabel);
  t->computes(d_lb->pDeformationMeasureLabel);

  sched->addTask(t, level->eachPatch(), mm->allMaterials("LineSegment"));
}

//__________________________________
//
void LineSegment::initialize(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset* lineseg_matls,
                             DataWarehouse* ,
                             DataWarehouse* new_dw)
{
  particleIndex totalLineSegments=0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    //printTask(patches,patch,cout_doing,"Doing initialize for LineSegments\t");

    for(int m=0;m<lineseg_matls->size();m++){
      LineSegmentMaterial* lineseg_matl = 
        (LineSegmentMaterial*) d_materialManager->getMaterial("LineSegment", m);
      string filename = lineseg_matl->getLineSegmentFilename();
      particleIndex numLineSegments = countLineSegments(patch,filename);
      totalLineSegments+=numLineSegments;

      createLineSegments(lineseg_matl, numLineSegments,
                    patch, new_dw, filename);
    }
    cout << "Total Line Segments " << totalLineSegments << endl;
  }
}
