/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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


#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Exceptions/InvalidGrid.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Math/UintahMiscMath.h>
#include <Core/Math/Primes.h>
#include <TauProfilerForSCIRun.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Geometry/BBox.h>
#include <Core/Math/MiscMath.h>
#include <iostream>
#include <sci_values.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;


struct StretchSpec {
  string shape;
  double from;
  double to;
  double fromSpacing;
  double toSpacing;

  int countCells() const;
  void fillCells(int& start, int lowCells, int highCells, OffsetArray1<double>& faces) const;
};

#ifdef _WIN32
inline double remainder(double x,double y) 
{
  double z = x/y;
  int mult = (int) z+.5;
  return x-y*mult;
}
#endif

int StretchSpec::countCells() const
{
  if(shape == "uniform"){
    return Round((to-from)/fromSpacing);
  } else {
    double a = Min(fromSpacing, toSpacing);
    double b = Max(fromSpacing, toSpacing);
    double totalDistance = to-from;
    double nn = log(b/a) / log((totalDistance+b)/(totalDistance+a)) -1;
    int n = (int)(nn+0.5);
    return n;
  }
}

static double xk(double a, double r, int k)
{
  if(r == 1)
    return a*k;
  return a*r*(1-pow(r, k))/(1-r);
}

void StretchSpec::fillCells(int& start, int lowExtra, int highExtra, OffsetArray1<double>& faces) const
{
  if(shape == "uniform"){
    int n = Round((to-from)/fromSpacing);
    for(int i=-lowExtra;i<n+highExtra;i++){
      faces[i+start] = from + i*fromSpacing;
    }
    start += n;
  } else {
    int n = countCells();
    double totalDistance = to-from;
    double a = fromSpacing;
    double b = toSpacing;
    bool switched = false;
    if(a > b){
      double tmp = a;
      a = b;
      b = tmp;
      switched = true;
    }

    double r = pow(b/a, 1./(n+1));

    // Now adjust the rate to ensure that there are an integer number of cells
    // We use a binary search because a newton solve doesn't alway converge very well,
    // and this is not performance critical
    double r1 = 1;
    double r2 = r * r * 2;
    for(int i=0;i<1000;i++){
      double newr = (r1+r2)/2;
      if(r == newr)
        break;
      r = newr;
      double residual = xk(a, r, n) - totalDistance;
      if(residual > 0)
        r2 = r;
      else
        r1 = r;
    }
    if(switched){
      a = a*pow(r, n+1);
      r = 1./r;
    }
    for(int i=-lowExtra;i<n+highExtra;i++){
      faces[i+start] = from + xk(a, r, i);
    }
    start += n;
  }
}

Grid::Grid()
{
  // Initialize values that may be uses for the autoPatching calculations
  af_ = 0;
  bf_ = 0;
  cf_ = 0;
  nf_ = -1;
  ares_ = 0;
  bres_ = 0;
  cres_ = 0;
}

Grid::~Grid()
{
}

int Grid::numLevels() const
{
  return (int)d_levels.size();
}

const LevelP& Grid::getLevel( int l ) const
{
  ASSERTRANGE(l, 0, numLevels());
  return d_levels[ l ];
}

Level* Grid::addLevel(const Point& anchor, const Vector& dcell, int id)
{
  // find the new level's refinement ratio
  // this should only be called when a new grid is created, so if this level index 
  // is > 0, then there is a coarse-fine relationship between this level and the 
  // previous one.

  IntVector ratio;
  if (d_levels.size() > 0) {
    Vector r = (d_levels[d_levels.size()-1]->dCell() / dcell) + Vector(1e-6, 1e-6, 1e-6);
    ratio = IntVector((int)r.x(), (int)r.y(), (int)r.z());
    Vector diff = r - ratio.asVector();
    if (diff.x() > 1e-5 || diff.y() > 1e-5 || diff.z() > 1e-5) {
      // non-integral refinement ratio
      ostringstream out;
      out << "Non-integral refinement ratio: " << r;
      throw InvalidGrid(out.str().c_str(), __FILE__, __LINE__);
    }
  }
  else
    ratio = IntVector(1,1,1);


  Level* level = scinew Level(this, anchor, dcell, (int)d_levels.size(), ratio, id);  

  d_levels.push_back( level );
  return level;
}

void Grid::performConsistencyCheck() const
{
#if SCI_ASSERTION_LEVEL > 0
  TAU_PROFILE("Grid::performConsistencyCheck()", " ", TAU_USER);
  // Verify that patches on a single level do not overlap
  for(int i=0;i<(int)d_levels.size();i++)
    d_levels[i]->performConsistencyCheck();

  // Check overlap between levels
  // See if patches on level 0 form a connected set (warning)
  // Compute total volume - compare if not first time

  //cerr << "Grid::performConsistencyCheck not done\n";
  
  //__________________________________
  //  bullet proofing with multiple levels
  if(d_levels.size() > 0) {
    for(int i=0;i<(int)d_levels.size() -1 ;i++) {
      LevelP level     = d_levels[i];
      LevelP fineLevel = level->getFinerLevel();
      //Vector dx_level     = level->dCell();
      Vector dx_fineLevel = fineLevel->dCell();
      
      //__________________________________
      // finer level can't lay outside of the coarser level
      BBox C_box,F_box;
      level->getSpatialRange(C_box);
      fineLevel->getSpatialRange(F_box);
      
      Point Cbox_min = C_box.min();
      Point Cbox_max = C_box.max(); 
      Point Fbox_min = F_box.min();
      Point Fbox_max = F_box.max();
      
      if(Fbox_min.x() < Cbox_min.x() ||
         Fbox_min.y() < Cbox_min.y() ||
         Fbox_min.z() < Cbox_min.z() ||
         Fbox_max.x() > Cbox_max.x() ||
         Fbox_max.y() > Cbox_max.y() ||
         Fbox_max.z() > Cbox_max.z() ) {
        ostringstream desc;
        desc << " The finer Level " << fineLevel->getIndex()
             << " "<< F_box.min() << " "<< F_box.max()
             << " can't lay outside of coarser level " << level->getIndex()
             << " "<< C_box.min() << " "<< C_box.max() << endl;
        throw InvalidGrid(desc.str(),__FILE__,__LINE__);
      }
      //__________________________________
      //  finer level must have a box width that is
      //  an integer of the cell spacing
      Vector integerTest_min(remainder(Fbox_min.x(),dx_fineLevel.x() ), 
                             remainder(Fbox_min.y(),dx_fineLevel.y() ),
                             remainder(Fbox_min.z(),dx_fineLevel.z() ) );
                             
      Vector integerTest_max(remainder(Fbox_max.x(),dx_fineLevel.x() ), 
                             remainder(Fbox_max.y(),dx_fineLevel.y() ),
                             remainder(Fbox_max.z(),dx_fineLevel.z() ) );
      
      Vector distance = Fbox_max.asVector() - Fbox_min.asVector();
      
      Vector integerTest_distance(remainder(distance.x(), dx_fineLevel.x() ),
                                  remainder(distance.y(), dx_fineLevel.y() ),
                                  remainder(distance.z(), dx_fineLevel.z() ) );
      Vector smallNum(1e-14,1e-14,1e-14);
      
      if( (integerTest_min >smallNum || integerTest_max > smallNum) && 
           integerTest_distance > smallNum){
        ostringstream desc;
        desc << " The finer Level " << fineLevel->getIndex()
             << " "<< Fbox_min << " "<< Fbox_max
             << " upper or lower limits are not divisible by the cell spacing "
             << dx_fineLevel << " \n Remainder of level box/dx: lower" 
             << integerTest_min << " upper " << integerTest_max<< endl;
        throw InvalidGrid(desc.str(),__FILE__,__LINE__);
      } 
    }
  }
#endif
}

void Grid::printStatistics() const
{
  cout << "Grid statistics:\n";
  cout << "Number of levels:\t\t" << numLevels() << '\n';
  unsigned long totalCells = 0;
  unsigned long totalPatches = 0;
  for(int i=0;i<numLevels();i++){
    LevelP l = getLevel(i);
    cout << "Level " << i << ":\n";
    if (l->getPeriodicBoundaries() != IntVector(0,0,0))
      cout << "  Periodic boundaries:\t\t" << l->getPeriodicBoundaries()
           << '\n';
    cout << "  Number of patches:\t\t" << l->numPatches() << '\n';
    totalPatches += l->numPatches();
    double ppc = double(l->totalCells())/double(l->numPatches());
    cout << "  Total number of cells:\t" << l->totalCells() << " (" << ppc << " avg. per patch)\n";
    totalCells += l->totalCells();
  }
  cout << "Total patches in grid:\t\t" << totalPatches << '\n';
  double ppc = double(totalCells)/double(totalPatches);
  cout << "Total cells in grid:\t\t" << totalCells << " (" << ppc << " avg. per patch)\n";
  cout << "\n";
}

//////////
// Computes the physical boundaries for the grid
void Grid::getSpatialRange(BBox& b) const
{
  // just call the same function for all the levels
  for(int l=0; l < numLevels(); l++) {
    getLevel(l)->getSpatialRange(b);
  }
}

////////// 
// Returns the boundary of the grid exactly (without
// extra cells).  The value returned is the same value
// as found in the .ups file.
void Grid::getInteriorSpatialRange(BBox& b) const
{
  // just call the same function for all the levels
  for(int l=0; l < numLevels(); l++) {
    getLevel(l)->getInteriorSpatialRange(b);
  }
}


//__________________________________
// Computes the length in each direction of the grid
void Grid::getLength(Vector& length, const string flag) const
{
  BBox b;
  // just call the same function for all the levels
  for(int l=0; l < numLevels(); l++) {
    getLevel(l)->getSpatialRange(b);
  }
  length = ( b.max() - b.min() );
  if (flag == "minusExtraCells") {
    Vector dx = getLevel(0)->dCell();
    IntVector extraCells = getLevel(0)->getExtraCells();
    Vector ec_length = IntVector(2,2,2) * extraCells * dx;
    length = ( b.max() - b.min() )  - ec_length;
  }
}

void 
Grid::problemSetup(const ProblemSpecP& params, const ProcessorGroup *pg, bool do_amr)
{
   ProblemSpecP grid_ps = params->findBlock("Grid");
   if(!grid_ps)
      return;

   // anchor/highpoint on the grid
   Point anchor(DBL_MAX, DBL_MAX, DBL_MAX);

   // time refinement between a level and the previous one

   int levelIndex = 0;

   for(ProblemSpecP level_ps = grid_ps->findBlock("Level");
       level_ps != 0; level_ps = level_ps->findNextBlock("Level")){
      // Make two passes through the boxes.  The first time, we
      // want to find the spacing and the lower left corner of the
      // problem domain.  Spacing can be specified with a dx,dy,dz
      // on the level, or with a resolution on the patch.  If a
      // resolution is used on a problem with more than one patch,
      // the resulting grid spacing must be consistent.

      // anchor/highpoint on the level
      Point levelAnchor(DBL_MAX, DBL_MAX, DBL_MAX);
      Point levelHighPoint(-DBL_MAX, -DBL_MAX, -DBL_MAX);

      Vector spacing;
      bool have_levelspacing=false;

      if(level_ps->get("spacing", spacing))
        have_levelspacing=true;
      bool have_patchspacing=false;
        

      // first pass - find upper/lower corner, find resolution/spacing and extraCells
      IntVector extraCells(0, 0, 0);
      for(ProblemSpecP box_ps = level_ps->findBlock("Box");
         box_ps != 0; box_ps = box_ps->findNextBlock("Box")){
        Point lower;
        box_ps->require("lower", lower);
        Point upper;
        box_ps->require("upper", upper);
        if (levelIndex == 0) {
          anchor=Min(lower, anchor);
        }
        levelAnchor=Min(lower, levelAnchor);
        levelHighPoint=Max(upper, levelHighPoint);
        
        IntVector resolution;
        if(box_ps->get("resolution", resolution)){
           if(have_levelspacing){
              throw ProblemSetupException("Cannot specify level spacing and patch resolution", 
                                          __FILE__, __LINE__);
           } else {
              // all boxes on same level must have same spacing
              Vector newspacing = (upper-lower)/resolution;
              if(have_patchspacing){
                Vector diff = spacing-newspacing;
                if(diff.length() > 1.e-14)
                   throw ProblemSetupException("Using patch resolution, and the patch spacings are inconsistent",
                                               __FILE__, __LINE__);
              } else {
                spacing = newspacing;
              }
              have_patchspacing=true;
           }
        }
        
        IntVector ec;
        box_ps->getWithDefault("extraCells", ec, IntVector(0,0,0));
        extraCells = Max(ec, extraCells);
        
        // bulletproofing
        if(have_levelspacing || have_patchspacing){
          for(int dir = 0; dir<3; dir++){
            if ((upper(dir)-lower(dir)) <= 0.0) {
              ostringstream msg;
              msg<< "\nComputational Domain Input Error: Level("<< levelIndex << ")"
                 << " \n The computational domain " << lower<<", " << upper
                 << " must have a positive distance in each coordinate direction  " << upper-lower << endl; 
              throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
            }
          
            if (spacing[dir] > (upper(dir)-lower(dir)) || spacing[dir] < 0){
              ostringstream msg;
              msg<< "\nComputational Domain Input Error: Level("<< levelIndex << ")"
                 << " \n The spacing " << spacing 
                 << " must be less than the upper - lower corner and positive " << upper << endl; 
              throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
            }
          }
        }
      }  // boxes loop

      // Look for stretched grid info
      vector<StretchSpec> stretch[3];
      for(ProblemSpecP stretch_ps = level_ps->findBlock("Stretch");
          stretch_ps != 0; stretch_ps = stretch_ps->findNextBlock("Stretch")){
        string axisName;
        if(!stretch_ps->getAttribute("axis", axisName))
          throw ProblemSetupException("Error, no specified axis for Stretch section: should be x, y, or z", __FILE__, __LINE__);
        int axis;
        if(axisName == "x" || axisName == "X")
          axis = 0;
        else if(axisName == "y" || axisName == "Y")
          axis = 1;
        else if(axisName == "z" || axisName == "Z")
          axis = 2;
        else {
          ostringstream msg;
          msg << "Error, invalid axis in Stretch section: " << axisName << ", should be x, y, or z";
          throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
        }
        if(stretch[axis].size() != 0){
          ostringstream msg;
          msg << "Error, stretch axis already specified: " << axisName;
          throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
        }
        double begin = -DBL_MAX;
        for(ProblemSpecP region_ps = stretch_ps->findBlock();
            region_ps != 0; region_ps = region_ps->findNextBlock()){
          StretchSpec spec;
          spec.shape = region_ps->getNodeName();
          if(spec.shape != "uniform" && region_ps->getNodeName() != "linear") {
            ostringstream msg;
            msg << "Error, invalid region shape in stretched grid: " << region_ps->getNodeName();
            throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
          }
          if(region_ps->getAttribute("from", spec.from)){
            if(spec.from < begin) {
              ostringstream msg;
              msg << "Error, stretched grid regions must be specified in ascending order";
              throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
            }
          } else {
            spec.from = DBL_MAX;
          }
          if(region_ps->getAttribute("to", spec.to)){
            begin = spec.to;
          } else {
            spec.to = DBL_MAX;
          }
          if(spec.shape == "uniform"){
            if(!region_ps->getAttribute("spacing", spec.fromSpacing))
              spec.fromSpacing = DBL_MAX;
            spec.toSpacing = spec.fromSpacing;
          } else if(spec.shape == "linear"){
            if(!region_ps->getAttribute("fromSpacing", spec.fromSpacing))
              spec.fromSpacing = DBL_MAX;
            if(!region_ps->getAttribute("toSpacing", spec.toSpacing))
              spec.toSpacing = DBL_MAX;
          }
          stretch[axis].push_back(spec);
          begin = spec.to;
        }
      }

      // Check the stretched grid specification
      int stretch_count = 0;
      for(int axis=0;axis<3;axis++){
        if(stretch[axis].size()){
          stretch_count++;
          spacing[axis] = getNan();
        }
        for(int i=0;i<(int)stretch[axis].size();i++){
          StretchSpec& spec = stretch[axis][i];
          if(spec.from == DBL_MAX){
            if(i > 0 && stretch[axis][i-1].to != DBL_MAX){
              spec.from = stretch[axis][i-1].to;
            } else if(i == 0){
              spec.from = levelAnchor(axis);
            } else {
              ostringstream msg;
              msg << "Stretch region from point not specified for region " << i << '\n';
              throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
            }
          }
          if(spec.to == DBL_MAX){
            if(i < (int)stretch[axis].size()-1 && stretch[axis][i+1].from != DBL_MAX){
              spec.to = stretch[axis][i+1].from;
            } else if(i == (int)stretch[axis].size()-1){
              spec.to = levelHighPoint(axis);
            } else {
              ostringstream msg;
              msg << "Stretch region to point not specified for region " << i << '\n';
              throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
            }
          }
          if(spec.fromSpacing <= 0 || spec.toSpacing <= 0){
            ostringstream msg;
            msg << "Grid spacing must be >= 0 (" << spec.fromSpacing << ", " << spec.toSpacing << ")";
            throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
          }

          if(spec.shape == "linear"){
            if(spec.fromSpacing == DBL_MAX){
              if(i > 0 && stretch[axis][i-1].toSpacing != DBL_MAX){
                spec.fromSpacing = stretch[axis][i-1].toSpacing;
              } else {
                ostringstream msg;
                msg << "Stretch region from spacing not specified for region " << i << '\n';
                throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
              }
            }
            if(spec.toSpacing == DBL_MAX){
              if(i < (int)stretch[axis].size()-1 && stretch[axis][i+1].fromSpacing != DBL_MAX){
                spec.toSpacing = stretch[axis][i+1].toSpacing;
              } else {
                ostringstream msg;
                msg << "Stretch region to spacing not specified";
                throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
              }
            }
          } else if(spec.shape == "uniform"){
            if(spec.fromSpacing == DBL_MAX){
              if(i > 0 && stretch[axis][i-1].toSpacing != DBL_MAX){
                spec.fromSpacing = spec.toSpacing = stretch[axis][i-1].toSpacing;
              } else {
                ostringstream msg;
                msg << "Stretch region uniform spacing not specified for region " << i << '\n';
                throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
              }
            }
          }
          if(i > 0 && stretch[axis][i-1].toSpacing != spec.fromSpacing){
            if(pg->myrank() == 0){
              cerr << "WARNING: specifying two uniform sections with a different spacing can cause erroneous grid (" << stretch[axis][i-1].toSpacing << ", " << spec.fromSpacing << "\n";
            }
          }
          if(i > 0 && stretch[axis][i-1].to != spec.from){
            ostringstream msg;
            msg << "Gap in strech region from: " << stretch[axis][i-1].to << " to " << spec.from;
            throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
          }
          if(spec.to < spec.from) {
            ostringstream msg;
            msg << "Error, stretched grid to must be larger then from";
            throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
          }
          if(spec.shape == "linear"){
            // If toSpacing == fromSpacing, then convert this into a uniform section, since
            // the grid generation numerics have a singularity at that point
            if(spec.fromSpacing == spec.toSpacing){
              spec.shape = "uniform";
            }
          }
          if(spec.shape == "uniform"){
            // Check that dx goes nicely into the range
            double ncells = (spec.to - spec.from)/spec.fromSpacing;
            if(Fraction(ncells) > 1e-4 && Fraction(ncells) < 1-1e-4){
              ostringstream msg;
              msg << "Error, uniform region not an integer multiple of the cell spacing";
              throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
            }
            int n = Round(ncells);
            // Recompute newdx to avoid roundoff issues
            double newdx = (spec.to - spec.from)/n;
            spec.toSpacing = spec.fromSpacing = newdx;
          }
        }

        if(stretch[axis].size() > 0){
          if(stretch[axis][0].from > levelAnchor(axis) || stretch[axis][stretch[axis].size()-1].to < levelHighPoint(axis)){
            ostringstream msg;
            msg << "Error, stretched grid specification does not cover entire axis";
            throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
          }
        }
      }

      if(pg->myrank() == 0 && stretch_count != 0){
        cerr << "Stretched grid information:\n";
        for(int axis=0;axis<3;axis++){
          if(axis == 0)
            cerr << "x";
          else if(axis == 1)
            cerr << "y";
          else
            cerr << "z";
          cerr << " axis\n";
          for(int i=0;i<(int)stretch[axis].size();i++){
            StretchSpec& spec = stretch[axis][i];
            cerr << spec.shape << ": from " << spec.from << "(" << spec.fromSpacing << ") to " << spec.to << "(" << spec.toSpacing << "), " << spec.countCells() << " cells\n";
          }
        }
        cerr << "\n";
      }

      if(!have_levelspacing && !have_patchspacing && stretch_count != 3)
        throw ProblemSetupException("Box resolution is not specified", __FILE__, __LINE__);

      LevelP level = addLevel(anchor, spacing);

      // Determine the interior cell limits.  For no extraCells, the limits
      // will be the same.  For extraCells, the interior cells will have
      // different limits so that we can develop a CellIterator that will
      // use only the interior cells instead of including the extraCell
      // limits.
      level->setExtraCells(extraCells);
        
      if(stretch_count != 0){
        for(int axis = 0; axis < 3; axis++){
          OffsetArray1<double> faces;
          if(stretch[axis].size() == 0){
            // Uniform spacing...
            int l = -extraCells[axis];
            double start = levelAnchor(axis);
            double end = levelHighPoint(axis);
            double dx = spacing[axis];
            int ncells = Round((start-end)/dx);
            int h = ncells + extraCells(axis);
            faces.resize(l, h+1);
            for(int i=l;i<=h;i++)
              faces[i] = start + faces[i] * dx;
          } else {
            int count = 0;
            for(int i=0;i<(int)stretch[axis].size();i++){
              StretchSpec& spec = stretch[axis][i];
              count += spec.countCells();
            }
            count += 2*extraCells[axis];
            int l = -extraCells[axis];
            faces.resize(l, count);
            int start = 0;
            for(int i=0;i<(int)stretch[axis].size();i++){
              StretchSpec& spec = stretch[axis][i];
              int lowExtra = (i == 0)? extraCells[axis] : 0;
              int highExtra = (i == (int)stretch[axis].size() -1)? extraCells[axis]+1 : 0;
              spec.fillCells(start, lowExtra, highExtra, faces);
            }
          }
          level->setStretched((Grid::Axis)axis, faces);
        }
      }

      IntVector anchorCell(level->getCellIndex(levelAnchor + Vector(1.e-14,1.e-14,1.e-14)));
      IntVector highPointCell(level->getCellIndex(levelHighPoint + Vector(1.e-14,1.e-14,1.e-14)));

      // second pass - set up patches and cells
      for(ProblemSpecP box_ps = level_ps->findBlock("Box");
         box_ps != 0; box_ps = box_ps->findNextBlock("Box")){
        Point lower, upper;
        box_ps->require("lower", lower);
        box_ps->require("upper", upper);
        
        //__________________________________
        // bullet proofing inputs
        for(int dir = 0; dir<3; dir++){
          if (lower(dir) >= upper(dir)){
            ostringstream msg;
            msg<< "\nComputational Domain Input Error: Level("<< levelIndex << ")"
               << " \n The lower corner " << lower 
               << " must be smaller than the upper corner " << upper << endl; 
            throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
          }
        }
        
        IntVector lowCell  = level->getCellIndex(lower+Vector(1.e-14,1.e-14,1.e-14));
        IntVector highCell = level->getCellIndex(upper+Vector(1.e-14,1.e-14,1.e-14));
        Point lower2 = level->getNodePosition(lowCell);
        Point upper2 = level->getNodePosition(highCell);
        double diff_lower = (lower2-lower).length();
        double diff_upper = (upper2-upper).length();
        
        if(diff_lower > 1.e-14) {
          cerr << "lower=" << lower << '\n';
          cerr << "lowCell =" << lowCell << '\n';
          cerr << "highCell =" << highCell << '\n';
          cerr << "lower2=" << lower2 << '\n';
          cerr << "diff=" << diff_lower << '\n';
          
          throw ProblemSetupException("Box lower corner does not coincide with grid", __FILE__, __LINE__);
        }
        if(diff_upper > 1.e-14){
          cerr << "upper=" << upper << '\n';
          cerr << "lowCell =" << lowCell << '\n';
          cerr << "highCell =" << highCell << '\n';
          cerr << "upper2=" << upper2 << '\n';
          cerr << "diff=" << diff_upper << '\n';
          throw ProblemSetupException("Box upper corner does not coincide with grid", __FILE__, __LINE__);
        }

        IntVector resolution(highCell-lowCell);
        if(resolution.x() < 1 || resolution.y() < 1 || resolution.z() < 1) {
          cerr << "highCell: " << highCell << " lowCell: " << lowCell << '\n';
          throw ProblemSetupException("Degenerate patch", __FILE__, __LINE__);
        }
        
        // Check if autoPatch is enabled, if it is ignore the values in the
        // patches tag and compute them based on the number or processors

        IntVector patches;          // Will store the partition dimensions returned by the
                                    // run_partition3D function
        IntVector tempPatches;      // For 2D case, stores the results returned by run_partition2D
                                    // before they are sorted into the proper dimensions in
                                    // the patches variable.
        double autoPatchValue = 0;  // This value represents the ideal ratio of patches per
                                    // processor.  Often this is one, but for some load balancing
                                    // schemes it will be around 1.5.  When in doubt, use 1.
        map<string, string> patchAttributes;  // Hash for parsing out the XML attributes


        if(box_ps->get("autoPatch", autoPatchValue)) {
          // autoPatchValue must be >= 1, else it will generate fewer patches than processors, and fail
          if( autoPatchValue < 1 )
            throw ProblemSetupException("autoPatch value must be greater than 1", __FILE__, __LINE__);

          patchAttributes.clear();
          box_ps->getAttributes(patchAttributes);
          if(pg->myrank() == 0) {
            cout << "Automatically performing patch layout.\n";
          }
          
          int numProcs = pg->size();
          int targetPatches = (int)(numProcs * autoPatchValue);
          
          Primes::FactorType factors;
          int numFactors = Primes::factorize(targetPatches, factors);
          list<int> primeList;
          for(int i=0; i<numFactors; ++i) {
            primeList.push_back(factors[i]);
          }

          // First check all possible values for a 2D partition.  If no valid value
          // is found, perform a normal 3D partition.
          if( patchAttributes["flatten"] == "x" || resolution.x() == 1 )
          {
            ares_ = resolution.y();
            bres_ = resolution.z();
            tempPatches = run_partition2D(primeList);
            patches = IntVector(1,tempPatches.x(), tempPatches.y());
          } 
          else if ( patchAttributes["flatten"] == "y" || resolution.y() == 1 )
          {
            ares_ = resolution.x();
            bres_ = resolution.z();
            tempPatches = run_partition2D(primeList);
            patches = IntVector(tempPatches.x(),1,tempPatches.y());
          }
          else if ( patchAttributes["flatten"] == "z" || resolution.z() == 1 )
          {
            ares_ = resolution.x();
            bres_ = resolution.y();
            tempPatches = run_partition2D(primeList);
            patches = IntVector(tempPatches.x(),tempPatches.y(),1);
          }
          else 
          {
            // 3D case
            // Store the resolution in member variables
            ares_ = resolution.x();
            bres_ = resolution.y();
            cres_ = resolution.z();

            patches = run_partition3D(primeList);
          }
        } 
        else { // autoPatching is not enabled, get the patch field 
          box_ps->getWithDefault("patches", patches, IntVector(1,1,1));
          nf_ = 0;
        }

        // If the value of the norm nf_ is too high, then user chose a 
        // bad number of processors, warn them.
        if( nf_ > 3 ) {
          cout << "\n********************\n";
          cout << "*\n";
          cout << "* WARNING:\n";
          cout << "* The patch to processor ratio you chose\n";
          cout << "* does not factor well into patches.  Consider\n";
          cout << "* using a differnt number of processors.\n";
          cout << "*\n";
          cout << "********************\n\n";
        }
  
        if(pg->myrank() == 0) {
          cout << "Patch layout: \t\t(" << patches.x() << ","
               << patches.y() << "," << patches.z() << ")\n";
        }
      
        IntVector refineRatio = level->getRefinementRatio();
        level->setPatchDistributionHint(patches);
        for(int i=0;i<patches.x();i++){
          for(int j=0;j<patches.y();j++){
            for(int k=0;k<patches.z();k++){
              IntVector startcell = resolution*IntVector(i,j,k)/patches+lowCell;
              IntVector endcell = resolution*IntVector(i+1,j+1,k+1)/patches+lowCell;
              IntVector inStartCell(startcell);
              IntVector inEndCell(endcell);
              
              // this algorithm for finding extra cells is not sufficient for AMR
              // levels - it only finds extra cells on the domain boundary.  The only 
              // way to find extra cells for them is to do neighbor queries, so we will
              // potentially adjust extra cells in Patch::setBCType (called from Level::setBCTypes)
              startcell -= IntVector(startcell.x() == anchorCell.x() ? extraCells.x():0,
                                     startcell.y() == anchorCell.y() ? extraCells.y():0,
                                     startcell.z() == anchorCell.z() ? extraCells.z():0);
              endcell += IntVector(endcell.x() == highPointCell.x() ? extraCells.x():0,
                                   endcell.y() == highPointCell.y() ? extraCells.y():0,
                                   endcell.z() == highPointCell.z() ? extraCells.z():0);
            
              if (inStartCell.x() % refineRatio.x() || inEndCell.x() % refineRatio.x() || 
                  inStartCell.y() % refineRatio.y() || inEndCell.y() % refineRatio.y() || 
                  inStartCell.z() % refineRatio.z() || inEndCell.z() % refineRatio.z()) {
                ostringstream desc;
                desc << "The finer patch boundaries (" << inStartCell << "->" << inEndCell 
                     << ") do not coincide with a coarse cell"
                     << "\n(i.e., they are not divisible by te refinement ratio " << refineRatio << ')';
                throw InvalidGrid(desc.str(),__FILE__,__LINE__);
              }
              level->addPatch(startcell, endcell, inStartCell, inEndCell,this);
            }
          }
        } // end for(int i=0;i<patches.x();i++){
      } // end for(ProblemSpecP box_ps = level_ps->findBlock("Box");

      if (pg->size() > 1 && (level->numPatches() < pg->size()) && !do_amr) {
        throw ProblemSetupException("Number of patches must >= the number of processes in an mpi run",
                                    __FILE__, __LINE__);
      }
      
      IntVector periodicBoundaries;
      if(level_ps->get("periodic", periodicBoundaries)){
       level->finalizeLevel(periodicBoundaries.x() != 0,
                          periodicBoundaries.y() != 0,
                          periodicBoundaries.z() != 0);
      }
      else {
       level->finalizeLevel();
      }
      //level->assignBCS(grid_ps,0);
      levelIndex++;
   }
   if(numLevels() >1 && !do_amr) {  // bullet proofing
     throw ProblemSetupException("Grid.cc:problemSetup: Multiple levels encountered in non-AMR grid",
                                __FILE__, __LINE__);
   }
} // end problemSetup()

namespace Uintah
{
  ostream& operator<<(ostream& out, const Grid& grid)
  {
    out.setf(ios::floatfield);
    out.precision(6);
    out << "Grid has " << grid.numLevels() << " level(s)" << endl;
    for ( int levelIndex = 0; levelIndex < grid.numLevels(); levelIndex++ ) {
      LevelP level = grid.getLevel( levelIndex );
      out << "  Level " << level->getID() 
          << ", indx: "<< level->getIndex()
          << " has " << level->numPatches() << " patch(es)" << endl;
      for ( Level::patchIterator patchIter = level->patchesBegin(); patchIter < level->patchesEnd(); patchIter++ ) {
        const Patch* patch = *patchIter;
        out <<"    "<< *patch << endl;
      }
    }
    return out;
  }
}

//This is O(p).
bool Grid::operator==(const Grid& othergrid) const
{
  if (numLevels() != othergrid.numLevels())
    return false;
  for (int i = 0; i < numLevels(); i++) {
    const Level* level = getLevel(i).get_rep();
    const Level* otherlevel = othergrid.getLevel(i).get_rep();
    if (level->numPatches() != otherlevel->numPatches())
      return false;
      
    // do the patches have the same number of cells and
    // cover the same physical domain?  
    Level::const_patchIterator iter = level->patchesBegin();
    Level::const_patchIterator otheriter = otherlevel->patchesBegin();
    for (; iter != level->patchesEnd(); iter++, otheriter++) {
      const Patch* patch = *iter;
      const Patch* otherpatch = *otheriter;
      
      IntVector lo, o_lo;
      IntVector hi, o_hi;
      lo   = patch->getCellLowIndex();
      o_lo = otherpatch->getCellLowIndex();
      hi   = patch->getCellHighIndex();
      o_hi = otherpatch->getCellHighIndex();
       
      if ( lo !=  o_lo || hi != o_hi ){
        return false;
      }
      if( patch->getCellPosition(lo) != otherpatch->getCellPosition(o_lo) ||
          patch->getCellPosition(hi) != otherpatch->getCellPosition(o_hi) ){
        return false;
      }
    }
      
  }
  return true;

}

//This seems to have performance issues when there are lots of patches. This could be 
//partially avoided by parallelizing it.  
bool Grid::isSimilar(const Grid& othergrid) const
{
  if(numLevels() != othergrid.numLevels())
     return false;

 
  for(int i=numLevels()-1;i>=0;i--)
  {
    vector<Region> r1, r2, difference;
    const Level* l1=getLevel(i).get_rep();
    const Level* l2=othergrid.getLevel(i).get_rep();
    int a1=0,a2=0;

    //fill deques
    Level::const_patchIterator iter;
    for(iter=l1->patchesBegin(); iter!=l1->patchesEnd();iter++)
    {
      const Patch* patch=*iter;
      a1+=Region::getVolume(patch->getCellLowIndex(),patch->getCellHighIndex());
      r1.push_back(Region(patch->getCellLowIndex(),patch->getCellHighIndex()));
    }
    
    for(iter=l2->patchesBegin(); iter!=l2->patchesEnd();iter++)
    {
      const Patch* patch=*iter;
      a2+=Region::getVolume(patch->getCellLowIndex(),patch->getCellHighIndex());
      r2.push_back(Region(patch->getCellLowIndex(),patch->getCellHighIndex()));
    }

    //if volumes are not the same the grids cannot be the same
    if(a1!=a2)
      return false;
    
    //compare regions
    difference=Region::difference(r1,r2);
    if(!difference.empty())  //if region in r1 that is not in r2
      return false;
    difference=Region::difference(r2,r1);
    if(!difference.empty())  //if region in r1 that is not in r2
      return false;
  }
  return true;
}

IntVector Grid::run_partition3D(list<int> primes)
{
  partition3D(primes, 1, 1, 1);
  return IntVector(af_, bf_, cf_);
}

void Grid::partition3D(list<int> primes, int a, int b, int c)
{
  // base case: no primes left, compute the norm and store values
  // of a,b,c if they are the best so far.
  if( primes.size() == 0 ) {
    double new_norm = sqrt( (double)(max(a,b)/min(a,b) - max(ares_,bres_)/min(ares_,bres_)) *
                            (max(a,b)/min(a,b) - max(ares_,bres_)/min(ares_,bres_)) + 
                            (max(b,c)/min(b,c) - max(bres_,cres_)/min(bres_,cres_)) *
                            (max(b,c)/min(b,c) - max(bres_,cres_)/min(bres_,cres_)) +
                            (max(a,c)/min(a,c) - max(ares_,cres_)/min(ares_,cres_)) *
                            (max(a,c)/min(a,c) - max(ares_,cres_)/min(ares_,cres_))
                          );

    if( new_norm < nf_ || nf_ == -1 ) { // negative 1 flags initial trash value of nf_, 
                                       // should always be overwritten
      nf_ = new_norm;
      af_ = a;
      bf_ = b;
      cf_ = c;
    }
    
    return;
  }

  int head = primes.front();
  primes.pop_front();
  partition3D(primes, a*head, b, c);
  partition3D(primes, a, b*head, c);
  partition3D(primes, a, b, c*head);

  return;
}

IntVector Grid::run_partition2D(std::list<int> primes)
{
  partition2D(primes, 1, 1);
  return IntVector(af_, bf_, cf_);
}

void Grid::partition2D(std::list<int> primes, int a, int b)
{
  // base case: no primes left, compute the norm and store values
  // of a,b if they are the best so far.
  if( primes.size() == 0 ) {
    double new_norm = (double)max(a,b)/min(a,b) - max(ares_,bres_)/min(ares_,bres_);

    if( new_norm < nf_ || nf_ == -1 ) { // negative 1 flags initial trash value of nf_, 
                                       // should always be overwritten
      nf_ = new_norm;
      af_ = a;
      bf_ = b;
    }
    
    return;
  }

  int head = primes.front();
  primes.pop_front();
  partition2D(primes, a*head, b);
  partition2D(primes, a, b*head);

  return;
}

const Patch* Grid::getPatchByID(int patchid, int startingLevel) const
{
  const Patch* patch = NULL;
  for (int i = startingLevel; i < numLevels(); i++) {
    LevelP checkLevel = getLevel(i);
    int levelBaseID = checkLevel->getPatch(0)->getID();
    if (patchid >= levelBaseID && patchid < levelBaseID+checkLevel->numPatches()) {
      patch = checkLevel->getPatch(patchid-levelBaseID);
      break;
    }
  }
  return patch;
}

void Grid::assignBCS(const ProblemSpecP &grid_ps,LoadBalancer *lb)
{
  TAU_PROFILE("Grid::assignBCS()", " ", TAU_USER);
  for(int l=0;l<numLevels();l++)
  {
    LevelP level= getLevel(l);
    level->assignBCS(grid_ps,lb);
  }
}
