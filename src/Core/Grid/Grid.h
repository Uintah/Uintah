/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#ifndef UINTAH_HOMEBREW_GRID_H
#define UINTAH_HOMEBREW_GRID_H

#include <CCA/Ports/LoadBalancer.h>

#include <Core/Containers/OffsetArray1.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/Handle.h>
#include <Core/Util/RefCounted.h>

#include <list>
#include <string>
#include <vector>

namespace Uintah {

  class ProcessorGroup;
  class Patch;
/**************************************

CLASS
   Grid
   
   This class manages the grid used to solve the CFD and MPM problems.

GENERAL INFORMATION

   Grid.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   Grid, Level

DESCRIPTION
   This class basicly manages the pointers to the levels that make
   up the grid.  
  
WARNING
  
****************************************/

  struct StretchSpec {
    std::string shape;
    double from;
    double to;
    double fromSpacing;
    double toSpacing;

    int countCells() const;
    void fillCells(int& start, int lowCells, int highCells, SCIRun::OffsetArray1<double>& faces) const;
  };

  class LevelBox {
    public:
      LevelBox(SCIRun::BBox         _extents,
               SCIRun::IntVector    _extraCells,
               SCIRun::Vector       _boxSpacing)
              :boxExtents(_extents), extraCells(_extraCells), boxSpacing(_boxSpacing) {};
     ~LevelBox() {};
      SCIRun::BBox getBoxExtents() const
      {
        return boxExtents;
      }
      SCIRun::IntVector getExtraCells() const
      {
        return extraCells;
      }
      SCIRun::Vector    getSpacing() const
      {
        return boxSpacing;
      }
      bool              hasPatchSpacing() const
      {
        return (boxSpacing[0] == -1.0 ? false : true); // Flag by first value being negative
      }
    private:
      SCIRun::BBox      boxExtents;
      SCIRun::IntVector extraCells;
      SCIRun::Vector    boxSpacing;
  };

  class Grid : public RefCounted {
  public:
    Grid();
    virtual ~Grid();
    
    //////////
    // Returns the number of levels in this grid.
    int numLevels() const { return d_levels.size(); }
    
    //////////
    // Returns a "Handle" to the "idx"th level 
    const LevelP& getLevel(int idx) const;
    
    //////////
    // Adds a level to the grid.
    Level* addLevel( const SCIRun::Point  & anchor,
                     const SCIRun::Vector & dcell, int id = -1, bool isAMR = false, bool isMultiscale = false);

    // Reads in XML data line by line to create a level...
    void readLevelsFromFile( FILE * fp, std::vector< std::vector<int> > & procMap );
   
    void performConsistencyCheck() const;
    void printStatistics() const;

    //////////
    // Computes the physical boundaries for the grid (including extra cells)
    void getSpatialRange(SCIRun::BBox& b) const;

    const Patch* getPatchByID(int id, int startLevel) const;

    ////////// 
    // Returns the boundary of the grid exactly (without
    // extra cells).  The value returned is the same value
    // as found in the .ups file.
    void getInteriorSpatialRange(SCIRun::BBox& b) const;
    
    //////////
    // Computes the length of the grid
    void getLength(       SCIRun::Vector & length,
                    const std::string    & flag = "plusExtraCells" ) const;
    
    //////////
    // Problem setup functions called from simulation controller
    void problemSetup(const ProblemSpecP& params, const ProcessorGroup *pg, bool do_AMR, bool do_MultiScale = false);

    // For comparing grids - level and patch structure must be equal
    bool operator==(const Grid& othergrid) const;

    // Returns true if the two grids cover the exact same region.  Patch structure can be different.
    bool isSimilar(const Grid& othergrid) const;

    //Assigns the boundary conditions to the grid
    void assignBCS( const ProblemSpecP &grid_ps, Uintah::LoadBalancer *lb );

    void setExtraCells( const IntVector & ex );
           
    friend std::ostream& operator<<(std::ostream& out, const Uintah::Grid& grid);

    // Used in Level and Patch for stretched grids
    enum Axis {
      XAxis, YAxis, ZAxis
    };


  private:
    std::vector<LevelP> d_levels;
    
    Grid& operator=(const Grid&);

    // For automatic patch layout.  run_partition() will initialize the values of
    // af_, bf_, cf_, and nf_, then start the recursive call.  You should never
    // explicitly call partition(), only run_partition().
    IntVector run_partition3D(std::list<int> primes);
    void      partition3D(std::list<int> primes, int a, int b, int c);
    
    IntVector run_partition2D(std::list<int> primes);
    void      partition2D(std::list<int> primes, int a, int b);

    // Helper function for reading in xml specification of the grid from timestep.xml.
    bool      parseGridFromFile(  FILE * fp, std::vector< std::vector<int> > & procMap );         // returns true if "</Grid>" found.
    bool      parseLevelFromFile( FILE * fp, std::vector<int> & procMapForLevel );                // returns true if "</Level>" found.
    bool      parsePatchFromFile( FILE * fp, LevelP level, std::vector<int> & procMapForLevel );  // returns true if "</Patch>" found.

    void      parseStretches(std::vector<StretchSpec> (&stretch)[3], const ProblemSpecP& stretch_ps);
    int       checkStretches(       std::vector<StretchSpec>   (&stretch)[3],
                                    SCIRun::Vector&             spacing,
                             const  Uintah::Point&              levelAnchor,
                             const  Uintah::Point&              levelHighPoint,
                                    int                         procRank);
    LevelBox  parseBox(      ProblemSpecP       box_ps,
                       const bool               haveLevelSpacing,
                       const bool               havePatchSpacing,
                       const SCIRun::Vector    &currentSpacing);

    Level*    parseLevel(      ProblemSpecP    &level_ps,
                         const int              levelIndex,
                         const int              myProcRank);

    void      newProblemSetup(const ProblemSpecP   &params,
                              const ProcessorGroup *pg,
                                    bool            do_AMR,
                                    bool            do_MultiScale);

    // The current (final) values of a,b,c, and norm for the partition function.
    // Used to hold data between recursive calls.
    int    af_;
    int    bf_;
    int    cf_;
    double nf_;

    // Temporary storage of the resolution values for use in the norm functions
    // In the 3D case, (a,b,c) corresponds to (x,y,z).  However in the 2D case
    // we do not know which dimensions (a,b) are, so the calling method will
    // place the results into the correct dimensions.
    int ares_;
    int bres_;
    int cres_;

    // Arbitrary desired maxium value for the norm.  If the norm of the best possible
    // patch layout exceeds this number, a warning message will be printed suggestion
    // the user run on a different number of processors.
    // static const double PATCH_TOLERANCE_ = 3;  
    
    IntVector d_extraCells;

  };

} // End namespace Uintah

#endif
