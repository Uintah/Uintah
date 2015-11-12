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

#ifndef UINTAH_CORE_GRID_GRID_H
#define UINTAH_CORE_GRID_GRID_H

#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>

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
#include <Core/Util/FancyAssert.h>

#include <list>
#include <string>
#include <vector>

namespace Uintah {

class ProcessorGroup;
class Patch;
class LevelFlags;

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
   This class basically manages the pointers to the levels that make
   up the grid.
  
WARNING
  
****************************************/


  class Grid : public RefCounted
  {
    private:

      class stretchRegion
      {
        public:

          stretchRegion(  std::string & _shape
                        , double      & _to
                        , double      & _from
                        , double      & _toSpacing
                        , double      & _fromSpacing)
           :  shape(_shape)
            , to(_to)
            , from(_from)
            , toSpacing(_toSpacing)
            , fromSpacing(_fromSpacing)
          { };

         ~stretchRegion() {};

          double getFrom() const {
            return from;
          }

          //////////
          //
          void setFrom( const double& value ) {
            from = value;
          }

          //////////
          //
          double getTo() const {
            return to;
          }

          //////////
          //
          void setTo( const double& value ) {
            to = value;
          }

          //////////
          //
          double getToSpacing() const {
            return toSpacing;
          }

          //////////
          //
          void setToSpacing( const double& value ) {
            toSpacing = value;
          }

          //////////
          //
          double getFromSpacing() const {
            return fromSpacing;
          }

          //////////
          //
          void setFromSpacing( const double& value ) {
            fromSpacing = value;
          }

          //////////
          //
          std::string getShape() const {
            return shape;
          }

          //////////
          //
          void setShape( const std::string& value ) {
            shape = value;
          }

          //////////
          //
          int  countCells() const;

          //////////
          //
          void fillCells(  int                          & start
                         , int                            lowCells
                         , int                            highCells
                         , SCIRun::OffsetArray1<double> & faces ) const;


        private:

          std::string   shape;
          double        to;
          double        from;
          double        toSpacing;
          double        fromSpacing;
      };

      class stretchDescription
      {
        public:

          stretchDescription() {};

         ~stretchDescription() {};

          //////////
          //
          void addRegion(  const int           axis
                         ,       stretchRegion region ) {
            axialStretches[axis].push_back(region);
          }

          //////////
          //
          SCIRun::Vector checkStretches(  const SCIRun::BBox & extents
                                        , const int          & procRank );

          //////////
          //
          int getRegionsPerAxis(const int& axis) const {
            return static_cast<int> (axialStretches[axis].size());
          }

          stretchRegion* getRegion(  const int & axis
                                   , const int & region ) {
            ASSERTRANGE( region, 0, getRegionsPerAxis(axis));
            return (&axialStretches[axis][region]);
          }

          //////////
          //
          int stretchCount() const {
            int count = 0;
            for (int axis = 0; axis < 3; ++axis) {
              if (axialStretches[axis].size()) {
                ++count;
              }
            }
            return count;
          }

          //////////
          //
          bool checkForPrevious( const int& axis ) const {
            if (axialStretches[axis].size() !=0){
              return true;
            }
            return false;
          }


        private:

          std::vector<stretchRegion>  axialStretches[3];
      };

      class LevelBox {

        public:

          LevelBox(  SCIRun::BBox         _extents
                   , SCIRun::IntVector    _extraCells
                   , SCIRun::Vector       _boxSpacing
                   , stretchDescription   _stretches )
        :  boxExtents(_extents)
         , extraCells(_extraCells)
         , boxSpacing(_boxSpacing)
         , stretches(_stretches)
        { };

         ~LevelBox() {};

          //////////
          //
          SCIRun::BBox getBoxExtents() const {
            return boxExtents;
          }

          //////////
          //
          SCIRun::IntVector getExtraCells() const {
            return extraCells;
          }

          //////////
          //
          SCIRun::Vector    getSpacing() const {
            return boxSpacing;
          }

          //////////
          //
          SCIRun::Vector    getAnchor() const {
            return boxExtents.min().asVector();
          }

          //////////
          //
          SCIRun::Vector    getHighPoint() const {
            return boxExtents.max().asVector();
          }

          //////////
          //
          bool hasSpacing() const {
            return (boxSpacing[0] == -1.0 ? false : true); // Flag by first value being negative
          }

          //////////
          //
          int stretchCount() const {
            return stretches.stretchCount();
          }

          //////////
          //
          stretchDescription* getStretchDescription() {
            return &stretches;
          }

        private:

          SCIRun::BBox          boxExtents;
          SCIRun::IntVector     extraCells;
          SCIRun::Vector        boxSpacing;
          stretchDescription    stretches;
      };

  public:

    Grid();

    virtual ~Grid();
    
    //////////
    // Returns the number of levels in this grid.
    int numLevels() const { return d_levels.size(); }
    
    //////////
    // Returns a "Handle" to the "idx"th level 
    const LevelP& getLevel( int idx ) const;
    
    //////////
    // Adds a level to the grid.
    Level* addLevel(  const SCIRun::Point  & anchor
                    , const SCIRun::Vector & dcell
                    , LevelFlags           & flags
                    , int                    id = -1 );

    // Reads in XML data line by line to create a level...
    void readLevelsFromFile( FILE * fp, std::vector< std::vector<int> > & procMap, const bool &do_AMR );
   
    //////////
    //
    void performConsistencyCheck() const;

    //////////
    //
    void printStatistics() const;

    //////////
    // Computes the physical boundaries for the grid (including extra cells)
    void getSpatialRange(SCIRun::BBox& b) const;

    //////////
    //
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

    //////////
    //
    void setExtraCells( const IntVector & ex );

    //////////
    //
    inline int numLevelSets() { return d_levelSet.size(); };

    //////////
    // The 0th LevelSubset will be the set of Levels AMR is occurring on
    const LevelSubset* getAMRLevelSubset() { return d_levelSet.getSubset(0); };

    //////////
    //
    const LevelSet* getLevelSet() const { return &d_levelSet; };

    //////////
    //
    void createLevelSubsets( int num_sets );

    //////////
    //
    friend std::ostream& operator<<(std::ostream& out, const Uintah::Grid& grid);

    //////////
    // Used in Level and Patch for stretched grids
    enum Axis {
        XAxis
      , YAxis
      , ZAxis
    };


  private:

    std::vector<LevelP> d_levels;
    
    Grid& operator=(const Grid&);

    // For automatic patch layout.  run_partition() will initialize the values of
    // af_, bf_, cf_, and nf_, then start the recursive call.  You should never
    // explicitly call partition(), only run_partition().
    IntVector run_partition3D(std::list<int> primes);

    //////////
    //
    void      partition3D(std::list<int> primes, int a, int b, int c);
    
    SCIRun::IntVector run_partition2D(std::list<int> primes);

    void      partition2D(std::list<int> primes, int a, int b);

    // Helper function for reading in xml specification of the grid from timestep.xml.
    bool      parseGridFromFile(  FILE * fp, std::vector< std::vector<int> > & procMap, const bool& doAMR );         // returns true if "</Grid>" found.

    bool      parseLevelFromFile( FILE * fp, std::vector<int> & procMapForLevel );                // returns true if "</Level>" found.

    bool      parsePatchFromFile( FILE * fp, LevelP level, std::vector<int> & procMapForLevel );  // returns true if "</Patch>" found.

    SCIRun::OffsetArray1<double> assignStretchedFaces(      stretchDescription* stretches,
                                                      const LevelBox*           levelInfo,
                                                      const SCIRun::IntVector&  extraCells,
                                                      const int&                axis);
    stretchDescription      parseStretches(const ProblemSpecP& stretch_ps);


    void parsePatches(  const ProblemSpecP      & level_ps
                      ,       LevelP            & level
                      , const SCIRun::IntVector & levelAnchorCell
                      , const SCIRun::IntVector & levelHighPointCell
                      , const SCIRun::IntVector & levelExtraCells
                      , const int                 numProcs
                      , const int                 myRank );

    LevelBox  parseBox(        ProblemSpecP     box_ps
                       , const bool             haveLevelSpacing
                       , const bool             havePatchSpacing
                       , const SCIRun::Vector & currentSpacing );

    LevelBox  parseLevel(        ProblemSpecP & level_ps
                         , const int            levelIndex
                         , const int            myProcRank);

    bool      specIsAMR(  const ProblemSpecP & ps) const;

    void parseLevelSet(   const ProblemSpecP & level_ps
                        , const int            numProcs
                        , const int            myProcRank
                        , const int            globalIndexOffset
                        , const int            levelSetIndex
                        , const bool           do_AMR
                        , const bool           do_MultiScale
                      );

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

    // Arbitrary desired maximum value for the norm.  If the norm of the best possible
    // patch layout exceeds this number, a warning message will be printed suggestion
    // the user run on a different number of processors.
    // static const double PATCH_TOLERANCE_ = 3;  
    
    SCIRun::IntVector d_extraCells;

    // Holds the level subsets for portions of the grid.
    LevelSet  d_levelSet;

    std::vector<const LevelSubset*> d_levelSubsetMap;

  };

} // End namespace Uintah

#endif  // end #ifndef UINTAH_CORE_GRID_GRID_H
