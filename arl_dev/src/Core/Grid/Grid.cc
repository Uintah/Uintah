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


#include <Core/Grid/Grid.h>

#include <Core/Exceptions/InvalidGrid.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/Primes.h>
#include <Core/Math/UintahMiscMath.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/XMLUtils.h>

#include <iomanip>
#include <ostream>
#include <sci_values.h>

using namespace Uintah;

static double xk(double a, double r, int k)
{
  if (r == 1) {
    return a * k;
  }
  return a * r * (1 - pow(r, k)) / (1 - r);
}

Grid::Grid()
  :  af_(0)
   , bf_(0)
   , cf_(0)
   , nf_(-1)
   , ares_(0)
   , bres_(0)
   , cres_(0)
   , d_extraCells(IntVector(0,0,0))
{
  // initialized values may be used for the autoPatching calculations
}

Grid::~Grid()
{
}

const LevelP &
Grid::getLevel( int l ) const
{
  ASSERTRANGE( l, 0, numLevels() );
  return d_levels[ l ];
}

//FIXME: TODO:  Need to fix parse*FromFile to account for parsing in old .xml files yet still work with the
// new isAMR/isMultiscale level structure.  JBH - 11-4-2015

//
// Parse in the <Patch> from the input file (most likely 'timestep.xml').  We should only need to parse
// three tags: <id>, <proc>, <lowIndex>, <highIndex>, <interiorLowIndex>, <nnodes>, <lower>, <upper>, 
// <totalCells>, and </Patch>
// as the XML should look like this (and we've already parsed <Patch> to get here - and the Level parsing
// will eat up </Patch>):
//
//      [Patch] <- eaten by caller
//        <id>0</id>
//        <proc>0</proc>
//        <lowIndex>[-1, -1, -1]</lowIndex>
//        <highIndex>[15, 31, 3]</highIndex>
//        <interiorLowIndex>[0, 0, 0]</interiorLowIndex>
//        <nnodes>2048</nnodes>
//        <lower>[-0.01, -0.01, -0.10000000000000001]</lower>
//        <upper>[0.14999999999999999, 0.31, 0.30000000000000004]</upper>
//        <totalCells>2048</totalCells>
//      </Patch>
//
bool
Grid::parsePatchFromFile( FILE * fp, LevelP level, std::vector<int> & procMapForLevel )
{
  bool      doneWithPatch   = false;

  int       id               = -1;
  bool      foundId          = false;

  int       proc             = -1;
  bool      foundProc        = false;

  IntVector lowIndex, highIndex;
  bool      foundLowIndex    = false;
  bool      foundHighIndex   = false;

  IntVector interiorLowIndex, interiorHighIndex;
  bool      foundInteriorLowIndex = false;
  bool      foundInteriorHighIndex = false;

  while( !doneWithPatch ) {
    std::string line = UintahXML::getLine( fp );
    
    if( line == "</Patch>" ) {
      doneWithPatch = true;
    }
    else if( line == "" ) {
      return false; // end of file reached
    }
    else {
      std::vector< std::string > pieces = UintahXML::splitXMLtag( line );
      if( pieces[0] == "<id>" ) {
        id = atoi( pieces[1].c_str() );
        foundId = true;
      }
      else if( pieces[0] == "<proc>" ) {
        proc = atoi( pieces[1].c_str() );
        foundProc = true;
      }
      else if( pieces[0] == "<lowIndex>" ) {
        lowIndex = IntVector::fromString( pieces[1] );
        foundLowIndex = true;
      }
      else if( pieces[0] == "<highIndex>" ) {
        highIndex = IntVector::fromString( pieces[1] );
        foundHighIndex = true;
      }
      else if( pieces[0] == "<interiorLowIndex>" ) {
        interiorLowIndex = IntVector::fromString( pieces[1] );
        foundInteriorLowIndex = true;
      }
      else if( pieces[0] == "<interiorHighIndex>" ) {
        interiorHighIndex = IntVector::fromString( pieces[1] );
        foundInteriorHighIndex = true;
      }
      else if( pieces[0] == "<nnodes>" ) {
        // FIXME: do I need to handle nnodes, totalCells, lower, and upper?
      }
      else if( pieces[0] == "<lower>" ) {
      }
      else if( pieces[0] == "<upper>" ) {
      }
      else if( pieces[0] == "<totalCells>" ) {
      }
      else {
        std::ostringstream msg;
        msg << "parsePatchFromFile(): Bad XML tag: " << pieces[0] << "\n";
        throw InternalError( msg.str(), __FILE__, __LINE__ );
      }
    }
  } // end while()

  if( !foundId || !foundProc || !foundHighIndex ) {
    std::cout << "I am here: " << foundId << ", " << foundProc << ", " << foundHighIndex << "\n";
    throw InternalError("Grid::parsePatchFromFile() - Missing a <Patch> child tag...", __FILE__, __LINE__ );
  }

  if( !foundInteriorLowIndex ) { interiorLowIndex = lowIndex; }
  if( !foundInteriorHighIndex ) { interiorHighIndex = highIndex; }

  level->addPatch( lowIndex, highIndex, interiorLowIndex, interiorHighIndex, this, id );

  procMapForLevel.push_back( proc ); // corresponds to DataArchive original timedata.d_patchInfo[levelIndex].push_back(pi);

  return doneWithPatch;

} // end parsePatchFromFile()

//
// Parse in the <Level> from the input file (most likely 'timestep.xml').  We should only need to parse
// these tags: <numPatches>, <totalCells>, <extraCells>, <anchor>, <id>, <cellspacing>, <Patch>, and </Level>
// as the XML should look like this (and we've already parsed <Level> to get here - and the Patch parsing
// will eat up </Patch>):
//
//    <Level>
//      <numPatches>8192</numPatches>
//      <totalCells>13104208</totalCells>
//      <extraCells>[1, 1, 1]</extraCells>
//      <anchor>[0, 0, 0]</anchor>
//      <id>0</id>
//      <cellspacing>[0.01, 0.01, 0.10000000000000001]</cellspacing>
//      <Patch>
//         ...
//      </Patch>
//    </Level>
//
bool
Grid::parseLevelFromFile( FILE * fp, std::vector<int> & procMapForLevel )
{
  int       numPatches       = 0;
  int       numPatchesRead   = 0;
  int       totalCells       = -1;

  bool      done_with_level  = false;

  IntVector periodicBoundaries;
  bool      foundPeriodicBoundaries = false;

  Point     anchor;
  bool      foundAnchor      = false;

  Vector    dcell;
  bool      foundCellSpacing = false;

  int       id               = -1;
  bool      foundId          = false;

  IntVector extraCells(0,0,0);
  bool      foundExtraCells  = false;

  bool      foundStretch     = false;

  bool      levelCreated     = false;

  LevelP    level;

  while( !done_with_level ) {
    std::string line = UintahXML::getLine( fp );
    
    if( line == "</Level>" ) {
      done_with_level = true;
      levelCreated = false;

    }
    else if( line == "" ) {
      break; // end of file reached.
    }
    else if( line == "<Patch>" ) {

      if( !levelCreated ) {

        levelCreated = true;

        //
        // When we hit the first <Patch>, then all the Level data will have been
        // read in, so we can go ahead and create the level... Also, we must create
        // the level here as it is needed by parsePatchFromFile so that the patch
        // can be added to the level.
        //
        
        //if( !foundStretch ) {
        //  dcell *= d_cell_scale;
        //}

        LevelFlags flags;
        level = this->addLevel( anchor, dcell, flags, id );
        // FIXME! File parsing needs to be fixed for AMR, MultiScale per level in a transparent way! JBH 9/2015

        level->setExtraCells( extraCells );

        //  if( foundStretch ) {
        //    level->setStretched((Grid::Axis)0, faces[0]);
        //    level->setStretched((Grid::Axis)1, faces[1]);
        //    level->setStretched((Grid::Axis)2, faces[2]);
        //  }
      }

      numPatchesRead++;

      // At this point, we should be done reading in <Level> information, so we should go ahead and
      // create the level... if for no other reason that the Patches have to be added to it...
      if( !foundStretch && !foundCellSpacing ) {
        throw InternalError("Grid::parseLevelFromFile() - Did not find <cellspacing> or <StretchPositions> point", __FILE__, __LINE__ );
      }
      else if( !foundAnchor ) {
        throw InternalError("Grid::parseLevelFromFile() - Did not find level anchor point", __FILE__, __LINE__ );
      }
      else if( !foundId ) {
        static bool warned_once = false;
        if( !warned_once ){
          std::cerr << "WARNING: Data archive does not have level ID.\n";
          std::cerr << "This is okay, as long as you aren't trying to do AMR.\n";
        }
        warned_once = true;
      }

      parsePatchFromFile( fp, level, procMapForLevel );
    }
    else {
      std::vector< std::string > pieces = UintahXML::splitXMLtag( line );
      if( pieces[0] == "<numPatches>" ) {
        numPatches = atoi( pieces[1].c_str() );
      }
      else if( pieces[0] == "<totalCells>" ) {
        totalCells = atoi( pieces[1].c_str() );
      }
      else if( pieces[0] == "<extraCells>" ) {
        extraCells = IntVector::fromString( pieces[1] );
        foundExtraCells = true;
      }
      else if( pieces[0] == "<anchor>" ) {
        Vector v = Vector::fromString( pieces[1] );
        anchor = Point( v );
        foundAnchor = true;
      }
      else if( pieces[0] == "<id>" ) {
         id = atoi( pieces[1].c_str() );
         foundId = true;
      }
      else if( pieces[0] == "<cellspacing>" ) {
        dcell = Vector::fromString( pieces[1] );
        foundCellSpacing = true;
      }
      else if( pieces[0] == "<StretchPositions>" ) {
        foundStretch = true;
        // FIXME - QWERTY - README - add in the code from original DataArchive.cc to handle <StretchPositions>

        throw InternalError( "Grid::getLine() fail - don't know how to handle StretchPositions yet.", __FILE__, __LINE__ );
      }
      else if( pieces[0] == "<periodic>" ) {

        periodicBoundaries = IntVector::fromString( pieces[1] );
        foundPeriodicBoundaries = true;
      }
      else {
        std::ostringstream msg;
        msg << "parseLevelFromFile(): Bad XML tag: " << pieces[0] << "\n";
        throw InternalError( msg.str(), __FILE__, __LINE__ );
      }
    }
  } // end while()

  if( foundPeriodicBoundaries ){
    level->finalizeLevel( periodicBoundaries.x() != 0,
                          periodicBoundaries.y() != 0,
                          periodicBoundaries.z() != 0 );
  }
  else {
    level->finalizeLevel();
  }

  if( numPatches != numPatchesRead ) {
    proc0cout << "numPatchesRead, numPatches: " << numPatchesRead << ", " << numPatches << "\n";
    throw InternalError( "XML file is corrupted, read different number of patches then it specifies.", __FILE__, __LINE__ );
  }

  return done_with_level;

} // end parseLevelFromFile()

//
// Parse in the <Grid> from the input file (most likely 'timestep.xml').  We should only need to parse
// three tags: <numLevels>, <Level>, and </Grid> as the XML should look like this (and we've already
// parsed <Grid> to get here - and the Level parsing will eat up </Level>):
//
//    <Grid>
//      <numLevels>1</numLevels>
//      <Level>
//         ...
//      </Level>
//    </Grid>
//
bool
Grid::parseGridFromFile( FILE * fp, std::vector< std::vector<int> > & procMap, const bool& doAMR )
{
  int  numLevelsRead  = 0;
  int  numLevels      = 0;
  bool doneWithGrid   = false;
  bool foundLevelTag  = false;
  bool hasLevelSet    = false;

  while( !doneWithGrid ) { 

    // Parse all of the Grid...  When we are done with it, we are done parsing the file, hence the reason
    // we can use "done" for both while loops.

    std::string line = UintahXML::getLine( fp );

    if( line == "</Grid>" ) {
      doneWithGrid = true;
    }
    else if( line == "<Level>" ) {
      foundLevelTag = true;

      procMap.push_back( std::vector<int>() );
      std::vector<int> & procMapForLevel = procMap[ numLevelsRead ];
 
      numLevelsRead++;

      parseLevelFromFile( fp, procMapForLevel );
    }
    else if( line == "" ) {
      break; // end of file reached.
    }
    else {
      std::vector< std::string > pieces = UintahXML::splitXMLtag( line );
      if( pieces[0] == "<numLevels>" ) {
        numLevels = atoi( pieces[1].c_str() );
      }
      else {
        std::ostringstream msg;
        msg << "parseGridFromFile(): Bad XML tag: " << pieces[0] << "\n";
        throw InternalError( msg.str(), __FILE__, __LINE__ );
      }
    }
  }
  if( !foundLevelTag ) {
    throw InternalError( "Grid.cc::parseGridFromFile(): Did not find '<Level>' tag in file.", __FILE__, __LINE__ );
  }

  // Verify that the <numLevels> tag matches the actual number of levels that we have parsed.
  // If not, then there is an error in the xml file (most likely it was corrupted or written out incorrectly.
  ASSERTEQ( numLevels, numLevelsRead );

  return doneWithGrid;

} // end parseGridFromFile()

//
// We are parsing the XML manually, line by line, because if we use the XML library function to read it into an XML tree
// data structure, then for large number of patches, too much memory is used by that data structure.  It is unclear
// whether the XML library frees up this memory when you "releaseDocument()", but it probably doesn't matter because
// we are concerned about the highwater mark (max memory per node), which the XML structure may push us over.
//
void
Grid::readLevelsFromFile( FILE * fp, std::vector< std::vector<int> > & procMap, const bool &do_AMR )
{
  bool done      = false;
  bool foundGrid = false;

  while( !done ) { // Start at the very top of the file (most likely 'timestep.xml').

    std::string line = UintahXML::getLine( fp );

    if( line == "<Grid>" ) {
      foundGrid = parseGridFromFile( fp, procMap, do_AMR );
      done = true;
    }
    else if( line == "" ) { // End of file reached.
      done= true;
    }
  }

  if( !foundGrid ) {
    throw InternalError( "Grid.cc: readLevelsFromFile: Did not find '<Grid>' in file.", __FILE__, __LINE__ );
  }

} // end readLevelsFromFile()


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Level *
Grid::addLevel( const Point& anchor, const Vector& dcell, LevelFlags& flags, int id )
{
  // Find the new level's refinement ratio.
  // This should only be called when a new grid is created, so if this level index 
  // is > 0, then there is a coarse-fine relationship between this level and the 
  // previous one.
  SCIRun::IntVector ratio;
  if (d_levels.size() > 0) {
    SCIRun::Vector r = (d_levels[d_levels.size()-1]->dCell() / dcell) + Vector(1e-6, 1e-6, 1e-6);
    ratio = SCIRun::IntVector((int)r.x(), (int)r.y(), (int)r.z());
    SCIRun::Vector diff = r - ratio.asVector();
    if (diff.x() > 1e-5 || diff.y() > 1e-5 || diff.z() > 1e-5) {
      // non-integral refinement ratio
      std::ostringstream out;
      out << "Non-integral refinement ratio: " << r;
      throw InvalidGrid(out.str().c_str(), __FILE__, __LINE__);
    }
  }
  else
    ratio = SCIRun::IntVector(1,1,1);


  Level* level = scinew Level(this, anchor, dcell, (int)d_levels.size(), ratio, flags, id);

  d_levels.push_back( level );
  return level;
}


void
Grid::performConsistencyCheck(const LevelSet & currentSet) const {

#if SCI_ASSERTION_LEVEL > 0
  size_t numSubsets = currentSet.size();
  for (size_t subsetIndex = 0; subsetIndex < numSubsets; ++subsetIndex) {

    // Verify that patches on a single level do not overlap
    const LevelSubset* currLevelSubset = currentSet.getSubset(subsetIndex);
    size_t numLevels = currLevelSubset->size();
    for (size_t levelIndex = 0; levelIndex < numLevels; ++levelIndex) {
      const Level* currLevel = currLevelSubset->get(levelIndex);
      currLevel->performConsistencyCheck();
    }
    // Check overlap between levels
    // See if patches on level 0 form a connected set (warning)
    // Compute total volume - compare if not first time
    //
    // FIXME TODO JBH APH 11-14-2015
    // For now, we only expect consistency within a single levelSubset since that is representative of
    // previous "grids".  At some point we need to add consistency checks for two subsets interacting with
    // each other.
    // Also this consistency check is only reasonable for AMR.
    if (numLevels > 0) {
      for (size_t levelIndex = 0; levelIndex < numLevels - 1; ++levelIndex) {
        // Current grid structure has coarser/finer as -/+1 level offsets
        // In a levelSet implementation, coarser is always before finer
        const Level* coarseLevel = currLevelSubset->get(levelIndex);
        const Level* fineLevel   = currLevelSubset->get(levelIndex + 1);

        SCIRun::Vector dx_fineLevel = fineLevel->dCell();

        // Finer level can't lay outside of the coarser level
        SCIRun::BBox C_box = coarseLevel->getSpatialRange();
        SCIRun::BBox F_box = fineLevel->getSpatialRange();
        if (!C_box.contains(F_box)) {
          std::ostringstream desc;
          desc << " The finer Level " << fineLevel->getIndex()
               << " "<< F_box.min() << " "<< F_box.max()
               << " can't lay outside of coarser level " << coarseLevel->getIndex()
               << " "<< C_box.min() << " "<< C_box.max() << std::endl;
          throw InvalidGrid(desc.str(),__FILE__,__LINE__);
        }
        //__________________________________
        //  finer level must have a box width that is
        //  an integer of the cell spacing
        SCIRun::Vector integerTest_min( remainder(F_box.min().x(),dx_fineLevel.x())
                                       ,remainder(F_box.min().y(),dx_fineLevel.y())
                                       ,remainder(F_box.min().z(),dx_fineLevel.z()));
        SCIRun::Vector integerTest_max( remainder(F_box.max().x(),dx_fineLevel.x())
                                       ,remainder(F_box.max().y(),dx_fineLevel.y())
                                       ,remainder(F_box.max().z(),dx_fineLevel.z()));

        SCIRun::Vector distance = F_box.max() - F_box.min();
        SCIRun::Vector integerTest_distance( remainder(distance.x(), dx_fineLevel.x())
                                            ,remainder(distance.y(), dx_fineLevel.y())
                                            ,remainder(distance.z(), dx_fineLevel.z())
                                           );

        SCIRun::Vector smallNum(1e-14,1e-14,1e-14);

        if( (integerTest_min > smallNum || integerTest_max > smallNum) && integerTest_distance > smallNum) {
          std::ostringstream desc;
          desc << " The finer Level " << fineLevel->getIndex()
               << " "<< F_box.min() << " "<< F_box.max()
               << " upper or lower limits are not divisible by the cell spacing "
               << dx_fineLevel << " \n Remainder of level box/dx: lower"
               << integerTest_min << " upper " << integerTest_max<< std::endl;
          throw InvalidGrid(desc.str(),__FILE__,__LINE__);
        }

      }
    }

  }



#endif
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
Grid::performConsistencyCheck() const
{
#if SCI_ASSERTION_LEVEL > 0

  // Verify that patches on a single level do not overlap
  for (int i = 0; i < (int)d_levels.size(); i++) {
    d_levels[i]->performConsistencyCheck();
  }

  // Check overlap between levels
  // See if patches on level 0 form a connected set (warning)
  // Compute total volume - compare if not first time

  //std::cerr << "Grid::performConsistencyCheck not done\n";
  
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
        std::ostringstream desc;
        desc << " The finer Level " << fineLevel->getIndex()
             << " "<< F_box.min() << " "<< F_box.max()
             << " can't lay outside of coarser level " << level->getIndex()
             << " "<< C_box.min() << " "<< C_box.max() << std::endl;
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
        std::ostringstream desc;
        desc << " The finer Level " << fineLevel->getIndex()
             << " "<< Fbox_min << " "<< Fbox_max
             << " upper or lower limits are not divisible by the cell spacing "
             << dx_fineLevel << " \n Remainder of level box/dx: lower"
             << integerTest_min << " upper " << integerTest_max<< std::endl;
        throw InvalidGrid(desc.str(),__FILE__,__LINE__);
      }
    }
  }
#endif
}

void
Grid::printStatistics() const
{
  std::cout << "Grid statistics:\n";
  std::cout << "Number of levels:\t\t" << numLevels() << '\n';

  unsigned long totalCells = 0;
  unsigned long totalPatches = 0;

  for( int i = 0; i < numLevels(); i++ ) {
    LevelP l = getLevel(i);
    std::cout << "Level " << i << ":\n";
    if (l->getPeriodicBoundaries() != IntVector(0,0,0))
      std::cout << "  Periodic boundaries:\t\t" << l->getPeriodicBoundaries()
           << '\n';
    std::cout << "  Number of patches:\t\t" << l->numPatches() << '\n';
    totalPatches += l->numPatches();
    double ppc = double(l->totalCells())/double(l->numPatches());
    std::cout << "  Total number of cells:\t" << l->totalCells() << " (" << ppc << " avg. per patch)\n";
    totalCells += l->totalCells();
  }
  std::cout << "Total patches in grid:\t\t" << totalPatches << '\n';
  double ppc = double(totalCells)/double(totalPatches);
  std::cout << "Total cells in grid:\t\t" << totalCells << " (" << ppc << " avg. per patch)\n";
  std::cout << "\n";
}

//////////
// Computes the physical boundaries for the grid
void
Grid::getSpatialRange( BBox & b ) const
{
  // just call the same function for all the levels
  for( int l = 0; l < numLevels(); l++ ) {
    getLevel( l )->getSpatialRange( b );
  }
}

////////// 
// Returns the boundary of the grid exactly (without
// extra cells).  The value returned is the same value
// as found in the .ups file.
void
Grid::getInteriorSpatialRange( BBox & b ) const
{
  // Just call the same function for all the levels.
  for( int l = 0; l < numLevels(); l++ ) {
    getLevel( l )->getInteriorSpatialRange( b );
  }
}

//__________________________________
// Computes the length in each direction of the grid
void
Grid::getLength( Vector & length, const std::string & flag ) const
{
  BBox b;
  // just call the same function for all the levels
  for( int l = 0; l < numLevels(); l++ ) {
    getLevel( l )->getSpatialRange( b );
  }
  length = ( b.max() - b.min() );
  if( flag == "minusExtraCells" ) {
    Vector dx = getLevel(0)->dCell();
    IntVector extraCells = getLevel(0)->getExtraCells();
    Vector ec_length = IntVector(2,2,2) * extraCells * dx;
    length = ( b.max() - b.min() )  - ec_length;
  }
}

Grid::stretchDescription
Grid::parseStretches(const ProblemSpecP& curr_ps)
{

  Grid::stretchDescription levelStretch;
  for (ProblemSpecP stretch_ps = curr_ps->findBlock("Stretch"); stretch_ps != 0; stretch_ps = stretch_ps->findNextBlock("Stretch"))
  {
    std::string axisName;
    if(!stretch_ps->getAttribute("axis", axisName))
      throw ProblemSetupException("Error, no specified axis for Stretch section: should be x, y, or z", __FILE__, __LINE__);
    int axis;
    if      (axisName == "x" || axisName == "X")
    {
      axis = 0;
    }
    else if (axisName == "y" || axisName == "Y")
    {
      axis = 1;
    }
    else if (axisName == "z" || axisName == "Z")
    {
      axis = 2;
    }
    else
    {
      throw ProblemSetupException("Error, invalid axis in Stretch section: " + axisName + ", should be x, y, or z.",
                                  __FILE__, __LINE__);
    }
    if (levelStretch.checkForPrevious(axis))
    {
      throw ProblemSetupException("Error, stretch axis already specified: "+axisName, __FILE__, __LINE__);
    }

    double globalUpperBound = -DBL_MAX;
    for(ProblemSpecP region_ps = stretch_ps->findBlock(); region_ps != 0; region_ps = region_ps->findNextBlock())
    {
      std::string spacingType;
      double regionUpper, regionLower;
      double spacingAtUpper, spacingAtLower;
      spacingType = region_ps->getNodeName();
      if (spacingType != "uniform" && spacingType != "linear")
      {
        throw ProblemSetupException("Error, invalid region shape in stretched grid: " + region_ps->getNodeName(),
                                    __FILE__, __LINE__);
      }
      if(region_ps->getAttribute("from",regionLower))
      {
        if(regionLower < globalUpperBound)
        {
          throw ProblemSetupException("Error, stretched grid regions must be specified in ascending order.",
                                      __FILE__, __LINE__);
        }
      }
      else
      {
        regionLower = DBL_MAX;
      }
      if (region_ps->getAttribute("to", regionUpper))
      {
        globalUpperBound = regionUpper;
      }
      else
      {
        regionUpper = DBL_MAX;
      }
      if (spacingType == "uniform")
      {
        if (!region_ps->getAttribute("spacing",spacingAtLower))
        {
          spacingAtLower = DBL_MAX;
        }
        spacingAtUpper = spacingAtLower;
      }
      else if (spacingType == "linear") // SHouldn't need the if per above error check
      {
        if (!region_ps->getAttribute("fromSpacing",spacingAtLower))
        {
          spacingAtLower = DBL_MAX;
        }
        if(!region_ps->getAttribute("toSpacing",spacingAtUpper))
        {
          spacingAtUpper = DBL_MAX;
        }
      }

      Grid::stretchRegion currRegion(spacingType,regionUpper,regionLower,spacingAtUpper,spacingAtLower);
      levelStretch.addRegion(axis,currRegion);
      globalUpperBound=regionUpper;
    }
  }
  return levelStretch;

}

void
Grid::stretchRegion::fillCells(int& start, int lowExtra, int highExtra, OffsetArray1<double>& faces) const
{
  if(shape == "uniform")
  {
    int n = SCIRun::Round((to-from)/fromSpacing);
    for(int i=-lowExtra;i<n+highExtra;i++)
    {
      faces[i+start] = from + i*fromSpacing;
    }
    start += n;
  }
  else
  {
    int n = countCells();
    double totalDistance = to-from;
    double a = fromSpacing;
    double b = toSpacing;
    bool switched = false;
    if(a > b)
    {
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
    for(int i=0;i<1000;i++)
    {
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
    if(switched)
    {
      a = a*pow(r, n+1);
      r = 1./r;
    }
    for(int i=-lowExtra;i<n+highExtra;i++)
    {
      faces[i+start] = from + xk(a, r, i);
    }
    start += n;
  }}

int
Grid::stretchRegion::countCells() const
{
  if (shape == "uniform")
  {
    return SCIRun::Round((to-from)/fromSpacing);
  }
  else
  {
    double a = SCIRun::Min(fromSpacing, toSpacing);
    double b = SCIRun::Max(fromSpacing, toSpacing);
    double totalDistance = to-from;
    double nn = log(b/a) / log((totalDistance+b)/(totalDistance+a)) -1;
    int n = static_cast<int>(nn+0.5);
    return n;
  }
}

SCIRun::Vector
Grid::stretchDescription::checkStretches(const SCIRun::BBox &extents, const int& procRank)
{

  Uintah::Point levelAnchor = extents.min();
  Uintah::Point levelHighPoint = extents.max();

  SCIRun::Vector spacing;

  for (int axis = 0; axis < 3; ++axis)
  {
    int numRegions = getRegionsPerAxis(axis);
    if (numRegions > 0)
    {
        spacing[axis] = getNan();
    }

    for (int region=0; region < numRegions; ++region)
    {
      stretchRegion* prevRegion = NULL;
      stretchRegion* nextRegion = NULL;
      if (region > 0)
      {
        prevRegion = getRegion(axis, region-1);
      }
      if (region < numRegions - 1)
      {
        nextRegion = getRegion(axis, region + 1);
      }
      stretchRegion* currRegion = getRegion(axis,region);
      if (currRegion->getFrom() == DBL_MAX)
      {
        if (prevRegion && prevRegion->getTo() != DBL_MAX)
        {
          currRegion->setFrom(prevRegion->getTo());
        }
        else if (region == 0)
        {
          currRegion->setFrom(levelAnchor(axis));
        }
        else
        {
          std::ostringstream msg;
          msg << "Stretch region from point not specified for region " << region << "." << std::endl;
          throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
        }
      }
      if (currRegion->getTo() == DBL_MAX)
      {
        if ( nextRegion && nextRegion->getFrom() != DBL_MAX )
        {
          currRegion->setTo(nextRegion->getFrom());
        }
        else if (region == getRegionsPerAxis(axis) - 1)
        {
          currRegion->setTo(levelHighPoint(axis));
        }
        else
        {
          std::ostringstream msg;
          msg << "Stretch region to point not specified for region " << region << "." << std::endl;
          throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
        }
      }
      if (currRegion->getFromSpacing() <= 0 || currRegion->getToSpacing() <= 0)
      {
        std::ostringstream msg;
        msg << "Grid spacing must be >= 0 (" << currRegion->getFromSpacing() << ", " << currRegion->getToSpacing() << ")." << std::endl;
        throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
      }
      if (currRegion->getShape() == "linear")
      {
        if (currRegion->getFromSpacing() == DBL_MAX)
        {
          if (prevRegion && prevRegion->getToSpacing() != DBL_MAX)
          {
            currRegion->setFromSpacing(prevRegion->getToSpacing());
          }
          else
          {
            std::ostringstream msg;
            msg << "Stretch region from spacing not specified for region " << region << "." << std::endl;
            throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
          }
        }
        if (currRegion->getToSpacing() == DBL_MAX)
        {
          if (nextRegion && nextRegion->getFromSpacing() != DBL_MAX)
          {
            // The original code has this as the equivalent of
            // currRegion->setToSpacing(nextRegion->getToSpacing());
            // I believe this to be a bug, so have 'corrected' it.  JBH - 10/31/2015
            currRegion->setToSpacing(nextRegion->getFromSpacing());
          }
          else
          {
            std::ostringstream msg;
            msg << "Stretch region to spacing not specified for region " << region << "." << std::endl;
            throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
          }
        }
      }
      else if (currRegion->getShape() == "uniform")
      {
        if (currRegion->getFromSpacing() == DBL_MAX)
        {
          if (prevRegion && prevRegion->getToSpacing() != DBL_MAX)
          {
            currRegion->setFromSpacing(prevRegion->getToSpacing());
          }
          else
          {
            std::ostringstream msg;
            msg << "Stretch region uniform spacing not specified for region " << region << "." << std::endl;
            throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
          }
        }
      }
      if (prevRegion && prevRegion->getToSpacing() != currRegion->getFromSpacing())
      {
        if (procRank == 0)
        {
          std::cerr << "WARNING: specifying two uniform sections with a different spacing can cause an erroneous grid ("
                    <<  prevRegion->getToSpacing() << ", " << currRegion->getFromSpacing() << "." << std::endl;
        }
      }
      if (prevRegion && prevRegion->getToSpacing() != currRegion->getFromSpacing())
      {
        std::ostringstream msg;
        msg << "Gap in stretch region from: " << prevRegion->getTo() << " to " << currRegion->getFrom() << "." << std::endl;
        throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
      }
      if (currRegion->getTo() < currRegion->getFrom())
      {
        std::ostringstream msg;
        msg << "Error, stretched grid to must be larger than from." << std::endl;
        throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
      }
      if (currRegion->getShape() == "linear")
      {
        // If toSpacing == fromSpacing then convert this into a uniform section,
        // since the grid generation numerics have a singularity at that point
        if (currRegion->getFromSpacing() == currRegion->getToSpacing())
        {
          currRegion->setShape("uniform");
        }
      }
      if (currRegion->getShape() == "uniform")
      {
        // Check that dx goes nicely into the range
        double ncells = (currRegion->getTo() - currRegion->getFrom())/currRegion->getFromSpacing();
        if (SCIRun::Fraction(ncells) > 1e-4 && SCIRun::Fraction(ncells) < 1-1e-4)
        {
          std::ostringstream msg;
          msg << "Error:  Uniform region is not an integer multiple of the cell spacing." << std::endl;
          throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
        }
        int n = SCIRun::Round(ncells);
        // Recompute newdx to avoid roundoff issues
        double newdx = (currRegion->getTo() - currRegion->getFrom())/n;
        currRegion->setToSpacing(newdx);
        currRegion->setFromSpacing(newdx);
      }
    }
    if (numRegions > 0)
    {
      stretchRegion* firstRegion = getRegion(axis,0);
      stretchRegion* lastRegion  = getRegion(axis,numRegions-1);
      if(firstRegion->getFrom() > levelAnchor(axis) || lastRegion->getTo() < levelHighPoint(axis))
      {
        std::ostringstream msg;
        msg << "Error, stretched grid specification does not cover entire axis." << std::endl;
        throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
      }
    }
  }
  if (procRank == 0 && stretchCount() !=0)
  {
    std::cerr << "Stretched grid information: " << std::endl;
    for (int axis = 0; axis < 3; ++axis)
    {
      if (axis == 0)
      {
        std::cerr << "x";
      }
      if (axis == 1)
      {
        std::cerr << "y";
      }
      if (axis == 2)
      {
        std::cerr << "z";
      }
      std::cerr << " axis: " << std::endl;
      int numRegions = getRegionsPerAxis(axis);
      for (int region = 0; region < numRegions; ++region)
      {
        stretchRegion* currRegion = getRegion(axis,region);
        std::cerr << "\t" << currRegion->getShape()
                  << ": from " << currRegion->getFrom() << "(" << currRegion->getFromSpacing()
                  << ") to "   << currRegion->getTo()   << "(" << currRegion->getToSpacing()
                  << "), " << currRegion->countCells() << " cells. " << std::endl;
      }
    }
    std::cerr << std::endl;
  }
  return (spacing);
}

Grid::LevelBox
Grid::parseBox(      ProblemSpecP         box_ps,
               const bool                 haveLevelSpacing,
               const bool                 havePatchSpacing,
               const SCIRun::Vector      &currentSpacing)
{
  SCIRun::Point lower, upper;
  box_ps->require("lower", lower);
  box_ps->require("upper", upper);

  SCIRun::BBox boxExtents(lower,upper);

  SCIRun::IntVector boxResolution;
  SCIRun::Vector    boxSpacing(-1.0, -1.0, -1.0);
  if (box_ps->get("resolution", boxResolution))
  {
    if (haveLevelSpacing)
    {
      throw ProblemSetupException("Cannot specify level spacing and patch resolution",
                                  __FILE__, __LINE__);
    }
    else
    {
      SCIRun::Vector newSpacing = (upper-lower)/boxResolution;
      if (havePatchSpacing)
      {
        SCIRun::Vector diff = currentSpacing - newSpacing;
        if (diff.length() > 1.e-14) // Exception: Mismatching resolutions
        {
          throw ProblemSetupException("Using patch resolution, and the patch spacings are inconsistent", __FILE__, __LINE__);
        }
      }
      else
      {
        boxSpacing = newSpacing;
      }
    }
  } // Resolution subsection parsed
  SCIRun::IntVector boxExtraCells;
  box_ps->getWithDefault("extraCells", boxExtraCells, d_extraCells);
  stretchDescription nullStretch;
  return LevelBox(boxExtents, boxExtraCells, boxSpacing, nullStretch);
}

Grid::LevelBox
Grid::parseLevel(ProblemSpecP& level_ps, const size_t levelIndex, const int myProcRank)
{

  // Create an intentionally inverted bounding box so that the first real bounding box from
  // a parse gets assigned properly.
  SCIRun::BBox levelExtents(SCIRun::Point(DBL_MAX,DBL_MAX,DBL_MAX),
                            SCIRun::Point(DBL_MIN,DBL_MIN,DBL_MIN));

  bool haveLevelSpacing = false;
  SCIRun::Vector currentSpacing = SCIRun::Vector(0.0, 0.0, 0.0);
  if (level_ps->get("spacing",currentSpacing))
  {
    haveLevelSpacing = true;
  }

  SCIRun::IntVector levelExtraCells = SCIRun::IntVector(0, 0, 0);
  bool havePatchSpacing = false;
  ProblemSpecP box_ps = level_ps->findBlock("Box");
  while (box_ps) // Loop through all boxes
  {
    LevelBox currentBox=parseBox(box_ps, haveLevelSpacing, havePatchSpacing, currentSpacing);
    // CHeck validity of box.
    SCIRun::Point upper = currentBox.getBoxExtents().max();
    SCIRun::Point lower = currentBox.getBoxExtents().min();
    for (int axis = 0; axis < 3; ++axis)
    {
      if (lower(axis) >= upper(axis))
      {
        std::ostringstream msg;
        msg << std::endl << "Computational Domain Input Error: Level("<< levelIndex << ")"
            << std::endl << "The lower corner " << lower
            << " must be smaller than the upper corner " << upper << std::endl;
        throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
      }
    }
    levelExtents.extend(currentBox.getBoxExtents()); // Extend the level to include the box

    // We can't get here without exception unless haveLevelSpacing is false, meaning we
    // either have spacing set at the patch level, or not at all.
    if ( currentBox.hasSpacing() )
    {
      havePatchSpacing = true;
      currentSpacing = currentBox.getSpacing();
    }
    // bulletproofing
    if (haveLevelSpacing || havePatchSpacing) {
      for(int dir = 0; dir < 3; dir ++)
      {
        if (upper(dir) - lower(dir) <= 0.0)
        {
          std::ostringstream msg;
          msg << "\nComputational Domain Input Error: Level("<< levelIndex << ")"
              << " \n The computational domain " << lower<<", " << upper
              << " must have a positive distance in each coordinate direction  " << upper-lower << std::endl;
          throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
        }
        if (currentSpacing[dir] > (upper(dir)-lower(dir)) || currentSpacing[dir] < 0)
        {
          std::ostringstream msg;
          msg << "\nComputational Domain Input Error: Level("<< levelIndex << ")"
              << " \n The spacing " << currentSpacing
              << " must be less than the upper - lower corner and positive " << upper << std::endl;
          throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
        }
      }
    }
    levelExtraCells = Max(levelExtraCells,currentBox.getExtraCells());

    // Done with all processing for this box, get the next one.
    box_ps = box_ps->findNextBlock("Box");
  } // Loop through all boxes

  stretchDescription     stretches = parseStretches(level_ps);
  SCIRun::Vector        newSpacing = stretches.checkStretches(levelExtents, myProcRank);

  int stretchCount = stretches.stretchCount();
  if (stretchCount == 3)
  {
    // Only instance in which we need to move over the spacing
    currentSpacing = newSpacing;
  }

  if (!haveLevelSpacing && !havePatchSpacing && stretchCount != 3)
  {
    throw ProblemSetupException("Box resolution is not specified", __FILE__, __LINE__);
  }
  return LevelBox(levelExtents, levelExtraCells, currentSpacing, stretches);

}

SCIRun::OffsetArray1<double>
Grid::assignStretchedFaces(      stretchDescription *stretches,
                           const LevelBox           *levelInfo,
                           const SCIRun::IntVector  &extraCells,
                           const int                &axis)
{
  SCIRun::OffsetArray1<double> faces;
  int numRegions = stretches->getRegionsPerAxis(axis);
  if (numRegions == 0)
  {
    // Uniform spacing.
    int     low     = -extraCells[axis];
    double  start   = (levelInfo->getAnchor())[axis];
    double  end     = (levelInfo->getHighPoint())[axis];
    double  dx      = (levelInfo->getSpacing())[axis];
    int     ncells  = SCIRun::Round((start-end)/dx);
    int     high    = ncells + extraCells[axis];
    faces.resize(low, high + 1);
    for (int index = low; index <= high; ++index)
    {
      // Shouldn't this be faces[index] = start + index*dx?? JBH - 10/31/2015
      faces[index] = start + faces[index]*dx;
    }
  }
  else
  {
    int count = 0;
    for (int region = 0; region < numRegions; ++region)
    {
      count += stretches->getRegion(axis,region)->countCells();
    }
    count += 2*extraCells[axis];
    int low = -extraCells[axis];
    faces.resize(low, count);
    int start = 0;
    for (int index = 0; index < numRegions; ++index)
    {
      int lowExtra = 0;
      int highExtra = 0;
      if (index == 0)
      {
        lowExtra=extraCells[axis];
      }
      if (index == numRegions - 1)
      {
        highExtra = extraCells[axis] + 1;
      }
      stretchRegion* currRegion = stretches->getRegion(axis,index);
      currRegion->fillCells(start, lowExtra, highExtra, faces);
    }
  }
  return faces;
}

void
Grid::parsePatches(const ProblemSpecP&      level_ps,
                         LevelP&            level,
                   const SCIRun::IntVector& levelLowCell,
                   const SCIRun::IntVector& levelHighCell,
                   const SCIRun::IntVector& extraCells,
                   const int                numProcs,
                   const int                myRank)
{
  double epsilon = 1.e-14;
  SCIRun::Vector epsVector(epsilon);

  ProblemSpecP box_ps = level_ps->findBlock("Box");
  while (box_ps)
  {
    Point lower, upper;
    box_ps->require("lower", lower);
    box_ps->require("upper", upper);

    SCIRun::Point lowerPoint = lower*epsilon + epsVector;
    SCIRun::Point upperPoint = upper*epsilon + epsVector;
    IntVector lowerCell = level->getCellIndex(lower + Abs(lowerPoint.asVector()));
    IntVector upperCell= level->getCellIndex(upper + Abs(upperPoint.asVector()));

    SCIRun::Point lowComp = level->getNodePosition(lowerCell);
    SCIRun::Point upperComp = level->getNodePosition(upperCell);

    double diffLow = (lowComp - lower).length();
    double diffUpper = (upperComp - upper).length();

    double maxComponentLower = Abs(Vector(lower)).maxComponent();
    double maxComponentUpper = Abs(Vector(upper)).maxComponent();

    if (diffLow > maxComponentLower * epsilon)
    {
      std::cerr << std::setprecision(16) << "lower     = " << lower << std::endl
                << std::setprecision(16) << "lowCell   = " << lowerCell << std::endl
                << std::setprecision(16) << "upperCell = " << upperCell << std::endl
                << std::setprecision(16) << "lowComp   = " << lowComp << std::endl;
      throw ProblemSetupException("Box lower corner does not align with grid.", __FILE__, __LINE__);
    }
    if (diffUpper > maxComponentUpper * epsilon)
    {

      std::cerr << std::setprecision(16) << "upper     = " << upper << std::endl
                << std::setprecision(16) << "lowCell   = " << lowerCell << std::endl
                << std::setprecision(16) << "upperCell = " << upperCell << std::endl
                << std::setprecision(16) << "upperComp = " << upperComp << std::endl;
      throw ProblemSetupException("Box upper corner does not align with grid.", __FILE__, __LINE__);
    }

    IntVector resolution(upperCell - lowerCell);
    if (resolution.x() < 1 || resolution.y() < 1 || resolution.z() < 1)
    {
      std::cerr << "Upper cell: " << upperCell << " lowerCell: " << lowerCell << std::endl;
      throw ProblemSetupException("Degenerate patch", __FILE__, __LINE__);
    }

    SCIRun::IntVector patches;          // Will store the partition dimensions returned by runPartition3D
    SCIRun::IntVector tempPatches;      // For 2D case, stores results of runPartition2D before sorting
    double  autoPatchValue = 0;         // The ideal ratio of patches per processor.  Usually 1, 1.5 for some load balancers

    std::map<std::string, std::string> patchAttributes;         // Hash for parsing XML attributes
    if (box_ps->get("autoPatch", autoPatchValue))
    {
      // AutoPatchValue must be >= 1, otherwise it will generate fewer patches than processors
      if ( autoPatchValue < 1)
      {
        throw ProblemSetupException("autoPatch value must be greater than 1", __FILE__, __LINE__);
      }

      patchAttributes.clear();
      box_ps->getAttributes(patchAttributes);
      proc0cout << "Automatically laying out patches." << std::endl;

      int targetPatches = static_cast<int> (numProcs * autoPatchValue);

      Primes::FactorType factors;
      int numFactors = Primes::factorize(targetPatches, factors);
      std::list<int> primeList;
      for (int index = 0; index < numFactors; ++index)
      {
        primeList.push_back(factors[index]);
      }

      // First check all possible values for a 2D partition.  If no valid value is found, perform a normal 3D partition
      if ( patchAttributes["flatten"] == "x" || resolution.x() ==1 )
      {
        ares_ = resolution.y();
        bres_ = resolution.z();
        tempPatches = run_partition2D(primeList);
        patches = SCIRun::IntVector(1, tempPatches.x(), tempPatches.y());
      }
      else if (patchAttributes["flatten"] == "y" || resolution.y() == 1)
      {
        ares_ = resolution.x();
        bres_ = resolution.z();
        tempPatches = run_partition2D(primeList);
        patches = SCIRun::IntVector(tempPatches.x(), 1, tempPatches.z());
      }
      else if (patchAttributes["flatten"] == "z" || resolution.z() == 1)
      {
        ares_ = resolution.x();
        bres_ = resolution.y();
        tempPatches = run_partition2D(primeList);
        patches = SCIRun::IntVector(tempPatches.x(), tempPatches.y(), 1);
      }
      else
      {
        // 3D partition case
        ares_ = resolution.x();
        bres_ = resolution.y();
        cres_ = resolution.z();

        patches = run_partition3D(primeList);
      }
    }
    else
    {
      // Autopatchin is not enabled, get th epatch field
      box_ps->getWithDefault("patches", patches, IntVector(1,1,1));
      nf_ = 0;
    }

    // If the value of the norm, nf_, is too high, then user chose a bad number of processors.
    if ( nf_ > 3 )
    {
          std::cout << std::endl << "********************"
                    << std::endl << "*"
                    << std::endl << "* WARNING:"
                    << std::endl << "* The patch to processor ratio you chose"
                    << std::endl << "* does not factor well into patches.  Consider"
                    << std::endl << "* using a different number of processors."
                    << std::endl << "*"
                    << std::endl << "********************"
                    << std::endl << std::endl;
    }

    proc0cout << "Level: " << level->getIndex() << " --> "
              << "Patch layout: \t\t(" << patches.x() << ", "
              << patches.y() << ", " << patches.z() << ")" << std::endl;

    IntVector refineRatio = level->getRefinementRatio();
    level->setPatchDistributionHint(patches);
    for (int x = 0; x < patches.x(); ++x)
    {
      for (int y = 0; y < patches.y(); ++y)
      {
        for (int z = 0; z < patches.z(); ++z)
        {
          SCIRun::IntVector startCell = resolution*SCIRun::IntVector(x,y,z)/patches + lowerCell;
          SCIRun::IntVector endCell   = resolution*SCIRun::IntVector(x+1, y+1, z+1)/patches + lowerCell;
          SCIRun::IntVector inStartCell(startCell);
          SCIRun::IntVector inEndCell(endCell);

          // This algorithm for finding extra cells is not sufficient for AMR levels
          // since it only finds extra cells on the domain boundary.
          // The only way to find extra cells for them is to do neighbor queries, so we will
          // potentially adjust extra cells in Patch::setBCType (called from Level::setBCTypes)
          startCell -= SCIRun::IntVector(startCell.x() == levelLowCell.x() ? extraCells.x():0,
                                         startCell.y() == levelLowCell.y() ? extraCells.y():0,
                                         startCell.z() == levelLowCell.z() ? extraCells.z():0);
          endCell   -= SCIRun::IntVector(endCell.x() == levelHighCell.x() ? extraCells.x():0,
                                         endCell.y() == levelHighCell.y() ? extraCells.y():0,
                                         endCell.z() == levelHighCell.z() ? extraCells.z():0);

          if (inStartCell.x() % refineRatio.x() || inEndCell.x() % refineRatio.x() ||
              inStartCell.y() % refineRatio.y() || inEndCell.y() % refineRatio.y() ||
              inStartCell.z() % refineRatio.z() || inEndCell.z() % refineRatio.z())
          {
            std::ostringstream msg;
            msg << "The finer patch boundaries (" << inStartCell << "->" << inEndCell
                << ") do not coincide with a coarse cell." << std::endl
                << "(i.e. they are not divisible by the refinement ratio " << refineRatio << ")"
                << std::endl;
            throw InvalidGrid(msg.str(), __FILE__, __LINE__);
          }
          level->addPatch(startCell, endCell, inStartCell, inEndCell, this);
        }
      }
    }
    // Move on to the next box
    box_ps = box_ps->findNextBlock("Box");
  }
  return;
}

bool
Grid::specIsAMR(const ProblemSpecP &ps) const
{
  bool isAMR = false;
  ProblemSpecP tmp_ps = ps->findBlock("doAMR");
  if (tmp_ps)
  {
    tmp_ps->get("doAMR", isAMR);
  }
  else if (ps->findBlock("AMR")) {
      isAMR = true;
  }
  return isAMR;
}

void
Grid::parseLevelSet(  const ProblemSpecP & grid_ps
                    , const int            numProcs
                    , const int            myProcRank
                    , const size_t         globalIndexOffset
                    , const size_t         levelSetIndex
                    , const bool           do_AMR
                    , const bool           do_MultiScale
                   )
{
  size_t localIndexOffset = 0;
  SCIRun::Point setAnchor(DBL_MAX, DBL_MAX, DBL_MAX);

  ProblemSpecP level_ps = grid_ps->findBlock("Level");
  while (level_ps) // Loop through all levels
  {
    size_t levelIndex = localIndexOffset + globalIndexOffset;
    Grid::LevelBox levelInfo = parseLevel(level_ps, levelIndex, myProcRank);
    if (!levelInfo.hasSpacing() && levelInfo.stretchCount() != 3)
    {
      throw ProblemSetupException("Level resolution is not specified", __FILE__, __LINE__);
    }
//    if (levelIndex == curreLevelSubset->getMinLevelIndex())
    if (localIndexOffset == 0)
    {
      setAnchor = levelInfo.getAnchor().asPoint();
    }

    SCIRun::IntVector extraCells = levelInfo.getExtraCells();
    if (extraCells != d_extraCells && d_extraCells != IntVector(0,0,0))
    {
      proc0cout << "Warning:  Input file overrides extraCells specification via level Set "
                << levelSetIndex << ", level " << localIndexOffset <<"." << std::endl
                << "\tCurrent extraCell: " << levelInfo.getExtraCells() << "." << std::endl;
    }
    LevelFlags flags;
    flags.set(LevelFlags::isAMR, do_AMR);
    flags.set(LevelFlags::isMultiScale, do_MultiScale);
    LevelP level = addLevel(setAnchor, levelInfo.getSpacing(), flags);
    level->setExtraCells(extraCells);

    if(levelInfo.stretchCount() != 0)
    {
      stretchDescription *stretches = levelInfo.getStretchDescription();
      for (int axis = 0; axis < 3; ++axis)
      {
        SCIRun::OffsetArray1<double> faces = assignStretchedFaces(stretches,&levelInfo,extraCells,axis);
        level->setStretched(static_cast<Grid::Axis> (axis), faces);
      }
    }

    // Second pass - set up patches and cells
    IntVector anchorCell(level->getCellIndex((levelInfo.getAnchor() + Vector(1.e-14,1.e-14,1.e-14)).asPoint()));
    IntVector highPointCell(level->getCellIndex((levelInfo.getHighPoint() + Vector(1.e-14,1.e-14,1.e-14)).asPoint()));
    parsePatches(level_ps, level, anchorCell, highPointCell, extraCells, numProcs, myProcRank );

    SCIRun::IntVector periodicBoundaries;
    if (level_ps->get("periodic",periodicBoundaries))
    {
      level->finalizeLevel(periodicBoundaries.x() != 0,
                           periodicBoundaries.y() != 0,
                           periodicBoundaries.z() != 0);
    }
    else
    {
      level->finalizeLevel();
    }
    ++localIndexOffset;

    // Done with all processing for this level, get the next one.
    level_ps = level_ps->findNextBlock("Level");
  }
}

void
Grid::problemSetup(  const ProblemSpecP   & params
                   , const ProcessorGroup * pg
                   ,       bool             do_AMR
                   ,       bool             do_MultiScale
                  )
{

  ProblemSpecP grid_ps = params->findBlock("Grid");
  if (!grid_ps) {
    throw ProblemSetupException("Error:  Grid description not found in .ups file!", __FILE__, __LINE__);
  }

  SCIRun::Point  gridAnchor; // Minimum point in grid
  bool           fileIsAMR = specIsAMR(params);
  ProblemSpecP   level_ps;

  ProblemSpecP    levelset_ps = params->findBlock("Grid")->findBlock("LevelSet");
  if (!levelset_ps) { // Only one level set parsed from the level section of the current block.
    level_ps = grid_ps;
    int levelIndex         = 0;
    parseLevelSet(level_ps, pg->size(), pg->myrank(), levelIndex, 0, fileIsAMR, do_MultiScale);

    // Determine size of newly parsed subset and create an empty subset to house it
    d_levelSet.createEmptySubsets(1);
    LevelSubset* currLevelSubset = d_levelSet.getSubset(0);

    size_t numLevels = d_levels.size();
    std::string componentName = "";
    ProblemSpecP simCompSpec = params->findBlock("SimulationComponent");
    simCompSpec->getAttribute("type",componentName);
    if (componentName == "") {
      throw ProblemSetupException("Error:  SimulationComponent not found.", __FILE__, __LINE__);
    }
    for (size_t currIndex = levelIndex; currIndex < numLevels; ++currIndex) {
      // Grab rep of the level and add it to the subset
      currLevelSubset->add(d_levels[currIndex].get_rep());
      // Place a pointer to the level subset in the level for easy retrieval
      d_levels[currIndex]->setLevelSubset(currLevelSubset);
      // Store a pointer to the subset to which this level is assigned
      d_subsetOfLevel.push_back(0);
    }
    d_levelSubsetLabels.push_back("Solo Level Set");
    d_levelSubsetComponentNames.push_back(componentName);


  }
  else // parse LevelSets
  {
    size_t levelIndex = 0;
    size_t currentSubsetIndex = 0;
    while (levelset_ps) {  // iterate through each level set tag

      // Set default levelSet name
      std::string componentName="NULL";
      levelset_ps->require("Component",componentName);
      std::ostringstream setNameStream;
      setNameStream << "Level Set " << std::left << levelIndex;
      std::string setName = setNameStream.str();
      levelset_ps->getAttribute("label",setName);

      // And override if present
      d_levelSubsetLabels.push_back(setName);
      d_levelSubsetComponentNames.push_back(componentName);
      parseLevelSet(levelset_ps, pg->size(), pg->myrank(), levelIndex, currentSubsetIndex, fileIsAMR, do_MultiScale);

      // Determine size of newly parsed subset and create an empty subset to house it
      d_levelSet.createEmptySubsets(1);
      LevelSubset* currLevelSubset = d_levelSet.getSubset(currentSubsetIndex);

      size_t numLevels = d_levels.size();
      for (size_t currIndex = levelIndex; currIndex < numLevels; ++currIndex) {

        // Grab rep of the level and add it to the subset
        currLevelSubset->add(d_levels[currIndex].get_rep());

        // Place a pointer to the level subset in the level for easy retrieval
        d_levels[currIndex]->setLevelSubset(currLevelSubset);
      }
      levelIndex += numLevels;
      ++currentSubsetIndex;
      levelset_ps = levelset_ps->findNextBlock("LevelSet"); // Find next Levelset block
    }
  }
  assignSubsetToLevels();

  int num_patches = 0;
  for (int i = 0; i < numLevels(); ++i) {
    num_patches += d_levels[i]->numPatches();
  }

  int num_threads = Uintah::Parallel::getNumThreads();
  int num_procs = pg->size();
  bool using_threads = (num_threads > 0) ? true : false;
  bool undersubscribed_procs   = (!using_threads) && (num_patches < num_procs);
  bool undersubscribed_threads = (using_threads)  && (num_patches < (num_procs * num_threads));

  // if not doing AMR and number of patches is less than available resources (threads or procs) - throw
  if (!do_AMR && (undersubscribed_threads  || undersubscribed_procs)) {
    throw ProblemSetupException("Number of patches must be >= the number of processes in an mpi run.", __FILE__, __LINE__);
  }

  proc0cout << "Level Sets: " << std::endl;
  for (int i=0; i < d_levelSet.size(); ++i) {
    proc0cout << "  Set # " << std::left << i << " (" << d_levelSubsetComponentNames[i]
              << ") :" << *(d_levelSet.getSubset(i))
              << "  \"" << d_levelSubsetLabels[i] << "\""
              << std::endl;
  }
}

namespace Uintah
{
  std::ostream& operator<<(std::ostream& out, const Grid& grid)
  {
    out.setf(std::ios::floatfield);
    out.precision(6);
    out << "Grid has " << grid.numLevels() << " level(s)" << std::endl;
    for ( int levelIndex = 0; levelIndex < grid.numLevels(); levelIndex++ ) {
      LevelP level = grid.getLevel( levelIndex );
      out << "  Level " << level->getID() 
          << ", indx: "<< level->getIndex()
          << " has " << level->numPatches() << " patch(es)" << std::endl;
      for ( Level::patchIterator patchIter = level->patchesBegin(); patchIter < level->patchesEnd(); patchIter++ ) {
        const Patch* patch = *patchIter;
        out <<"    "<< *patch << std::endl;
      }
    }
    return out;
  }
}

// This is O(p).
bool
Grid::operator==( const Grid & othergrid ) const
{
  if( numLevels() != othergrid.numLevels() ) {
    return false;
  }
  for( int i = 0; i < numLevels(); i++ ) {
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
    std::vector<Region> r1, r2, difference;
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

IntVector Grid::run_partition3D(std::list<int> primes)
{
  partition3D(primes, 1, 1, 1);
  return IntVector(af_, bf_, cf_);
}

void Grid::partition3D(std::list<int> primes, int a, int b, int c)
{
  // base case: no primes left, compute the norm and store values
  // of a,b,c if they are the best so far.
  int ratio_ab, ratio_bc, ratio_ac, ratio_abres, ratio_acres, ratio_bcres, abdiff, acdiff, bcdiff;

  ratio_ab = std::max(a,b)/std::min(a,b);
  ratio_bc = std::max(b,c)/std::min(b,c);
  ratio_ac = std::max(a,c)/std::min(a,c);

  ratio_abres = std::max(ares_,bres_)/std::min(ares_,bres_);
  ratio_acres = std::max(ares_,cres_)/std::min(ares_,cres_);
  ratio_bcres = std::max(bres_,cres_)/std::min(bres_,cres_);

  abdiff = ratio_ab - ratio_abres;
  acdiff = ratio_ac - ratio_acres;
  bcdiff = ratio_bc - ratio_bcres;
  if( primes.size() == 0 ) {
    double new_norm = sqrt( static_cast<double> (abdiff*abdiff + bcdiff * bcdiff + acdiff*acdiff));
//  if( primes.size() == 0 ) {
//    double new_norm = sqrt( (double)(max(a,b)/min(a,b) - max(ares_,bres_)/min(ares_,bres_)) *
//                            (max(a,b)/min(a,b) - max(ares_,bres_)/min(ares_,bres_)) +
//                            (max(b,c)/min(b,c) - max(bres_,cres_)/min(bres_,cres_)) *
//                            (max(b,c)/min(b,c) - max(bres_,cres_)/min(bres_,cres_)) +
//                            (max(a,c)/min(a,c) - max(ares_,cres_)/min(ares_,cres_)) *
//                            (max(a,c)/min(a,c) - max(ares_,cres_)/min(ares_,cres_))
//                          );

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
    int ratio_ab = std::max(a,b)/std::min(a,b);
    int ratio_abres = std::max(ares_,bres_)/std::min(ares_,bres_);

    double new_norm = static_cast<double> (ratio_ab - ratio_abres);
    //double new_norm = (double)max(a,b)/min(a,b) - max(ares_,bres_)/min(ares_,bres_);

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

const Patch *
Grid::getPatchByID( int patchid, int startingLevel ) const
{
  const Patch* patch = NULL;
  for( int i = startingLevel; i < numLevels(); i++ ) {
    LevelP checkLevel = getLevel(i);
    int levelBaseID = checkLevel->getPatch(0)->getID();
    if (patchid >= levelBaseID && patchid < levelBaseID+checkLevel->numPatches()) {
      patch = checkLevel->getPatch(patchid-levelBaseID);
      break;
    }
  }
  return patch;
}

void
Grid::assignBCS( const ProblemSpecP & grid_ps, LoadBalancer * lb )
{
  for( int l = 0; l < numLevels(); l++ )
  {
    LevelP level = getLevel( l );
    level->assignBCS( grid_ps, lb );
  }
}

void
Grid::assignBCS( const LevelSet &currLevelSet, const ProblemSpecP & grid_ps, LoadBalancer * lb)
{
  size_t numSubsets = currLevelSet.size();
  for (size_t setIndex = 0; setIndex < numSubsets; ++setIndex) {
    const LevelSubset* currLevelSubset=currLevelSet.getSubset(setIndex);
    size_t levelsInSubset = currLevelSubset->size();
    for (size_t indexInSubset = 0; indexInSubset < levelsInSubset; ++indexInSubset)
    {
      LevelP level = getLevel(currLevelSubset->get(indexInSubset)->getIndex());
      level->assignBCS(grid_ps, lb);
    }
  }
}

void
Grid::setExtraCells( const IntVector & ex )
{
  if( numLevels() > 0 ) {
     throw ProblemSetupException("Cannot set extraCells after grid setup",
                                 __FILE__, __LINE__);
     return;
  }
  d_extraCells = ex;
}

void
Grid::createLevelSubsets(int num_sets)
{
  d_levelSet.createEmptySubsets(num_sets);
  ASSERTEQ(num_sets, d_levelSet.size());
}

void
Grid::assignSubsetToLevels()
{
  int numSubsets = d_levelSet.size();
  d_subsetOfLevel.clear();
  d_subsetOfLevel.resize(d_levels.size());

  for (int subsetIndex = 0; subsetIndex < numSubsets; ++subsetIndex) {
    const LevelSubset* currSubset = d_levelSet.getSubset(subsetIndex);
    int levelsInSubset = currSubset->size();
    for (int indexInSubset = 0; indexInSubset < levelsInSubset; ++indexInSubset) {
      int levelIndex = currSubset->get(indexInSubset)->getIndex();
      d_subsetOfLevel[levelIndex] = subsetIndex;
      d_levels[levelIndex]->setSubsetIndex(subsetIndex);
    }
  }
}

void
Grid::copySubsetData(const GridP& copyFrom) {
  int destinationSubsetCount = copyFrom->numLevelSets();
  ASSERTEQ(this->numLevelSets(), destinationSubsetCount);
  d_levelSubsetLabels.clear();
  d_levelSubsetComponentNames.clear();
  d_levelSubsetLabels.resize(destinationSubsetCount);
  d_levelSubsetComponentNames.resize(destinationSubsetCount);
  for (int subsetIndex = 0; subsetIndex < destinationSubsetCount; ++subsetIndex) {
    int levelIndex = copyFrom->getLevelSubset(subsetIndex)->get(0)->getIndex();
    d_levelSubsetLabels[subsetIndex] = copyFrom->getSubsetLabel(levelIndex);
    d_levelSubsetComponentNames[subsetIndex] = copyFrom->getSubsetComponentName(levelIndex);
  }
}
