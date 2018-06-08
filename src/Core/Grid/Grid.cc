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
#include <Core/Util/DOUT.hpp>
#include <Core/Util/StringUtil.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/XMLUtils.h>

#include <climits>
#include <cfloat>
#include <iomanip>
#include <iostream>

using namespace Uintah;

namespace {
  Dout g_dbg("GRID", "Grid", "prints out each patch's index space", false);
}

//______________________________________________________________________
//

Grid::Grid()
{
  // Initialize values that may be used for the autoPatching calculations
  af_   =  0;
  bf_   =  0;
  cf_   =  0;
  nf_   = -1;
  ares_ =  0;
  bres_ =  0;
  cres_ =  0;
  d_extraCells = IntVector(0,0,0);
}

Grid::~Grid()
{
}
//______________________________________________________________________
//
const LevelP &
Grid::getLevel( int l ) const
{
  ASSERTRANGE( l, 0, numLevels() );
  return d_levels[ l ];
}

//______________________________________________________________________
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

  IntVector lowIndex;
  IntVector highIndex;
  bool      foundLowIndex    = false;
  bool      foundHighIndex   = false;

  IntVector interiorLowIndex, interiorHighIndex;
  bool      foundInteriorLowIndex = false;
  bool      foundInteriorHighIndex = false;

  // int       nnodes           = -1;
  // bool      foundNNodes      = false;

  // Std::Vector    lower, upper;
  // bool      foundLower       = false;
  // bool      foundUpper       = false;


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

  if( !foundId || !foundProc || !foundLowIndex || !foundHighIndex ) {
    throw InternalError("Grid::parsePatchFromFile() - Missing a <Patch> child tag...", __FILE__, __LINE__ );
  }

  if( !foundInteriorLowIndex )  { interiorLowIndex = lowIndex; }
  if( !foundInteriorHighIndex ) { interiorHighIndex = highIndex; }

  level->addPatch( lowIndex, highIndex, interiorLowIndex, interiorHighIndex, this, id );

  procMapForLevel.push_back( proc ); // corresponds to DataArchive original timedata.d_patchInfo[levelIndex].push_back(pi);

  return doneWithPatch;

} // end parsePatchFromFile()


//______________________________________________________________________
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
  // int       totalCells       = -1;

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
  // bool      foundExtraCells  = false;

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

        level = this->addLevel( anchor, dcell, id );

        level->setExtraCells( extraCells );
      }

      numPatchesRead++;

      // At this point, we should be done reading in <Level> information, so we should go ahead and
      // create the level... if for no other reason that the Patches have to be added to it...
      if( !foundCellSpacing ) {
        throw InternalError("Grid::parseLevelFromFile() - Did not find <cellspacing> point", __FILE__, __LINE__ );
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
      
/*`==========TESTING==========*/
      if( pieces[0] == "<nonCubic>" ) {
        // This conditional is not necessary and is here for backwards compatibility.  
        //  Remove it after 03/2018  -Todd
      } 
/*===========TESTING==========`*/
      else if( pieces[0] == "<numPatches>" ) {
        numPatches = atoi( pieces[1].c_str() );
      }
      else if( pieces[0] == "<totalCells>" ) {
        // totalCells = atoi( pieces[1].c_str() );
      }
      else if( pieces[0] == "<extraCells>" ) {
        extraCells = IntVector::fromString( pieces[1] );
        // foundExtraCells = true;
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
            
//______________________________________________________________________
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
Grid::parseGridFromFile( FILE * fp, std::vector< std::vector<int> > & procMap )
{
  int  num_levelsRead  = 0;
  int  num_levels      = 0;
  (void) num_levels; // Removes an unused var warning
  
  bool doneWithGrid   = false;
  bool foundLevelTag  = false;

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
      std::vector<int> & procMapForLevel = procMap[ num_levelsRead ];
 
      num_levelsRead++;

      parseLevelFromFile( fp, procMapForLevel );
    }
    else if( line == "" ) {
      break; // end of file reached.
    }
    else {
      std::vector< std::string > pieces = UintahXML::splitXMLtag( line );
      if( pieces[0] == "<numLevels>" ) {
        num_levels = atoi( pieces[1].c_str() );
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

  // Verify that the <numLevels> tag matches the actual number of
  // levels parsed.  If not, then there is an error in the xml file
  // Most likely it was corrupted or written out incorrectly.
  ASSERTEQ( num_levels, num_levelsRead );

  return doneWithGrid;

} // end parseGridFromFile()


//______________________________________________________________________
// Read in the grid information (from grid.xml) in binary.
void
Grid::readLevelsFromFileBinary( FILE * fp, std::vector< std::vector<int> > & procMap )
{
  int    num_levels, num_patches;
  long   num_cells;
  int    extra_cells[3], period[3];
  double anchor[3], cell_spacing[3];
  int    l_id;
  
  fread( & num_levels,    sizeof(int),    1, fp );

  for( int lev = 0; lev < num_levels; lev++ ) {
    fread( & num_patches,  sizeof(int),    1, fp );    // Number of Patches -  100
    fread( & num_cells,    sizeof(long),   1, fp );    // Number of Cells   - 8000
    fread(   extra_cells,  sizeof(int),    3, fp );    // Extra Cell Info   - [1,1,1]
    fread(   anchor,       sizeof(double), 3, fp );    // Anchor Info       - [0,0,0]
    fread(   period,       sizeof(int),    3, fp );    // 
    fread( & l_id,         sizeof(int),    1, fp );    // ID of Level       -    0
    fread(   cell_spacing, sizeof(double), 3, fp );    // Cell Spacing      - [0.1,0.1,0.1]

    bool foundPeriodicBoundaries = false;
    if( period[0] != 0 || period[1] != 0 || period[2] != 0 ) {
      foundPeriodicBoundaries = true;
    }

    procMap.push_back( std::vector<int>() );
    std::vector<int> & procMapForLevel = procMap[ lev ];

    const Point  anchor_p( anchor[0], anchor[1], anchor[2] );
    const Vector dcell( cell_spacing[0], cell_spacing[1], cell_spacing[2] );

    LevelP level = this->addLevel( anchor_p, dcell, l_id );

    const IntVector extraCells( extra_cells[0], extra_cells[1], extra_cells[2] );
    level->setExtraCells( extraCells );

    for( int patch = 0; patch < num_patches; patch++ ) {
      int    p_id, rank, nnodes, total_cells;
      int    low_index[3], high_index[3], i_low_index[3], i_high_index[3];
      double lower[3], upper[3];
     
      fread( & p_id,         sizeof(int),    1, fp );
      fread( & rank,         sizeof(int),    1, fp );
      fread(   low_index,    sizeof(int),    3, fp );    // <lowIndex>[-1,-1,-1]</lowIndex>
      fread(   high_index,   sizeof(int),    3, fp );    // <highIndex>[20,20,4]</highIndex>
      fread(   i_low_index,  sizeof(int),    3, fp );    // <interiorLowIndex></interiorLowIndex>
      fread(   i_high_index, sizeof(int),    3, fp );    // <interiorHighIndex>[20,20,3]</interiorHighIndex>
      fread( & nnodes,       sizeof(int),    1, fp );    // <nnodes>2646</nnodes>
      fread(   lower,        sizeof(double), 3, fp );    // <lower>[-0.025000000000000001,-0.025000000000000001,-0.049999999999999996]</lower>
      fread(   upper,        sizeof(double), 3, fp );    // <upper>[0.5,0.5,0.19999999999999998]</upper>
      fread( & total_cells,  sizeof(int),    1, fp );    // <totalCells>2205</totalCells>

      const IntVector lowIndex(   low_index[0],  low_index[1],  low_index[2] );
      const IntVector highIndex( high_index[0], high_index[1], high_index[2] );
      const IntVector interiorLowIndex(   i_low_index[0],  i_low_index[1],  i_low_index[2] );
      const IntVector interiorHighIndex( i_high_index[0], i_high_index[1], i_high_index[2] );

      level->addPatch( lowIndex, highIndex, interiorLowIndex, interiorHighIndex, this, p_id );

      procMapForLevel.push_back( rank );

    } // end for patch loop
    
    if( foundPeriodicBoundaries ) {
      level->finalizeLevel( period[0] != 0, period[1] != 0, period[2] != 0 );
    }
    else {
      level->finalizeLevel();
    }

  } // end for level loop

} // end readLevelsFromFileBinary()


//______________________________________________________________________
//
// We are parsing the XML manually, line by line, because if we use the XML library function to read it into an XML tree
// data structure, then for large number of patches, too much memory is used by that data structure.  It is unclear
// whether the XML library frees up this memory when you "releaseDocument()", but it probably doesn't matter because
// we are concerned about the highwater mark (max memory per node), which the XML structure may push us over.
//
void
Grid::readLevelsFromFile( FILE * fp, std::vector< std::vector<int> > & procMap )
{
  bool done      = false;
  bool foundGrid = false;

  while( !done ) { // Start at the very top of the file (most likely 'timestep.xml').

    std::string line = UintahXML::getLine( fp );

    if( line == "<Grid>" ) {
      foundGrid = parseGridFromFile( fp, procMap );
      done = true;
    }
    else if( line == "" ) { // End of file reached.
      done= true;
    }
  }

  if( !foundGrid ) {
    throw InternalError( "Grid.cc: readLevelsFromFile: Did not find '<Grid>' in file.", __FILE__, __LINE__ );
  }


  //      timedata.d_patchInfo.push_back(std::vector<PatchData>());
  //      timedata.d_matlInfo.push_back(std::vector<bool>());

  //          r->get("proc", pi.proc); // defaults to -1 if not available
  //          timedata.d_patchInfo[levelIndex].push_back(pi);


        // Ups only provided when queryGrid() is called for a restart.  The <Grid> is not necessary on non-restarts..
  //ProblemSpecP grid_ps = ups->findBlock("Grid");
  //    level->assignBCS( grid_ps, 0 );

} // end readLevelsFromFile()


//______________________________________________________________________
//
Level *
Grid::addLevel( const Point & anchor, const Vector & dcell, int id /* = -1 */ )
{
  // Find the new level's refinement ratio.
  // This should only be called when a new grid is created, so if this level index 
  // is > 0, then there is a coarse-fine relationship between this level and the 
  // previous one.

  IntVector ratio;
  if (d_levels.size() > 0) {
    Vector r = (d_levels[d_levels.size()-1]->dCell() / dcell) + Vector(1e-6, 1e-6, 1e-6);
    ratio = IntVector((int)r.x(), (int)r.y(), (int)r.z());
    Vector diff = r - ratio.asVector();
    if (diff.x() > 1e-5 || diff.y() > 1e-5 || diff.z() > 1e-5) {
      // non-integral refinement ratio
      std::ostringstream out;
      out << "Non-integral refinement ratio: " << r;
      throw InvalidGrid(out.str().c_str(), __FILE__, __LINE__);
    }
  }
  else {
    ratio = IntVector(1,1,1);
  }

  Level* level = scinew Level(this, anchor, dcell, (int)d_levels.size(), ratio, id);  

  d_levels.push_back( level );
  return level;
}


//______________________________________________________________________
//
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

//______________________________________________________________________
//
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
    if (l->getPeriodicBoundaries() != IntVector(0,0,0)){
      std::cout << "  Periodic boundaries:\t\t" << l->getPeriodicBoundaries()<< '\n';
    }
    if( l->isNonCubic() ){
      std::cout << "  isNonCubic:\t\t\t" << l->isNonCubic() << '\n';
    }
    std::cout << "  Number of patches:\t\t" << l->numPatches() << '\n';
    totalPatches += l->numPatches();
    double ppc = double(l->totalCells())/double(l->numPatches());
    std::cout << "  Total number of cells:\t" << l->totalCells() << " (" << ppc << " avg. per patch)\n";
    totalCells += l->totalCells();

    //__________________________________
    //  debugging
    if( g_dbg.active() ){
      Level::const_patch_iterator iter;
      printf("  patches: \n");
      for(iter = l->patchesBegin(); iter != l->patchesEnd(); iter++) {
        const Patch* patch = *iter;
        std::ostringstream msg;
        
        msg << "   Patch: " << patch->getID();
        msg << " Interior Cells " << patch->getCellLowIndex() << " " << patch->getCellHighIndex();
        msg.width(15);
        msg << " \tExtra Cells " << patch->getExtraCellLowIndex() << " " << patch->getExtraCellHighIndex(); 
        printf( "%s\n", msg.str().c_str() );
        //DOUT( true,  msg.str() );
      }
    }
    
  }
  std::cout << "Total patches in grid:\t\t" << totalPatches << '\n';
  double ppc = double(totalCells)/double(totalPatches);
  std::cout << "Total cells in grid:\t\t" << totalCells << " (" << ppc << " avg. per patch)\n";
  std::cout << "\n";
}

//______________________________________________________________________
// Computes the physical boundaries for the grid
void
Grid::getSpatialRange( BBox & b ) const
{
  // just call the same function for all the levels
  for( int l = 0; l < numLevels(); l++ ) {
    getLevel( l )->getSpatialRange( b );
  }
}

//______________________________________________________________________
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

//______________________________________________________________________
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

//______________________________________________________________________
//
void 
Grid::problemSetup(const ProblemSpecP& params, const ProcessorGroup *pg, bool do_amr)
{
   ProblemSpecP grid_ps = params->findBlock("Grid");
   if( !grid_ps ) {
      return;
   }
      
   // anchor/highpoint on the grid
   Point anchor(DBL_MAX, DBL_MAX, DBL_MAX);

   // Time refinement between a level and the previous one

   int levelIndex = 0;

   for( ProblemSpecP level_ps = grid_ps->findBlock("Level"); level_ps != nullptr; level_ps = level_ps->findNextBlock("Level") ) {
      // Make two passes through the boxes.  The first time, we
      // want to find the spacing and the lower left corner of the
      // problem domain.  Spacing can be specified with a dx,dy,dz
      // on the level, or with a resolution on the patch.  If a
      // resolution is used on a problem with more than one patch,
      // the resulting grid spacing must be consistent.

      // anchor/highpoint on the level
      Point levelAnchor(    DBL_MAX,  DBL_MAX,  DBL_MAX);
      Point levelHighPoint(-DBL_MAX, -DBL_MAX, -DBL_MAX);

      Vector spacing;
      bool have_levelspacing=false;

      if( level_ps->get( "spacing", spacing ) ) {
        have_levelspacing = true;
      }
      bool have_patchspacing=false;
        

      //__________________________________
      // first pass - find upper/lower corner, find resolution/spacing and extraCells
      IntVector extraCells(0, 0, 0);
      for(ProblemSpecP box_ps = level_ps->findBlock("Box"); box_ps != nullptr; box_ps = box_ps->findNextBlock("Box")){
         
        std::string boxLabel = "";
        box_ps->getAttribute("label",boxLabel);
         
        Point lower;
        box_ps->require("lower", lower);
        Point upper;
        box_ps->require("upper", upper);
        
        if (levelIndex == 0) {
          anchor=Min(lower, anchor);
        }
        
        levelAnchor   =Min( lower, levelAnchor );
        levelHighPoint=Max( upper, levelHighPoint );
        
        IntVector resolution;
        if( box_ps->get("resolution", resolution) ){
          if( have_levelspacing ){
            throw ProblemSetupException("Cannot specify level spacing and patch resolution" + boxLabel, 
                                        __FILE__, __LINE__);
          } else {
            // all boxes on same level must have same spacing
            Vector newspacing = (upper-lower)/resolution;
            if( have_patchspacing ){
              Vector diff = spacing-newspacing;
              if( diff.length() > 1.e-14 ) {
                std::ostringstream msg;
                msg<< "\nAll boxes on same level must have same cell spacing. Box (" << boxLabel 
                   << ") is inconsistent with the previous box by: (" << diff << ").  Box spacing is ("
                   << newspacing << ")\n";
                throw ProblemSetupException( msg.str(), __FILE__, __LINE__ );
              }
            } else {
              spacing = newspacing;
            }
            have_patchspacing=true;
          }
        }  // resolution
        
        IntVector ec;
        box_ps->getWithDefault( "extraCells", ec, d_extraCells );
        extraCells = Max(ec, extraCells);
        
        // bulletproofing
        if( have_levelspacing || have_patchspacing ){
          for(int dir = 0; dir<3; dir++){
            if ( (upper(dir)-lower(dir)) <= 0.0 ) {
              std::ostringstream msg;
              msg<< "\nComputational Domain Input Error: Level("<< levelIndex << ")"
                 << " \n The computational domain " << lower<<", " << upper
                 << " must have a positive distance in each coordinate direction  " << upper-lower << std::endl; 
              throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
            }
          
            if ( spacing[dir] > (upper(dir)-lower(dir)) || spacing[dir] < 0 ){
              std::ostringstream msg;
              msg<< "\nComputational Domain Input Error: Level("<< levelIndex << ")"
                 << " \n The spacing " << spacing 
                 << " must be less than the upper - lower corner and positive " << upper << std::endl; 
              throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
            }
          }
        }
      }  // boxes loop

      if ( extraCells != d_extraCells && d_extraCells != IntVector(0,0,0) ) {
        proc0cout << "Warning:: Input file overrides extraCells specification, current extraCell: " << extraCells << "\n";
      }

      if( !have_levelspacing && !have_patchspacing ) {
        throw ProblemSetupException("Box resolution is not specified", __FILE__, __LINE__);
      }

      LevelP level = addLevel( anchor, spacing );

      // Determine the interior cell limits.  For no extraCells, the limits
      // will be the same.  For extraCells, the interior cells will have
      // different limits so that we can develop a CellIterator that will
      // use only the interior cells instead of including the extraCell
      // limits.
      level->setExtraCells(extraCells);

      IntVector anchorCell(level->getCellIndex(    levelAnchor   + Vector(1.e-14,1.e-14,1.e-14)) );
      IntVector highPointCell(level->getCellIndex(levelHighPoint + Vector(1.e-14,1.e-14,1.e-14)) );
        
      //______________________________________________________________________
      // second pass - set up patches and cells
      for(ProblemSpecP box_ps = level_ps->findBlock("Box"); box_ps != nullptr; box_ps = box_ps->findNextBlock("Box")){
         
        std::string boxLabel="";
        box_ps->getAttribute( "label", boxLabel );
         
        Point lower, upper;
        box_ps->require("lower", lower);
        box_ps->require("upper", upper);
        
        //__________________________________
        // bullet proofing inputs
        for(int dir = 0; dir<3; dir++){
          if (lower(dir) >= upper(dir)){
            std::ostringstream msg;
            msg<< "\nComputational Domain Input Error: Level("<< levelIndex << ")"
               << " \n The lower corner " << lower 
               << " must be smaller than the upper corner " << upper << std::endl; 
            throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
          }
        }
        // The following few lines of code add a small epsilon to a number.  If the number
        // is very large, then adding a fixed epsilon does not do anything (as there
        // would not be enough precision to account for this).  Therefore, we use the
        // following 'magic' (multiply by '1+epsilon').  In all cases except when the
        // number is 0, multiplying by (1 + epsilon) has the desired effect... However,
        // when the number is 0 this doesn't work, so we have the extra '+epsilon'.
        //
        // For example: (Numbers are simplified for illustrative purposes and may not
        //               be actually large enough to exhibit the actual behavior.)
        //
        //        1 + 0.000000001 =>        1.000000001  (This works correctly!)
        // 99999999 + 0.000000001 => 99999999            (This returns the 'wrong' answer!)

        double epsilon = 1.e-14;

        Vector ep_v  = Vector( epsilon, epsilon, epsilon );
        IntVector lowCell  = level->getCellIndex( lower + Vector( fabs(lower.x())*epsilon,
                                                                  fabs(lower.y())*epsilon,
                                                                  fabs(lower.z())*epsilon ) + ep_v );
        IntVector highCell = level->getCellIndex( upper + Vector( fabs(upper.x())*epsilon,
                                                                  fabs(upper.y())*epsilon,
                                                                  fabs(upper.z())*epsilon ) + ep_v );

        // bulletproofing
        Point lower2 = level->getNodePosition( lowCell );
        Point upper2 = level->getNodePosition( highCell );
        double diff_lower = (lower2-lower).length();
        double diff_upper = (upper2-upper).length();

        double max_component_lower = Abs(Vector(lower)).maxComponent();
        double max_component_upper = Abs(Vector(upper)).maxComponent();

        if( diff_lower > max_component_lower * epsilon ) {
          std::cerr << " boxLabel: " << boxLabel << '\n';
          std::cerr << std::setprecision(16) << "epsilon: " << epsilon << "\n";

          std::cerr << "diff_lower: " << diff_lower << "\n";
          std::cerr << "max_component_lower: " << max_component_lower << "\n";

          std::cerr << std::setprecision(16) << "lower    = " << lower << '\n';
          std::cerr << std::setprecision(16) << "lowCell  = " << lowCell << '\n';
          std::cerr << std::setprecision(16) << "highCell = " << highCell << '\n';
          std::cerr << std::setprecision(16) << "lower2   = " << lower2 << '\n';
          std::cerr << std::setprecision(16) << "diff     = " << diff_lower << '\n';
        
          throw ProblemSetupException("Box lower corner does not coincide with grid", __FILE__, __LINE__);
        }

        if( diff_upper > max_component_upper * epsilon ){
          
          std::cerr << " boxLabel: " << boxLabel << '\n'
		    << "upper    = " << upper << '\n'
		    << "lowCell  = " << lowCell << '\n'
		    << "highCell = " << highCell << '\n'
		    << "upper2   = " << upper2 << '\n'
		    << "diff     = " << diff_upper << '\n';
          throw ProblemSetupException("Box upper corner does not coincide with grid", __FILE__, __LINE__);
        }

        IntVector resolution( highCell - lowCell );
        if(resolution.x() < 1 || resolution.y() < 1 || resolution.z() < 1) {
          std::ostringstream warn;
          warn << "Resolution is negative: " << resolution
               << " high Cell: " << highCell << " lowCell: " << lowCell << '\n';
          throw ProblemSetupException(warn.str() , __FILE__, __LINE__);
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
        std::map<std::string, std::string> patchAttributes;  // Hash for parsing out the XML attributes


        if(box_ps->get("autoPatch", autoPatchValue)) {
          // autoPatchValue must be >= 1, else it will generate fewer patches than processors, and fail
          if( autoPatchValue < 1 )
            throw ProblemSetupException("autoPatch value must be greater than 1", __FILE__, __LINE__);

          patchAttributes.clear();
          box_ps->getAttributes(patchAttributes);
          proc0cout << "Automatically performing patch layout.\n";
          
          int numProcs = pg->nRanks();
          int targetPatches = (int)(numProcs * autoPatchValue);
          
          Primes::FactorType factors;
          int numFactors = Primes::factorize(targetPatches, factors);
	  std::list<int> primeList;
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

        // bulletproofing: catch patches > resolution 
        for (int d=0; d<3; d++) {
          if ( patches[d] > resolution[d] ){
	    std::ostringstream desc;
            desc << "   ERROR: The number of patches in direction (" << d << ") is greater than the number of cells."
                 << " (patches: " << patches << ", cells: " << resolution << ")";
            throw InvalidGrid(desc.str(),__FILE__,__LINE__);
          }
        }

        // If the value of the norm nf_ is too high, then user chose a 
        // bad number of processors, warn them.
        if( nf_ > 3 ) {
          std::cout << "\n********************\n";
          std::cout << "*\n";
          std::cout << "* WARNING:\n";
          std::cout << "* The patch to processor ratio you chose\n";
          std::cout << "* does not factor well into patches.  Consider\n";
          std::cout << "* using a different number of processors.\n";
          std::cout << "*\n";
          std::cout << "********************\n\n";
        }
  
        proc0cout << "Patch layout: \t\t(" << patches.x() << ","
                  << patches.y() << "," << patches.z() << ")\n";

        IntVector refineRatio = level->getRefinementRatio();
        level->setPatchDistributionHint(patches);
        
        IntVector boxLo_cell( SHRT_MAX, SHRT_MAX, SHRT_MAX );
        IntVector boxHi_cell(-SHRT_MAX,-SHRT_MAX,-SHRT_MAX );
        
        for(int i=0;i<patches.x();i++){
          for(int j=0;j<patches.y();j++){
            for(int k=0;k<patches.z();k++){
              IntVector startcell = resolution * IntVector(i,j,k)/patches + lowCell;
              IntVector endcell   = resolution * IntVector(i+1,j+1,k+1)/patches + lowCell;
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
                Vector startRatio = inStartCell.asVector()/refineRatio.asVector();
                Vector endRatio   = inEndCell.asVector()/refineRatio.asVector();
                std::ostringstream desc;
                desc << "Level Box: (" << boxLabel << "), the finer patch boundaries (" << inStartCell << "->" << inEndCell 
                     << ") does not coincide with a coarse cell"
                     << "\n(i.e., they are not divisible by the refinement ratio " << refineRatio << ')'
                     << "\n startCell/refineRatio (" << startRatio << "), endCell/refineRatio ("<<endRatio << ")\n";
                throw InvalidGrid(desc.str(),__FILE__,__LINE__);
              }
              level->addPatch(startcell, endcell, inStartCell, inEndCell, this);
              
              boxLo_cell = Uintah::Min( boxLo_cell, inStartCell );
              boxHi_cell = Uintah::Max( boxHi_cell, inEndCell );
            }
          }
        } // end for(int i=0;i<patches.x();i++){
      } // end for(ProblemSpecP box_ps = level_ps->findBlock("Box");

      if (pg->nRanks() > 1 && (level->numPatches() < pg->nRanks()) && !do_amr) {
        std::ostringstream warn;
        warn << "Number of patches (" << level->numPatches() << ") must >= the number of processes in an mpi run";
        throw ProblemSetupException(warn.str(),__FILE__, __LINE__);
      }
      
      IntVector periodicBoundaries;
      if( level_ps->get( "periodic", periodicBoundaries ) ) {
        level->finalizeLevel( periodicBoundaries.x() != 0,
                              periodicBoundaries.y() != 0,
                              periodicBoundaries.z() != 0 );
      }
      else {
        level->finalizeLevel();
      }
      //level->assignBCS(grid_ps,0);
      levelIndex++;
   }

  if(numLevels() > 1 && !do_amr) {  // bullet proofing
     throw ProblemSetupException("Grid.cc:problemSetup: Multiple levels encountered in non-AMR grid", __FILE__, __LINE__);
   }
} // end problemSetup()

//______________________________________________________________________
//
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
      for ( Level::patch_iterator patchIter = level->patchesBegin(); patchIter < level->patchesEnd(); patchIter++ ) {
        const Patch* patch = *patchIter;
        out <<"    "<< *patch << std::endl;
      }
    }
    return out;
  }
}

//______________________________________________________________________
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
    Level::const_patch_iterator iter = level->patchesBegin();
    Level::const_patch_iterator otheriter = otherlevel->patchesBegin();
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
//______________________________________________________________________
//
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
    Level::const_patch_iterator iter;
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
//______________________________________________________________________
//
IntVector Grid::run_partition3D(std::list<int> primes)
{
  partition3D(primes, 1, 1, 1);
  return IntVector(af_, bf_, cf_);
}
//______________________________________________________________________
//
void Grid::partition3D(std::list<int> primes, int a, int b, int c)
{
  // base case: no primes left, compute the norm and store values
  // of a,b,c if they are the best so far.
  if( primes.size() == 0 ) {
    double new_norm = sqrt( (double)(std::max(a,b)/std::min(a,b) - std::max(ares_,bres_)/std::min(ares_,bres_)) *
                            (std::max(a,b)/std::min(a,b) - std::max(ares_,bres_)/std::min(ares_,bres_)) + 
                            (std::max(b,c)/std::min(b,c) - std::max(bres_,cres_)/std::min(bres_,cres_)) *
                            (std::max(b,c)/std::min(b,c) - std::max(bres_,cres_)/std::min(bres_,cres_)) +
                            (std::max(a,c)/std::min(a,c) - std::max(ares_,cres_)/std::min(ares_,cres_)) *
                            (std::max(a,c)/std::min(a,c) - std::max(ares_,cres_)/std::min(ares_,cres_))
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
//______________________________________________________________________
//
IntVector Grid::run_partition2D(std::list<int> primes)
{
  partition2D(primes, 1, 1);
  return IntVector(af_, bf_, cf_);
}
//______________________________________________________________________
//
void Grid::partition2D(std::list<int> primes, int a, int b)
{
  // base case: no primes left, compute the norm and store values
  // of a,b if they are the best so far.
  if( primes.size() == 0 ) {
    double new_norm = (double)std::max(a,b)/std::min(a,b) - std::max(ares_,bres_)/std::min(ares_,bres_);

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
//______________________________________________________________________
//
const Patch *
Grid::getPatchByID( int patchid, int startingLevel ) const
{
  const Patch * patch = nullptr;
  for( int i = startingLevel; i < numLevels(); i++ ) {
    LevelP level       = getLevel( i );
    int    levelBaseID = level->getPatch( 0 )->getID();
    if( patchid >= levelBaseID && patchid < levelBaseID + level->numPatches() ) {
      patch = level->getPatch( patchid - levelBaseID );
      break;
    }
  }
  return patch;
}
//______________________________________________________________________
//
void
Grid::assignBCS( const ProblemSpecP & grid_ps, LoadBalancer * lb )
{
  for( int l = 0; l < numLevels(); l++ )
  {
    LevelP level = getLevel( l );
    level->assignBCS( grid_ps, lb );
  }
}
//______________________________________________________________________
//
void
Grid::setExtraCells( const IntVector & ex )
{
  if( numLevels() > 0 ) {
     throw ProblemSetupException( "Cannot set extraCells after grid setup", __FILE__, __LINE__ );
     return;
  }
  d_extraCells = ex;
}
