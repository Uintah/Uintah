/*
 *
 * ViewFrameDriver: Tests ViewFrame message
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#include <Message/ViewFrame.h>

using namespace SemotusVisum;

/** Test XML String */
static const string XML1 = "<viewFrame size=\"6\">Following</viewFrame>";
  
  /** Test XML String */
static const string XML2 = "<viewFrame size=\"6\" indexed=\"true\" vertices=\"678\">Following</viewFrame>";

  /** Test XML String */
static const string XML3 = "<viewFrame size=\"6\" indexed=\"true\" replace=\"false\" vertices=\"454\" indices=\"443\" polygons=\"4\">Following</viewFrame>";

/** Test XML String */
static const string XML4 = "<viewFrame size=\"6\" indexed=\"true\" replace=\"false\" vertices=\"454\" indices=\"443\" polygons=\"4\" texture=\"256\">Following</viewFrame>";

/**
 * Main function for driver.
 *
 * @param   argv      Arguments if needed
 *
 * @throws Exception  If there is a problem creating the message
 */	
int
main() {
  ViewFrame *v = NULL;


  // Test 1
  v = ViewFrame::mkViewFrame( (void *)XML1.data() );
  
  if ( v == NULL )
    cerr <<  "Error creating viewframe in test 1" << endl;
  else {
    if ( v->getFrameSize() != 6 || v->isIndexSet() || v->isReplaceSet() ||
	 v->getVertexCount() != -1 || v->getIndexCount() != -1 ||
	 v->getPolygonCount() != -1 )
      cerr <<  "Error in parameters in viewframe, test 1" << endl;
  }
  
  // Test 2
  v = ViewFrame::mkViewFrame( (void *)XML2.data() );
  
  if ( v == NULL )
    cerr <<  "Error creating viewframe in test 2" << endl;
  else {
    if ( v->getFrameSize() != 6 || !v->isIndexSet() || !v->isIndexed() ||
	 v->isReplaceSet() ||
	 v->getVertexCount() != 678 || v->getIndexCount() != -1 ||
	 v->getPolygonCount() != -1 )
      cerr <<  "Error in parameters in viewframe, test 2" << endl;
  }
  
  // Test 3
  v = ViewFrame::mkViewFrame( (void *)XML3.data() );
  
  if ( v == NULL )
    cerr <<  "Error creating viewframe in test 3" << endl;
  else {
    if ( v->getFrameSize() != 6 || !v->isIndexSet() || !v->isIndexed() ||
	 !v->isReplaceSet() || v->isReplace() ||
	 v->getVertexCount() != 454 || v->getIndexCount() != 443 ||
	 v->getPolygonCount() != 4 )
      cerr <<  "Error in parameters in viewframe, test 3" << endl;
  }

  // Test 4
  v = ViewFrame::mkViewFrame( (void *)XML4.data() );
  
  if ( v == NULL )
    cerr <<  "Error creating viewframe in test 4" << endl;
  else {
    if ( v->getFrameSize() != 6 || !v->isIndexSet() || !v->isIndexed() ||
	 !v->isReplaceSet() || v->isReplace() ||
	 v->getVertexCount() != 454 || v->getIndexCount() != 443 ||
	 v->getPolygonCount() != 4 || v->getTextureDimension() != 256 )
      cerr <<  "Error in parameters in viewframe, test 4" << endl;
  }
}
