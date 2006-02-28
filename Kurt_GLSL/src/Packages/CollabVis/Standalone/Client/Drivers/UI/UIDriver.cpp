/*
 *
 * UIDriver: Driver for UI.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2001
 *
 */

#include <iostream>
#include <UI/glutUI.h>

using namespace SemotusVisum;

int
main(int argc, char ** argv) {
  glutUI::initialize( argc, argv );

  glutUI::getInstance().show();
}
