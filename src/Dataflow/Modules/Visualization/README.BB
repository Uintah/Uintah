1. How is my program organized?

My module is organized like any standard SCIRun module.  It contains
several functions common to all modules such as a constructor,
destructor, execute, and tcl_command.  In order to work as expected,
through its input ports, the BB module must be connected to a
ScalarFieldReader and a GenTransferMap modules which provide it with
volume data and transfer map information.  Information about the scalar
field can be accessed through a handle aScalarFieldH, while the mapping
provided by GenTransferMap can be accessed through a handle aColorMapH.

Also, currently my module must be connected to the Salmon module through
its output port.  The salmon module provides me the viewing information
and also displays the rendered image (via texture mapping).  It is
important to _always_ open the screen Salmon interface before opening
the BB's interface, because otherwise BB will crash (i think) due to
undefined viewing parameters.  In Salmon, the round ball mouse
events allow the user to adjust the rendered plane in a new position.
Then, by clicking on the button "New view" in the UI, volume rendering
by the BB module is triggered.

The UI consists of an OpenGL window and a window with modifiable parameters.
The OpenGL window displays the rendered image.  The other window provides
a slider for the raster size, a radiobutton for volume rendering that
uses and does not use the octree (initialized to not using the octree),
and a button that a new view has been adjusted in the Salmon window
(i don't know exactly how to make my module re-execute every time Salmon
changes the view of the volume rendered plane).

The UI talks to the c code via a routine tcl_command.  any time
"$this-c blah" is called from the tcl code, control moves to the c-code
where it gets handled inside the tcl_command procedure.  i don't think
it is a good idea to execute time consuming pieces of c-code within
this routine.  in my module, tcl_command is called whenever a variable
has been changed therefore image recalculation must take place.  so,
variable "flag" (which is a bit flag) is assigned based on what parameter
changed.  also, want_to_execute() fnc is called which tells the scheduler
to call my module's function execute next time it gets around to it.

The execute function is most interesting because it does the final
image calculations.  execute will take place only if variables affecting
the final image have changed; these changes are stored in the flag
variable.  based on the flag value, appropriate pieces of code will
be executed.  The following routines are called if:
* PreProcess()    - scalar field change
* Classify()      - transfer map classification method change
* Render()        - scalar field or viewing info change
* RenderBRaster() - raster size change
* UseOctree()     - use of octree request
* DestroyOctree() - do not use octree request

Make sure to reset_vars() whenever variables from the UI are to be used
(they will not be updated to new values otherwise).

My code uses the VolPack library.  the rendering context "vpc" variable
(of type vpContext) is the point of contact between my code and the
shear-warp rendering code.  Routines called by execute (described above)
and Setup are the routines in my module that talk to the VolPack library.
In the Setup routine, the rendering context is created and constant
variables are assigned.  The other routines deal with changes described
above.

The VolPack library was easy to use, although numerous variables must
be set in order for things to work (otherwise a black screen is
rendered).  The only good lesson that i have learned is to follow the
directions in the User's guide very carefully.  examples in the
VolPack directory are also of great help.  every call to the library
stores the error code in the vpres variable which i check to make sure
no errors have been encountered.

for representing the volume within VolPack, i decided to use the same
format as described in the examples and the User guide (basically,
each voxel has a byte long scalar field, a short long normal, and a
byte long gradient).  volume data provided by the ScalarFieldReader
must be converted from the available 3d format to a 1d format so
VolPack library can use it.  for the transfer map specification, i use
the GenTransferMap parameters with little modification (i convert
percentages to scalar values).

2. How is there stuff organized? -- are there readme files, where can
   they find info.

the info about VolPack can be found at:
http://www-graphics.stanford.edu/papers/lacroute_thesis/

the VolPack code is placed in SCIRun/VolPack
the VolPack examples are placed in SCIRun/VolPack/examples
the VolPack compiled library (libvolpack.a) has a symbolic link
from the SCIRun/lib directory.

README files are available in the directory and the subdirectories
of SCIRun/VolPack directory.

3. things that are still to be done:
- use all 4 (rgb and opacity) transfer map lines in the volume rendering
  (right now i use only the opacity line)
- convert the raster size slider to a "type in number"
- add depth cueing with a user interface (so the image does not end
  up totally white like it did the last time i assigned constant values
  to the depth cueing call)
- allow the user to specify the min/max opacity/transparency (these
  are currently initialized to constant values in the Setup routine)
- add another light so the image looks better
- allow the user to change the position of lights
- change the module name

4. Known bugs
- minX,minY, maxX,maxY which specify the projection of the bbox onto
the view plane is incorrect because sometimes the volume rendering
is off-center and part of it is sliced off.
