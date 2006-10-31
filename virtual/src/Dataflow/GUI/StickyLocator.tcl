#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#


#
#  makeStickyLocator parent pos_x_var pos_y_var width height labeltext args
#
#  Author: James Bigler

# Creates a frame named parent and places a canvas in it with the
# given width height and label.  pos_x_var and pos_y_var are global
# variables that are updated with [-1,1] range suitable for placing
# GeomSticky objects in the viewer.

# You can override some of the look of the module by adding optional
# arguments (i.e. -width 100 -height 200 -bgcolor black).  These are
# pretty much any argument you could add to a canvas.  Whatever
# arguments you provide will override the ones I've specified.
#
#   pos_x_var, pos_y_var  Global variables that are to contain the
#                         normalized position of the node in [-1,1].
#   width, height         Dimensions of the canvas
#   labeltext             Text to put in label on top
#   args                  optional arguments to be passed on to the canvas

# There are two functions that are bound to the node's movement.
# StickyLocatorMark and StickyLocatorDrag.  These are called when the
# node is selected and moved.  You can override these functions by
# rebinding the mouse events like this:

# set can [Node_Canvas .c timeposition_x1 timeposition_y1 70 70 \
#              "Time Step Position"]
# pack .c -fill both -expand true
# $can bind movable <Button-1> {My_new_click_function}

# Keep in mind that you generally don't want to do this.  You should
# however bind a function to the release.

# $can bind movable <ButtonRelease> {$this-c needexecute}

proc makeStickyLocator { c pos_x_var pos_y_var width height args } {
    frame $c 

    # BTW, if you change the borderwidth or other similar things you
    # will need to update the code in StickyLocatorDrag or be more
    # clever and discover it there.  Currently it is hard coded.
    eval {canvas $c.canvas -bg "#ffffff" -height $height -width $width \
          -borderwidth 1 -relief solid} $args
    pack $c.canvas -side top -anchor c -expand no

    # These are the global variables that we will keep track of.  We
    # will do this by creating a copy of the variable name that we can
    # get access to later by way of the name of the canvas.
    global $pos_x_var
    global $pos_y_var
    # Copy of global variable name based on name of the canvas.  Since
    # this is a global variable pointing to a global variable, you
    # have to use two levels of sets to get to actual variable.  See
    # below.
    global $c.canvas.node_pos_x_var
    global $c.canvas.node_pos_y_var
    set $c.canvas.node_pos_x_var $pos_x_var
    set $c.canvas.node_pos_y_var $pos_y_var

#     set val [set $c.canvas.node_pos_x_var]
#     puts "val = $val"

    # Denormalize the positions to screen locations
    set x [expr $width * (([set [set $c.canvas.node_pos_x_var]] + 1) / 2)]
    set y [expr $height * ((-[set [set $c.canvas.node_pos_y_var]] + 1) / 2)]

    # Do some bounds checks
    if { $x < 0 } {
        set x 0
    } elseif { $x > $width } {
        set x $width
    }
    if { $y < 0 } {
        set y 0
    } elseif { $y > $height } {
        set y $height
    }

    # Create the node.  This returns an ID for the object created for
    # the display list.  You can cache it out and save it in a global
    # variable if you want.
    $c.canvas create oval \
        [expr $x - 5] [expr $y - 5] \
        [expr $x + 5] [expr $y + 5]\
        -outline black \
        -fill red -tags movable

#     global $c.canvas.name
#     set $c.canvas.name $labeltext

    # Do some default bindings
    $c.canvas bind movable <Button-1> {StickyLocatorMark %x %y %W}
    $c.canvas bind movable <B1-Motion> {StickyLocatorDrag %x %y %W}

    return $c.canvas
}

# This will locate the node and save some state.  This state is a
# global shared array between all canvases, but you can only drag in
# one at a time anyway.
proc StickyLocatorMark { x y can} {
    global canvas
#     global $can.name
#     puts "Hit [set $can.name]"

    # Map from view coordinates to canvas coordinates
    set x [$can canvasx $x]
    set y [$can canvasy $y]

    # Remember the object and its location
    set canvas($can,obj) [$can find closest $x $y]
    set canvas($can,x) $x
    set canvas($can,y) $y
}

# Move the node that was located in StickyLocatorMark and update the
# global variables' values.
proc StickyLocatorDrag { x y can} {
    global canvas
    # Map from view coordinates to canvas coordinates
    set x [$can canvasx $x]
    set y [$can canvasy $y]

    # Get the position of the node.  This is the bounding box of the node.
    set pos [$can coords $canvas($can,obj)]
    # Now compute the center of the node from the bounding box.
    set center_x [expr ([lindex $pos 2] + [lindex $pos 0])/2]
    set center_y [expr ([lindex $pos 3] + [lindex $pos 1])/2]

    # Get the width and height of the canvas for bounds checks
    set cw [winfo width $can]
    set ch [winfo height $can]
    # The following code will cap movements to inside the right range
    if {$x < 0 } {
        set x 0
    } elseif {$x > $cw} {
        set x [expr $cw]
    }
    if {$y < 0 } {
        set y 0
    } elseif {$y > $ch} {
        set y [expr $ch]
    }

    # Move the current object
    set dx [expr $x - $canvas($can,x)]
    set dy [expr $y - $canvas($can,y)]

    # See if the center will be out of bounds and cap it off
    set new_cx [expr $center_x + $dx]

    # Now these values of 2 are based on the width of the borders of
    # the canvas.  If you change the width then you will need to
    # change these values.
    if {$new_cx < 2} {
        set dx [expr 2 - $center_x]
    } elseif {$new_cx > $cw-2} {
        set dx [expr $cw-2 - $center_x]
    }
    set new_cy [expr $center_y + $dy]
    if {$new_cy < 2} {
        set dy [expr 2 - $center_y]
    } elseif {$new_cy > $ch-2} {
        set dy [expr $ch-2 - $center_y]
    }

    # Update the position
    $can move $canvas($can,obj) $dx $dy

    # Get the new location of the node, and compute the new center
    set pos [$can coords $canvas($can,obj)]
    set center_x [expr ([lindex $pos 2] + [lindex $pos 0])/2]
    set center_y [expr ([lindex $pos 3] + [lindex $pos 1])/2]

#     puts "center(x,y) = ($center_x, $center_y)"

    # Convert the center to the normalized coordinates
    global $can.node_pos_x_var
    global $can.node_pos_y_var
    set [set $can.node_pos_x_var] [expr ($center_x-2)/double($cw-4)*2 - 1]
    set [set $can.node_pos_y_var] [expr -($center_y-2)/double($ch-4)*2 + 1]
#    puts "node_pos(x,y) = ([set [set $can.node_pos_x_var]], [set [set $can.node_pos_y_var]])"

    # Store off these values for the next iteration of movement.
    set canvas($can,x) $x
    set canvas($can,y) $y
}

# Example usage

# proc Go {} {
#     global timeposition_x1
#     global timeposition_y1
#     global timeposition_x2
#     global timeposition_y2

#     set timeposition_x1  0.5
#     set timeposition_y1  0.6
#     set timeposition_x2  0.8
#     set timeposition_y2 -0.9

#     set can [makeStickyLocator .c timeposition_x1 timeposition_y1 70 70 \
#                  "Time Step Position"]
#     pack .c -fill both -expand true
#     $can bind movable <ButtonRelease> {$this-c needexecute}


#     set can [makeStickyLocator .c2 timeposition_x2 timeposition_y2 80 80 \
#                  "Clock Position"]
#     pack .c2 -fill both -expand true
# }

