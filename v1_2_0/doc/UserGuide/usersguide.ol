+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+
+
+ SCIRun Users Guide Outline
+
+ $Id$
+
+
+ Follows conventions of emacs outline mode, briefly:
+
+
+
+ '*' = Section heading
+
+ '**' = Subsection heading
+
+ '***' = Subsubsection heading
+
+
+
+ Text following a [sub[sub]]section heading briefly describes content of
+ section.
+
+ Use of font-lock mode within outline mode is recommended.
+
+
+
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Last update: Wed Feb 21 23:39:41 2001 by Rob MacLeod
#    - Rob added changes after our meeting today

* About This Guide (DO this at the end)

Purpose and scope of this users guide.

** Document roadmap

Brief introduction to each major section of the guide.

** Other Sources of Information

Information not in this guide.

*** Local online help

How to find and use local online help distributed with SCIRun.

*** SCIRun website

How to access the SCIRun web site and why you would bother--what you will
find there.

*** SCIRun mailing list

Mailings lists and procedures for subscribing.

*** Reporting bugs and suggesting improvements to SCIRun

Procedure for reporting bugs and suggesting improvements.

* Concepts  (Rob, starting with papers and text from the NCRR grant)

SCIRun's concept of data flow and other philosophical issues.  Much of this
will be culled from existing papers and grants.  Possible topics:

** Data flow

** Components, data, connections, and ports

* Starting up (Ted)
** The scirun command
*** Anatomy of the Main Window 
**** Window menu bar
**** Popup menu
**** Network Navigator pane
**** Error pane
**** Network pane
** The terminal app window.
** The SCIRUN_DATA environment variable.
** Stopping

* Building and Editing Networks (Ted and Matt)
** Creating modules.
*** Anatomy of a Module
Describe common elements of all modules.
**** Name
**** Timer
**** Progress
**** UI
**** Popup menu
**** Ports (and Connections)
***** Port colors
***** Port info
** Connecting modules.
** Modifying module properties.
** Deleting connections.
** Deleting a module.
** Writing module specific notes.
** Viewing a module's log.
** Saving a network
** Opening an existing network
*** Using the Open menu item in File menu.
*** Using the Terminal window app.
** Executing a network

* Visualization (using the Viewer module) (Rob with Dave W.)
Use of the Viewer module.  Possible topics:
** Overview and list of controls
** Mouse controls
** Menu controls
** Control widgets
** Saving images and creating animations 

* Packages

** SCIRun Package
Brief overview of contents of SCIRun package.
*** DataIO
*** Fields
*** Math
*** Render
*** Visualization

** BioPSE Package

* User Interface control (Ted)

What the generic UI elements look like and how they
work.

* Importing/Exporting Foreign Data (Ted)

Steps required to import and save foreign data.  Description of existing
importers.  Pointer to the programming guide's section on writing an
importer.

* Sample Networks and Data Sets (Rob)

Will there be sample networks and data for use with them?  If so this
section shows the way.

* Glossary of Terms (Rob and Ted)

* Appendicies (Not settled yet what this will contain)

* Bibliography (Rob and Ted)

* Index (Rob and Ted)

