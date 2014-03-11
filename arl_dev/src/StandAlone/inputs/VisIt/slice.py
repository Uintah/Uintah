#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import sys
import os

if '-MANUAL' not in sys.argv and '-help' not in sys.argv:
    #OPEN THE DATABASE
 if '-uda' in sys.argv:
  uda_path = os.path.abspath(sys.argv[sys.argv.index('-uda')+1])
  status = OpenDatabase(uda_path+'/index.xml',0,"udaReaderMTMD")
  #state = np.linspace(0,1.35,675)
  spacing = .002 #in meters
  AddPlot("Pseudocolor", "press_CC/0") #Choose what variable you want to visualize
  AddOperator("Slice") #Add Operator
  os.mkdir("./" 'tmp') #Make A Temporary directory to store all Photos in
  silr = SILRestriction()
  silr.TurnOffAll()
  silr.TurnSet(3L,1)
  SetPlotSILRestriction(silr)
  Leg = GetAnnotationObject("Plot0000")
  Leg.managePosition = 0
  Leg.numberFormat = "%# -9.2g"
  Leg.position = (0.02, 0.9) #Default is 0.05, 0.9
  Leg.fontHeight = 0.02
  Leg.drawTitle = 0
  Leg.yScale = 3
  Leg.drawMinMax = 0
  Leg.orientation = "VerticalRight" # VerticalRight, VerticalLeft, HorizontalTop, HorizontalBottom
  Pressure = CreateAnnotationObject("Text2D")
  Pressure.SetText("Pressure")
  Pressure.SetFontBold(1)
  Pressure.SetPosition(0.02, 0.92)
  Pressure.SetHeight(0.02)
  AnnAtts = AnnotationAttributes()
  AnnAtts.axes3D.xAxis.title.visible = 0
  AnnAtts.axes3D.yAxis.title.visible = 0
  AnnAtts.axes3D.zAxis.title.visible = 0
  AnnAtts.axes3D.zAxis.label.visible = 0
  AnnAtts.axes2D.yAxis.title.visible = 0
  AnnAtts.axes3D.triadFlag = 0
  AnnAtts.userInfoFlag = 0
  AnnAtts.databaseInfoFlag = 0
  SetAnnotationAttributes(AnnAtts)  
  for i in range(0,674):

   s = SliceAttributes()
   SetActivePlots(0)
   s.originType = s.Intercept  # Options are: Point, Intercept, Percent, Zone, Node (Default is Intercept)
   s.originIntercept = spacing*float(i)
   s.axisType = s.ZAxis  # XAxis, YAxis, ZAxis, Arbitrary, ThetaPhi
   s.project2d = 1
   SetOperatorOptions(s, 1)
   TimeSliderSetState(0)
   DrawPlots()
  
   att=SaveWindowAttributes() #get attributes object
   att.SetFormat(att.PNG) #Can also be PNG,JPEG, TIFF, etc.
   #set image name
   att.SetFileName('./tmp/photo')
   #Don't do a screen capture as no GUI is open, actually render frame
   att.SetScreenCapture(0)
   #set image resolution (currently all square images)
   att.SetWidth(1080)
   att.SetHeight(1080) 
   #Dont do a screen capture, actually render the frame
   att.SetScreenCapture(0)
   #Set theses settings and save
   SetSaveWindowAttributes(att)   #Set these attributes 
   status = SaveWindow()		 #save the window
 
  sys.exit()


 else:
  status = 0
  print("ERROR:No uda file specified")
  sys.exit()



  
