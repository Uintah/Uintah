import pickle

# INPUTS:
tstart = 300
tend = 614
tdiff = 20 #size of the window (always looking backwards)
vname = "temperature_running_sum/0" #Add the /0 for native grid variables


#this is the class with the function to make the plot
#your class must have a create_plot() function, otherwise, you
#are free to do what you want in there

pos = 98 #for x_slice, it is the % position along the axis
#Paste between the dotted lines
#-----------paste class here------------------------------------------------------------------------
class x_slice:

    def __init__(self, vname, slice_pos):

        self.vname = vname
        self.pos = slice_pos

    def create_plot(self):

        AddPlot("Pseudocolor", vname, 1, 0)
        AddOperator("Slice", 0)
        SetActivePlots(0)
        SetActivePlots(0)
        SliceAtts = SliceAttributes()
        SliceAtts.originType = SliceAtts.Percent  # Point, Intercept, Percent, Zone, Node
        SliceAtts.originPoint = (0, 0, 0)
        SliceAtts.originIntercept = 0
        SliceAtts.originPercent = self.pos
        SliceAtts.originZone = 0
        SliceAtts.originNode = 0
        SliceAtts.normal = (-1, 0, 0)
        SliceAtts.axisType = SliceAtts.XAxis  # XAxis, YAxis, ZAxis, Arbitrary, ThetaPhi
        SliceAtts.upAxis = (0, 1, 0)
        SliceAtts.project2d = 1
        SliceAtts.interactive = 1
        SliceAtts.flip = 0
        SliceAtts.originZoneDomain = 0
        SliceAtts.originNodeDomain = 0
        SliceAtts.meshName = "CC_Mesh"
        SliceAtts.theta = 90
        SliceAtts.phi = 0
        SetOperatorOptions(SliceAtts, 0)
        AddOperator("Threshold", 0)
        ThresholdAtts = ThresholdAttributes()
        ThresholdAtts.outputMeshType = 0
        ThresholdAtts.listedVarNames = ("cellType/0")
        ThresholdAtts.zonePortions = (1)
        ThresholdAtts.lowerBounds = (-1e+37)
        ThresholdAtts.upperBounds = (-1)
        ThresholdAtts.defaultVarName = vname
        ThresholdAtts.defaultVarIsScalar = 1
        SetOperatorOptions(ThresholdAtts, 0)
        DrawPlots()

#---------stop paste -------------------------------------------------------------------------------
myfunc = x_slice(vname, pos)
#END INPUTS

time = []
runningave = []

print "starting loop"
for i in range(tstart,tend):

    myfunc.create_plot()

    SetTimeSliderState(i)
    Query("Time")
    T0=GetQueryOutputValue()
    time.append(T0)
    SetTimeSliderState(i-tdiff)
    Query("Time")
    T1 = GetQueryOutputValue()
    dt = T0-T1
    print 'current dt = ', dt
    SetTimeSliderState(i)

    DefineScalarExpression("sliding_ave", "(<"+vname+">"+
    " - conn_cmfe(<[" + str(i-tdiff) + "]i:"+vname+">, <CC_Mesh>))/"+str(dt))

    ChangeActivePlotsVar("sliding_ave")

    QueryOverTimeAtts = GetQueryOverTimeAttributes()
    QueryOverTimeAtts.timeType = QueryOverTimeAtts.DTime  # Cycle, DTime, Timestep
    QueryOverTimeAtts.startTimeFlag = 0
    QueryOverTimeAtts.startTime = 0
    QueryOverTimeAtts.endTimeFlag = 0
    QueryOverTimeAtts.endTime = 1
    QueryOverTimeAtts.strideFlag = 0
    QueryOverTimeAtts.stride = 1
    QueryOverTimeAtts.createWindow = 1
    QueryOverTimeAtts.windowId = 2
    SetQueryOverTimeAttributes(QueryOverTimeAtts)
    SetQueryFloatFormat("%g")
    value = Query("Average Value")
    vsplit = value.split()
    avevalue = vsplit[len(vsplit)-1]
    avevalue = float(avevalue)
    runningave.append(avevalue)
    DeleteActivePlots()

print "done!"

final = [time,runningave]
pickle.dump( final, open("sliding_ave.pkl","wb") )

print "Output file written as sliding_ave.pkl"
print "read/plot file like (in python):"
print "import pickle "
print "import matplotlib.pyplot as plt "
print "r=pickle.load(open(\"sliding_ave.pkl\",\"rb\"))"
print "plt.plot(r[0],r[1],\'b--o\') "
print "plt.show()"
