#hard coded font path barf!
set term png enhanced font '/usr/share/fonts/truetype/ttf-liberation/LiberationSans-Regular.ttf' 9 size 1400,700
#set output '| display png:-'
set output "simRunTimeHistory.png"

set grid
set ylabel 'Simulation Run Time [s]'
set xdata time
set format x "%m/%d"
set timefmt "%m-%d-%Y"

set size 0.5,0.33 # for three plots
set label "1 week"   at screen 0.2,0.2
set label "2 weeks"  at screen 0.2,0.5
set label "Month"    at screen 0.2,0.8

set multiplot
TODAY  = system('date +%m-%d-%Y')
#__________________________________
# last week
set origin 0.0,0.0
FIRST_DAY = system('date -d " $(date +%d) -1week" +%m-%d-%Y')

set xlabel 'date'
set xrange [FIRST_DAY:TODAY]

plot 'simRunTimeHistory' using 1:2 with lines t ''


#__________________________________
# last 2 weeks
set origin 0.0,0.33
FIRST_DAY = system('date -d " $(date +%d) -2week" +%m-%d-%Y')
set xlabel ''
set xrange [FIRST_DAY:TODAY]

plot 'simRunTimeHistory' using 1:2 with lines t ''


#__________________________________
# last month
set origin 0.0,0.66
FIRST_DAY = system('date -d " $(date +%d) -4week" +%m-%d-%Y')
set xrange [FIRST_DAY:TODAY]

plot 'simRunTimeHistory' using 1:2 with lines t ''


#______________________________________________________________________
set label "3 months"  at screen 0.7,0.2
set label "6 months"  at screen 0.7,0.5
set label "1 year"    at screen 0.7,0.8
set label TODAY       at screen 0.93,0.95

#__________________________________
# 3months
set origin 0.5,0.0
FIRST_DAY = system('date -d " $(date +%d) -3month" +%m-%d-%Y')

set xlabel 'date'
set xrange [FIRST_DAY:TODAY]

plot 'simRunTimeHistory' using 1:2 with lines t ''


#__________________________________
# 6 months
set origin 0.5,0.33
FIRST_DAY = system('date -d " $(date +%d) -6month" +%m-%d-%Y')
set xlabel ''
set xrange [FIRST_DAY:TODAY]

plot 'simRunTimeHistory' using 1:2 with lines t ''


#__________________________________
# 1 year
set origin 0.5,0.66
FIRST_DAY = system('date -d " $(date +%d) -12 month" +%m-%d-%Y')
set xrange [FIRST_DAY:TODAY]

plot 'simRunTimeHistory' using 1:2 with lines t ''

set nomultiplot

#pause -1
