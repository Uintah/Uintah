

set term png enhanced font '/usr/share/fonts/truetype/ttf-liberation/LiberationSans-Regular.ttf' 9 size 700,700
#set output '| display png:-'
set output "me.png"

set grid
set ylabel 'Bytes'
set xdata time
set format x "%m/%d"
set timefmt "%m-%d-%Y"

set size 1.0,0.33 # for three plots
set label "last week"     at screen 0.2,0.2
set label "last 2 weeks"  at screen 0.2,0.5
set label "last month"    at screen 0.2,0.8

set multiplot
TODAY  = system('date +%m-%d-%Y')
#__________________________________
# last week
set origin 0.0,0.0
FIRST_DAY = system('date -d " $(date +%d) -1week" +%m-%d-%Y')

set xlabel 'date'
set xrange [FIRST_DAY:TODAY]

plot 'advect.clean' using 1:2 with linespoints t 'highwater usage (bytes)'


#__________________________________
# last 2 weeks
set origin 0.0,0.33
FIRST_DAY = system('date -d " $(date +%d) -2week" +%m-%d-%Y')
set xlabel ''
set xrange [FIRST_DAY:TODAY]

plot 'advect.clean' using 1:2 with linespoints t ''


#__________________________________
# last month
set origin 0.0,0.66
FIRST_DAY = system('date -d " $(date +%d) -4week" +%m-%d-%Y')
set xrange [FIRST_DAY:TODAY]

plot 'advect.clean' using 1:2 with linespoints t ''

set nomultiplot


pause -1
