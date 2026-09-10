#______________________________________________________________________
#  plotTimestepProfiles.gp
#
#  Template for plotting Uintah dataAnalysis/lineExtract-style profile data.
#  Reads one file per timestep from <uda>/<line>/L-0/timesteps/
#  ( e.g. "t00062" ) and writes one 4-panel PNG per PAGE, per timestep.
#
#  A "page" is one 2x2 window of 4 panels.  Define as many pages as you
#  like in the PAGE DEFINITIONS section below -- each produces its own
#  PNG:  <out_dir>/profiles_<page_suffix>_<timestep>.png
#
# X_line/                    << lineName
# `-- L-0
#     `-- timesteps
#         |-- t00001
#         |-- t00049
#         `-- t00097
#
#
#
#  Where X_line/L-0/timesteps/t00097 contains
# X_CC           Y_CC             Z_CC             Timestep         Time [s]        press_CC_0      delP_MassX_0    delPDilatate_0  rho_CC_1        temp_CC_1       vel_CC_1.x       vel_CC_1.y       vel_CC_1.z      
# 2.500000E-04     0.000000E+00     0.000000E+00     62               1.004886E-05    1.013250E+05    0.000000E+00    2.202184E-20    1.179295E-12    3.000000E+02    5.051760E-43     0.000000E+00     0.000000E+00    
# 7.500000E-04     0.000000E+00     0.000000E+00     62               1.004886E-05    1.013250E+05    0.000000E+00    4.223903E-18    1.179295E-12    3.000000E+02    2.580332E-42     0.000000E+00     0.000000E+00    
# 1.250000E-03     0.000000E+00     0.000000E+00     62               1.004886E-05    1.013250E+05    0.000000E+00    8.142830E-16    1.179295E-12    3.000000E+02    -3.912157E-26    0.000000E+00     0.000000E+00    
# 1.750000E-03     0.000000E+00     0.000000E+00     62               1.004886E-05    1.013250E+05    0.000000E+00    1.569935E-13    1.179295E-12    3.000000E+02    -9.970645E-26    0.000000E+00     0.000000E+00    
# 2.250000E-03     0.000000E+00     0.000000E+00     62               1.004886E-05    1.013250E+05    0.000000E+00    6.679376E-13    1.179295E-12    3.000000E+02    -1.355292E-17    0.000000E+00     0.000000E+00 

#
#  If the column layout differs (different number of materials, extra
#  variables, ...) update the PAGE DEFINITIONS below to match the
#  header line of the data files.
#
#  Usage:
#      gnuplot plotTimestepProfiles.gp
#      gnuplot -e "uda='StickMPMICE.uda.000'" -e "lineName='domain'"  plotTimestepProfiles.gp
#______________________________________________________________________

#______________________________________________________________________
#  USER SETTINGS
#______________________________________________________________________
if ( !exists( "uda" ) ) uda = "."

if ( !exists( "lineName" ) ) lineName = "."

data_dir  = uda . "/" . lineName . "/L-0/timesteps"
out_dir   = uda . "/" . lineName . "/plots"

col_x = 1        # what column is the X axis of the plot

#______________________________________________________________________
#  PAGE DEFINITIONS
#
#  n_pages        -- how many 4-panel PNGs to make per timestep
#  page_suffix[p] -- basename output filename for page p
#  cols[i]        -- the data column for panel i of page p, where
#                     i = (p-1)*4 + panel_number  ( panel_number = 1..4 )
#
#  Panel titles are NOT typed here -- they are pulled automatically from
#  the column descriptor in the data file's header line (see below).
#
#  Add another page by bumping n_pages, adding a page_suffix entry, and
#  adding 4 more cols entries.
#______________________________________________________________________
n_pages = 2

array page_suffix[n_pages]
array cols[n_pages*4]

page_suffix[1] = "A"
cols[1] = 6   # press_CC_0
cols[2] = 10  # temp_CC_1
cols[3] = 11  # vel_CC_1.x
cols[4] = 9   # rho_CC_1

page_suffix[2] = "B"
cols[5] = 7   # delP_MassX_0
cols[6] = 8   # delPDilatate_0
cols[7] = 12  # vel_CC_1.y
cols[8] = 13  # vel_CC_1.z

system( sprintf( "mkdir -p %s", out_dir ) )

#______________________________________________________________________
#  derive each panel's title from the column descriptor on the data
#  file's header line, e.g.:
#      # X_CC  Y_CC  Z_CC  Timestep  Time [s]  press_CC_0  delP_MassX_0 ...
#
#  Because the header starts with "#" and "Time [s]" is split into two
#  whitespace-separated words, the descriptor for data column N sits at
#  awk field ( N + header_offset ).
#______________________________________________________________________
header_offset = 2

file_list   = system( sprintf( "ls %s", data_dir ) )
sample_file = sprintf( "%s/%s", data_dir, word( file_list, 1 ) )

array titles[n_pages*4]

do for [idx=1:n_pages*4] {
    titles[idx] = system( sprintf( "awk 'NR==1{print $%d}' %s", cols[idx] + header_offset, sample_file ) )
}

#______________________________________________________________________
#  common plot settings
#______________________________________________________________________
set terminal pngcairo size 1200,900 noenhanced font "Helvetica,10"

set autoscale
set xtics
set ytics
set mxtics
set mytics
set grid xtics ytics
set pointsize 0.5
set xlabel titles[col_x]
unset key

#______________________________________________________________________
#  loop over every timestep file, and over every page, making 
#  4-panel plots per page per timestep
#______________________________________________________________________
do for [ts in file_list] {

    infile = sprintf( "%s/%s", data_dir, ts )
    print sprintf( "Working on %s", infile)

    t_val  = system( sprintf( "awk '!/^#/{print $5; exit}' %s", infile ) )

    do for [p=1:n_pages] {

        outfile = sprintf( "%s/%s_%s.png", out_dir, page_suffix[p], ts )

        set output outfile
        set multiplot layout 2,2 title sprintf( "timestep: %s      time = %s [s]", ts, t_val ) font ",12"

        do for [panel=1:4] {
            idx = ( p - 1 ) * 4 + panel
            set title titles[idx]
            plot infile using col_x:cols[idx] with linespoints
        }

        unset multiplot
    }
}

print sprintf( "Done.  PNGs written to %s/", out_dir )
