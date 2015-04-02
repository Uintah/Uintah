# uncomment below for post script output
#set terminal postscript color solid "Times-Roman" 14

# Latex fonts
set fontpath "/usr/share/texmf/fonts/type1/public/cm-super/"
set terminal postscript color landscape solid fontfile "sfrm1000.pfb" "SFRM1000" 14

set output "Error.ps"
 
set autoscale
set grid xtics ytics

set title " RMCRT:Adaptive Multi-level"
set xlabel "Fine Level Patches"
set ylabel "Normalized Div Q Error (L2norm)"
#set logscale x

set label "\nBurns and Christen Benchmark \n100 Rays per cell, 2 Levels\nHalo [1,1,1]\nResolution:\nCoarse level: 40,40,40 \nFine level: 80,80,80 \nDiv Q compared on L-1" at screen 0.45,0.4
#set key at screen 0.35,0.175

# these value were obtained from running RMCRT:1L on a 80^3 mesh.
x_1L=0.0451608
y_1L=0.0433944
z_1L=0.0450517


plot 'L2norm.dat' using 1:($2/x_1L) lc 1 lt 4 lw 4 pointsize 2 t 'X Error' with points, \
     'L2norm.dat' using 1:($3/y_1L) lc 2 lt 6 lw 4 pointsize 2 t 'Y Error' with points, \
     'L2norm.dat' using 1:($4/z_1L) lc 3 lt 8 lw 4 pointsize 2 t 'Z Error' with points

!ps2pdf Error.ps
