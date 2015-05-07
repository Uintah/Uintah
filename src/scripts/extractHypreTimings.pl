#! /usr/bin/perl

#______________________________________________________________________
#  This script parses sus output and extracts timing information
#  from Hypre.  Specifically, it computes the average time spend in different 
#  parts of the hypreSolve task.  To activate the hypre timers you must compile
#  hypre with -enable-timing.  After that run a problem that utilizes hypre
#  and capture the output.  In the output, for each timestep, there should be
#  
#  =============================================
#  Uintah->Struct Interface:
#  =============================================
#  Setup grid:
#    wall clock time = 0.090000 seconds
#    wall MFLOPS     = 0.000000
#    cpu clock time  = 0.090000 seconds
#    cpu MFLOPS      = 0.000000
#
#  <snip>
#
#  =============================================
#  Setup phase times:
#  =============================================
#  PCG:
#    wall clock time = 0.490000 seconds
#    wall MFLOPS     = 0.000000
#    cpu clock time  = 0.500000 seconds
#    cpu MFLOPS      = 0.000000
#
#  PFMG:
#  <snip>
# 
# This script will output 3 different files
#     hypreTiming_all_HumanRead: the total and averages values for each timer (human readable)
#     hypreTiming_ave_HumanRead: the average values for each timer (human readable)
#     hypreTiming_ave:            the average values for certain timers
#
# The way this script has been implemented each 'timer' or field has both a 'solve' time and a 'setup' time,
# even if that timer doesn't have a solve/setup phase.  For eample, the timer 'setup X' only has setup time,
#  not a solve time.
#______________________________________________________________________

open(MYFILE,$ARGV[0]);
$start_timestep = $ARGV[1];

if (defined($start_timestep)){
  ;
} else {
  $start_timestep = 1;
}

my(@lines) = <MYFILE>;

my($line);
$num_lines = @lines;

#Arrays for storing the data
@mean_v;
@so_v;  # Solver only
@it_v;  # iteration count

# setup phase arrays
@pcgSetup;
@pfmgSetup;
@semi_interpSetup;
@semi_restrictSetup;
@point_relaxSetup;
@rbgsSetup;
@setupGrid;
@createMatrix;
@setupRHS;
@setupX;


# Solution
@pcg;
@pfmg;
@semi_interp;
@semi_restrict;
@point_relax;
@rbgs;


#__________________________________
sub print_line {
#    print "$i", " ", "$_";
}

$i= 0;
$found_solveonly = 0;

#__________________________________
#Parse the solve phase times and setup phase times
while ( $i < $num_lines) {
    $line=@lines[$i];
    $_=$line;

    while ($i < $num_lines) {
        
        if (/Setup phase times:/) {
          print "setup Phase Times:\n--------------------\n";
          $inSetup = 1;
          $inSolve = 0;
        }
        if (/Solve phase times:/) {
          print "Solve phase Times:\n--------------------\n";
          $inSetup = 0;
          $inSolve = 1;
        }
        
        if (/PCG/) {
            print "found the PCG\n";
            $i++;    
            $_ = @lines[$i];
            print_line;
            my @wallClockTime = split / /, @lines[$i];
            
            if( $inSetup ){
              push(@pcgSetup,$wallClockTime[6]); 
            }
            if( $inSolve ){
              push(@pcg,     $wallClockTime[6]);
            }
        } elsif (/PFMG/) {
            print "found the PFMG\n";
            # increment count and get the wall clock time
            $i++;    
            $_ = @lines[$i];
            print_line;
            my @wallClockTime = split / /, @lines[$i];
            
            if( $inSetup ){
              push(@pfmgSetup,$wallClockTime[6]); 
            }
            if( $inSolve ){
              push(@pfmg,     $wallClockTime[6]);
            }

        } elsif (/SemiInterp/) {
            print "found the SemiInterp\n";
            $i++;    
            $_ = @lines[$i];
            print_line;
            my @wallClockTime = split / /, @lines[$i];
            
            if( $inSetup ){
              push(@semi_interpSetup,$wallClockTime[6]); 
            }
            if( $inSolve ){
              push(@semi_interp,     $wallClockTime[6]);
            }
        } elsif (/SemiRestrict/) {
            print "found the SemiRestrict\n";
            $i++;    
            $_ = @lines[$i];
            print_line;
            my @wallClockTime = split / /, @lines[$i];

            if( $inSetup ){
              push(@semi_restrictSetup,$wallClockTime[6]); 
            }
            if( $inSolve ){
              push(@semi_restrict,     $wallClockTime[6]);
            }
        } elsif (/PointRelax/) {
            print "found the PointRelax\n";

            $i++;    
            $_ = @lines[$i];
            print_line;
            my @wallClockTime = split / /, @lines[$i];

            push(@point_relaxSetup,@fields[6]);
            
            if( $inSetup ){
              push(@point_relaxSetup,$wallClockTime[6]); 
            }
            if( $inSolve ){
              push(@point_relax,     $wallClockTime[6]);
            }
        } elsif (/RedBlackGS/) {
            print "found the RedBlackGS\n";
            $i++;    
            $_ = @lines[$i];
            print_line;
            my @wallClockTime = split / /, @lines[$i];

            if( $inSetup ){
              push(@rbgsSetup,$wallClockTime[6]); 
            }
            if( $inSolve ){
              push(@rbgs,     $wallClockTime[6]);
            }
        } elsif (/solve only:/) {
            print "found Solve line\n";
            $so=$_;
            
            my $where_begin = index($so,"(");
            my $so_string   = substr($so,$where_begin+1,-1); # last line
            @so_string = split / /, $so_string;

            push(@so_v,@so_string[2]);
            push(@it_v,@so_string[4]);
            $i++;    
            $_ = @lines[$i];

        } elsif (/Create matrix:/) {
            print "found creatMatrix line\n";
            $i++;    
            $_ = @lines[$i];
            print_line;
            my @wallClockTime = split / /, @lines[$i];
            push(@createMatrix,     $wallClockTime[6])
        } elsif (/Setup RHS:/) {
            print "found setup RHS line\n";
            $i++;    
            $_ = @lines[$i];
            print_line;
            my @wallClockTime = split / /, @lines[$i];
            push(@setupRHS,         $wallClockTime[6])
        } elsif (/Setup X:/) {
            print "found setup X line\n";
            $i++;    
            $_ = @lines[$i];
            print_line;
            my @wallClockTime = split / /, @lines[$i];
            push(@setupX,           $wallClockTime[6])
        }elsif (/Setup grid:/) {
            print "found setup Grid line\n";
            $i++;    
            $_ = @lines[$i];
            print_line;
            my @wallClockTime = split / /, @lines[$i];
            push(@setupGrid,     $wallClockTime[6])
        }else {
            $i++;
            $line = @lines[$i];
            $_=$line;    
        }
#            print "End of while loop\n";
        print_line;
    }
     print "\n----------------------------------\n";
    $i++;
}   

#______________________________________________________________________
#  compute the totals and averages for each field/timer

# put all the arrays into an array of arrays
#                        0         1        2   3        4          5             6        7       8         9           10      11
my @names     = ( qw/iterations totalSolve pcg pfmg semi_interp semi_restrict point_relax rbgs setupGrid createMatrix setupRHS setupX/ );
my @AOA_solve = ( [@it_v], [@so_v], [@pcg], [@pfmg], [@semi_interp], [@semi_restrict], [@point_relax], [@rbgs], [0], [0], [0], [0]);
my @AOA_setup = ( [0], [0], [@pcgSetup], [@pfmgSetup], [@semi_interpSetup], [@semi_restrictSetup], [@point_relaxSetup], [@rbgsSetup],
                  [@setupGrid], [@createMatrix], [@setupRHS], [@setupX] );
my $numArrays = @names;


# compute totals and averages:
my @totalSolve;
my @totalSetup;
my @aveSolve;
my @aveSetup;
$num_timesteps = @so_v;
$nTimesteps = ($num_timesteps-$start_timestep);

for ($i= 0; $i < $numArrays; $i ++) {
  @totalSetup[$i] = 0;
  @totalSolve[$i] = 0;
  for ($j = $start_timestep; $j < $num_timesteps; $j++) {
  
    @totalSolve[$i] += $AOA_solve[$i][$j];
    @totalSetup[$i] += $AOA_setup[$i][$j];
  }
  @aveSolve[$i] = @totalSolve[$i]/$nTimesteps;
  @aveSetup[$i] = @totalSetup[$i]/$nTimesteps;
  #print "$names[$i]:\t\t totalSolve: $totalSolve[$i], \t aveSolve: $aveSolve[$i], \t totalSetup $totalSetup[$i] \t aveSetup: $aveSetup[$i]\n";
  
}

#__________________________________
# write data to files
open(output,">hypreTiming_all_HumanRead");
for ($i= 0; $i < $numArrays; $i ++) {
  print output "$names[$i] totalSolveTime $totalSolve[$i] aveSolveTime $aveSolve[$i] totalSetupTime $totalSetup[$i] aveSetupTime $aveSetup[$i]\n";
}
close(output);
    
open(output,">hypreTiming_ave_HumanRead");
for ($i= 0; $i < $numArrays; $i ++) {
  print output "$names[$i] aveSolveTime $aveSolve[$i] aveSetupTime $aveSetup[$i]\n";
}
close(output);

# only write out data for certain columns
my @columns = ( 1, 2, 3, 7, 8, 9, 10, 11);
open(output,">hypreTiming_ave");
$nCols = @columns;

print output "# sol: average solve time,  set: averate setup time\n#";
for ($i= 0; $i < $nCols; $i ++) {
  print output " $names[ $columns[$i] ](sol, set), ";
}
print output "\n";
for ($i= 0; $i < $nCols; $i ++) {
  print output " $aveSolve[ $columns[$i] ] $aveSetup[ $columns[$i] ] ";
} 

print output "\n";

close(output);    
    
close (MYFILE); 
