#!/usr/bin/perl
$srcroot = $ARGV[0];

#print $srcroot,"\n";
if($srcroot eq "") {
  print " You need to specify the path to SCIRun \n";
  print " For example:  /usr/sci/projects/Uintah/tester/Linux/SCIRun.062602 \n";
  print " now exiting \n";
  exit;
}

$filename = "$srcroot" . "/src/Packages/Uintah/StandAlone/sub.mk";

  print "\nNow working on $filename \n";
  open(IN, $filename);
  #__________________________________
  #  find and replace 1
  $search_str      = ".*sus.cc";
  $replacement_str = "SRCS := \$\(SRCDIR\)/sus.cc
SRCS := \$\(SRCS\) \$\(SRCDIR\)/FakeArches.cc";
  #__________________________________
  #  find and replace 2
   $search_str2     = ".
        Packages/Uintah/CCA/Components/Arches/Mixing .
        Packages/Uintah/CCA/Components/Arches/fortran .
        Packages/Uintah/CCA/Components/Arches/Radiation .
        Packages/Uintah/CCA/Components/Arches/Radiation/fortran";
  $replacement_str2 = "";
  #__________________________________
  #  find and replace 3
  $search_str3     = "
        Packages/Uintah/CCA/Components/Arches .
        Packages/Uintah/CCA/Components/MPMArches .";
  $replacement_str3 = "";
  
  @lines = <IN>;
  $text = join("", @lines);
  
  $changed1      = "false";
  if ($text =~ s/$search_str/$replacement_str/g) {
       $changed1 = "true";
  }
  $changed2      = "false";
  if ($changed1 == "true" &&$text =~ s/$search_str2/$replacement_str2/g) {
       $changed2 = "true";
  }
  $changed3      = "false";
  if ($changed2 == "true" &&$text =~ s/$search_str3/$replacement_str3/g) {
       $changed3 = "true";
  }

  close IN;
  print "Successful find and replace ",$changed1," ", $changed2," ", $changed3, "\n";
  
  
  if ($changed1 eq "true" && $changed2 eq "true" && $changed3 eq "true") {
    print " This sub.mk file was changed \n";
    open(OUT, "> " . $filename);
    print $filename .  "\n";
    print OUT $text;
    close OUT;
  } else {
    print " This sub.mk file WAS NOT changed \n";
  }

#.............................
# now take care of Components sub.mk
$filename = "$srcroot" . "/src/Packages/Uintah/CCA/Components/sub.mk";

  print "\nNow working on $filename \n";
  open(IN, $filename);

  $search_str       = "
	..SRCDIR./MPMArches .
	..SRCDIR./Arches .
	..SRCDIR./Arches/fortran .
	..SRCDIR./Arches/Mixing .
	..SRCDIR./Arches/Radiation .
	..SRCDIR./Arches/Radiation/fortran .";
  $replacement_str  = "";

  @lines = <IN>;
  $text = join("", @lines);
  
  $changed1 = "false";
  if ($text =~ s/$search_str/$replacement_str/g) {
    $changed1 = "true";
  }

  close IN;
  
  print "Successful find and replace ",$changed1, "\n";
  if ($changed1 eq "false") {
    print " The sub.mk file WAS NOT changed \n";
  }
  if ($changed1 eq "true") {
    open(OUT, "> " . $filename);    
    print $filename .  "\n";        
    print OUT $text;                
    close OUT;                      
  }

