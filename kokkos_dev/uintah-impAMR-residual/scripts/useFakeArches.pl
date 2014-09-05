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

  print "\nWorking on $filename \n";
  open(IN, $filename);
  #__________________________________
  #  find and replace 1
   $search_str1     = ".
        Packages/Uintah/CCA/Components/Arches/Mixing    .
        Packages/Uintah/CCA/Components/Arches/fortran   .
        Packages/Uintah/CCA/Components/Arches/Radiation .
        Packages/Uintah/CCA/Components/Arches/Radiation/fortran";
   $replacement_str1 = "";

  #__________________________________
  #  find and replace 2
  $search_str2     = "
        Packages/Uintah/CCA/Components/Arches        .
        Packages/Uintah/CCA/Components/MPMArches     .";
  $replacement_str2 = "
        Packages/Uintah/CCA/Components/Dummy         \\";
  
  @lines = <IN>;
  $text = join("", @lines);
  
  $changed1      = "false";
  if ($text =~ s/$search_str1/$replacement_str1/g) {
       $changed1 = "true";
  }
  $changed2      = "false";
  if ($changed1 == "true" && $text =~ s/$search_str2/$replacement_str2/g) {
       $changed2 = "true";
  }

  close IN;
  print "   Find and replaces results: 1) ",$changed1,", 2) ", $changed2,"\n";
  
  
  if ($changed1 eq "true" && $changed2 eq "true" ) {
    print "\n";
    print "   The following sub.mk file was changed:\n";
    print "           " . $filename .  "\n";

    open(OUT, "> " . $filename);
    print OUT $text;
    close OUT;

  } else {
    print "   The sub.mk file WAS NOT changed.\n";
  }

###################################################################
# now take care of Components sub.mk

$filename = "$srcroot" . "/src/Packages/Uintah/CCA/Components/sub.mk";

  print "\nWorking on $filename \n";
  open(IN, $filename);

  #  find and replace 1
  $search_str       = "
        ..SRCDIR./MPMArches .
        ..SRCDIR./Arches .
        ..SRCDIR./Arches/fortran .
        ..SRCDIR./Arches/Mixing .
        ..SRCDIR./Arches/Radiation .
        ..SRCDIR./Arches/Radiation/fortran .";
  $replacement_str  = "
        \$\(SRCDIR\)/Dummy \\";

  @lines = <IN>;
  $text = join("", @lines);
  
  $changed1 = "false";
  if ($text =~ s/$search_str/$replacement_str/g) {
    $changed1 = "true";
  }

  close IN;
  
  print "   Find and replace results: 1) ",$changed1, "\n";
  if ( $changed1 eq "false" ) {
    print "   The sub.mk file WAS NOT changed.\n";
  } else {

    print "   The following sub.mk file was changed:\n";
    print "           " . $filename .  "\n";        

    open(OUT, "> " . $filename);    
    print OUT $text;                
    close OUT;                      
  }
  print "\n";


###################################################################
# now take care of Components/Parent sub.mk

$filename = "$srcroot" . "/src/Packages/Uintah/CCA/Components/Parent/sub.mk";

  print "\nWorking on $filename \n";
  open(IN, $filename);

  #  find and replace 1
  $search_str       = "
        Packages/Uintah/CCA/Components/Arches    .
        Packages/Uintah/CCA/Components/MPMArches .";

  $replacement_str  = "
        Packages/Uintah/CCA/Components/Dummy \\";

  @lines = <IN>;
  $text = join("", @lines);
  
  $changed1 = "false";
  if ($text =~ s/$search_str/$replacement_str/g) {
    $changed1 = "true";
  }

  close IN;
  
  print "   Find and replace results: 1) ",$changed1, "\n";
  if ( $changed1 eq "false" ) {
    print "   The sub.mk file WAS NOT changed.\n";
  } else {

    print "   The following sub.mk file was changed:\n";
    print "           " . $filename .  "\n";        

    open(OUT, "> " . $filename);    
    print OUT $text;                
    close OUT;                      
  }
  print "\n";


