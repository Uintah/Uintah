-----------------------------------------------------------------------
Instructions for running mixing tables created by stand alone programs
-----------------------------------------------------------------------

1. Make sure that you change the mixing model in the UPS file. A typical
   block looks like this

           <Properties>
                <denUnderrelax>1.0</denUnderrelax>
                <ref_point>[3,3,3]</ref_point>
                <radiation>true</radiation>
                <mixing_model>StaticMixingTable</mixing_model>
                <StaticMixingTable>
                  <adiabatic>false</adiabatic>
                  <mixstatvars>1</mixstatvars>
                  <rxnvars>0</rxnvars>
                  <inputfile>nonadeqlb.tbl</inputfile>
                </StaticMixingTable>
            </Properties>

2. Put the non-adiabatic mixing table in the runs directory with file name
   "nonadeqlb.tbl"

3. Copy/move the SCIRun/SCIRun/src/Packages/Uintah/CCA/Components/Arches/
   Mixing/Stream.h.static to Stream.h (Caution: This has to be done before 
   making a build)

4. The non-adiabatic tables and ups files are located in the sub-directories 
   from this level, named by fuel types
  
----------------------------------------------------------------------------

-Compiled by Padmabhushana R. Desam
 
