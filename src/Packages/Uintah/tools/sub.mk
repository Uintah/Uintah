
FSPEC := $(OBJTOP)/Packages/Uintah/tools/fspec.pl
%_fort.h: %.fspec $(FSPEC)
	$(FSPEC) $< $@

