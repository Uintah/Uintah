
SRCDIR := Packages/Uintah/tools

FSPEC := $(SRCDIR)/genfspec.pl
FSPEC_ORIG := $(SRCDIR)/fspec.pl

%_fort.h: %.fspec $(FSPEC)
	$(FSPEC) $< $@

$(FSPEC): $(SRCDIR)/stamp-fspec

$(SRCDIR)/stamp-fspec: $(SRCTOP)/$(SRCDIR)/fspec.pl.in ${OBJTOP}/config.status
	@( CONFIG_FILES="Packages/Uintah/tools/fspec.pl" CONFIG_HEADERS="" ./config.status ) 1>&2
	@if cmp -s $(FSPEC) $(FSPEC_ORIG) 2>/dev/null; then echo "$(FSPEC) is unchanged"; else  mv $(FSPEC_ORIG) $(FSPEC); chmod +x $(FSPEC); echo "$(FSPEC) is changed"; fi
	echo timestamp > Packages/Uintah/tools/stamp-fspec
