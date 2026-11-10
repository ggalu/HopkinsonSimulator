# Build a HopkinsonSimulator report with latexmk; `make clean` removes the auxiliary files.
# The layout file hopkinson_report.sty lives in reports/template/ and is found through TEXINPUTS,
# so a report folder next to the template needs no copy of it.
#
# revision.tex is written before every build. It names the last commit that touched this folder
# or the template, and adds "with uncommitted changes" when either differs from that commit, so
# that a PDF can be traced to its source. A PDF built before the report is committed therefore
# says "uncommitted"; build it again after the commit. The file is rewritten only when its text
# changes, so that it does not force a LaTeX run by itself.
MAIN = $(firstword $(basename $(filter-out revision.tex,$(wildcard *.tex))))
export TEXINPUTS := ../template//:$(TEXINPUTS)

REV_HASH := $(shell git log -1 --format=%h -- . ../template 2>/dev/null)
REV_DIRTY := $(shell git status --porcelain -- . ../template 2>/dev/null)
REV_TEXT := $(if $(REV_HASH),$(REV_HASH)$(if $(REV_DIRTY), with uncommitted changes),uncommitted)

$(MAIN).pdf: $(MAIN).tex revision.tex $(wildcard figures/*) ../template/hopkinson_report.sty
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

revision.tex: FORCE
	@printf '%s\n' '\reportrevision{$(REV_TEXT)}' > revision.tex.new
	@cmp -s revision.tex.new revision.tex && rm revision.tex.new || mv revision.tex.new revision.tex

clean:
	latexmk -c $(MAIN).tex
	rm -f revision.tex

FORCE:
.PHONY: clean FORCE
