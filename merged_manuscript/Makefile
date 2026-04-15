# ================================================================
#  Makefile for merged_manuscript/merged.tex
#  Usage:
#    make          → full build (pdflatex + bibtex + pdflatex x2)
#    make quick    → single pdflatex pass (faster, no bib update)
#    make clean    → remove build artifacts
#    make sync     → re-copy figures from source output directories
# ================================================================

TEX      = merged
PDFLATEX = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX   = bibtex

# Source output directories (relative to repo root, one level up)
AC1_OUT     = ../ac1/output
EULER_OUT   = ../euler_ews/output
EICU_OUT    = ../eicu_validation/output

.PHONY: all quick clean sync

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex references.bib figures/*.png
	$(PDFLATEX) $(TEX)
	$(BIBTEX)   $(TEX)
	$(PDFLATEX) $(TEX)
	$(PDFLATEX) $(TEX)

quick:
	$(PDFLATEX) $(TEX)

clean:
	rm -f $(TEX).aux $(TEX).bbl $(TEX).blg $(TEX).log \
	       $(TEX).out $(TEX).toc $(TEX).synctex.gz \
	       $(TEX).fdb_latexmk $(TEX).fls

# Re-copy figures from source output directories
sync:
	cp $(EULER_OUT)/fig_sampen_4way_forest.png     figures/fig3_forest.png
	cp $(EULER_OUT)/fig_euler_vs_ac1_forest.png    figures/fig5_vs_ac1_forest.png
	cp $(EULER_OUT)/fig_euler_delta.png            figures/figS3_delta_euler.png
	cp $(EULER_OUT)/fig_euler_timeseries.png       figures/fig2b_euler_timeseries.png
	cp $(EICU_OUT)/fig_t0strat_sensitivity.png     figures/fig4_t0strat.png
	cp $(AC1_OUT)/fig3_recovery.png                figures/figS1_recovery.png
	cp $(AC1_OUT)/figS1_subgroup.png               figures/figS2_subgroup.png
	cp $(AC1_OUT)/fig2_delta_boxplot.png           figures/figS3_delta_ac1.png
	cp $(AC1_OUT)/fig1_timeseries.png              figures/fig2a_ac1_timeseries.png
	@echo "Figures synced from source output directories."
