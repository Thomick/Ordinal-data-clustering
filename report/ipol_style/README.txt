LaTeX Class and Example for IPOL Articles
=========================================

Rafael Grompone von Gioi <grompone@cmla.ens-cachan.fr>
Nicolas Limare <nicolas.limare@cmla.ens-cachan.fr>
José-Luis Lisani Roca <joseluis.lisani@uib.es>


Files
-----

README.txt                This file
ipol_style_guide.pdf      IPOL LaTeX style guide and example
ipol_style_guide.tex      IPOL LaTeX source style guide and example
ipol.cls                  IPOL LaTeX class
article.bib               BibTeX bibliography for the style guide
siam.bst                  BibTeX style
ipol_class_manual.pdf     IPOL LaTeX class manual
ipol_class_manual.tex     IPOL LaTeX class manual source
ipol_logo.eps             IPOL logo needed when using ipol.cls with latex
ipol_logo.pdf             IPOL logo needed when using ipol.cls with pdflatex
siims_logo.eps            Logo needed for SIIMS companion article with latex
siims_logo.jpg            Logo needed for SIIMS companion article with pdflatex


Getting Started
---------------

There is no need for installation to use this class, just two files need to be
copied to the same directory were the LaTeX source files are. These files are
the class itself, 'ipol.cls', and the logo file, that can be either
'ipol_logo.eps' if you compile with LaTeX, or 'ipol_logo.pdf' if you compile
with pdfLaTeX. If not sure, just copy the three of these files. You may also
need 'siims_logo.eps' or 'siims_logo.jpg' when preparing a SIIMS companion
article.

The minimal example of use of this class is as follows:

\documentclass{ipol}
\begin{document}
\end{document}

It will only generate the IPOL header, including its logo, the words 'title'
and 'authors' where the title and authors should be placed, and a red
'PREPRINT' label including the compilation date. This example is useless but
can be used to test your system. The LaTeX source file 'ipol_style_guide.tex'
uses the IPOL class and provides an example of how to use it.

IPOL class is based on the standard 'article' class of LaTeX and it is used
essentially in the same way. There are two main restrictions: the layout must
not be changed and the title is not generated with the usual '\title',
'\author', '\date' and '\maketitle' commands; special IPOL commands must be
used instead.

For more information, IPOL authors should read and use the IPOL Style
Guide and Example, ipol_style_guide.pdf

For explanations about the class itself, see the IPOL LaTeX Class Manual,
ipol_class_manual.pdf


Thanks
------

Comments about errors, omission or suggestions are warmly appreciated.
