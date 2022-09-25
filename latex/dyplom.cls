\NeedsTeXFormat{LaTeX2e} % dla pełnej jasności
\ProvidesClass{dyplom} % jedyny niezbędny element klasy :-)

\LoadClass[12pt,oneside]{mwrep} % ładujemy klasę bazową, do opcji % drukowanie 2 stronne//todo twoside vs oneside
% podanych w dokumencie dodajemy 12pt

\RequirePackage[OT4]{polski}      % odpowiednik \RequirePackage
%\RequirePackage[utf8]{inputenc} % to właściwie powinno być na zewnątrz
\RequirePackage{array,graphicx}

\RequirePackage[a4paper,left=30mm,right=20mm,top=25mm,bottom=25mm]{geometry}

\RequirePackage{tgtermes}
\RequirePackage[absolute]{textpos} % to na potrzeby strony tytułowej
\RequirePackage[table,xcdraw]{xcolor} % na potrzeby strony tytułowej i nagłówków tabel
\def\ifundefined#1{\expandafter\ifx\csname#1\endcsname\relax}
%%% Tytularia
%\renewcommand*\sectsettings{}

% Głębokość numerowania sekcji /section /subsection /subsubsection ...
\setcounter{secnumdepth}{4}

%% chapter -- pierwszy stopień
\SetSectionFormatting[breakbefore,wholewidth]{chapter}
{56\p@}
{\FormatBlockHeading{\fontsize{14pt}{14pt}\selectfont}}
{24\p@}
%% section -- drugi stopień
\SetSectionFormatting{section}
{16\p@}
{\FormatHangHeading{\fontsize{13pt}{13pt}\selectfont}}
{6\p@}
%% subsection -- ewentualny trzeci poziom
\SetSectionFormatting{subsection}
{16\p@}
{\FormatHangHeading{\fontsize{12pt}{12pt}\selectfont}}
{6\p@}
\SetSectionFormatting{subsubsection}
{16\p@}
{\FormatHangHeading{\fontsize{12pt}{12pt}\selectfont}}
{6\p@}
%%% Floaty
%% Rysunki
\renewcommand\figurename{Rysunek}
\RequirePackage{microtype}
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{scrextend}
\RequirePackage{amsmath,amsfonts,amssymb,amsthm}
\RequirePackage{enumerate}
\RequirePackage{xurl}
%\PassOptionsToPackage{hyphens}{url}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage{afterpage}
\RequirePackage{multirow}
\RequirePackage{enumitem,etoolbox}
\RequirePackage{array}
\RequirePackage{float}
\RequirePackage{xstring,xifthen}
\RequirePackage{stackengine}
\RequirePackage{tikz}
\RequirePackage{arydshln}
\RequirePackage{frontpage}
%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[titletoc,title]{appendix}

% \renewcommand{\appendixtocname}{Dodatki}
% \renewcommand{\appendixpagename}{Dodatki}
\renewcommand{\appendixname}{Dodatek}

% pakiet do składu listingów w razie potrzeby można odblokować możliwość numerowania linii lub zmienić wielkość czcionki w listingu
% na windowsie trzeba zainstalować: pip install pygments a następnie kompilacje uruchamiać z opcją: --shell-escape
\RequirePackage[cache=false]{minted}
\setminted{breaklines,
frame=lines,
framesep=5mm,
baselinestretch=1,
fontsize=\small,
linenos
}

% nowe otoczenie do składania listingów
\RequirePackage{float}
%\newfloat{listing}{!htp}{lop}
%\floatname{listing}{Kod}
\RequirePackage{chngcntr}
%\counterwithin{listing}{chapter}
\usepackage{listings}
\renewcommand\lstlistingname{Listing}
\renewcommand\lstlistlistingname{Spis kodów źródłowych}
\RequirePackage{floatrow} % listing caption above listing

% patch wyrównujący spis listingów do lewego marginesu
%https://tex.stackexchange.com/questions/58469/why-are-listof-and-listoffigures-styled-differently
\makeatletter
\renewcommand*{\listof}[2]{%
\@ifundefined{ext@#1}{\float@error{#1}}{%
\expandafter\let\csname l@#1\endcsname \l@figure  % <- use layout of figure
\float@listhead{#2}%
\begingroup
\setlength\parskip{0pt plus 1pt}%               % <- or drop this line completely
\@starttoc{\@nameuse{ext@#1}}%
\endgroup}}
\makeatother

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\pagestyle{outer} % numery stron po lewej i prawej stronie
\RequirePackage[font=normalsize,justification=raggedright,singlelinecheck=false]{caption}

\setlength{\parindent}{7mm} % wcięcie w pierwszym wierszu akapitu
\widowpenalty=1000 % żeby nagłówki nie zostawały na dole strony
\clubpenalty=1000
\hyphenpenalty=5000

\usepackage[defaultlines=4,all]{nowidow}

\textfloatsep=10pt % odstęp pommiędzy rysunkiem, tabelą, listingiem a tekstem
\renewcommand{\baselinestretch}{1} % interlinia 1

\setlength{\skip\footins}{1cm}

\setlist[itemize]{noitemsep, topsep=0pt, partopsep=0pt}
\setlist[enumerate]{noitemsep, topsep=0pt, partopsep=0pt}
\renewcommand{\labelitemi}{$\bullet$}
\renewcommand{\labelitemii}{$\circ$}
\renewcommand{\labelitemiii}{$\circ$}
\renewcommand{\labelitemiv}{$\circ$}

\renewcommand{\arraystretch}{2}

\def\threedigits#1{%
\ifnum#1<100 0\fi
\ifnum#1<10 0\fi
\number#1}

\def\twodigits#1{%
\ifnum#1<10 0\fi
\number#1}

%https://tex.stackexchange.com/questions/120756/referencing-enumeration-items-by-number-and-name
\makeatletter
\newcommand*{\req}[1]{%
\item
\appto\@currentlabel{\textbf{ #1}}%
~#1}
\makeatother

\newcommand\capmystring[1]{\capmystringaux#1\relax}
\def\capmystringaux#1#2\relax{\uppercase{#1}#2}
\RequirePackage{truncate}
\makeatletter
\newcommand*{\atr}[2][160pt]{%
\item
\def\@currentlabel{\textbf{#2}}%
\truncate{#1}{~#2}}
\makeatother

\newcommand{\role}[1]{\emph{#1}}

\RequirePackage{bookmark}

\RequirePackage{tocloft}
\renewcommand\cftloftitlefont{\fontsize{14pt}{14pt}\bfseries}
\renewcommand\cftlottitlefont{\fontsize{14pt}{14pt}\bfseries}

%\image[source]{scale}{file path}{caption}{label}
\newcommand{\imagescale}[5][\ownwork]{
\noindent\begin{minipage}{\textwidth}
    \begin{figure}[H]
        \centering\includegraphics[scale=#2]{#3}
        \caption{#4}\label{fig:#5}
        \raggedright\source{#1}
    \end{figure}
\end{minipage}
}

\newcommand{\imagewidth}[5][\ownwork]{
    \noindent\begin{minipage}{\textwidth}
                 \begin{figure}[H]
                     \centering\includegraphics[width=#2\textwidth]{#3}
                     \caption{#4}\label{fig:#5}
                     \raggedright\source{#1}
                 \end{figure}
    \end{minipage}
}

\newcommand{\imagewide}[4][\ownwork]{
\noindent\begin{minipage}{\textwidth}
    \begin{figure}[H]
        \centering\includegraphics[width=.85\textwidth]{#2}
        \caption{#3}\label{fig:#4}
        \raggedright\source{#1}
    \end{figure}
\end{minipage}
}

\newcommand{\source}[1]{{\footnotesize Źródło: #1}}
\newcommand{\ownwork}{opracowanie własne}
\newcommand{\workbasedon}[1]{\ownwork~na podstawie: #1}

\newcommand{\longcite}[1]{\bibentry{#1}\nolink{#1}}

\newcommand{\inlinetodo}[1]{
\textcolor{red}{todo: (#1)}
}

\newcommand{\todo}[1]{
\inlinetodo{#1}\par
}

\newcommand{\cellred}[1]{
\cellcolor[HTML]{E74C3C}\textcolor{white}{#1}
}

\newcommand{\cellgreen}[1]{
\cellcolor[HTML]{18BC9C}\textcolor{white}{#1}
}

\newcommand{\cellgray}[1]{
\cellcolor[HTML]{DDDDDD}#1
}

\makeatletter
\newcommand{\listnormal}[2]{%
\clearpage
\begingroup
\phantomsection
\addcontentsline{toc}{chapter}{#2}
\let\ps@plain\ps@normal
\pagestyle{normal}
#1
\clearpage
\endgroup}
\makeatother
\makeatletter
\newcommand\mynobreakpar{\par\nobreak\@afterheading}
\makeatother

\RequirePackage{splitbib}

\AtBeginEnvironment{tabular}{\footnotesize}

\newcommand*{\nolink}[1]{%
        {\protect\NoHyper\cite{#1}\protect\endNoHyper}%
}

\usepackage[bottom]{footmisc}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.8}
\newcommand{\bargraph}[6][\ownwork]{
\begin{minipage}{\textwidth}
    \begin{figure}[H]
        \centering\begin{tikzpicture}
\begin{axis}[
height=#6,
%width=10cm,
xbar, xmin=0,
axis y line=left,
%yticklabel style={text width=5cm},
axis line style={-},
axis x line=bottom,
% xlabel={#2},
symbolic y coords={#2},
% ytick=data,
% nodes near coords,
nodes near coords={
    \pgfmathtruncatemacro\iszero{\originalvalue==0}
    \ifnum\iszero=0
        \pgfmathprintnumber{\originalvalue}$\,\%$
    \fi},
    visualization depends on={x \as \originalvalue},
    nodes near coords align=horizontal,
    xticklabel=\pgfmathprintnumber{\tick}\,$\%$,
ytick=data,
%xtick={10,20,30,40,50,60,70},
  enlarge y limits=0.1,
]
\addplot coordinates {#3};
\end{axis}
\end{tikzpicture}
        \caption{#4}\label{fig:#5}
        \raggedright\source{#1}
    \end{figure}
\end{minipage}
}

\RequirePackage{multicol}
\endinput % dla porządku
