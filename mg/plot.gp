set terminal pdf enhanced font "Arial,10"
set output 'error.pdf'
set xlabel "h"
set ylabel "L^{2} error"
set format x '2^{%L}'
set format y '2^{%L}'
set grid
set logscale x 2
set logscale y 2
set key bottom right

set linetype  1 lc rgb '#1B9E77' lw 2
set linetype  2 lc rgb '#D95F02' lw 2
set linetype  3 lc rgb '#7570B3' lw 2
set linetype  4 lc rgb '#E7298A' lw 2
set linetype  5 lc rgb '#66A61E' lw 2
set linetype  6 lc rgb '#E6AB02' lw 2
set linetype  7 lc rgb '#A6761D' lw 2
set linetype  8 lc rgb '#666666' lw 2
set linetype cycle  8

plot "error.dat" index 0 with linespoints ps 0.5 title "Polynomial order 1", \
"" index 1 with linespoints ps 0.5 title "Polynomial order 2"